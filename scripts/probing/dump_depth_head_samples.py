#!/usr/bin/env python
"""Dump depth-head predictions against pooled CUT3R camera-depth targets."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llava.model.geometry.depth_supervision import build_depth_targets_from_point_maps  # noqa: E402
from scripts.diagnose_layerwise_spatial_hidden_scan import load_model, make_data_args, move_to_device  # noqa: E402


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def bool_config(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def make_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name=args.model_name,
        runtime_root=args.runtime_root,
        siglip_path=args.siglip_path,
        cut3r_weights=None,
        skip_spatial_tower_load=args.skip_spatial_tower_load,
        zero_spatial_features=False,
        attn_implementation=args.attn_implementation,
        mm_spatial_pool_stride=args.mm_spatial_pool_stride,
        pool_mode=args.pool_mode,
        train_data_json=args.data_path,
        image_folder=args.data_root,
        video_folder=args.data_root,
        spatial_feature_dir=args.spatial_features_root,
        spatial_features_subdir=args.spatial_features_subdir,
        frames_upbound=args.frames_upbound,
        add_time_instruction=args.add_time_instruction,
        seed=args.seed,
    )


def build_dataset(args: argparse.Namespace, tokenizer: Any, image_processor: Any):
    from llava import conversation as conversation_lib
    from llava.train.train import DataCollatorForSupervisedDataset, LazySupervisedDataset

    if "qwen_1_5" in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates["qwen_1_5"]

    runtime_args = make_runtime_args(args)
    data_args = make_data_args(runtime_args, image_processor)
    data_args.deterministic_data_order = True
    data_args.train_data_shuffle = False
    data_args.spatial_tower_type = "cut3r"
    data_args.spatial_features_root = args.spatial_features_root
    data_args.spatial_features_subdir = args.spatial_features_subdir
    data_args.require_spatial_features = True
    data_args.geometry_spatial_tower_type = "cut3r"
    data_args.geometry_spatial_features_root = args.geometry_spatial_features_root
    data_args.geometry_spatial_features_subdir = args.geometry_spatial_features_subdir
    data_args.require_geometry_spatial_features = True

    dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=args.data_path, data_args=data_args)
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dataset, collator


def first_frame_debug(depth_debug: dict[str, Any]) -> dict[str, Any] | None:
    samples = depth_debug.get("samples") if isinstance(depth_debug, dict) else None
    if not samples:
        return None
    frames = samples[0].get("frames") if isinstance(samples[0], dict) else None
    return frames[0] if frames else None


def save_frame_plot(
    output_path: Path,
    frame_id: int,
    grid_shape: tuple[int, int],
    frame_ids: torch.Tensor,
    pred_meter: torch.Tensor,
    gt_meter: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    mask = frame_ids == int(frame_id)
    h, w = int(grid_shape[0]), int(grid_shape[1])
    pred = pred_meter[mask].float().cpu().reshape(h, w)
    gt = gt_meter[mask].float().cpu().reshape(h, w)
    valid = valid_mask[mask].bool().cpu().reshape(h, w)
    err = (pred - gt).abs()

    def colorize(values: torch.Tensor, mask_values: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
        denom = max(float(vmax) - float(vmin), 1e-6)
        x = ((values - float(vmin)) / denom).clamp(0.0, 1.0)
        r = (255.0 * x).round()
        g = (255.0 * (1.0 - (2.0 * x - 1.0).abs()).clamp(0.0, 1.0)).round()
        b = (255.0 * (1.0 - x)).round()
        rgb = torch.stack([r, g, b], dim=-1).to(dtype=torch.uint8)
        rgb[~mask_values] = torch.tensor([32, 32, 32], dtype=torch.uint8)
        return rgb

    shared_values = torch.cat([pred[valid], gt[valid]])
    if shared_values.numel() == 0:
        return
    depth_min = float(shared_values.min().item())
    depth_max = float(shared_values.max().item())
    err_max = float(err[valid].max().item()) if valid.any() else 1.0
    pred_rgb = colorize(pred, valid, depth_min, depth_max)
    gt_rgb = colorize(gt, valid, depth_min, depth_max)
    err_rgb = colorize(err, valid, 0.0, max(err_max, 1e-6))
    separator = torch.full((h, 2, 3), 255, dtype=torch.uint8)
    image = torch.cat([pred_rgb, separator, gt_rgb, separator, err_rgb], dim=1)

    # Binary PPM keeps the artifact image-like without pulling in matplotlib/PIL.
    with output_path.open("wb") as f:
        f.write(f"P6\n{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii"))
        f.write(image.contiguous().numpy().tobytes())


def dump_one_sample(
    args: argparse.Namespace,
    model: torch.nn.Module,
    collator: Any,
    dataset: Any,
    dataset_index: int,
    output_dir: Path,
    sample_ordinal: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    item = dataset[dataset_index]
    batch = collator([item])
    batch = move_to_device(batch, device, dtype)

    prepare_fn = getattr(model, "prepare_inputs_labels_for_multimodal", None)
    if prepare_fn is None:
        raise RuntimeError("Model does not expose prepare_inputs_labels_for_multimodal().")
    if "return_visual_metadata" not in inspect.signature(prepare_fn).parameters:
        raise RuntimeError("prepare_inputs_labels_for_multimodal() lacks return_visual_metadata support.")

    with torch.no_grad():
        prepared = prepare_fn(
            input_ids=batch["input_ids"],
            position_ids=None,
            attention_mask=batch["attention_mask"],
            past_key_values=None,
            labels=batch.get("labels"),
            images=batch["images"],
            spatial_features=batch.get("spatial_features"),
            point_maps=batch.get("point_maps"),
            modalities=batch.get("modalities"),
            image_sizes=batch.get("image_sizes"),
            return_visual_metadata=True,
            geometry_spatial_features=batch.get("geometry_spatial_features"),
        )
        input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, _labels, visual_metadata = prepared
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

    sequence_hidden = outputs[0]
    visual_hidden = model._gather_aux_visual_hidden(sequence_hidden, visual_metadata, aux_name="DepthSample")
    payloads = model._select_depth_point_map_payloads(
        batch.get("spatial_features"),
        batch.get("point_maps"),
        batch.get("geometry_spatial_features"),
    )
    if payloads is None:
        raise RuntimeError("No camera-space point-map payloads found for depth sample.")

    depth_gt_log, depth_valid_mask, depth_debug = build_depth_targets_from_point_maps(
        payloads,
        visual_metadata,
        depth_point_map_key=str(getattr(model.config, "depth_point_map_key", "point_maps_cam")),
        use_geometry_confidence_mask=bool_config(getattr(model.config, "use_geometry_confidence_mask", True), True),
        depth_conf_threshold=float(getattr(model.config, "depth_conf_threshold", 0.0)),
        depth_max_gt=float(getattr(model.config, "depth_max_gt", 20.0)),
        depth_allow_generic_camera_assumed=bool_config(getattr(model.config, "depth_allow_generic_camera_assumed", False), False),
        depth_allow_tensor_camera_assumed=bool_config(getattr(model.config, "depth_allow_tensor_camera_assumed", False), False),
    )
    depth_gt_log = depth_gt_log.to(device=visual_hidden.device, dtype=visual_hidden.dtype)
    depth_valid_mask = depth_valid_mask.to(device=visual_hidden.device, dtype=torch.bool)

    depth_head = getattr(model, "depth_head", None)
    if depth_head is None:
        depth_head = model.initialize_depth_head(device=visual_hidden.device, dtype=visual_hidden.dtype)
    depth_pred_log = depth_head(visual_hidden)

    valid = depth_valid_mask[0] & torch.isfinite(depth_gt_log[0]) & torch.isfinite(depth_pred_log[0])
    pred_meter = torch.expm1(depth_pred_log[0].float()).detach().cpu()
    gt_meter = torch.expm1(depth_gt_log[0].float()).detach().cpu()
    pred_log = depth_pred_log[0].float().detach().cpu()
    gt_log = depth_gt_log[0].float().detach().cpu()
    valid_cpu = valid.detach().cpu()

    if int(valid_cpu.sum().item()) == 0:
        raise RuntimeError("Depth target builder returned zero valid visual tokens.")

    sample_dir = output_dir / f"sample_{sample_ordinal:03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    raw_item = dataset.list_data_dict[dataset_index]
    metadata = visual_metadata[0]
    frame_ids = metadata["visual_frame_ids"].detach().cpu().to(dtype=torch.long)

    csv_path = sample_dir / "tokens.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["token_idx", "frame_id", "valid", "pred_log", "gt_log", "pred_meter", "gt_meter", "abs_error_meter"],
        )
        writer.writeheader()
        for token_idx in range(int(pred_meter.shape[0])):
            abs_err = abs(float(pred_meter[token_idx]) - float(gt_meter[token_idx]))
            writer.writerow({
                "token_idx": token_idx,
                "frame_id": int(frame_ids[token_idx].item()) if token_idx < frame_ids.numel() else -1,
                "valid": int(bool(valid_cpu[token_idx].item())),
                "pred_log": float(pred_log[token_idx]),
                "gt_log": float(gt_log[token_idx]),
                "pred_meter": float(pred_meter[token_idx]),
                "gt_meter": float(gt_meter[token_idx]),
                "abs_error_meter": abs_err,
            })

    mae_meter = (pred_meter[valid_cpu] - gt_meter[valid_cpu]).abs().mean().item()
    mae_log = (pred_log[valid_cpu] - gt_log[valid_cpu]).abs().mean().item()
    summary = {
        "dataset_index": int(dataset_index),
        "sample_ordinal": int(sample_ordinal),
        "id": raw_item.get("id"),
        "video": raw_item.get("video"),
        "num_tokens": int(pred_meter.shape[0]),
        "num_valid_tokens": int(valid_cpu.sum().item()),
        "valid_token_ratio": float(valid_cpu.float().mean().item()),
        "mae_meter": float(mae_meter),
        "mae_log": float(mae_log),
        "pred_meter_mean": float(pred_meter[valid_cpu].mean().item()),
        "gt_meter_mean": float(gt_meter[valid_cpu].mean().item()),
        "csv": str(csv_path),
        "depth_debug": jsonable(depth_debug),
    }

    frame_info = first_frame_debug(depth_debug)
    if frame_info is not None:
        frame_id = int(frame_info["frame_id"])
        grid_shape = tuple(int(x) for x in frame_info["target_grid_shape"])
        plot_path = sample_dir / f"frame_{frame_id:02d}_pred_gt.ppm"
        save_frame_plot(plot_path, frame_id, grid_shape, frame_ids, pred_meter, gt_meter, valid_cpu)
        summary["plot"] = str(plot_path)
        summary["plot_frame_id"] = frame_id
        summary["plot_grid_shape"] = list(grid_shape)

    summary_path = sample_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/cut3r_depth_loss_43817021")
    parser.add_argument("--model-base", default="/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/LLaVA-NeXT-Video-7B-Qwen2")
    parser.add_argument("--model-name", default="vlm-3r-llava-qwen2-lora")
    parser.add_argument("--siglip-path", default="/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/siglip-so400m-patch14-384")
    parser.add_argument("--data-path", default="scripts/VLM_3R/vsibench_data.yaml")
    parser.add_argument("--data-root", default="/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r")
    parser.add_argument("--spatial-features-root", default="/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r")
    parser.add_argument("--spatial-features-subdir", default="spatial_features")
    parser.add_argument("--geometry-spatial-features-root", default="/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r")
    parser.add_argument("--geometry-spatial-features-subdir", default="spatial_features_points")
    parser.add_argument("--output-dir", default="/leonardo_scratch/fast/EUHPC_D32_006/eval/depth_head_samples/cut3r_depth_loss_43817021")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--frames-upbound", type=int, default=32)
    parser.add_argument("--mm-spatial-pool-stride", type=int, default=2)
    parser.add_argument("--pool-mode", choices=["bilinear", "average", "max"], default="bilinear")
    parser.add_argument("--add-time-instruction", type=str2bool, default=None)
    parser.add_argument("--skip-spatial-tower-load", type=str2bool, default=True)
    parser.add_argument("--runtime-root", default=str(REPO_ROOT / ".offline_runtime"))
    args = parser.parse_args()

    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)

    runtime_args = make_runtime_args(args)
    print("[INFO] Loading model", flush=True)
    tokenizer, model, image_processor = load_model(runtime_args, device, dtype)
    model.eval()
    print("[INFO] Building dataset", flush=True)
    dataset, collator = build_dataset(args, tokenizer, image_processor)
    print(f"[INFO] Dataset ready: {len(dataset)} samples", flush=True)

    summaries = []
    skip_reasons: dict[str, int] = {}
    for dataset_index in range(min(len(dataset), int(args.max_candidates))):
        if len(summaries) >= int(args.num_samples):
            break
        try:
            summary = dump_one_sample(
                args,
                model,
                collator,
                dataset,
                dataset_index,
                output_dir,
                len(summaries),
                device,
                dtype,
            )
        except Exception as exc:
            reason = type(exc).__name__
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            print(f"[WARN] Skipping dataset index {dataset_index}: {reason}: {exc}", flush=True)
            continue
        summaries.append(summary)
        print(f"[INFO] Wrote sample {summary['sample_ordinal']} from dataset index {dataset_index}: MAE={summary['mae_meter']:.4f}m", flush=True)

    manifest = {
        "model_path": args.model_path,
        "num_requested": int(args.num_samples),
        "num_written": len(summaries),
        "output_dir": str(output_dir),
        "skip_reasons": skip_reasons,
        "samples": summaries,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    if len(summaries) < int(args.num_samples):
        raise RuntimeError(f"Only wrote {len(summaries)} samples out of requested {args.num_samples}; see {manifest_path}")
    print(f"[INFO] Done. Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
