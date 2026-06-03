#!/usr/bin/env python
"""Extract cached frame-level features for the VLM-3R depth probing experiment."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from depth_probe_common import (
    DEFAULT_DATA_YAML,
    DEFAULT_FAST_FEATURE_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_POINT_MAPS_SUBDIR,
    DEFAULT_SPATIAL_FEATURES_SUBDIR,
    LLM_LAYERS,
    MODEL_PRESETS,
    PRE_LLM_FEATURES,
    coerce_cache_dtype,
    depth_from_point_maps,
    downsample_depth_to_grid,
    frame_depth_metadata,
    grid_shape_for_frame,
    load_frame_records,
    load_point_map_sidecar,
    read_json,
    reshape_tokens_to_grid,
    resolve_sidecar_path,
    select_point_maps,
    torch_dtype_from_name,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnose_layerwise_spatial_hidden_scan import load_model, make_data_args, move_to_device  # noqa: E402


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def json_ready_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu()
        elif isinstance(value, (list, tuple)):
            out[key] = list(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
    return out


def capture_hook(name: str, captured: dict[str, torch.Tensor]):
    def _hook(_module, _inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} hook expected tensor output, got {type(value)}")
        captured[name] = value.detach().cpu()

    return _hook


def register_pre_llm_hooks(model: torch.nn.Module, model_label: str, captured: dict[str, torch.Tensor]):
    if model_label == "zero_spatial":
        return []
    base = model.get_model()
    handles = []
    handles.append(base.get_fusion_block().register_forward_hook(capture_hook("fusion_output", captured)))
    handles.append(base.mm_projector.register_forward_hook(capture_hook("projected_features", captured)))
    return handles


def normalize_captured_video_tokens(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    *,
    num_frames: int,
    target_grid_shape: tuple[int, int],
) -> torch.Tensor:
    tensor = tensor.float()
    if tensor.ndim == 2:
        if tensor.shape[0] % int(num_frames) != 0:
            raise ValueError(f"Cannot unflatten captured tensor {tuple(tensor.shape)} into {num_frames} frames")
        tensor = tensor.reshape(int(num_frames), tensor.shape[0] // int(num_frames), tensor.shape[-1])
    if tensor.ndim != 3:
        raise ValueError(f"Expected captured tensor [F,N,D] or [F*N,D], got {tuple(tensor.shape)}")
    if tensor.shape[0] != int(num_frames):
        raise ValueError(f"Captured frame count mismatch: tensor={tuple(tensor.shape)} num_frames={num_frames}")

    target_tokens = int(target_grid_shape[0]) * int(target_grid_shape[1])
    if int(tensor.shape[1]) == target_tokens:
        return tensor

    pooled = None
    get_2d_pool = getattr(model, "get_2dPool", None)
    if callable(get_2d_pool):
        try:
            pooled = get_2d_pool(tensor)
        except Exception:
            pooled = None
    if pooled is not None and int(pooled.shape[1]) == target_tokens:
        return pooled.float()

    source_side = int(np.sqrt(int(tensor.shape[1])))
    if source_side * source_side != int(tensor.shape[1]):
        raise ValueError(
            f"Cannot resize non-square captured token grid {tuple(tensor.shape)} to {target_grid_shape}"
        )
    x = tensor.reshape(tensor.shape[0], source_side, source_side, tensor.shape[-1]).permute(0, 3, 1, 2)
    x = F.interpolate(x, size=target_grid_shape, mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).reshape(tensor.shape[0], target_tokens, tensor.shape[-1]).contiguous()


def selected_frame_hidden_grids(
    hidden: torch.Tensor,
    metadata: dict[str, Any],
    selected_frames: list[int],
) -> dict[int, torch.Tensor]:
    visual_indices = metadata["visual_token_indices"].to(device=hidden.device)
    frame_ids = metadata["visual_frame_ids"].to(device=hidden.device)
    out: dict[int, torch.Tensor] = {}
    for frame_idx in selected_frames:
        indices = visual_indices[frame_ids == int(frame_idx)]
        if indices.numel() == 0:
            raise RuntimeError(f"No visual tokens found for selected frame {frame_idx}")
        grid_shape = grid_shape_for_frame(metadata, int(frame_idx), token_count=int(indices.numel()))
        out[int(frame_idx)] = reshape_tokens_to_grid(hidden[0, indices].detach().float().cpu(), grid_shape)
    return out


def save_frame_outputs(
    *,
    output_root: Path,
    model_label: str,
    frame_record: dict[str, Any],
    llm_features: dict[str, torch.Tensor],
    pre_llm_features: dict[str, torch.Tensor],
    gt_depth: torch.Tensor,
    gt_valid: torch.Tensor,
    metadata: dict[str, Any],
    cache_dtype: torch.dtype,
) -> None:
    fsid = str(frame_record["frame_sample_id"])
    (output_root / "gt_depth").mkdir(parents=True, exist_ok=True)
    (output_root / "metadata").mkdir(parents=True, exist_ok=True)
    gt_path = output_root / "gt_depth" / f"frame_{fsid}.pt"
    metadata_path = output_root / "metadata" / f"frame_{fsid}.pt"
    if not gt_path.exists():
        torch.save(gt_depth.float().cpu(), gt_path)

    if not metadata_path.exists():
        meta_payload = dict(frame_record)
        meta_payload.update(metadata)
        meta_payload["gt_valid_mask"] = gt_valid.cpu()
        meta_payload["gt_depth_map_downsampled"] = gt_depth.float().cpu()
        torch.save(meta_payload, metadata_path)

    llm_dir = output_root / "features" / model_label / "llm_layers"
    llm_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {key: coerce_cache_dtype(value, cache_dtype) for key, value in llm_features.items()},
        llm_dir / f"frame_{fsid}.pt",
    )
    for feature_name, value in pre_llm_features.items():
        feature_dir = output_root / "features" / model_label / feature_name
        feature_dir.mkdir(parents=True, exist_ok=True)
        torch.save(coerce_cache_dtype(value, cache_dtype), feature_dir / f"frame_{fsid}.pt")


def output_complete(output_root: Path, model_label: str, frame_sample_id: str, include_pre_llm: bool) -> bool:
    if not (output_root / "gt_depth" / f"frame_{frame_sample_id}.pt").exists():
        return False
    if not (output_root / "metadata" / f"frame_{frame_sample_id}.pt").exists():
        return False
    if not (output_root / "features" / model_label / "llm_layers" / f"frame_{frame_sample_id}.pt").exists():
        return False
    if include_pre_llm:
        for feature_name in PRE_LLM_FEATURES:
            if not (output_root / "features" / model_label / feature_name / f"frame_{frame_sample_id}.pt").exists():
                return False
    return True


def build_dataset(args: argparse.Namespace, tokenizer: Any, image_processor: Any):
    from llava.train.train import DataCollatorForSupervisedDataset, LazySupervisedDataset

    data_args = make_data_args(args, image_processor)
    data_args.deterministic_data_order = True
    data_args.train_data_shuffle = False
    data_args.spatial_features_root = args.feature_root
    data_args.spatial_features_subdir = args.spatial_features_subdir
    data_args.spatial_tower_type = "cut3r"
    data_args.require_spatial_features = True
    dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=args.train_data_json, data_args=data_args)
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    by_video = {}
    for idx, item in enumerate(dataset.list_data_dict):
        video = item.get("video")
        if video is not None and video not in by_video:
            by_video[str(video)] = idx
    return dataset, collator, by_video


def extract_for_video(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    collator: Any,
    dataset: Any,
    dataset_index: int,
    video_record: dict[str, Any],
    selected_frames: list[int],
    captured: dict[str, torch.Tensor],
    cache_dtype: torch.dtype,
    device: torch.device,
    model_dtype: torch.dtype,
) -> None:
    output_root = Path(args.output_root)
    item = dataset[dataset_index]

    point_maps_path = resolve_sidecar_path(
        str(video_record["video_path"]),
        Path(args.feature_root),
        args.point_maps_subdir,
    )
    if point_maps_path is None:
        raise FileNotFoundError(f"Missing point-map sidecar for {video_record['video_path']}")
    point_payload = load_point_map_sidecar(point_maps_path)
    point_maps, point_key, depth_mode = select_point_maps(point_payload, allow_euclidean_depth=args.allow_euclidean_depth)
    depths = depth_from_point_maps(point_maps, depth_mode)
    if point_maps.shape[-1] == 3:
        model_point_maps = point_maps.permute(0, 3, 1, 2).contiguous()
    else:
        model_point_maps = point_maps.contiguous()
    item["point_maps"] = model_point_maps.float()

    batch = collator([item])
    batch = move_to_device(batch, device, model_dtype)
    prepare_fn = getattr(model, "prepare_inputs_labels_for_multimodal", None)
    if prepare_fn is None:
        raise RuntimeError("Model does not expose prepare_inputs_labels_for_multimodal().")
    if "return_visual_metadata" not in inspect.signature(prepare_fn).parameters:
        raise RuntimeError("prepare_inputs_labels_for_multimodal() lacks return_visual_metadata support.")

    captured.clear()
    with torch.no_grad():
        prepared = prepare_fn(
            input_ids=batch["input_ids"],
            position_ids=None,
            attention_mask=batch["attention_mask"],
            past_key_values=None,
            labels=None,
            images=batch["images"],
            spatial_features=batch.get("spatial_features"),
            point_maps=batch.get("point_maps"),
            modalities=batch.get("modalities"),
            image_sizes=batch.get("image_sizes"),
            return_visual_metadata=True,
        )
    input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, _labels, visual_metadata = prepared
    metadata = visual_metadata[0]
    visual_indices = metadata["visual_token_indices"]
    frame_ids = metadata["visual_frame_ids"]
    if visual_indices.numel() == 0:
        raise RuntimeError("No visual tokens returned by metadata")
    available_frames = {int(x) for x in frame_ids.detach().cpu().tolist()}
    missing = [idx for idx in selected_frames if int(idx) not in available_frames]
    if missing:
        raise RuntimeError(f"Selected frame ids not present in visual metadata: {missing}")

    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states")

    num_frames = int(max(available_frames)) + 1
    target_grid_shape = grid_shape_for_frame(
        metadata,
        int(selected_frames[0]),
        token_count=int((frame_ids == int(selected_frames[0])).sum().item()),
    )

    normalized_pre_llm: dict[str, torch.Tensor] = {}
    if args.model_label != "zero_spatial":
        for feature_name in PRE_LLM_FEATURES:
            if feature_name not in captured:
                raise RuntimeError(f"{feature_name} hook did not capture an output")
            normalized_pre_llm[feature_name] = normalize_captured_video_tokens(
                model,
                captured[feature_name],
                num_frames=num_frames,
                target_grid_shape=target_grid_shape,
            )

    llm_by_layer: dict[str, dict[int, torch.Tensor]] = {}
    for layer in LLM_LAYERS:
        hidden_index = int(layer) + 1
        if hidden_index >= len(hidden_states):
            raise ValueError(f"Requested layer {layer}, but hidden_states length is {len(hidden_states)}")
        llm_by_layer[f"layer_{layer}"] = selected_frame_hidden_grids(
            hidden_states[hidden_index],
            metadata,
            selected_frames,
        )

    for frame_record in video_record["frames"]:
        frame_idx = int(frame_record["frame_index"])
        grid_shape = grid_shape_for_frame(
            metadata,
            frame_idx,
            token_count=int((frame_ids == frame_idx).sum().item()),
        )
        gt_depth, gt_valid = downsample_depth_to_grid(depths[frame_idx], grid_shape)
        depth_meta = frame_depth_metadata(gt_depth, gt_valid)
        selected_indices = metadata["visual_token_indices"][metadata["visual_frame_ids"] == frame_idx].detach().cpu()
        frame_metadata = {
            "sample_id": frame_record["frame_sample_id"],
            "model_label": args.model_label,
            "num_frames": 1,
            "source_video_num_frames": num_frames,
            "visual_grid_shape": tuple(int(x) for x in grid_shape),
            "visual_token_indices": selected_indices,
            "point_maps_path": str(point_maps_path),
            "point_map_key": point_key,
            "depth_mode": depth_mode,
            **depth_meta,
            "visual_metadata": json_ready_metadata(metadata),
        }
        llm_features = {layer_name: frames[frame_idx] for layer_name, frames in llm_by_layer.items()}
        pre_llm_features = {
            feature_name: reshape_tokens_to_grid(value[frame_idx].float().cpu(), grid_shape)
            for feature_name, value in normalized_pre_llm.items()
        }
        save_frame_outputs(
            output_root=output_root,
            model_label=args.model_label,
            frame_record={**video_record, **frame_record},
            llm_features=llm_features,
            pre_llm_features=pre_llm_features,
            gt_depth=gt_depth,
            gt_valid=gt_valid,
            metadata=frame_metadata,
            cache_dtype=cache_dtype,
        )
    captured.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-label", choices=sorted(MODEL_PRESETS), required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-base", default="/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/LLaVA-NeXT-Video-7B-Qwen2")
    parser.add_argument("--model-name", default="vlm-3r-llava-qwen2-lora")
    parser.add_argument("--sample-indices", default=str(DEFAULT_OUTPUT_ROOT / "sample_indices.json"))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--train-data-json", "--data-yaml", dest="train_data_json", default=str(DEFAULT_DATA_YAML))
    parser.add_argument("--feature-root", default=str(DEFAULT_FAST_FEATURE_ROOT))
    parser.add_argument("--spatial-features-subdir", default=DEFAULT_SPATIAL_FEATURES_SUBDIR)
    parser.add_argument("--point-maps-subdir", default=DEFAULT_POINT_MAPS_SUBDIR)
    parser.add_argument("--image-folder", default=str(DEFAULT_FAST_FEATURE_ROOT))
    parser.add_argument("--video-folder", default=str(DEFAULT_FAST_FEATURE_ROOT))
    parser.add_argument("--frames-upbound", type=int, default=32)
    parser.add_argument("--mm-spatial-pool-stride", type=int, default=2)
    parser.add_argument("--pool-mode", choices=["bilinear", "average", "max"], default="bilinear")
    parser.add_argument("--add-time-instruction", type=str2bool, default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--cache-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--runtime-root", default=str(REPO_ROOT / ".offline_runtime"))
    parser.add_argument("--siglip-path", default=None)
    parser.add_argument("--cut3r-weights", default=None)
    parser.add_argument("--skip-spatial-tower-load", type=str2bool, default=None)
    parser.add_argument("--allow-euclidean-depth", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.model_path = args.model_path or MODEL_PRESETS[args.model_label]
    args.spatial_feature_dir = args.feature_root
    args.zero_spatial_features = args.model_label == "zero_spatial"
    if args.skip_spatial_tower_load is None:
        args.skip_spatial_tower_load = True

    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    try:
        torch._dynamo.disable()
    except Exception:
        pass

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "features" / args.model_label / "extraction_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model_dtype = torch_dtype_from_name(args.dtype)
    cache_dtype = torch_dtype_from_name(args.cache_dtype)

    print(f"[INFO] Loading model {args.model_label}: {args.model_path}")
    tokenizer, model, image_processor = load_model(args, device, model_dtype)
    if args.model_label == "zero_spatial":
        model.config.zero_spatial_features = True
    model.eval()

    print("[INFO] Building dataset")
    dataset, collator, by_video = build_dataset(args, tokenizer, image_processor)
    if args.shard_count < 1:
        raise ValueError(f"--shard-count must be >= 1, got {args.shard_count}")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError(
            f"--shard-index must be in [0, shard_count), got {args.shard_index}/{args.shard_count}"
        )

    sample_payload = read_json(Path(args.sample_indices))
    all_videos = list(sample_payload.get("videos", []))
    if args.limit_videos is not None:
        all_videos = all_videos[: int(args.limit_videos)]
    videos = [
        video
        for global_index, video in enumerate(all_videos)
        if global_index % int(args.shard_count) == int(args.shard_index)
    ]
    print(
        f"[INFO] Shard {args.shard_index}/{args.shard_count}: "
        f"{len(videos)}/{len(all_videos)} videos selected from fixed sample_indices",
        flush=True,
    )

    captured: dict[str, torch.Tensor] = {}
    handles = register_pre_llm_hooks(model, args.model_label, captured)
    try:
        with log_path.open("a", encoding="utf-8") as log_f:
            for idx, video in enumerate(videos):
                video_path = str(video["video_path"])
                selected_frames = [int(frame["frame_index"]) for frame in video["frames"]]
                include_pre_llm = args.model_label != "zero_spatial"
                if args.resume and all(
                    output_complete(output_root, args.model_label, str(frame["frame_sample_id"]), include_pre_llm)
                    for frame in video["frames"]
                ):
                    print(f"[SKIP] {idx + 1}/{len(videos)} {video_path} already complete")
                    continue
                if video_path not in by_video:
                    payload = {"ok": False, "video_path": video_path, "error": "video not found in dataset"}
                    print(json.dumps(payload), file=log_f, flush=True)
                    print(f"[WARN] {payload}", file=sys.stderr)
                    continue
                try:
                    print(f"[INFO] {idx + 1}/{len(videos)} extracting {video_path} frames={selected_frames}")
                    extract_for_video(
                        args=args,
                        model=model,
                        collator=collator,
                        dataset=dataset,
                        dataset_index=by_video[video_path],
                        video_record=video,
                        selected_frames=selected_frames,
                        captured=captured,
                        cache_dtype=cache_dtype,
                        device=device,
                        model_dtype=model_dtype,
                    )
                    print(json.dumps({"ok": True, "video_path": video_path, "frames": selected_frames}), file=log_f, flush=True)
                except Exception as exc:
                    payload = {"ok": False, "video_path": video_path, "frames": selected_frames, "error": str(exc)}
                    print(json.dumps(payload), file=log_f, flush=True)
                    print(f"[ERROR] {payload}", file=sys.stderr)
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()
