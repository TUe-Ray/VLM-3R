#!/usr/bin/env python
"""Dump VLM-3R raw H1 visual tokens for spatial-rank diagnostics.

The output .pt is intended for scripts/spatial_rank_diagnostics.py and contains:
  - cut3r_patch_tokens: [F, 729, C]
  - baseline_h1:        [F, N, D]
  - ours_h1:            [F, N, D]
  - ours_pgeo_h1:       [F, N, 256], when the CE+rank checkpoint has P_geo
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


def patch_runtime_checkpoint(
    checkpoint: str,
    runtime_root: Path | None,
    siglip_path: str | None,
    cut3r_weights: str | None,
) -> str:
    if runtime_root is None and siglip_path is None and cut3r_weights is None:
        return checkpoint

    src = Path(checkpoint)
    runtime_root = runtime_root or (REPO_ROOT / ".offline_runtime")
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in src.name)
    dst = runtime_root / f"{safe_name}_h1diag_runtime"
    dst.mkdir(parents=True, exist_ok=True)

    for child in src.iterdir():
        target = dst / child.name
        if child.name == "config.json":
            continue
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(child)

    with open(src / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if siglip_path is not None:
        cfg["mm_vision_tower"] = siglip_path
        cfg["vision_tower"] = siglip_path
    if cut3r_weights is not None:
        cfg["weights_path"] = cut3r_weights
    with open(dst / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    return str(dst)


def move_to_device(value: Any, device: torch.device, dtype: torch.dtype) -> Any:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            return value.to(device=device, dtype=dtype, non_blocking=True)
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device, dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device, dtype) for item in value)
    return value


def make_data_args(args: argparse.Namespace, image_processor: Any) -> SimpleNamespace:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.ours_checkpoint)
    return SimpleNamespace(
        data_path=args.data_path,
        lazy_preprocess=True,
        is_multimodal=True,
        early_mix_text=False,
        image_folder=args.image_folder,
        image_aspect_ratio=getattr(cfg, "image_aspect_ratio", "anyres_max_9"),
        image_grid_pinpoints=getattr(cfg, "image_grid_pinpoints", None),
        image_crop_resolution=getattr(cfg, "image_crop_resolution", None),
        image_split_resolution=getattr(cfg, "image_split_resolution", None),
        video_folder=args.video_folder,
        video_fps=1,
        frames_upbound=args.frames_upbound,
        add_time_instruction=str2bool(getattr(cfg, "add_time_instruction", True)),
        force_sample=str2bool(getattr(cfg, "force_sample", True)),
        train_data_percentage=args.data_percentage,
        train_data_percentage_seed=args.seed,
        train_data_shuffle=False,
        zero_spatial_features=False,
        spatial_tower_type="cut3r",
        spatial_features_root=args.spatial_features_root,
        spatial_features_subdir=args.spatial_features_subdir,
        image_processor=image_processor,
        mm_use_im_start_end=False,
    )


def load_one_model(
    checkpoint: str,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
):
    from llava.model.builder import load_pretrained_model

    runtime_checkpoint = patch_runtime_checkpoint(
        checkpoint,
        Path(args.runtime_root) if args.runtime_root else None,
        args.siglip_path,
        args.cut3r_weights,
    )
    tokenizer, model, image_processor, _ = load_pretrained_model(
        runtime_checkpoint,
        args.model_base,
        args.model_name,
        device_map=str(device),
        torch_dtype="bfloat16" if dtype == torch.bfloat16 else "float16",
        attn_implementation=args.attn_implementation,
        overwrite_config={
            "delay_load": False,
            "mm_spatial_pool_stride": args.mm_spatial_pool_stride,
            "mm_spatial_pool_mode": args.pool_mode,
            "zero_spatial_features": False,
        },
    )
    model.to(device=device, dtype=dtype)
    model.eval()
    model.config.spatial_rank_loss_enable = False
    model.config.use_cache = False
    return tokenizer, model, image_processor


def get_first_block(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[0]
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return get_first_block(model.base_model.model)
    raise RuntimeError("Could not locate model.layers[0] for H1 hook.")


def extract_h1(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    compute_pgeo: bool,
) -> dict[str, torch.Tensor]:
    batch = move_to_device(batch, device, dtype)
    h1_holder: dict[str, torch.Tensor] = {}

    prepare_fn = getattr(model, "prepare_inputs_labels_for_multimodal", None)
    if prepare_fn is None:
        raise RuntimeError("Model does not expose prepare_inputs_labels_for_multimodal().")
    prepare_kwargs = {
        "input_ids": batch["input_ids"],
        "position_ids": None,
        "attention_mask": batch["attention_mask"],
        "past_key_values": None,
        "labels": None,
        "images": batch["images"],
        "spatial_features": batch.get("spatial_features"),
        "point_maps": batch.get("point_maps"),
        "modalities": batch.get("modalities"),
        "image_sizes": batch.get("image_sizes"),
    }
    if "return_visual_metadata" in inspect.signature(prepare_fn).parameters:
        prepare_kwargs["return_visual_metadata"] = True
    else:
        raise RuntimeError(
            "prepare_inputs_labels_for_multimodal() does not support return_visual_metadata. "
            "Please use a checkpoint/code version with visual metadata support."
        )
    with torch.inference_mode():
        prepared = prepare_fn(**prepare_kwargs)
    (
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        inputs_embeds,
        _labels,
        visual_metadata,
    ) = prepared
    if not visual_metadata:
        raise RuntimeError("Model did not return visual metadata.")

    def capture_h1(_module, _inputs, output):
        h1_holder["h1"] = output[0] if isinstance(output, (tuple, list)) else output

    handle = get_first_block(model).register_forward_hook(capture_h1)
    try:
        with torch.inference_mode():
            model.model(
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
    finally:
        handle.remove()

    if "h1" not in h1_holder:
        raise RuntimeError("H1 hook did not fire.")
    metadata = visual_metadata[0]

    h1 = h1_holder["h1"]
    visual_indices = metadata["visual_token_indices"].to(device=h1.device)
    frame_ids = metadata["visual_frame_ids"].to(device=h1.device)
    frame_order = [int(x) for x in metadata.get("frame_order", [])]
    if not frame_order:
        frame_order = sorted({int(x) for x in frame_ids.detach().cpu().tolist()})

    h1_frames = []
    pgeo_frames = []
    rank_head = getattr(model, "spatial_rank_head", None)
    for frame_id in frame_order:
        frame_indices = visual_indices[frame_ids == frame_id]
        frame_h1 = h1[0, frame_indices].detach()
        h1_frames.append(frame_h1.cpu())
        if compute_pgeo and rank_head is not None:
            pgeo_frames.append(rank_head(frame_h1.to(dtype=dtype)).detach().cpu())

    token_counts = {int(frame.shape[0]) for frame in h1_frames}
    if len(token_counts) != 1:
        raise RuntimeError(f"Variable visual token counts per frame are not supported: {sorted(token_counts)}")

    result = {"h1": torch.stack(h1_frames, dim=0)}
    if pgeo_frames:
        result["pgeo_h1"] = torch.stack(pgeo_frames, dim=0)
    return result


def find_valid_samples(dataset: LazySupervisedDataset, collator: DataCollatorForSupervisedDataset, args: argparse.Namespace):
    end = min(len(dataset), args.sample_index + max(1, args.max_sample_tries))
    last_error = None
    samples = []
    seen_media = set()
    for idx in range(args.sample_index, end):
        try:
            item = dataset[idx]
            if "spatial_features" not in item:
                continue
            sf = item["spatial_features"]
            if not isinstance(sf, dict) or "patch_tokens" not in sf:
                continue
            raw_item = dataset.list_data_dict[idx]
            media_key = tuple(source_media_paths(raw_item, args)) or (str(raw_item.get("id", idx)),)
            if not args.allow_duplicate_media and media_key in seen_media:
                continue
            batch = collator([item])
            samples.append((idx, item, batch, raw_item))
            seen_media.add(media_key)
            if len(samples) >= args.num_samples:
                return samples
        except Exception as exc:
            last_error = exc
    if samples:
        return samples
    raise RuntimeError(f"No valid CUT3R sample found from index {args.sample_index} to {end - 1}. Last error: {last_error}")


def sample_output_path(output: Path, ordinal: int, num_samples: int) -> Path:
    if num_samples == 1 and output.suffix == ".pt":
        return output
    root = output.parent / output.stem if output.suffix == ".pt" else output
    return root / f"sample_{ordinal:03d}" / "h1_dump.pt"


def source_media_paths(raw_item: dict[str, Any], args: argparse.Namespace) -> list[str]:
    if "video" in raw_item:
        video = raw_item["video"]
        if isinstance(video, str):
            return [video if Path(video).is_absolute() else str(Path(args.video_folder) / video)]
    if "image" in raw_item:
        images = raw_item["image"]
        if not isinstance(images, list):
            images = [images]
        return [image if Path(image).is_absolute() else str(Path(args.image_folder) / image) for image in images]
    return []


def tensor_to_uint8_image(frame: torch.Tensor, image_processor: Any) -> np.ndarray:
    frame = frame.detach().float().cpu()
    mean = torch.tensor(getattr(image_processor, "image_mean", [0.5, 0.5, 0.5])).view(3, 1, 1)
    std = torch.tensor(getattr(image_processor, "image_std", [0.5, 0.5, 0.5])).view(3, 1, 1)
    frame = (frame * std + mean).clamp(0.0, 1.0)
    return (frame.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def save_input_frames(
    batch: dict[str, Any],
    image_processor: Any,
    output_dir: Path,
    frame_indices: list[int],
) -> list[str]:
    images = batch.get("images")
    if images is None:
        return []
    if isinstance(images, (list, tuple)):
        if not images:
            return []
        tensor = images[0]
    else:
        tensor = images
        if tensor.dim() >= 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 5 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.dim() != 4:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    frame_count = int(tensor.shape[0])
    for frame_idx in frame_indices:
        if frame_idx < 0 or frame_idx >= frame_count:
            continue
        frame_path = output_dir / f"frame_{frame_idx:03d}.png"
        Image.fromarray(tensor_to_uint8_image(tensor[frame_idx], image_processor)).save(frame_path)
        saved.append(str(frame_path))
    return saved


def write_sample_metadata(
    output_path: Path,
    sample_idx: int,
    raw_item: dict[str, Any],
    payload: dict[str, Any],
    saved_frames: list[str],
    args: argparse.Namespace,
) -> None:
    metadata = {
        "sample_index": sample_idx,
        "sample_id": payload.get("sample_id"),
        "source_media_paths": source_media_paths(raw_item, args),
        "saved_input_frames": saved_frames,
        "diagnostic_frame_index": args.saved_frame_index,
        "saved_input_frames_note": (
            "These PNGs are the denormalized model input frame(s) used for diagnostics after "
            "dataset decoding/preprocessing, not byte-for-byte copies of the original source media."
        ),
        "data_keys": sorted(str(key) for key in raw_item.keys()),
        "video": raw_item.get("video"),
        "image": raw_item.get("image"),
        "conversations": raw_item.get("conversations"),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--ours-checkpoint", required=True)
    parser.add_argument("--model-base", required=True)
    parser.add_argument("--model-name", default="vlm-3r-llava-qwen2-lora")
    parser.add_argument("--siglip-path", default=None)
    parser.add_argument("--cut3r-weights", default=None)
    parser.add_argument("--data-path", default="scripts/VLM_3R/vsibench_data.yaml")
    parser.add_argument("--image-folder", default="/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r")
    parser.add_argument("--video-folder", default="/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r")
    parser.add_argument("--spatial-features-root", default="/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r")
    parser.add_argument("--spatial-features-subdir", default="spatial_features")
    parser.add_argument("--output", required=True)
    parser.add_argument("--runtime-root", default=str(REPO_ROOT / ".offline_runtime"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--allow-duplicate-media", action="store_true")
    parser.add_argument("--max-sample-tries", type=int, default=200)
    parser.add_argument("--data-percentage", type=float, default=100.0)
    parser.add_argument("--frames-upbound", type=int, default=32)
    parser.add_argument("--mm-spatial-pool-stride", type=int, default=2)
    parser.add_argument("--pool-mode", default="bilinear", choices=["bilinear", "average", "max"])
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--save-input-frames", action="store_true")
    parser.add_argument("--saved-frame-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from llava import conversation as conversation_lib
    from llava.train.train import DataCollatorForSupervisedDataset, LazySupervisedDataset

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    conversation_lib.default_conversation = conversation_lib.conv_templates["qwen_1_5"]

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    tokenizer, baseline_model, image_processor = load_one_model(args.baseline_checkpoint, args, device, dtype)
    data_args = make_data_args(args, image_processor)
    dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=args.data_path, data_args=data_args)
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    samples = find_valid_samples(dataset, collator, args)

    baseline_results = []
    for _sample_idx, _item, batch, _raw_item in samples:
        baseline_results.append(extract_h1(baseline_model, batch, device, dtype, compute_pgeo=False))
    del baseline_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _, ours_model, _ = load_one_model(args.ours_checkpoint, args, device, dtype)
    output_arg = Path(args.output)
    for ordinal, ((sample_idx, item, batch, raw_item), baseline) in enumerate(zip(samples, baseline_results)):
        ours = extract_h1(ours_model, batch, device, dtype, compute_pgeo=True)

        spatial_features = item["spatial_features"]
        patch_tokens = spatial_features["patch_tokens"]
        if patch_tokens.dim() == 4 and patch_tokens.shape[0] == 1:
            patch_tokens = patch_tokens[0]
        if patch_tokens.dim() != 3:
            raise RuntimeError(f"Expected cut3r patch_tokens [F,729,C], got {tuple(patch_tokens.shape)}")

        payload = {
            "cut3r_patch_tokens": patch_tokens.detach().cpu(),
            "baseline_h1": baseline["h1"],
            "ours_h1": ours["h1"],
            "sample_index": sample_idx,
            "sample_id": item.get("id", sample_idx),
            "baseline_checkpoint": args.baseline_checkpoint,
            "ours_checkpoint": args.ours_checkpoint,
        }
        if "pgeo_h1" in ours:
            payload["ours_pgeo_h1"] = ours["pgeo_h1"]

        output = sample_output_path(output_arg, ordinal, len(samples))
        output.parent.mkdir(parents=True, exist_ok=True)
        saved_frames = []
        if args.save_input_frames:
            saved_frames = save_input_frames(
                batch,
                image_processor,
                output.parent / "input_frames",
                [args.saved_frame_index],
            )
        write_sample_metadata(output.parent / "sample_metadata.json", sample_idx, raw_item, payload, saved_frames, args)
        torch.save(payload, output)
        print(f"[DONE] wrote {output}")
        print(f"sample_index={sample_idx} sample_id={payload['sample_id']}")
        for key in ("cut3r_patch_tokens", "baseline_h1", "ours_h1", "ours_pgeo_h1"):
            if key in payload and isinstance(payload[key], torch.Tensor):
                print(f"{key}: {tuple(payload[key].shape)}")


if __name__ == "__main__":
    main()
