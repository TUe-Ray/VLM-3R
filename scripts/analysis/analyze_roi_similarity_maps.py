#!/usr/bin/env python
"""ROI similarity-map visualization for VLM-3R representation flow.

This script is inference-only. It uses VLM-3R's own multimodal preparation
metadata to select real visual tokens and excludes text, answer, padding,
newline, special, camera/prefix, and other non-grid tokens from LLM hidden maps.
"""

from __future__ import annotations

import argparse
import csv
import gc
import inspect
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - handled at runtime
    plt = None


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None or value == "":
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_positive_int_list(value: str | None) -> list[int] | None:
    values = parse_int_list(value)
    if values is None:
        return None
    if any(x <= 0 for x in values):
        raise ValueError(f"Expected positive one-based layer numbers, got {values}.")
    return values


def parse_str_list(value: str | None) -> set[str] | None:
    if value is None or value == "":
        return None
    return {x.strip() for x in value.split(",") if x.strip()}


def parse_anchor_coords(value: str | None) -> list[tuple[float, float]]:
    if value is None or value.strip() == "":
        return []
    chunks = [x.strip() for x in value.replace(";", " ").split() if x.strip()]
    coords = []
    if len(chunks) == 1 and chunks[0].count(",") == 1:
        chunks = [chunks[0]]
    for chunk in chunks:
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid anchor coordinate '{chunk}', expected x,y.")
        x, y = float(parts[0]), float(parts[1])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"Anchor coordinates must be normalized to [0,1], got {chunk}.")
        coords.append((x, y))
    return coords


def safe_id(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)[:180]


def patch_runtime_checkpoint(
    checkpoint: str,
    runtime_root: Path | None,
    siglip_path: str | None,
    cut3r_weights: str | None,
) -> str:
    if runtime_root is None and siglip_path is None and cut3r_weights is None:
        return checkpoint
    src = Path(checkpoint)
    if not (src / "config.json").exists():
        return checkpoint
    runtime_root = runtime_root or (REPO_ROOT / ".offline_runtime")
    dst = runtime_root / f"{safe_id(src.name)}_roi_runtime_{os.getpid()}"
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
        return {k: move_to_device(v, device, dtype) for k, v in value.items()}
    if isinstance(value, list):
        return [move_to_device(v, device, dtype) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device, dtype) for v in value)
    return value


def detach_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu()
    if isinstance(value, (tuple, list)) and value:
        return detach_cpu(value[0])
    return value


def disable_training_runtime(model: torch.nn.Module) -> None:
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = False
    if hasattr(model, "config"):
        model.config.use_cache = False
        model.config.spatial_rank_loss_enable = False


def load_model(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    from llava.model.builder import load_pretrained_model

    runtime_checkpoint = patch_runtime_checkpoint(
        args.model_path,
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
            # Avoid eager-loading the heavy CUT3R tower in LlavaQwen.from_pretrained.
            # The analysis path uses pre-extracted sidecars, then flips this back
            # before multimodal preparation so fusion still consumes geometry.
            "zero_spatial_features": True,
            "mm_spatial_pool_stride": args.mm_spatial_pool_stride,
            "mm_spatial_pool_mode": args.pool_mode,
        },
    )
    model.to(device=device, dtype=dtype)
    disable_training_runtime(model)
    model.config.zero_spatial_features = False
    return tokenizer, model, image_processor


def make_data_args(args: argparse.Namespace, model: torch.nn.Module, image_processor: Any) -> SimpleNamespace:
    sf_dir = Path(args.spatial_feature_dir)
    spatial_root = str(sf_dir.parent if sf_dir.name else sf_dir)
    spatial_subdir = sf_dir.name or "spatial_features"
    cfg = model.config
    return SimpleNamespace(
        data_path=args.data_json,
        lazy_preprocess=True,
        is_multimodal=True,
        early_mix_text=False,
        image_folder=args.image_folder,
        image_aspect_ratio=getattr(cfg, "image_aspect_ratio", "anyres_max_9"),
        image_grid_pinpoints=getattr(cfg, "image_grid_pinpoints", None),
        image_crop_resolution=getattr(cfg, "image_crop_resolution", None),
        image_split_resolution=getattr(cfg, "image_split_resolution", None),
        video_folder=args.video_folder,
        video_fps=args.video_fps,
        frames_upbound=args.frames_upbound,
        add_time_instruction=str2bool(getattr(cfg, "add_time_instruction", True)),
        force_sample=str2bool(getattr(cfg, "force_sample", True)),
        train_data_percentage=100.0,
        train_data_percentage_seed=args.seed,
        train_data_shuffle=False,
        deterministic_data_order=True,
        zero_spatial_features=False,
        spatial_tower_type=getattr(cfg, "spatial_tower", "cut3r"),
        spatial_features_root=spatial_root,
        spatial_features_subdir=spatial_subdir,
        image_processor=image_processor,
        mm_use_im_start_end=False,
    )


def extract_question(raw_item: dict[str, Any]) -> str:
    for key in ("question", "prompt"):
        if raw_item.get(key):
            return str(raw_item[key])
    for msg in raw_item.get("conversations", []) or []:
        role = str(msg.get("from", msg.get("role", ""))).lower()
        if role in {"human", "user"}:
            text = str(msg.get("value", msg.get("content", "")))
            return text.replace("<image>", "").replace("<video>", "").strip()
    return ""


def extract_category(raw_item: dict[str, Any]) -> str:
    for key in ("category", "type", "question_type", "task_type", "data_source"):
        if raw_item.get(key) is not None:
            return str(raw_item[key])
    return ""


def sample_identifier(raw_item: dict[str, Any], idx: int) -> str:
    for key in ("id", "question_id", "sample_id", "uid"):
        if raw_item.get(key) is not None:
            return str(raw_item[key])
    if raw_item.get("video"):
        return Path(str(raw_item["video"])).stem
    return str(idx)


def find_spatial_feature_path(raw_item: dict[str, Any], spatial_feature_dir: str, sample_id: str) -> Path | None:
    root = Path(spatial_feature_dir)
    candidates: list[Path] = []
    for key in ("spatial_features", "spatial_feature", "feature_path"):
        if raw_item.get(key):
            candidates.append(Path(str(raw_item[key])))
    if raw_item.get("video"):
        video = Path(str(raw_item["video"]))
        candidates.extend([
            root / video.with_suffix(".pt").name,
            root / f"{video.stem}.pt",
            root / video.with_suffix(".pt"),
        ])
        parts = list(video.with_suffix(".pt").parts)
        if "videos" in parts:
            parts[parts.index("videos")] = root.name
            candidates.append(root.parent / Path(*parts))
    candidates.append(root / f"{sample_id}.pt")
    for path in candidates:
        if not path.is_absolute():
            path = root / path
        if path.exists():
            return path
    return None


def ensure_spatial_features(item: dict[str, Any], raw_item: dict[str, Any], args: argparse.Namespace, sample_id: str) -> bool:
    sf = item.get("spatial_features")
    if isinstance(sf, dict) and "patch_tokens" in sf:
        return True
    path = find_spatial_feature_path(raw_item, args.spatial_feature_dir, sample_id)
    if path is None:
        return False
    item["spatial_features"] = torch.load(path, map_location="cpu")
    return isinstance(item["spatial_features"], dict) and "patch_tokens" in item["spatial_features"]


def select_samples(dataset: Any, collator: Any, args: argparse.Namespace) -> tuple[list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]], list[str]]:
    categories = parse_str_list(args.categories)
    requested_ids = parse_str_list(args.sample_ids)
    selected = []
    skipped = []
    matched = 0
    for idx, raw_item in enumerate(dataset.list_data_dict):
        sid = sample_identifier(raw_item, idx)
        category = extract_category(raw_item)
        if requested_ids is not None and sid not in requested_ids and str(raw_item.get("question_id", "")) not in requested_ids:
            continue
        if categories is not None and category not in categories:
            continue
        if matched < args.sample_offset:
            matched += 1
            continue
        matched += 1
        try:
            item = dataset[idx]
            if not ensure_spatial_features(item, raw_item, args, sid):
                skipped.append(f"{sid}: missing CUT3R patch_tokens sidecar")
                continue
            batch = collator([item])
            selected.append((idx, item, batch, raw_item))
            if len(selected) >= args.num_samples:
                break
        except Exception as exc:
            skipped.append(f"{sid}: {exc}")
    return selected, skipped


def get_language_layers(model: torch.nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return get_language_layers(model.base_model.model)
    raise RuntimeError("Could not locate model.model.layers for LLM hidden-state hooks.")


def capture_representations(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    llm_layers: list[int],
) -> dict[str, Any]:
    batch = move_to_device(batch, device, dtype)
    captures: dict[str, Any] = {}
    handles = []

    def hook(name: str):
        def _capture(_module, _inputs, output):
            captures[name] = detach_cpu(output)
        return _capture

    vision_tower = model.get_model().get_vision_tower()
    fusion_block = model.get_model().get_fusion_block()
    if vision_tower is not None:
        handles.append(vision_tower.register_forward_hook(hook("siglip_raw")))
    if fusion_block is not None:
        handles.append(fusion_block.register_forward_hook(hook("fusion_raw")))

    prepare_fn = getattr(model, "prepare_inputs_labels_for_multimodal", None)
    if prepare_fn is None:
        raise RuntimeError("Model does not expose prepare_inputs_labels_for_multimodal().")
    if "return_visual_metadata" not in inspect.signature(prepare_fn).parameters:
        raise RuntimeError("This checkout lacks return_visual_metadata support; cannot safely identify visual tokens.")

    try:
        with torch.inference_mode():
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
    finally:
        for handle in handles:
            handle.remove()

    input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, _labels, visual_metadata = prepared
    if not visual_metadata:
        raise RuntimeError("Model did not return visual metadata.")

    h_holder: dict[str, torch.Tensor] = {}
    layers = get_language_layers(model)
    h_handles = []

    def h_hook(name: str):
        def _capture(_module, _inputs, output):
            h_holder[name] = (output[0] if isinstance(output, (tuple, list)) else output).detach().float().cpu()
        return _capture

    for layer_num in llm_layers:
        layer_idx = layer_num - 1
        if layer_idx >= len(layers):
            raise RuntimeError(
                f"Requested H{layer_num}, but the language model exposes only {len(layers)} transformer layers."
            )
        h_handles.append(layers[layer_idx].register_forward_hook(h_hook(f"h{layer_num}")))
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
        for handle in h_handles:
            handle.remove()

    missing_layers = [f"H{layer_num}" for layer_num in llm_layers if f"h{layer_num}" not in h_holder]
    if missing_layers:
        raise RuntimeError(f"LLM hidden-state hooks did not fire: {missing_layers}")
    captures.update(h_holder)
    captures["llm_input"] = inputs_embeds.detach().float().cpu()
    captures["visual_metadata"] = [
        {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in md.items()}
        for md in visual_metadata
    ]
    captures["batch_cpu"] = {k: detach_cpu(v) for k, v in batch.items() if k in {"images", "spatial_features"}}
    return captures


def tensor_to_uint8_image(frame: torch.Tensor, image_processor: Any) -> np.ndarray:
    frame = frame.detach().float().cpu()
    mean = torch.tensor(getattr(image_processor, "image_mean", [0.5, 0.5, 0.5])).view(3, 1, 1)
    std = torch.tensor(getattr(image_processor, "image_std", [0.5, 0.5, 0.5])).view(3, 1, 1)
    frame = (frame * std + mean).clamp(0.0, 1.0)
    return (frame.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def get_video_tensor(batch: dict[str, Any]) -> torch.Tensor:
    images = batch["images"]
    if isinstance(images, (list, tuple)):
        tensor = images[0]
    else:
        tensor = images
    if tensor.dim() == 5 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 4:
        raise RuntimeError(f"Expected video tensor [F,C,H,W], got {tuple(tensor.shape)}")
    return tensor


def pool_grid(tokens: torch.Tensor, in_hw: tuple[int, int], out_hw: tuple[int, int], pool_mode: str) -> torch.Tensor:
    if in_hw == out_hw:
        return tokens
    grid = tokens.view(in_hw[0], in_hw[1], -1).permute(2, 0, 1).unsqueeze(0).float()
    pool_mode = pool_mode.lower()
    if pool_mode == "bilinear":
        pooled = F.interpolate(grid, size=out_hw, mode="bilinear", align_corners=False)
    elif pool_mode in {"average", "avg"}:
        pad_h = (out_hw[0] * 2) - in_hw[0] if in_hw == (27, 27) and out_hw == (14, 14) else 0
        pad_w = (out_hw[1] * 2) - in_hw[1] if in_hw == (27, 27) and out_hw == (14, 14) else 0
        if pad_h or pad_w:
            grid = F.pad(grid, (0, pad_w, 0, pad_h), value=0.0)
            valid = F.pad(torch.ones(1, 1, *in_hw), (0, pad_w, 0, pad_h), value=0.0)
            summed = F.avg_pool2d(grid, 2, 2) * 4.0
            counts = F.avg_pool2d(valid, 2, 2) * 4.0
            pooled = summed / counts.clamp_min(1.0)
        else:
            pooled = F.adaptive_avg_pool2d(grid, out_hw)
    elif pool_mode == "max":
        pooled = F.adaptive_max_pool2d(grid, out_hw)
    else:
        raise ValueError(f"Unsupported pool mode: {pool_mode}")
    return pooled[0].permute(1, 2, 0).reshape(out_hw[0] * out_hw[1], -1).to(dtype=tokens.dtype)


def tokens_for_frame_from_sequence(sequence: torch.Tensor, metadata: dict[str, Any], frame_id: int) -> torch.Tensor:
    visual_indices = metadata["visual_token_indices"].long()
    frame_ids = metadata["visual_frame_ids"].long()
    frame_positions = visual_indices[frame_ids == int(frame_id)]
    if sequence.dim() == 3:
        return sequence[0, frame_positions]
    return sequence[frame_positions]


def tokens_for_frame_from_framed(
    framed: torch.Tensor,
    local_frame_idx: int,
    target_hw: tuple[int, int],
    raw_hw: tuple[int, int],
    pool_mode: str,
) -> tuple[torch.Tensor, str]:
    if framed is None:
        raise KeyError
    if framed.dim() == 4 and framed.shape[0] == 1:
        framed = framed[0]
    if framed.dim() != 3:
        raise RuntimeError(f"Expected framed tokens [F,N,D], got {tuple(framed.shape)}")
    frame_tokens = framed[local_frame_idx]
    target_n = target_hw[0] * target_hw[1]
    raw_n = raw_hw[0] * raw_hw[1]
    if frame_tokens.shape[0] == target_n:
        return frame_tokens, "direct"
    if frame_tokens.shape[0] == raw_n:
        return pool_grid(frame_tokens, raw_hw, target_hw, pool_mode), pool_mode
    if frame_tokens.shape[0] > raw_n:
        prefix = frame_tokens.shape[0] - raw_n
        return pool_grid(frame_tokens[prefix:], raw_hw, target_hw, pool_mode), f"strip_prefix_{prefix}+{pool_mode}"
    raise RuntimeError(
        f"Cannot map frame tokens to grid: tokens={frame_tokens.shape[0]}, raw={raw_hw}, target={target_hw}"
    )


def similarity_map(features: torch.Tensor, anchor_index: int, grid_hw: tuple[int, int]) -> torch.Tensor:
    if features.shape[0] != grid_hw[0] * grid_hw[1]:
        raise RuntimeError(f"Grid {grid_hw} expects {grid_hw[0] * grid_hw[1]} tokens, got {features.shape[0]}.")
    feats = F.normalize(features.float(), dim=-1, eps=1e-6)
    sim = feats @ feats[int(anchor_index)].unsqueeze(-1)
    return sim.squeeze(-1).view(*grid_hw).detach().cpu()


def normalized_for_vis(maps: dict[str, torch.Tensor], mode: str) -> dict[str, torch.Tensor]:
    if mode == "per_map":
        out = {}
        for key, value in maps.items():
            v = value.float()
            out[key] = (v - v.min()) / (v.max() - v.min()).clamp_min(1e-6)
        return out
    values = torch.cat([v.float().flatten() for v in maps.values()])
    lo, hi = values.min(), values.max()
    return {k: (v.float() - lo) / (hi - lo).clamp_min(1e-6) for k, v in maps.items()}


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def corr_values(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    x = a.float().flatten().numpy()
    y = b.float().flatten().numpy()
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan"), float("nan")
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    return pearson, spearman


def build_anchors(mode: str, coords: list[tuple[float, float]], grid_hw: tuple[int, int], raw_item: dict[str, Any]) -> list[dict[str, Any]]:
    h, w = grid_hw
    if mode == "center":
        coords = [(0.5, 0.5)]
    elif mode == "grid":
        coords = [(0.5, 0.5), (0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]
    elif mode == "manual":
        if not coords:
            raise ValueError("--anchor_mode manual requires --anchor_coords.")
    elif mode == "object":
        boxes = raw_item.get("boxes") or raw_item.get("bboxes") or raw_item.get("objects")
        coords = []
        if isinstance(boxes, list):
            for box in boxes[:3]:
                candidate = box.get("bbox") if isinstance(box, dict) else box
                if isinstance(candidate, (list, tuple)) and len(candidate) >= 4:
                    x1, y1, x2, y2 = [float(v) for v in candidate[:4]]
                    if max(x1, y1, x2, y2) > 1.0:
                        continue
                    coords.append(((x1 + x2) * 0.5, (y1 + y2) * 0.5))
        if not coords:
            return []
    anchors = []
    for idx, (x, y) in enumerate(coords):
        token_x = min(w - 1, max(0, int(math.floor(x * w))))
        token_y = min(h - 1, max(0, int(math.floor(y * h))))
        anchors.append({
            "anchor_id": f"{mode}{idx}",
            "image_xy": (float(x), float(y)),
            "token_xy": (int(token_x), int(token_y)),
            "token_index": int(token_y * w + token_x),
        })
    return anchors


def select_frames(args: argparse.Namespace, frame_order: list[int], raw_item: dict[str, Any]) -> list[int]:
    frame_count = len(frame_order)
    requested = parse_int_list(args.frames)
    if requested is not None:
        selected = []
        for value in requested:
            if value in frame_order:
                selected.append(frame_order.index(value))
            elif 0 <= value < frame_count:
                selected.append(value)
        return selected
    for key in ("frame_idx", "frame_id", "target_frame", "question_frame"):
        if raw_item.get(key) is not None:
            idx = int(raw_item[key])
            if idx in frame_order:
                return [frame_order.index(idx)]
            if 0 <= idx < frame_count:
                return [idx]
    return [frame_count // 2]


def save_figure(
    path: Path,
    rgb: np.ndarray,
    anchor: dict[str, Any],
    raw_maps: dict[str, torch.Tensor],
    normalize_mode: str,
) -> None:
    if plt is None:
        print("[WARN] matplotlib is not installed; skipping PNG figure output.")
        return
    ncols = 1 + len(raw_maps)
    fig, axes = plt.subplots(1, ncols, figsize=(3.1 * ncols, 3.4), squeeze=False)
    ax = axes[0, 0]
    ax.imshow(rgb)
    x = anchor["image_xy"][0] * (rgb.shape[1] - 1)
    y = anchor["image_xy"][1] * (rgb.shape[0] - 1)
    ax.scatter([x], [y], s=52, c="red", edgecolors="white", linewidths=1.2)
    ax.set_title("RGB")
    ax.axis("off")
    if normalize_mode == "global_per_figure":
        all_values = torch.cat([v.float().flatten() for v in raw_maps.values()])
        vmin, vmax = float(all_values.min()), float(all_values.max())
    else:
        vmin = vmax = None
    for idx, (name, value) in enumerate(raw_maps.items(), start=1):
        ax = axes[0, idx]
        heat = F.interpolate(
            value[None, None].float(),
            size=rgb.shape[:2],
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        if normalize_mode == "per_map":
            im = ax.imshow(heat, cmap="viridis")
        else:
            im = ax.imshow(heat, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_rgb_frame(path: Path, rgb: np.ndarray, anchor: dict[str, Any] | None = None) -> None:
    image = Image.fromarray(rgb)
    if anchor is not None:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(image)
        x = anchor["image_xy"][0] * (rgb.shape[1] - 1)
        y = anchor["image_xy"][1] * (rgb.shape[0] - 1)
        radius = max(4, int(min(rgb.shape[:2]) * 0.018))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline="white", width=3)
        draw.ellipse((x - radius + 2, y - radius + 2, x + radius - 2, y + radius - 2), outline="red", width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def jsonable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in metadata.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu().tolist()
        elif isinstance(value, (list, tuple)):
            out[key] = [jsonable_metadata(v) if isinstance(v, dict) else v for v in value]
        else:
            out[key] = value
    return out


def analyze_sample(
    args: argparse.Namespace,
    model: torch.nn.Module,
    image_processor: Any,
    item: dict[str, Any],
    batch: dict[str, Any],
    raw_item: dict[str, Any],
    sample_idx: int,
    visualize: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[dict[str, Any]], list[str], int]:
    sample_id = sample_identifier(raw_item, sample_idx)
    captures = capture_representations(model, batch, device, dtype, args.llm_layers)
    metadata = captures["visual_metadata"][0]
    frame_order = [int(x) for x in metadata.get("frame_order", [])]
    if not frame_order:
        frame_order = sorted(set(int(x) for x in metadata["visual_frame_ids"].tolist()))
    frame_count = len(frame_order)
    selected_frames = select_frames(args, frame_order, raw_item)
    video_tensor = get_video_tensor(batch)
    sf = item["spatial_features"]
    cut3r_patch = sf["patch_tokens"]
    if cut3r_patch.dim() == 4 and cut3r_patch.shape[0] == 1:
        cut3r_patch = cut3r_patch[0]
    cut3r_patch = cut3r_patch.detach().float().cpu()

    rows = []
    skipped = []
    figure_count = 0
    pool_records: set[str] = set()
    figures_dir = Path(args.output_dir) / "figures"
    raw_dir = Path(args.output_dir) / "raw"
    frames_dir = Path(args.output_dir) / "frames"
    figures_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    for local_frame_idx in selected_frames:
        frame_id = frame_order[local_frame_idx]
        grid_hw = tuple(int(v) for v in metadata["visual_grid_shapes"][local_frame_idx])
        raw_hw = tuple(int(v) for v in metadata["raw_visual_grid_shapes"][local_frame_idx])
        anchors = build_anchors(args.anchor_mode, parse_anchor_coords(args.anchor_coords), grid_hw, raw_item)
        if not anchors:
            skipped.append(f"{sample_id}: no anchors for mode {args.anchor_mode}")
            continue

        frame_features: dict[str, torch.Tensor] = {}
        pool_used: dict[str, str] = {}
        try:
            frame_features["SigLIP"], pool_used["SigLIP"] = tokens_for_frame_from_framed(
                captures.get("siglip_raw"), local_frame_idx, grid_hw, raw_hw, args.pool_mode
            )
        except Exception as exc:
            skipped.append(f"{sample_id} frame {frame_id}: SigLIP unavailable ({exc})")
        try:
            frame_features["CUT3R teacher"], pool_used["CUT3R teacher"] = tokens_for_frame_from_framed(
                cut3r_patch, local_frame_idx, grid_hw, raw_hw, args.pool_mode
            )
        except Exception as exc:
            skipped.append(f"{sample_id} frame {frame_id}: CUT3R unavailable ({exc})")
            continue
        if "fusion_raw" in captures:
            try:
                frame_features["Fusion"] , pool_used["Fusion"] = tokens_for_frame_from_framed(
                    captures["fusion_raw"], local_frame_idx, grid_hw, raw_hw, args.pool_mode
                )
            except Exception as exc:
                skipped.append(f"{sample_id} frame {frame_id}: fusion unavailable ({exc})")
        frame_features["LLM input"] = tokens_for_frame_from_sequence(captures["llm_input"], metadata, frame_id)
        for layer_num in args.llm_layers:
            capture_key = f"h{layer_num}"
            if capture_key in captures:
                frame_features[f"H{layer_num}"] = tokens_for_frame_from_sequence(captures[capture_key], metadata, frame_id)
        pgeo_exists = False
        if args.include_aligned_projection and getattr(model, "spatial_rank_head", None) is not None:
            if "H1" not in frame_features:
                raise RuntimeError("--include_aligned_projection requires H1 in --llm_layers.")
            pgeo_exists = True
            rank_head = model.spatial_rank_head.to(device=device, dtype=dtype).eval()
            h1_device = frame_features["H1"].to(device=device, dtype=dtype)
            with torch.inference_mode():
                frame_features["P_geo(H1)"] = rank_head(h1_device).detach().float().cpu()

        excluded = parse_str_list(args.exclude_representations) or set()
        for name in list(frame_features.keys()):
            if name in excluded:
                frame_features.pop(name)
                pool_used.pop(name, None)

        token_counts = {name: int(value.shape[0]) for name, value in frame_features.items()}
        if len(set(token_counts.values())) != 1:
            raise RuntimeError(f"Representation token counts differ for {sample_id} frame {frame_id}: {token_counts}")
        for name in frame_features:
            pool_records.add(f"{name}:{pool_used.get(name, 'metadata_direct')}")

        rgb = tensor_to_uint8_image(video_tensor[local_frame_idx], image_processor)
        if visualize:
            save_rgb_frame(frames_dir / f"{safe_id(sample_id)}_frame{frame_id}.png", rgb)
        for anchor in anchors:
            raw_maps = {
                name: similarity_map(value, anchor["token_index"], grid_hw)
                for name, value in frame_features.items()
            }
            vis_maps = normalized_for_vis(raw_maps, args.normalize_mode)
            stem = f"{safe_id(sample_id)}_frame{frame_id}_anchor{anchor['anchor_id']}"
            figure_path = figures_dir / f"{stem}.png"
            if visualize:
                save_figure(figure_path, rgb, anchor, raw_maps, args.normalize_mode)
                save_rgb_frame(frames_dir / f"{stem}_rgb_anchor.png", rgb, anchor)
                figure_count += 1

            if args.save_raw:
                torch.save(
                    {
                        "raw_similarity_maps": raw_maps,
                        "visualization_normalized_maps": vis_maps,
                        "features": {k: v.cpu() for k, v in frame_features.items()} if args.save_features else {},
                    },
                    raw_dir / f"{stem}.pt",
                )

            teacher_map = raw_maps["CUT3R teacher"]
            siglip_map = raw_maps.get("SigLIP")
            for name, sim in raw_maps.items():
                pearson, spearman = corr_values(sim, teacher_map)
                siglip_pearson = siglip_spearman = float("nan")
                if siglip_map is not None:
                    siglip_pearson, siglip_spearman = corr_values(sim, siglip_map)
                rows.append({
                    "sample_id": sample_id,
                    "category": extract_category(raw_item),
                    "frame_idx": frame_id,
                    "anchor_id": anchor["anchor_id"],
                    "representation": name,
                    "pearson_with_cut3r": pearson,
                    "spearman_with_cut3r": spearman,
                    "pearson_with_siglip_selected_m2": siglip_pearson,
                    "spearman_with_siglip_selected_m2": siglip_spearman,
                    "spearman_cut3r_minus_siglip": (
                        spearman - siglip_spearman
                        if not math.isnan(spearman) and not math.isnan(siglip_spearman)
                        else float("nan")
                    ),
                    "mean_similarity": float(sim.float().mean().item()),
                    "std_similarity": float(sim.float().std(unbiased=False).item()),
                })

            sanity = {
                "number_of_frames": frame_count,
                "number_of_valid_visual_tokens": int(metadata["visual_token_indices"].numel()),
                "visual_tokens_per_frame": token_counts,
                "cut3r_tokens_per_frame_before_pooling": int(cut3r_patch.shape[1]),
                "cut3r_tokens_per_frame_after_pooling": int(frame_features["CUT3R teacher"].shape[0]),
                "siglip_tokens_per_frame_before_pooling": int(captures["siglip_raw"].shape[1]) if "siglip_raw" in captures else None,
                "siglip_tokens_per_frame_after_pooling": int(frame_features["SigLIP"].shape[0]) if "SigLIP" in frame_features else None,
                "h1_visual_token_count": int(frame_features["H1"].shape[0]) if "H1" in frame_features else None,
                "h4_visual_token_count": int(frame_features["H4"].shape[0]) if "H4" in frame_features else None,
                "llm_visual_token_counts": {
                    f"H{layer_num}": int(frame_features[f"H{layer_num}"].shape[0])
                    for layer_num in args.llm_layers
                    if f"H{layer_num}" in frame_features
                },
                "all_maps_same_grid_shape": len({tuple(v.shape) for v in raw_maps.values()}) == 1,
            }
            meta = {
                "sample_id": sample_id,
                "sample_index": sample_idx,
                "question": extract_question(raw_item),
                "category": extract_category(raw_item),
                "frame_idx": frame_id,
                "local_frame_idx": local_frame_idx,
                "anchor_coordinate_image_space": anchor["image_xy"],
                "anchor_token_xy": anchor["token_xy"],
                "anchor_token_index": anchor["token_index"],
                "visual_token_count": int(token_counts.get("H1", next(iter(token_counts.values())))),
                "grid_shape": list(grid_hw),
                "raw_grid_shape": list(raw_hw),
                "representation_names_included": list(raw_maps.keys()),
                "model_checkpoint_path": args.model_path,
                "pooling_method_used": sorted(pool_records),
                "p_geo_exists": pgeo_exists,
                "visual_metadata": jsonable_metadata(metadata),
                "sanity_checks": sanity,
            }
            with open(raw_dir / f"{stem}.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print(json.dumps({"sample_id": sample_id, "frame_idx": frame_id, "anchor": anchor["anchor_id"], **sanity}, indent=2))
    return rows, skipped, figure_count


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> dict[str, float]:
    csv_path = output_dir / "roi_similarity_summary.csv"
    fieldnames = [
        "sample_id",
        "category",
        "frame_idx",
        "anchor_id",
        "representation",
        "pearson_with_cut3r",
        "spearman_with_cut3r",
        "pearson_with_siglip_selected_m2",
        "spearman_with_siglip_selected_m2",
        "spearman_cut3r_minus_siglip",
        "mean_similarity",
        "std_similarity",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    by_rep = defaultdict(list)
    for row in rows:
        value = row["spearman_with_cut3r"]
        if not math.isnan(value):
            by_rep[row["representation"]].append(float(value))
    return {rep: float(np.mean(values)) for rep, values in by_rep.items() if values}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_json", required=True)
    parser.add_argument("--spatial_feature_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--sample_ids", default=None)
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument("--categories", default=None)
    parser.add_argument("--frames", default=None)
    parser.add_argument("--anchor_mode", choices=["manual", "center", "object", "grid"], default="center")
    parser.add_argument("--anchor_coords", default=None)
    parser.add_argument("--include_aligned_projection", type=str2bool, default=False)
    parser.add_argument("--save_raw", type=str2bool, default=True)
    parser.add_argument("--max_visualized_samples", type=int, default=-1)
    parser.add_argument("--normalize_mode", choices=["global_per_figure", "per_map"], default="global_per_figure")
    parser.add_argument("--exclude_representations", default=None)
    parser.add_argument(
        "--llm_layers",
        type=parse_positive_int_list,
        default=[1, 4],
        help="Comma-separated one-based LLM block outputs to capture, e.g. 1,4,8,12.",
    )

    parser.add_argument("--model_base", default=None)
    parser.add_argument("--model_name", default="vlm-3r-llava-qwen2-lora")
    parser.add_argument("--image_folder", default=".")
    parser.add_argument("--video_folder", default=".")
    parser.add_argument("--runtime_root", default=str(REPO_ROOT / ".offline_runtime"))
    parser.add_argument("--siglip_path", default=None)
    parser.add_argument("--cut3r_weights", default=None)
    parser.add_argument("--frames_upbound", type=int, default=32)
    parser.add_argument("--video_fps", type=int, default=1)
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=2)
    parser.add_argument("--pool_mode", choices=["bilinear", "average", "max"], default="bilinear")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_features", action="store_true")
    args = parser.parse_args()
    if args.llm_layers is None:
        args.llm_layers = [1, 4]

    if plt is None:
        raise RuntimeError("matplotlib is required for figure generation.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from llava import conversation as conversation_lib
    from llava.train.train import DataCollatorForSupervisedDataset, LazySupervisedDataset

    conversation_lib.default_conversation = conversation_lib.conv_templates["qwen_1_5"]
    device = torch.device(args.device)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    tokenizer, model, image_processor = load_model(args, device, dtype)
    data_args = make_data_args(args, model, image_processor)
    dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=args.data_json, data_args=data_args)
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    samples, skipped = select_samples(dataset, collator, args)
    if not samples:
        raise RuntimeError("No valid samples found. Skips:\n" + "\n".join(skipped[:20]))

    all_rows: list[dict[str, Any]] = []
    figure_count = 0
    for selected_ordinal, (sample_idx, item, batch, raw_item) in enumerate(samples):
        try:
            visualize = args.max_visualized_samples < 0 or selected_ordinal < args.max_visualized_samples
            rows, sample_skips, count = analyze_sample(
                args, model, image_processor, item, batch, raw_item, sample_idx, visualize, device, dtype
            )
            all_rows.extend(rows)
            skipped.extend(sample_skips)
            figure_count += count
        except Exception as exc:
            skipped.append(f"{sample_identifier(raw_item, sample_idx)}: {exc}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_spearman = write_summary(output_dir, all_rows)
    with open(output_dir / "skipped_samples.json", "w", encoding="utf-8") as f:
        json.dump(skipped, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("\n[DONE] ROI similarity analysis")
    print(f"figures_generated={figure_count}")
    print(f"output_dir={output_dir}")
    print(f"skipped_samples={len(skipped)}")
    if skipped:
        for reason in skipped[:20]:
            print(f"  SKIP: {reason}")
    print("average_spearman_with_cut3r:")
    for rep, value in sorted(avg_spearman.items()):
        print(f"  {rep}: {value:.4f}")


if __name__ == "__main__":
    main()
