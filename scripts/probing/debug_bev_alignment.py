#!/usr/bin/env python
"""Debug BEV target extraction and optional visual-token alignment for VLM-3R.

GT-only mode is the default and does not load the LLM. Use
--run-model-forward for the slower no-grad metadata/hidden-state check.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PROBING_DIR = Path(__file__).resolve().parent
if str(PROBING_DIR) not in sys.path:
    sys.path.insert(0, str(PROBING_DIR))

from depth_probe_common import load_point_map_sidecar, load_yaml_dataset_records, resolve_sidecar_path, torch_dtype_from_name
from llava.model.geometry.bev_supervision import (
    CAMERA_POINT_MAP_KEYS,
    GENERIC_POINT_MAP_KEYS,
    REFERENCE_POINT_MAP_KEYS,
    build_bev_targets_from_point_maps,
)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def parse_grid(value: str) -> Tuple[int, int]:
    text = str(value).lower().replace(",", "x")
    parts = [part for part in text.split("x") if part]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected grid as HxW, got {value!r}")
    return int(parts[0]), int(parts[1])


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def first_point_map(payload: Dict[str, Any], requested_key: str) -> torch.Tensor:
    key_groups = {
        "point_maps_ref": REFERENCE_POINT_MAP_KEYS,
        "pts3d_in_other_view": ("pts3d_in_other_view", "point_maps_ref"),
        "ref": REFERENCE_POINT_MAP_KEYS,
        "point_maps_cam": CAMERA_POINT_MAP_KEYS,
        "pts3d_in_self_view": ("pts3d_in_self_view", "point_maps_cam"),
        "cam": CAMERA_POINT_MAP_KEYS,
    }
    keys = key_groups.get(str(requested_key).lower(), (requested_key,) + GENERIC_POINT_MAP_KEYS)
    for key in keys:
        value = payload.get(key)
        if isinstance(value, torch.Tensor):
            return value
    raise KeyError(f"Could not find point map for key {requested_key!r}. Available keys: {sorted(payload.keys())}")


def point_map_shape_fhw(point_map: torch.Tensor) -> Tuple[int, int, int]:
    if point_map.dim() == 5 and point_map.shape[0] == 1:
        point_map = point_map[0]
    if point_map.dim() != 4:
        raise ValueError(f"Expected point map [F,H,W,3] or [F,3,H,W], got {tuple(point_map.shape)}")
    if point_map.shape[-1] == 3:
        return int(point_map.shape[0]), int(point_map.shape[1]), int(point_map.shape[2])
    if point_map.shape[1] == 3:
        return int(point_map.shape[0]), int(point_map.shape[2]), int(point_map.shape[3])
    raise ValueError(f"Point map does not expose 3 coordinate channels: {tuple(point_map.shape)}")


def infer_visual_grid_shape(args: argparse.Namespace) -> Tuple[int, int]:
    if args.visual_grid_shape is not None:
        return parse_grid(args.visual_grid_shape)
    raw_h, raw_w = parse_grid(args.raw_visual_grid_shape)
    stride = int(args.mm_spatial_pool_stride)
    if stride <= 0:
        raise ValueError(f"--mm-spatial-pool-stride must be positive, got {stride}")
    return int(math.ceil(raw_h / stride)), int(math.ceil(raw_w / stride))


def build_gt_only_visual_metadata(
    *,
    num_frames: int,
    grid_shape: Tuple[int, int],
    raw_grid_shape: Tuple[int, int],
    newline_position: str,
) -> Dict[str, Any]:
    h, w = int(grid_shape[0]), int(grid_shape[1])
    frame_order = list(range(int(num_frames)))
    visual_positions = []
    frame_ids = []
    newline_positions = []
    if newline_position == "grid":
        frame_len = h * (w + 1)
        for frame_idx in frame_order:
            base = frame_idx * frame_len
            for row in range(h):
                row_base = base + row * (w + 1)
                for col in range(w):
                    visual_positions.append(row_base + col)
                    frame_ids.append(frame_idx)
                newline_positions.append(row_base + w)
    elif newline_position == "frame":
        frame_len = h * w + 1
        for frame_idx in frame_order:
            base = frame_idx * frame_len
            for token_idx in range(h * w):
                visual_positions.append(base + token_idx)
                frame_ids.append(frame_idx)
            newline_positions.append(base + h * w)
    elif newline_position in {"one_token", "no_token"}:
        frame_len = h * w
        for frame_idx in frame_order:
            base = frame_idx * frame_len
            for token_idx in range(h * w):
                visual_positions.append(base + token_idx)
                frame_ids.append(frame_idx)
        if newline_position == "one_token":
            newline_positions.append(num_frames * frame_len)
    else:
        raise ValueError(f"Unsupported --mm-newline-position: {newline_position}")

    empty = torch.empty(0, dtype=torch.long)
    return {
        "visual_token_indices": torch.tensor(visual_positions, dtype=torch.long),
        "visual_frame_ids": torch.tensor(frame_ids, dtype=torch.long),
        "frame_order": frame_order,
        "visual_grid_shapes": [tuple(grid_shape) for _ in frame_order],
        "raw_visual_grid_shapes": [tuple(raw_grid_shape) for _ in frame_order],
        "newline_token_indices": torch.tensor(newline_positions, dtype=torch.long),
        "padding_token_indices": empty,
        "answer_token_indices": empty,
        "text_token_indices": empty,
        "special_token_indices": empty,
        "camera_prefix_token_indices": empty,
        "tokens_per_frame": [h * w for _ in frame_order],
        "layout": f"gt_only_{newline_position}",
    }


def select_record(records: list[dict[str, Any]], args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    if args.video is not None:
        for idx, record in enumerate(records):
            if str(record.get("video", "")) == args.video:
                return idx, record
        raise ValueError(f"Could not find video={args.video!r} in {args.train_data_json}")
    idx = int(args.sample_index)
    if idx < 0 or idx >= len(records):
        raise IndexError(f"--sample-index {idx} out of range for {len(records)} records")
    return idx, records[idx]


def _select_valid_bev(coords: torch.Tensor, valid_mask: torch.Tensor, metadata: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    coords = coords[0].detach().cpu() if coords.dim() == 3 else coords.detach().cpu()
    valid = valid_mask[0].detach().cpu().bool() if valid_mask.dim() == 2 else valid_mask.detach().cpu().bool()
    frame_ids = metadata["visual_frame_ids"].detach().cpu()
    n = min(int(coords.shape[0]), int(valid.numel()), int(frame_ids.numel()))
    coords = coords[:n]
    valid = valid[:n]
    frame_ids = frame_ids[:n]
    return coords[valid], frame_ids[valid]


def bev_minmax(coords: torch.Tensor, valid_mask: torch.Tensor) -> Dict[str, Any]:
    coords = coords[0].detach().cpu() if coords.dim() == 3 else coords.detach().cpu()
    valid = valid_mask[0].detach().cpu().bool() if valid_mask.dim() == 2 else valid_mask.detach().cpu().bool()
    n = min(int(coords.shape[0]), int(valid.numel()))
    coords = coords[:n][valid[:n]]
    if coords.numel() == 0:
        return {"x": None, "z": None}
    return {
        "x": [float(coords[:, 0].min().item()), float(coords[:, 0].max().item())],
        "z": [float(coords[:, 1].min().item()), float(coords[:, 1].max().item())],
    }


def valid_ratio_by_frame(valid_mask: torch.Tensor, metadata: Dict[str, Any]) -> Dict[str, float]:
    valid = valid_mask[0].detach().cpu().bool() if valid_mask.dim() == 2 else valid_mask.detach().cpu().bool()
    frame_ids = metadata["visual_frame_ids"].detach().cpu()
    n = min(int(valid.numel()), int(frame_ids.numel()))
    valid = valid[:n]
    frame_ids = frame_ids[:n]
    ratios = {}
    for frame_id in torch.unique(frame_ids, sorted=True).tolist():
        frame_mask = frame_ids == int(frame_id)
        denom = int(frame_mask.sum().item())
        ratios[str(int(frame_id))] = float((valid & frame_mask).sum().item() / denom) if denom else 0.0
    return ratios


def save_bev_scatter(
    path: Path,
    coords: torch.Tensor,
    valid_mask: torch.Tensor,
    metadata: Dict[str, Any],
    *,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required to save BEV debug plots") from exc

    coords, frame_ids = _select_valid_bev(coords, valid_mask, metadata)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    if coords.numel() == 0:
        ax.text(0.5, 0.5, "No valid BEV tokens", ha="center", va="center")
    else:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=frame_ids, s=10, cmap="viridis", alpha=0.85)
        fig.colorbar(scatter, ax=ax, label="frame")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_bev_overlay(
    path: Path,
    bev_gt: torch.Tensor,
    bev_pred: torch.Tensor,
    valid_mask: torch.Tensor,
    metadata: Dict[str, Any],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required to save BEV debug plots") from exc

    gt_coords, frame_ids = _select_valid_bev(bev_gt, valid_mask, metadata)
    pred_coords, _ = _select_valid_bev(bev_pred, valid_mask, metadata)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    if gt_coords.numel() == 0 or pred_coords.numel() == 0:
        ax.text(0.5, 0.5, "No valid BEV tokens", ha="center", va="center")
    else:
        scatter = ax.scatter(gt_coords[:, 0], gt_coords[:, 1], c=frame_ids, s=10, cmap="viridis", alpha=0.7, label="GT")
        ax.scatter(pred_coords[:, 0], pred_coords[:, 1], c=frame_ids, s=12, cmap="viridis", alpha=0.65, marker="x", label="Pred")
        fig.colorbar(scatter, ax=ax, label="frame")
        step = max(int(gt_coords.shape[0] // 128), 1)
        for gt, pred in zip(gt_coords[::step], pred_coords[::step]):
            ax.plot([gt[0], pred[0]], [gt[1], pred[1]], color="0.55", linewidth=0.4, alpha=0.35)
        ax.legend(loc="best")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title("BEV GT / prediction overlay")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_bev_plot(path: Path, bev_gt: torch.Tensor, valid_mask: torch.Tensor, metadata: Dict[str, Any]) -> None:
    save_bev_scatter(path, bev_gt, valid_mask, metadata, title="BEV GT visual-token targets")


def run_optional_model_forward(
    args: argparse.Namespace,
    dataset_index: int,
    payload: Dict[str, Any],
    gt_only_metadata: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    if not args.model_path:
        raise ValueError("--run-model-forward requires --model-path")

    from llava.train.train import DataCollatorForSupervisedDataset, LazySupervisedDataset
    from scripts.diagnose_layerwise_spatial_hidden_scan import load_model, make_data_args, move_to_device

    device = torch.device(args.device)
    dtype = torch_dtype_from_name(args.dtype)
    tokenizer, model, image_processor = load_model(args, device, dtype)
    if hasattr(model, "config"):
        model.config.bev_point_map_key = args.bev_point_map_key
        model.config.bev_coord_scale = args.bev_coord_scale
        model.config.bev_conf_threshold = args.bev_conf_threshold
    data_args = make_data_args(args, image_processor)
    data_args.deterministic_data_order = True
    data_args.train_data_shuffle = False
    data_args.spatial_features_root = args.feature_root
    data_args.spatial_features_subdir = args.spatial_features_subdir
    data_args.require_spatial_features = True
    data_args.geometry_spatial_tower_type = "cut3r"
    data_args.geometry_spatial_features_root = args.feature_root
    data_args.geometry_spatial_features_subdir = args.point_maps_subdir
    data_args.require_geometry_spatial_features = True
    dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=args.train_data_json, data_args=data_args)
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    item = dataset[dataset_index]
    batch = collator([item])
    batch = move_to_device(batch, device, dtype)

    prepare_fn = getattr(model, "prepare_inputs_labels_for_multimodal", None)
    if prepare_fn is None:
        raise RuntimeError("Model does not expose prepare_inputs_labels_for_multimodal().")
    with torch.no_grad():
        prepared = prepare_fn(
            input_ids=batch["input_ids"],
            position_ids=None,
            attention_mask=batch["attention_mask"],
            past_key_values=None,
            labels=None,
            images=batch["images"],
            spatial_features=batch.get("spatial_features"),
            geometry_spatial_features=batch.get("geometry_spatial_features"),
            point_maps=batch.get("point_maps"),
            modalities=batch.get("modalities"),
            image_sizes=batch.get("image_sizes"),
            return_visual_metadata=True,
        )
    input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, _labels, visual_metadata = prepared
    actual_metadata = visual_metadata[0]
    actual_gt, actual_mask, actual_debug = build_bev_targets_from_point_maps(
        [payload],
        [actual_metadata],
        bev_point_map_key=args.bev_point_map_key,
        use_geometry_confidence_mask=args.use_geometry_confidence_mask,
        bev_conf_threshold=args.bev_conf_threshold,
    )
    with torch.no_grad():
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
    hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
    visual_indices = actual_metadata["visual_token_indices"].to(device=hidden.device)
    visual_hidden = hidden[0, visual_indices].unsqueeze(0)
    if visual_hidden.shape[:2] != actual_gt.shape[:2]:
        raise RuntimeError(f"visual_hidden {tuple(visual_hidden.shape[:2])} != bev_gt {tuple(actual_gt.shape[:2])}")
    if visual_hidden.shape[:2] != actual_mask.shape[:2]:
        raise RuntimeError(f"visual_hidden {tuple(visual_hidden.shape[:2])} != bev_mask {tuple(actual_mask.shape[:2])}")

    prediction_report: Dict[str, Any] = {
        "prediction_available": False,
        "reason": None,
    }
    if args.save_prediction_plots:
        bev_head = getattr(model, "bev_head", None)
        if bev_head is None and args.allow_random_bev_head:
            init_fn = getattr(model, "initialize_bev_head", None)
            if not callable(init_fn):
                raise RuntimeError("Model does not expose initialize_bev_head(); cannot create a BEV head.")
            bev_head = init_fn(device=device, dtype=dtype)
            prediction_report["random_bev_head_initialized"] = True

        if bev_head is None:
            prediction_report["reason"] = (
                "Model checkpoint/config does not include bev_head. Re-run with a BEV-supervised "
                "checkpoint or pass --allow-random-bev-head for a wiring-only diagnostic."
            )
        else:
            bev_head.eval()
            head_param = next(bev_head.parameters(), None)
            head_device = head_param.device if head_param is not None else visual_hidden.device
            head_dtype = head_param.dtype if head_param is not None else visual_hidden.dtype
            with torch.no_grad():
                pred_norm = bev_head(visual_hidden.to(device=head_device, dtype=head_dtype))
            pred_meter = (pred_norm.detach().cpu().float() * float(args.bev_coord_scale))
            pred_path = output_dir / "bev_pred_debug_sample.png"
            overlay_path = output_dir / "bev_overlay_debug_sample.png"
            save_bev_scatter(pred_path, pred_meter, actual_mask, actual_metadata, title="BEV predicted visual-token coordinates")
            save_bev_overlay(overlay_path, actual_gt, pred_meter, actual_mask, actual_metadata)
            prediction_report.update({
                "prediction_available": True,
                "reason": None,
                "bev_pred_shape": list(pred_meter.shape),
                "bev_pred_xz_min_max": bev_minmax(pred_meter, actual_mask),
                "outputs": {
                    "pred_png": str(pred_path),
                    "overlay_png": str(overlay_path),
                },
            })

    return {
        "ran_model_forward": True,
        "actual_visual_tokens": int(actual_metadata["visual_token_indices"].numel()),
        "gt_only_visual_tokens": int(gt_only_metadata["visual_token_indices"].numel()),
        "visual_hidden_shape": list(visual_hidden.shape),
        "bev_gt_shape": list(actual_gt.shape),
        "bev_mask_shape": list(actual_mask.shape),
        "bev_valid_shape": list(actual_mask.shape),
        "valid_ratio_overall": float(actual_mask.float().mean().item()) if actual_mask.numel() else 0.0,
        "valid_ratio_by_frame": valid_ratio_by_frame(actual_mask, actual_metadata),
        "bev_gt_xz_min_max": bev_minmax(actual_gt, actual_mask),
        "prediction": prediction_report,
        "actual_bev_debug": actual_debug,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data-json", "--data-yaml", dest="train_data_json", default=str(REPO_ROOT / "scripts" / "VLM_3R" / "vsibench_data.yaml"))
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--spatial-features-subdir", default="spatial_features")
    parser.add_argument("--point-maps-subdir", default="spatial_features_points")
    parser.add_argument("--bev-point-map-key", default="point_maps_ref")
    parser.add_argument("--bev-conf-threshold", type=float, default=0.0)
    parser.add_argument("--bev-coord-scale", type=float, default=10.0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--video", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-visual-grid-shape", default="27x27")
    parser.add_argument("--visual-grid-shape", default=None)
    parser.add_argument("--mm-spatial-pool-stride", type=int, default=2)
    parser.add_argument("--mm-newline-position", default="grid", choices=["grid", "frame", "one_token", "no_token"])
    parser.add_argument("--use-geometry-confidence-mask", type=str2bool, default=True)
    parser.add_argument("--run-model-forward", action="store_true")
    parser.add_argument("--save-prediction-plots", action="store_true")
    parser.add_argument("--allow-random-bev-head", action="store_true")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-base", default="/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/LLaVA-NeXT-Video-7B-Qwen2")
    parser.add_argument("--model-name", default="vlm-3r-llava-qwen2-lora")
    parser.add_argument("--image-folder", default=None)
    parser.add_argument("--video-folder", default=None)
    parser.add_argument("--frames-upbound", type=int, default=32)
    parser.add_argument("--add-time-instruction", type=str2bool, default=None)
    parser.add_argument("--force-sample", type=str2bool, default=True)
    parser.add_argument("--pool-mode", default="bilinear", choices=["bilinear", "average", "max"])
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--runtime-root", default=str(REPO_ROOT / ".offline_runtime"))
    parser.add_argument("--siglip-path", default=None)
    parser.add_argument("--cut3r-weights", default=None)
    parser.add_argument("--skip-spatial-tower-load", type=str2bool, default=True)
    parser.add_argument("--zero-spatial-features", type=str2bool, default=False)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.spatial_feature_dir = args.feature_root
    args.image_folder = args.image_folder or args.feature_root
    args.video_folder = args.video_folder or args.feature_root

    records = load_yaml_dataset_records(Path(args.train_data_json))
    dataset_index, record = select_record(records, args)
    video_rel = str(record.get("video", ""))
    if not video_rel:
        raise ValueError(f"Selected record {dataset_index} does not contain a video path")
    sidecar_path = resolve_sidecar_path(video_rel, Path(args.feature_root), args.point_maps_subdir)
    if sidecar_path is None:
        raise FileNotFoundError(
            f"Missing CUT3R point-map sidecar for {video_rel} under root={args.feature_root}, "
            f"subdir={args.point_maps_subdir}"
        )
    payload = load_point_map_sidecar(sidecar_path)
    target_pm = first_point_map(payload, args.bev_point_map_key)
    num_frames, source_h, source_w = point_map_shape_fhw(target_pm)
    grid_shape = infer_visual_grid_shape(args)
    raw_grid_shape = parse_grid(args.raw_visual_grid_shape)
    metadata = build_gt_only_visual_metadata(
        num_frames=num_frames,
        grid_shape=grid_shape,
        raw_grid_shape=raw_grid_shape,
        newline_position=args.mm_newline_position,
    )
    bev_gt, bev_mask, debug = build_bev_targets_from_point_maps(
        [payload],
        [metadata],
        bev_point_map_key=args.bev_point_map_key,
        use_geometry_confidence_mask=args.use_geometry_confidence_mask,
        bev_conf_threshold=args.bev_conf_threshold,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "bev_gt_debug_sample.png"
    json_path = output_dir / "bev_debug_sample.json"
    legacy_json_path = output_dir / "bev_gt_debug_sample.json"
    save_bev_plot(png_path, bev_gt, bev_mask, metadata)

    report = {
        "dataset_index": dataset_index,
        "video": video_rel,
        "sidecar_path": str(sidecar_path),
        "spatial_features_subdir": args.spatial_features_subdir,
        "point_maps_subdir": args.point_maps_subdir,
        "bev_target_key": args.bev_point_map_key,
        "bev_conf_threshold": args.bev_conf_threshold,
        "bev_coord_scale": args.bev_coord_scale,
        "coordinate_axis_mapping": {
            "bev_x": "point[..., 0]",
            "height_assumption": "point[..., 1]",
            "bev_z": "point[..., 2]",
        },
        "source_hw": [source_h, source_w],
        "gt_only_grid_shape": list(grid_shape),
        "gt_only_raw_visual_grid_shape": list(raw_grid_shape),
        "gt_only_metadata_note": (
            "GT-only mode infers metadata from grid args. Run with --run-model-forward "
            "once before training to verify actual model visual-token metadata."
        ),
        "bev_gt_shape": list(bev_gt.shape),
        "bev_mask_shape": list(bev_mask.shape),
        "bev_valid_shape": list(bev_mask.shape),
        "valid_ratio_overall": float(bev_mask.float().mean().item()) if bev_mask.numel() else 0.0,
        "valid_ratio_by_frame": valid_ratio_by_frame(bev_mask, metadata),
        "bev_gt_xz_min_max": bev_minmax(bev_gt, bev_mask),
        "bev_debug": debug,
        "outputs": {
            "gt_png": str(png_path),
            "json": str(json_path),
            "legacy_json": str(legacy_json_path),
        },
    }
    if args.run_model_forward:
        report["model_forward_check"] = run_optional_model_forward(args, dataset_index, payload, metadata, output_dir)
    else:
        report["model_forward_check"] = {"ran_model_forward": False}

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(report), f, indent=2, sort_keys=True)
        f.write("\n")
    if legacy_json_path != json_path:
        with legacy_json_path.open("w", encoding="utf-8") as f:
            json.dump(jsonable(report), f, indent=2, sort_keys=True)
            f.write("\n")
    print(json.dumps({"ok": True, "gt_png": str(png_path), "json": str(json_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
