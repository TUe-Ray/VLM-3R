#!/usr/bin/env python
"""Check semantic GT tensors against cached feature grids and write final usable subset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from depth_probe_common import DEFAULT_OUTPUT_ROOT, read_json, write_json
from semantic_probe_common import squeeze_singleton_feature_batch, write_csv
from train_depth_probes import available_feature_levels, feature_tensor_path, load_feature_tensor


def parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return parts or None


def iter_frame_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for video in payload.get("videos", []):
        for frame in video.get("frames", []):
            record = dict(video)
            record.pop("frames", None)
            record.update(frame)
            records.append(record)
    return records


def filter_payload_frames(payload: dict[str, Any], kept_frame_ids: set[str]) -> dict[str, Any]:
    videos = []
    for video in payload.get("videos", []):
        frames = [dict(frame) for frame in video.get("frames", []) if str(frame.get("frame_sample_id")) in kept_frame_ids]
        if not frames:
            continue
        cur = dict(video)
        cur["frames"] = frames
        videos.append(cur)
    out = dict(payload)
    out["videos"] = videos
    out["train_videos"] = sum(1 for video in videos if video.get("split") == "train")
    out["val_videos"] = sum(1 for video in videos if video.get("split") == "val")
    out["train_frames"] = sum(len(video.get("frames", [])) for video in videos if video.get("split") == "train")
    out["val_frames"] = sum(len(video.get("frames", [])) for video in videos if video.get("split") == "val")
    return out


def check_feature_shape(feature: torch.Tensor, gt_shape: tuple[int, int]) -> tuple[bool, str]:
    feature = squeeze_singleton_feature_batch(feature)
    h_tok, w_tok = gt_shape
    if feature.ndim == 3:
        if tuple(feature.shape[:2]) == gt_shape:
            return True, "grid_tensor"
        return False, f"grid_shape_mismatch:feature={tuple(feature.shape[:2])}:gt={gt_shape}"
    if feature.ndim == 2:
        if int(feature.shape[0]) == h_tok * w_tok:
            return True, "flat_tokens"
        return False, f"token_count_mismatch:feature={int(feature.shape[0])}:gt={h_tok * w_tok}"
    return False, f"unsupported_feature_rank:{feature.ndim}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--sample-indices",
        default=str(DEFAULT_OUTPUT_ROOT / "semantic_probe_scannet_semantic_aligned_sample_indices.json"),
    )
    parser.add_argument("--model-labels", default="zero_spatial,vlm3r_baseline")
    parser.add_argument("--feature-levels", default=None)
    parser.add_argument("--require-depth-gt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-all-feature-levels", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    payload = read_json(Path(args.sample_indices))
    records = iter_frame_records(payload)
    model_labels = [part.strip() for part in args.model_labels.split(",") if part.strip()]
    level_override = parse_csv(args.feature_levels)
    rows: list[dict[str, Any]] = []
    frame_ok: dict[str, bool] = {str(record["frame_sample_id"]): True for record in records}
    checked_by_frame: dict[str, int] = {str(record["frame_sample_id"]): 0 for record in records}

    for record in records:
        fsid = str(record["frame_sample_id"])
        gt_path = output_root / "semantic_gt_scannet" / f"frame_{fsid}.pt"
        depth_gt_path = output_root / "gt_depth" / f"frame_{fsid}.pt"
        if not gt_path.exists():
            frame_ok[fsid] = False
            rows.append({
                "frame_sample_id": fsid,
                "sample_id": fsid,
                "model_label": "",
                "feature_level": "",
                "feature_cache_path": "",
                "semantic_gt_path": str(gt_path),
                "alignment_successful": False,
                "reason_if_failed": "missing_semantic_gt",
            })
            continue
        gt = torch.load(gt_path, map_location="cpu")
        if gt.ndim != 2:
            frame_ok[fsid] = False
            rows.append({
                "frame_sample_id": fsid,
                "sample_id": fsid,
                "model_label": "",
                "feature_level": "",
                "feature_cache_path": "",
                "semantic_gt_path": str(gt_path),
                "alignment_successful": False,
                "reason_if_failed": f"semantic_gt_rank:{gt.ndim}",
            })
            continue
        if args.require_depth_gt and not depth_gt_path.exists():
            frame_ok[fsid] = False
        frame_failure_reason = "missing_depth_gt" if args.require_depth_gt and not depth_gt_path.exists() else ""
        for model_label in model_labels:
            levels = level_override if level_override is not None else available_feature_levels(model_label)
            for feature_level in levels:
                feature_path = feature_tensor_path(output_root, model_label, feature_level, fsid)
                row = {
                    "frame_sample_id": fsid,
                    "sample_id": fsid,
                    "scene_id": str(record.get("scene_id", "")),
                    "frame_index": int(record.get("raw_frame_index", record.get("frame_index", -1))),
                    "model_label": model_label,
                    "feature_level": feature_level,
                    "feature_cache_path": str(feature_path),
                    "semantic_gt_path": str(gt_path),
                    "depth_gt_path": str(depth_gt_path),
                    "depth_gt_exists": depth_gt_path.exists(),
                    "frame_failure_reason": frame_failure_reason,
                    "semantic_grid_shape": list(gt.shape),
                    "feature_shape": "",
                    "feature_layout": "",
                    "alignment_successful": False,
                    "reason_if_failed": "",
                }
                if not feature_path.exists():
                    row["reason_if_failed"] = "missing_feature_cache"
                    if frame_failure_reason and not row["reason_if_failed"]:
                        row["reason_if_failed"] = frame_failure_reason
                    if args.require_all_feature_levels:
                        frame_ok[fsid] = False
                    rows.append(row)
                    continue
                try:
                    feature = load_feature_tensor(output_root, model_label, feature_level, fsid)
                    feature = squeeze_singleton_feature_batch(feature)
                    row["feature_shape"] = list(feature.shape)
                    ok, layout_or_reason = check_feature_shape(feature, tuple(int(x) for x in gt.shape))
                    row["alignment_successful"] = ok
                    if ok:
                        row["feature_layout"] = layout_or_reason
                        checked_by_frame[fsid] += 1
                        if frame_failure_reason:
                            row["alignment_successful"] = False
                            row["reason_if_failed"] = frame_failure_reason
                    else:
                        row["reason_if_failed"] = layout_or_reason
                        if args.require_all_feature_levels:
                            frame_ok[fsid] = False
                except Exception as exc:
                    row["reason_if_failed"] = f"{type(exc).__name__}:{exc}"
                    if args.require_all_feature_levels:
                        frame_ok[fsid] = False
                rows.append(row)

    for fsid, count in checked_by_frame.items():
        if count == 0:
            frame_ok[fsid] = False
    kept_frame_ids = {fsid for fsid, ok in frame_ok.items() if ok}
    final_payload = filter_payload_frames(payload, kept_frame_ids)
    final_payload["feature_alignment_model_labels"] = model_labels
    final_payload["feature_alignment_feature_levels"] = level_override or "default_per_model"
    final_payload["require_depth_gt"] = bool(args.require_depth_gt)
    final_payload["require_all_feature_levels"] = bool(args.require_all_feature_levels)
    final_payload["final_usable_frames"] = len(kept_frame_ids)

    write_json(output_root / "semantic_feature_alignment_report.json", rows)
    write_csv(output_root / "semantic_feature_alignment_report.csv", rows)
    write_json(output_root / "semantic_probe_scannet_final_usable_sample_indices.json", final_payload)
    print(
        f"[INFO] Final usable frames: {len(kept_frame_ids)}/{len(records)}. "
        f"Wrote {output_root / 'semantic_probe_scannet_final_usable_sample_indices.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
