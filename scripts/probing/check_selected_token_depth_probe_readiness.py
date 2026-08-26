#!/usr/bin/env python
"""Fail-closed readiness check for the EoMT selected-token depth comparison.

This check deliberately does *not* convert a CUT3R K/V mask into a VLM
visual-token mask.  The post-SFT selective-fusion forward applies its EoMT
mask to 27x27 CUT3R keys/values, before cross-attention.  The cached depth
probe representation instead consists of 14x14 ordinary VLM visual-query
tokens.  There is no checkpoint-defined one-to-one selected-VLM-token support
to fabricate by resizing or thresholding the soft mask.

It also verifies the checkpoint-defined query selector (things-only,
confidence >= 0.8) over every authoritative selected probe frame.  An empty
support, a missing baseline feature cache, or a domain mismatch results in a
BLOCKED report instead of training a misleading probe.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(
    "/home/shaoruei/probe_provenance/scannet_baseline_L6/"
    "scannet_baseline_L6_depth_provenance/splits/"
    "semantic_probe_scannet_final_usable_sample_indices.json"
)
DEFAULT_SELECTIVE_ROOT = Path("/home/shaoruei/probe_outputs/post_sft_eomt_selective_full_20260825")
DEFAULT_EOMT_CACHE_ROOT = Path("/home/shaoruei/probe_cache/eomt_consumer_grid_v1")
DEFAULT_BASELINE_ROOT = Path(
    "/home/shaoruei/probe_cache/scannet_baseline_replicates_v1/full/baseline_apr05_reproduction"
)
DEFAULT_BASELINE_PROVENANCE = Path(
    "/home/shaoruei/probe_outputs/scannet_baseline_replicates_v1/provenance/"
    "baseline_apr05_reproduction/extraction_provenance.json"
)
EXPECTED_LEVELS = (
    "fusion_output",
    "projected_features",
    "layer_0",
    "layer_1",
    "layer_2",
    "layer_3",
    "layer_6",
    "layer_9",
    "layer_12",
    "layer_15",
    "layer_18",
    "layer_21",
    "layer_24",
    "layer_27",
)
NUM_FRAMES = 32
NUM_QUERIES = 200
NUM_CLASSES_WITH_NO_OBJECT = 134
THING_CLASS_COUNT = 80
SELECTIVE_SCORE_THRESHOLD = 0.8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return payload


def frame_records(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for video in manifest.get("videos", []):
        for frame in video.get("frames", []):
            fsid = str(frame["frame_sample_id"])
            if fsid in records:
                raise RuntimeError(f"Duplicate frame sample ID in manifest: {fsid}")
            records[fsid] = {
                "frame_index": int(frame["frame_index"]),
                "scene_id": str(video["scene_id"]),
                "split": str(frame["split"]),
                "video_sample_id": str(video["video_sample_id"]),
            }
    return records


def selective_cache_inventory(root: Path, records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metadata_dir = root / "metadata"
    gt_dir = root / "gt_depth"
    metadata_paths = sorted(metadata_dir.glob("frame_*.pt"))
    metadata_ids = {path.stem.removeprefix("frame_") for path in metadata_paths}
    result: dict[str, Any] = {
        "root": str(root),
        "metadata_frames": len(metadata_paths),
        "gt_frames": len(list(gt_dir.glob("frame_*.pt"))),
        "manifest_frames": len(records),
        "missing_metadata_ids": sorted(set(records) - metadata_ids)[:10],
        "unexpected_metadata_ids": sorted(metadata_ids - set(records))[:10],
        "feature_levels": {},
        "metadata_smoke": [],
    }
    for level in EXPECTED_LEVELS:
        feature_dir = root / "features" / "eomt_selective" / level
        result["feature_levels"][level] = len(list(feature_dir.glob("frame_*.pt")))

    if len(metadata_paths) != len(records) or set(records) != metadata_ids:
        return result

    smoke_indices = (0, len(metadata_paths) // 2, len(metadata_paths) - 1)
    for index in smoke_indices:
        path = metadata_paths[index]
        fsid = path.stem.removeprefix("frame_")
        meta = torch.load(path, map_location="cpu")
        expected = records[fsid]
        visual_indices = meta.get("visual_token_indices")
        gt_valid = meta.get("gt_valid_mask")
        gt = torch.load(gt_dir / path.name, map_location="cpu")
        row = {
            "frame_sample_id": fsid,
            "scene_id_matches_manifest": str(meta.get("scene_id")) == expected["scene_id"],
            "frame_index_matches_manifest": int(meta.get("frame_index", -1)) == expected["frame_index"],
            "split_matches_manifest": str(meta.get("split")) == expected["split"],
            "ordinary_visual_grid": list(meta.get("visual_grid_shape", ())),
            "ordinary_visual_token_count": int(visual_indices.numel()) if isinstance(visual_indices, torch.Tensor) else None,
            "gt_shape": list(gt.shape) if isinstance(gt, torch.Tensor) else None,
            "gt_valid_shape": list(gt_valid.shape) if isinstance(gt_valid, torch.Tensor) else None,
            "gt_valid_tokens": int(gt_valid.sum()) if isinstance(gt_valid, torch.Tensor) else None,
        }
        result["metadata_smoke"].append(row)
    return result


def one_scene_selection(
    scene_id: str,
    frames: list[dict[str, Any]],
    class_root: Path,
) -> list[dict[str, Any]]:
    payload = torch.load(class_root / f"{scene_id}.pt", map_location="cpu")
    logits = payload.get("class_logits") if isinstance(payload, dict) else None
    if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (
        NUM_FRAMES,
        NUM_QUERIES,
        NUM_CLASSES_WITH_NO_OBJECT,
    ):
        raise RuntimeError(f"{scene_id}: invalid cached class logits shape {getattr(logits, 'shape', None)}")
    if logits.dtype != torch.float32 or not torch.isfinite(logits).all():
        raise RuntimeError(f"{scene_id}: cached class logits are not finite FP32")
    probabilities = torch.softmax(logits.float(), dim=-1)[..., :-1]
    scores, class_ids = probabilities.max(dim=-1)
    rows = []
    for frame in frames:
        frame_index = int(frame["frame_index"])
        things = class_ids[frame_index] < THING_CLASS_COUNT
        selected = (scores[frame_index] >= SELECTIVE_SCORE_THRESHOLD) & things
        rows.append(
            {
                "frame_sample_id": frame["frame_sample_id"],
                "selected_query_count": int(selected.sum().item()),
                "max_things_confidence": float(scores[frame_index][things].max().item()),
            }
        )
    return rows


def selector_summary(records: dict[str, dict[str, Any]], class_root: Path) -> dict[str, Any]:
    scenes: dict[str, list[dict[str, Any]]] = {}
    for fsid, record in records.items():
        scenes.setdefault(record["scene_id"], []).append({"frame_sample_id": fsid, **record})
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        batches = list(executor.map(lambda item: one_scene_selection(item[0], item[1], class_root), scenes.items()))
    rows = [row for batch in batches for row in batch]
    counts = [row["selected_query_count"] for row in rows]
    maxima = [row["max_things_confidence"] for row in rows]
    return {
        "selector": {
            "things_only": True,
            "score_threshold": SELECTIVE_SCORE_THRESHOLD,
            "zero_3d_empty_fallback": True,
        },
        "selected_frames_checked": len(rows),
        "selected_query_total": sum(counts),
        "frames_with_selected_queries": sum(count > 0 for count in counts),
        "frames_without_selected_queries": sum(count == 0 for count in counts),
        "max_things_confidence": {
            "min": min(maxima),
            "mean": sum(maxima) / len(maxima),
            "max": max(maxima),
        },
    }


def mask_smoke(records: dict[str, dict[str, Any]], mask_root: Path) -> list[dict[str, Any]]:
    ordered = sorted(records.items())
    rows = []
    for index in (0, len(ordered) // 2, len(ordered) - 1):
        fsid, record = ordered[index]
        path = mask_root / f"{record['scene_id']}.pt"
        payload = torch.load(path, map_location="cpu")
        masks = payload.get("soft_masks") if isinstance(payload, dict) else None
        expected_shape = (NUM_FRAMES, NUM_QUERIES, 27, 27)
        rows.append(
            {
                "frame_sample_id": fsid,
                "scene_id": record["scene_id"],
                "shape": list(masks.shape) if isinstance(masks, torch.Tensor) else None,
                "dtype": str(masks.dtype) if isinstance(masks, torch.Tensor) else None,
                "finite": bool(torch.isfinite(masks).all()) if isinstance(masks, torch.Tensor) else False,
                "matches_expected_cut3r_kv_shape": bool(isinstance(masks, torch.Tensor) and tuple(masks.shape) == expected_shape),
            }
        )
    return rows


def baseline_inventory(root: Path, label: str, expected_frame_ids: set[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"root": str(root), "label": label, "exists": root.is_dir()}
    if not root.is_dir():
        result["reason"] = "The formal reproduction feature cache directory is absent."
        return result
    metadata_ids = {path.stem.removeprefix("frame_") for path in (root / "metadata").glob("frame_*.pt")}
    result.update(
        {
            "metadata_frames": len(metadata_ids),
            "same_frame_ids_as_selective": metadata_ids == expected_frame_ids,
            "missing_frame_ids": sorted(expected_frame_ids - metadata_ids)[:10],
            "feature_levels": {
                level: len(list((root / "features" / label / level).glob("frame_*.pt")))
                for level in EXPECTED_LEVELS
            },
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--selective-root", type=Path, default=DEFAULT_SELECTIVE_ROOT)
    parser.add_argument("--eomt-cache-root", type=Path, default=DEFAULT_EOMT_CACHE_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--baseline-label", default="baseline_apr05_reproduction")
    parser.add_argument("--baseline-provenance", type=Path, default=DEFAULT_BASELINE_PROVENANCE)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "logs" / "selected_token_depth_probe" / "readiness_20260826.json",
    )
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    records = frame_records(manifest)
    selective = selective_cache_inventory(args.selective_root, records)
    selection = selector_summary(records, args.eomt_cache_root / "class_logits" / "scannet")
    masks = mask_smoke(records, args.eomt_cache_root / "selective_masks" / "scannet")
    baseline = baseline_inventory(args.baseline_root, args.baseline_label, set(records))
    baseline_provenance = read_json(args.baseline_provenance) if args.baseline_provenance.is_file() else None

    blockers = []
    if selective["metadata_frames"] != len(records) or selective["gt_frames"] != len(records):
        blockers.append("SELECTIVE_FRAME_CACHE_INCOMPLETE")
    if any(count != len(records) for count in selective["feature_levels"].values()):
        blockers.append("SELECTIVE_FEATURE_LEVEL_INCOMPLETE")
    if not baseline["exists"]:
        blockers.append("BASELINE_REPRODUCTION_FEATURE_CACHE_MISSING")
    elif not baseline.get("same_frame_ids_as_selective", False):
        blockers.append("BASELINE_SELECTIVE_FRAME_ID_MISMATCH")
    elif any(count != len(records) for count in baseline["feature_levels"].values()):
        blockers.append("BASELINE_FEATURE_LEVEL_INCOMPLETE")
    if selection["selected_query_total"] == 0:
        blockers.append("EMPTY_CHECKPOINT_DEFINED_SELECTIVE_SUPPORT")
    blockers.append("NO_CHECKPOINT_DEFINED_27X27_KV_TO_14X14_VISUAL_TOKEN_MAPPING")

    report = {
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "manifest": {
            "path": str(args.manifest),
            "sha256": sha256(args.manifest),
            "videos": len(manifest.get("videos", [])),
            "selected_frames": len(records),
        },
        "selective_cache": selective,
        "baseline_cache": baseline,
        "baseline_provenance": {
            "path": str(args.baseline_provenance),
            "exists": args.baseline_provenance.is_file(),
            "sample_indices": baseline_provenance.get("sample_indices") if baseline_provenance else None,
            "sample_indices_sha256": baseline_provenance.get("sample_indices_sha256") if baseline_provenance else None,
        },
        "checkpoint_defined_selector": selection,
        "mask_smoke": masks,
        "representation_contract": {
            "depth_probe_tokens": "ordinary frame-aligned VLM visual-query tokens [14,14] / 196",
            "selective_fusion_mask_tokens": "CUT3R cross-attention K/V tokens [27,27] / 729",
            "mapping": "none in the checkpoint forward path; all visual queries use cross-attention to the gated K/V bank",
            "decision": "Do not resize, threshold, or otherwise manufacture a 14x14 selected-visual-token mask.",
        },
        "decision": (
            "Do not train selected-token probes: the actual selector has an empty support and the stored masks do not index "
            "the ordinary visual tokens used by the probe."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "blockers": blockers, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
