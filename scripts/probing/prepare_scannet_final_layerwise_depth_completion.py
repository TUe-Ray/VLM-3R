#!/usr/bin/env python
"""CPU-only inventory and command planner for final ScanNet layer-wise probes.

This intentionally does not import a model or touch CUDA.  It identifies only
complete, durable probe points and emits one isolated cache namespace per
model; the companion shell runner executes those namespaces sequentially.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from depth_probe_common import load_frame_records


EXPECTED_SPLIT_SHA256 = "d478cb684958dfc25066821ec83d5216469577c9e282e33bdf87d3c88b200d8e"
EXPECTED_TOKENS = 75_656
ROOT = Path("/mnt/DATA_SSD/shaoruei/models/vlm3r_runs")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def variants() -> list[dict[str, Any]]:
    ss_depth = ROOT / "cut3r_spatialstack_d2_pointmap_45457911"
    ss_depth_label = "cut3r_spatialstack_d2_pointmap_45457911"
    return [
        {"display": "SS", "label": "cut3r_spatialstack_44323703", "checkpoint": ROOT / "cut3r_spatialstack_44323703", "preset": "spatialstack", "levels": ["siglip_output", "projected_features", "layer_12", "layer_18", "layer_24"]},
        {"display": "SS at L1/L2/L3", "label": "cut3r_spatialstack_token_mlp_dec6_9_12_llm1_2_3_4n", "checkpoint": ROOT / "cut3r_spatialstack_token_mlp_dec6_9_12_llm1_2_3_4n", "preset": "spatialstack", "levels": ["siglip_output", "projected_features", "layer_12", "layer_18", "layer_24"]},
        {"display": "SS + depth", "label": ss_depth_label, "checkpoint": ss_depth, "preset": "spatialstack", "levels": ["siglip_output", "projected_features", "layer_12", "layer_18", "layer_24"], "depth_supervision_alias": "point_map_supervision"},
        {"display": "SS cross-attention", "label": "cut3r_spatialstack_cross_attn_45303862", "checkpoint": ROOT / "cut3r_spatialstack_cross_attn_45303862", "preset": "spatialstack", "levels": ["siglip_output", "projected_features", "layer_12", "layer_18", "layer_24"]},
        {"display": "baseline + depth", "label": "cut3r_depth_loss_43817021", "checkpoint": ROOT / "cut3r_depth_loss_43817021", "preset": "original", "levels": ["layer_1", "layer_2", "layer_12", "layer_18", "layer_24"]},
    ]


def valid_metric(path: Path, label: str, level: str) -> bool:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    required = [path.parent / "history.json", path.parent / "best.pt"]
    return (
        all(item.is_file() for item in required)
        and row.get("model_label") == label
        and row.get("feature_level") == level
        and int(row.get("num_tokens", -1)) == EXPECTED_TOKENS
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-indices", required=True)
    parser.add_argument("--durable-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--print-commands", action="store_true")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()

    split = Path(args.sample_indices)
    split_hash = sha256(split)
    records = load_frame_records(split)
    report: dict[str, Any] = {
        "mode": "cpu_preflight_only",
        "cuda_or_model_loaded": False,
        "sample_indices": str(split),
        "split_sha256": split_hash,
        "split_identity_pass": split_hash == EXPECTED_SPLIT_SHA256,
        "frame_records": len(records),
        "expected_frame_records": 2398,
        "expected_validation_tokens": EXPECTED_TOKENS,
        "models": [],
    }
    durable = Path(args.durable_root)
    for spec in variants():
        checkpoint = Path(spec["checkpoint"])
        item = {**spec, "checkpoint": str(checkpoint)}
        item["checkpoint_exists"] = checkpoint.is_dir()
        config_path = checkpoint / "config.json"
        item["checkpoint_config_sha256"] = sha256(config_path) if config_path.is_file() else None
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            item["spatialstack_depth_contract"] = {
                "use_cut3r_spatialstack": config.get("use_cut3r_spatialstack"),
                "use_depth_supervision": config.get("use_depth_supervision"),
            }
        present, missing = [], []
        for level in spec["levels"]:
            metric = durable / "probes" / spec["label"] / level / "metrics.json"
            (present if valid_metric(metric, spec["label"], level) else missing).append(level)
        item["already_present_valid"] = present
        item["actually_missing"] = missing
        item["cache_namespace"] = str(Path(args.cache_root) / "final_layerwise" / spec["label"])
        item["durable_namespace"] = str(durable / "probes" / spec["label"])
        report["models"].append(item)
    report["assessment"] = "PASS" if report["split_identity_pass"] and len(records) == 2398 else "FAIL"
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.write_summary:
        rows = []
        for item in report["models"]:
            for level in item["levels"]:
                metric = durable / "probes" / item["label"] / level / "metrics.json"
                if valid_metric(metric, item["label"], level):
                    rows.append({"display": item["display"], **json.loads(metric.read_text(encoding="utf-8"))})
        summary_root = durable / "summary" / "final_layerwise"
        summary_root.mkdir(parents=True, exist_ok=True)
        (summary_root / "graph_data.json").write_text(
            json.dumps({"feature_order": ["siglip_output", "projected_features", "layer_1", "layer_2", "layer_12", "layer_18", "layer_24"], "metric_order": ["mae", "absrel", "delta125"], "rows": rows}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        lines = ["# ScanNet final layer-wise depth probes", "", "| Variant | Feature | MAE | AbsRel | δ<1.25 |", "|---|---|---:|---:|---:|"]
        for row in rows:
            lines.append(f"| {row['display']} | {row['feature_level']} | {row['mae']:.6f} | {row['absrel']:.6f} | {row['delta125']:.6f} |")
        (summary_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] CPU-only preflight: {output}")
    for item in report["models"]:
        print(f"[MODEL] {item['display']}: label={item['label']}")
        print(f"  present={','.join(item['already_present_valid']) or '-'}")
        print(f"  missing={','.join(item['actually_missing']) or '-'}")
        if item.get("unresolved_reason") and not item["checkpoint_exists"]:
            print(f"  UNRESOLVED: {item['unresolved_reason']}")
        if args.print_commands and item["actually_missing"] and item["checkpoint_exists"]:
            print(
                "  launch: "
                f"GPU=<physical_gpu> CUDA_DEVICES=<visible_gpus> bash scripts/probing/run_scannet_final_layerwise_depth_completion_local.sh "
                f"run-one {item['label']}"
            )


if __name__ == "__main__":
    main()
