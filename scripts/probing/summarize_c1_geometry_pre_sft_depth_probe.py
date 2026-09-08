#!/usr/bin/env python
"""Write compact official C1 GeoRoPE pre-SFT depth-probe comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


LAYERS = [0, 1, 2, 3, 6, 9, 15, 21, 27]
PRE_LLM = ["fusion_output", "projected_features"]
EXPECTED_VALIDATION_TOKENS = 75_656


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def read_baseline(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_json(path)
    require(isinstance(rows, list), f"Expected result rows in {path}")
    result = {str(row["feature_level"]): row for row in rows}
    required = {f"layer_{layer}" for layer in LAYERS}
    require(required <= set(result), f"C1 VLM3R baseline lacks requested layers: {sorted(required - set(result))}")
    return result


def metric(path: Path) -> dict[str, Any]:
    row = load_json(path)
    require(int(row.get("num_tokens", -1)) == EXPECTED_VALIDATION_TOKENS, f"Not the official validation population: {path}")
    for key in ("mae", "absrel", "delta125"):
        require(math.isfinite(float(row[key])), f"Non-finite {key}: {path}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--architecture", required=True, choices=("geo_rope_fusion", "visual_geo_rope"))
    parser.add_argument("--activation-json", required=True)
    parser.add_argument("--c1-reference-json", required=True)
    parser.add_argument("--baseline-results", default="/home/shaoruei/probe_cache/c1_vlm3r_v1/full/c1_vlm3r_depth_probe_results.json")
    parser.add_argument("--baseline-provenance", default="/home/shaoruei/probe_cache/c1_vlm3r_v1/full/features/c1_vlm3r/extraction_provenance.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = Path(args.output_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    provenance_path = root / "features" / args.model_label / "extraction_provenance.json"
    provenance = load_json(provenance_path)
    activation = load_json(Path(args.activation_json))
    reference = load_json(Path(args.c1_reference_json))
    baseline_provenance = load_json(Path(args.baseline_provenance))
    require(provenance.get("model_loading_mode") == "pre_sft_fusion", "Not a pre-SFT extraction")
    require(provenance.get("c1_calibration_sha256") == activation.get("reference_c1_artifact_sha256"), "C1 dense maps differ from calibration")
    require(provenance.get("sample_indices_sha256") == baseline_provenance.get("sample_indices_sha256"), "Probe split differs from C1 VLM3R")
    require(math.isclose(float(activation["r0"]), float(reference["r0"]), rel_tol=0.0, abs_tol=1e-12), "Wrong C1 r0")

    baseline = read_baseline(Path(args.baseline_results))
    rows: list[dict[str, Any]] = []
    for level in PRE_LLM + [f"layer_{layer}" for layer in LAYERS]:
        candidate = metric(root / "probes" / args.model_label / level / "metrics.json")
        row = {
            "feature_level": level,
            "candidate_mae": candidate["mae"],
            "candidate_absrel": candidate["absrel"],
            "candidate_delta125": candidate["delta125"],
            "num_tokens": candidate["num_tokens"],
        }
        if level in baseline:
            base = baseline[level]
            row.update({
                "c1_vlm3r_mae": base["mae"], "delta_mae": candidate["mae"] - base["mae"],
                "c1_vlm3r_absrel": base["absrel"], "delta_absrel": candidate["absrel"] - base["absrel"],
                "c1_vlm3r_delta125": base["delta125"], "delta_delta125": candidate["delta125"] - base["delta125"],
            })
        rows.append(row)

    stem = f"c1_{args.architecture}_depth_probe"
    with (output / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / f"{stem}.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = [
        f"# C1 {args.architecture} pre-SFT depth probe",
        "",
        "| Feature | Candidate MAE | Candidate AbsRel | Candidate δ<1.25 | C1 VLM3R MAE | C1 VLM3R AbsRel | C1 VLM3R δ<1.25 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        baseline_cells = (
            f"{row['c1_vlm3r_mae']:.6f} | {row['c1_vlm3r_absrel']:.6f} | {row['c1_vlm3r_delta125']:.6f}"
            if "c1_vlm3r_mae" in row else "— | — | —"
        )
        markdown.append(
            f"| {row['feature_level']} | {row['candidate_mae']:.6f} | {row['candidate_absrel']:.6f} | "
            f"{row['candidate_delta125']:.6f} | {baseline_cells} |"
        )
    (output / f"{stem}.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    metadata = {
        "schema_version": "c1_geometry_pre_sft_depth_probe_v1",
        "architecture": args.architecture,
        "pre_sft_initialization": "C1 structured-isometric dense maps; no post-SFT architecture weights",
        "geometry_source": "predicted CUT3R point_maps_ref",
        "activation": activation["activation"],
        "r0": activation["r0"],
        "achieved_ratio": activation["achieved_ratio"],
        "activation_diagnostics": activation.get("diagnostics"),
        "c1_artifact": str(Path(args.c1_reference_json).resolve()),
        "c1_artifact_sha256": provenance.get("c1_calibration_sha256"),
        "activation_artifact": str(Path(args.activation_json).resolve()),
        "cache_path": str(root.resolve()),
        "layer_list": LAYERS,
        "pre_llm_levels": PRE_LLM,
        "probe_protocol": {"seed": 0, "epochs": 50, "batch_size": 32, "lr": 1e-3, "patience": 10},
        "ordinary_visual_token_readout": "32 x 196 only; no geometry/object tokens",
        "validation_tokens_per_level": EXPECTED_VALIDATION_TOKENS,
        "extraction_provenance": str(provenance_path.resolve()),
    }
    (output / f"{stem}_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\n".join(markdown))
    print(json.dumps({"metadata": str(output / f'{stem}_metadata.json')}, indent=2))


if __name__ == "__main__":
    main()
