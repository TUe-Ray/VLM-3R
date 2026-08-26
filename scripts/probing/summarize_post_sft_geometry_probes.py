#!/usr/bin/env python
"""Write the four-model post-SFT geometry depth-probe comparison."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Allow direct execution via an absolute script path (as used by local
# experiment wrappers) without requiring callers to pre-set PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probing.post_sft_geometry_probe_specs import (
    MODEL_SPECS,
    POST_SFT_DEPTH_FEATURE_LEVELS,
)


ROOTS = {
    "eomt_object": Path("/home/shaoruei/probe_outputs/post_sft_eomt_object_full_20260825"),
    "eomt_selective": Path("/home/shaoruei/probe_outputs/post_sft_eomt_selective_full_20260825"),
    "geo_rope_fusion": Path("/home/shaoruei/probe_outputs/post_sft_geo_rope_fusion_full_20260823"),
    "visual_3d_rope": Path("/home/shaoruei/probe_outputs/post_sft_visual_3d_rope_full_20260823"),
}


def read_metrics(root: Path, model: str, feature: str) -> dict:
    path = root / "probes" / model / feature / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if int(value.get("num_tokens", -1)) != 75656:
        raise RuntimeError(f"Unexpected validation-token count in {path}")
    return value


def architecture_record(root: Path, model: str) -> dict:
    path = root / "features" / model / "fusion_output" / "provenance.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    provenance = json.loads(path.read_text(encoding="utf-8"))
    contract = provenance.get("post_sft_config_contract", {})
    return {
        "model": model,
        "checkpoint": MODEL_SPECS[model].checkpoint_name,
        "architecture": provenance.get("post_sft_architecture"),
        "config_assessment": contract.get("assessment"),
        "fusion_block": contract.get("fusion_block"),
        "use_geometry_aware_projection": contract.get("use_geometry_aware_projection"),
        "geometry_point_map_key": contract.get("point_map_key"),
        "hidden_state_indexing": provenance.get("hidden_state_indexing"),
        "main_representation": "ordinary frame-aligned visual tokens only (32x196)",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/home/shaoruei/probe_outputs/post_sft_geometry_comparison_20260825"))
    args = parser.parse_args()
    rows = []
    for model, root in ROOTS.items():
        spec = MODEL_SPECS[model]
        for feature in POST_SFT_DEPTH_FEATURE_LEVELS:
            metric = read_metrics(root, model, feature)
            rows.append(
                {
                    "model": model,
                    "checkpoint": spec.checkpoint_name,
                    "architecture": spec.architecture,
                    "feature_level": feature,
                    "mae": float(metric["mae"]),
                    "absrel": float(metric["absrel"]),
                    "delta125": float(metric["delta125"]),
                    "best_epoch": int(metric["best_epoch"]),
                    "num_tokens": int(metric["num_tokens"]),
                }
            )
    if len(rows) != 56:
        raise RuntimeError(f"Expected 56 post-SFT metrics rows, got {len(rows)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "results.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (args.output_root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| Model | Level | MAE | AbsRel | δ<1.25 |", "|---|---|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['feature_level']} | {row['mae']:.6f} | {row['absrel']:.6f} | {row['delta125']:.6f} |"
        )
    (args.output_root / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    architectures = [architecture_record(root, model) for model, root in ROOTS.items()]
    (args.output_root / "architecture_verification.json").write_text(
        json.dumps(architectures, indent=2) + "\n", encoding="utf-8"
    )
    verification_lines = [
        "# Architecture verification",
        "",
        "All primary probe features exclude EoMT/object/camera/geometry auxiliary sequence tokens.",
        "",
        "| Model | Fusion | Geometry projection | Point-map key | Contract |",
        "|---|---|---|---|---|",
    ]
    for item in architectures:
        verification_lines.append(
            f"| {item['model']} | {item['fusion_block']} | {item['use_geometry_aware_projection']} | "
            f"{item['geometry_point_map_key']} | {item['config_assessment']} |"
        )
    (args.output_root / "architecture_verification.md").write_text(
        "\n".join(verification_lines) + "\n", encoding="utf-8"
    )
    ranked = sorted(rows, key=lambda row: (row["mae"], row["absrel"], -row["delta125"], row["feature_level"]))
    winner = ranked[0]
    (args.output_root / "c1_candidate_recommendation.md").write_text(
        "# C1 candidate\n\n"
        f"Recommend `{winner['model']}` at `{winner['feature_level']}`: "
        f"MAE={winner['mae']:.6f}, AbsRel={winner['absrel']:.6f}, "
        f"delta<1.25={winner['delta125']:.6f}.\n",
        encoding="utf-8",
    )
    interpretation = ["# Geometry propagation interpretation", ""]
    for model in ROOTS:
        model_rows = [row for row in rows if row["model"] == model]
        best = min(model_rows, key=lambda row: row["mae"])
        early = next(row for row in model_rows if row["feature_level"] == "layer_0")
        late = next(row for row in model_rows if row["feature_level"] == "layer_27")
        direction = "improves" if late["mae"] < early["mae"] else "degrades"
        interpretation.append(
            f"- `{model}` reaches its lowest MAE at `{best['feature_level']}` "
            f"({best['mae']:.6f}); L0→L27 {direction} MAE "
            f"({early['mae']:.6f}→{late['mae']:.6f})."
        )
    interpretation.append("")
    interpretation.append(
        "This is a representation-probe trend, not a causal claim about the model's geometric mechanism."
    )
    (args.output_root / "geometry_propagation_interpretation.md").write_text(
        "\n".join(interpretation) + "\n", encoding="utf-8"
    )
    try:
        import matplotlib.pyplot as plt

        labels = list(POST_SFT_DEPTH_FEATURE_LEVELS)
        for metric, filename, ylabel in (
            ("mae", "depth_probe_mae.png", "MAE (m)"),
            ("absrel", "depth_probe_absrel.png", "AbsRel"),
            ("delta125", "depth_probe_delta125.png", "delta < 1.25"),
        ):
            figure, axis = plt.subplots(figsize=(11, 5))
            for model in ROOTS:
                by_feature = {row["feature_level"]: row for row in rows if row["model"] == model}
                axis.plot(range(len(labels)), [by_feature[label][metric] for label in labels], marker="o", label=model)
            axis.set_xticks(range(len(labels)), [label.replace("layer_", "L") for label in labels], rotation=30, ha="right")
            axis.set_ylabel(ylabel)
            axis.legend()
            figure.tight_layout()
            figure.savefig(args.output_root / filename, dpi=180)
            plt.close(figure)
    except ImportError:
        pass
    print(args.output_root)


if __name__ == "__main__":
    main()
