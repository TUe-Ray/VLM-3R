#!/usr/bin/env python3
"""Compare effective SigLIP-reference and CUT3R-token-only wrapper arguments."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ASSIGNMENT = re.compile(r"^([A-Z][A-Z0-9_]*)=(.*)$")
ARRAY_ENTRY = re.compile(r'^\s*\[([^]]+)\]="?([^"\n]*)"?')
SBATCH = re.compile(r"^#SBATCH\s+--([^=\s]+)(?:=(.*)|\s+(.*))$")
VARIABLE = re.compile(r"\$(?:\{)?([A-Z][A-Z0-9_]*)(?::-[^}]*)?\}?")

INTENTIONAL = {
    "visual_token_source", "cut3r_token_sidecar_key", "cut3r_token_feature_dim",
    "cut3r_token_projector_layernorm", "tune_cut3r_token_projector",
    "cut3r_token_debug_telemetry", "cut3r_token_debug_first_n",
    "spatial_tower_preextracted_only", "fusion_block", "geo_rope_fusion_mode",
    "geo_rope_fusion_max_depth", "geo_rope_fusion_group_split", "tune_fusion_block",
    "tune_mm_mlp_adapter", "require_spatial_features", "strict_video_loading",
}
RUNTIME = {
    "run_name", "output_dir", "max_steps", "report_to", "logging_steps", "save_steps",
    "train_data_max_samples", "cut3r_token_smoke_telemetry", "cut3r_token_smoke_full_scan_steps",
    "slurm_nodes", "slurm_gpus_per_node", "slurm_ntasks_per_node",
}


def _resolve(value: str, variables: dict[str, str]) -> str:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    for _ in range(5):
        updated = VARIABLE.sub(lambda match: variables.get(match.group(1), match.group(0)), value)
        if updated == value:
            break
        value = updated
    return value


def _parse(path: Path) -> dict[str, str]:
    variables: dict[str, str] = {}
    arrays: dict[str, str] = {}
    in_array = False
    for line in path.read_text(encoding="utf-8").splitlines():
        directive = SBATCH.match(line)
        if directive:
            name = directive.group(1).replace("-", "_")
            value = directive.group(2) if directive.group(2) is not None else directive.group(3)
            if name in {"nodes", "gpus_per_node", "ntasks_per_node"}:
                arrays[f"slurm_{name}"] = value
        assignment = ASSIGNMENT.match(line)
        if assignment and not in_array:
            variables[assignment.group(1)] = assignment.group(2)
        if line.startswith("declare -A MODEL_ARGS=(") or line.startswith("declare -A DATA_ARGS=(") or line.startswith("declare -A TRAINING_ARGS=("):
            in_array = True
            continue
        if in_array and line == ")":
            in_array = False
            continue
        if in_array:
            entry = ARRAY_ENTRY.match(line)
            if entry:
                arrays[entry.group(1)] = _resolve(entry.group(2), variables)
    return arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="train_cut3r_Baseline.sh")
    parser.add_argument("--cut3r", default="train_cut3r_token_only_vsi.sh")
    parser.add_argument("--output", default="diagnostics/cut3r_token_only_wrapper_diff.json")
    args = parser.parse_args()
    baseline, cut3r = _parse(Path(args.baseline)), _parse(Path(args.cut3r))
    differences = []
    for key in sorted(set(baseline) | set(cut3r)):
        before, after = baseline.get(key), cut3r.get(key)
        if before == after:
            continue
        if key in INTENTIONAL:
            classification = "intentional CUT3R input change"
        elif key in RUNTIME:
            classification = "runtime/output naming difference"
        else:
            classification = "unintended difference"
        differences.append({"argument": key, "baseline": before, "cut3r_token_only": after, "classification": classification})
    report = {
        "baseline": str(Path(args.baseline)),
        "cut3r_token_only": str(Path(args.cut3r)),
        "differences": differences,
        "unintended_difference_count": sum(item["classification"] == "unintended difference" for item in differences),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["unintended_difference_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
