#!/usr/bin/env python
"""Validate and summarize full controlled-fusion pre-SFT depth probes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llava.model.controlled_fusion_pre_sft import CONTROLLED_FUSION_PRE_SFT_SPECS  # noqa: E402
from scripts.probing.probe_layer_policy import (  # noqa: E402
    COMMON_PROBE_LAYERS,
    PRE_SFT_PRE_LLM_FEATURES,
)
from scripts.probing.verify_controlled_fusion_c1 import SCHEMA_VERSION as C1_MANIFEST_SCHEMA  # noqa: E402


SCHEMA_VERSION = "controlled_fusion_pre_sft_depth_probe_summary_v1"
EXPECTED_VALIDATION_TOKENS = 75_656


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--sample-indices", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_json = args.output_dir / "results.json"
    output_csv = args.output_dir / "metrics.csv"
    output_md = args.output_dir / "summary.md"
    existing = [str(path) for path in (output_json, output_csv, output_md) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite controlled-fusion summary files: {existing}")
    artifact_manifest = read_json(args.artifact_manifest)
    if artifact_manifest.get("schema_version") != C1_MANIFEST_SCHEMA:
        raise ValueError("Summary received an incompatible C1 artifact manifest")
    if artifact_manifest.get("post_sft_state_loaded") is not False:
        raise ValueError("Summary C1 manifest does not prove absence of post-SFT state")
    sample_sha256 = sha256_file(args.sample_indices)
    feature_levels = [*PRE_SFT_PRE_LLM_FEATURES, *(f"layer_{layer}" for layer in COMMON_PROBE_LAYERS)]
    entries = {
        "BASE": {"label": "pre_sft_base_vlm", "display": "Plain pre-SFT base VLM"},
        **{
            identifier: {"label": spec.pre_sft_variant, "display": spec.display_name}
            for identifier, spec in CONTROLLED_FUSION_PRE_SFT_SPECS.items()
        },
    }
    rows: list[dict[str, Any]] = []
    provenance_records: dict[str, Any] = {}
    for identifier, entry in entries.items():
        root = args.results_root / identifier
        provenance_path = root / "extraction_provenance.json"
        provenance = read_json(provenance_path)
        if provenance.get("git_worktree_dirty") is not False:
            raise ValueError(f"{identifier} full extraction was run from a dirty worktree")
        if provenance.get("git_commit") != artifact_manifest.get("git_commit"):
            raise ValueError(f"{identifier} full extraction commit differs from C1 manifest")
        if provenance.get("sample_indices_sha256") != sample_sha256:
            raise ValueError(f"{identifier} does not use the authoritative ScanNet split")
        if provenance.get("no_vlm3r_sft_adapter_loaded") is not True:
            raise ValueError(f"{identifier} lacks proof that no post-SFT adapter was loaded")
        if set(provenance.get("requested_feature_levels", [])) != set(feature_levels):
            raise ValueError(f"{identifier} does not cover the complete pre-SFT feature policy")
        if identifier != "BASE":
            expected_hash = artifact_manifest["artifacts"][identifier]["sha256"]
            if provenance.get("c1_calibration_sha256") != expected_hash:
                raise ValueError(f"{identifier} C1 hash differs from the locked artifact")
        elif provenance.get("c1_calibration_json") is not None:
            raise ValueError("BASE full extraction unexpectedly loaded C1 state")
        provenance_records[identifier] = {
            "path": str(provenance_path),
            "sha256": sha256_file(provenance_path),
        }
        for level in feature_levels:
            metric_path = root / "probes" / level / "metrics.json"
            metric = read_json(metric_path)
            values = {name: float(metric.get(name, float("nan"))) for name in ("mae", "absrel", "delta125")}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"{identifier}/{level} contains non-finite metrics")
            if int(metric.get("num_tokens", -1)) != EXPECTED_VALIDATION_TOKENS:
                raise ValueError(
                    f"{identifier}/{level} has {metric.get('num_tokens')} validation tokens; "
                    f"expected {EXPECTED_VALIDATION_TOKENS}"
                )
            rows.append(
                {
                    "candidate": identifier,
                    "display_name": entry["display"],
                    "model_label": entry["label"],
                    "feature_level": level,
                    **values,
                    "num_tokens": int(metric["num_tokens"]),
                    "metrics_path": str(metric_path),
                }
            )
    by_level: dict[str, list[str]] = {}
    for level in feature_levels:
        selected = [row for row in rows if row["feature_level"] == level]
        by_level[level] = [row["candidate"] for row in sorted(selected, key=lambda row: row["mae"])]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_label": "controlled-fusion pre-SFT extension",
        "formal_existing_five_candidate_roster_modified": False,
        "artifact_manifest": str(args.artifact_manifest.resolve()),
        "artifact_manifest_sha256": sha256_file(args.artifact_manifest),
        "sample_indices": str(args.sample_indices.resolve()),
        "sample_indices_sha256": sample_sha256,
        "feature_levels": feature_levels,
        "expected_validation_tokens": EXPECTED_VALIDATION_TOKENS,
        "provenance": provenance_records,
        "rows": rows,
        "mae_ranking_ascending_by_feature": by_level,
        "post_sft_state_loaded": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Controlled-fusion pre-SFT depth probes",
        "",
        "This is a separate B/C/D/E/H extension; it does not alter the existing formal five-candidate roster.",
        "",
        "| Candidate | Feature | MAE | AbsRel | delta<1.25 | Validation tokens |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {row['feature_level']} | {row['mae']:.6g} | "
            f"{row['absrel']:.6g} | {row['delta125']:.6g} | {row['num_tokens']} |"
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output_json), "rows": len(rows)}))


if __name__ == "__main__":
    main()
