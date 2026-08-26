#!/usr/bin/env python
"""CPU-only verification for one isolated ScanNet final-layer smoke run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--feature-levels", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    levels = [part.strip() for part in args.feature_levels.split(",") if part.strip()]
    manifest = read_json(Path(args.manifest))
    errors: list[str] = []
    train_videos = int(manifest.get("train_videos", 0))
    val_videos = int(manifest.get("val_videos", 0))
    if train_videos < 1 or val_videos < 1:
        errors.append(f"smoke manifest must contain train and val videos, got {train_videos}/{val_videos}")

    extraction_path = output_root / "features" / args.model_label / "extraction_provenance.json"
    extraction: dict[str, Any] = {}
    if not extraction_path.is_file():
        errors.append(f"missing extraction provenance: {extraction_path}")
    else:
        extraction = read_json(extraction_path)
        if extraction.get("model_label") != args.model_label:
            errors.append(f"wrong extraction model label: {extraction.get('model_label')!r}")
        if extraction.get("hidden_state_indexing") != "requested_L -> hidden_states[L + 1]":
            errors.append("wrong hidden-state indexing provenance")
        samples = extraction.get("extraction_samples") or []
        if not samples:
            errors.append("extraction provenance has no sample records")
        else:
            assertion = samples[0].get("first_video_runtime_assertions")
            if not isinstance(assertion, dict) or assertion.get("assessment") != "PASS":
                errors.append("first-video runtime assertion did not pass")

    metrics: list[dict[str, Any]] = []
    for level in levels:
        probe_dir = output_root / "probes" / args.model_label / level
        metrics_path = probe_dir / "metrics.json"
        if not metrics_path.is_file():
            errors.append(f"missing metrics: {metrics_path}")
            continue
        row = read_json(metrics_path)
        metrics.append(row)
        if row.get("model_label") != args.model_label or row.get("feature_level") != level:
            errors.append(f"wrong metric identity for {level}")
        if int(row.get("num_tokens", 0)) <= 0:
            errors.append(f"no validation tokens for {level}")
        for name in ("mae", "absrel", "delta125"):
            if not math.isfinite(float(row.get(name, float("nan")))):
                errors.append(f"non-finite {name} for {level}")
        for required in ("history.json", "best.pt"):
            if not (probe_dir / required).is_file():
                errors.append(f"missing {required}: {probe_dir / required}")

    report = {
        "assessment": "PASS" if not errors else "FAIL",
        "model_label": args.model_label,
        "output_root": str(output_root),
        "manifest": str(Path(args.manifest).resolve()),
        "feature_levels": levels,
        "metrics": metrics,
        "extraction_provenance": str(extraction_path),
        "errors": errors,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[SMOKE VERIFY] {args.model_label}: {report['assessment']} ({report_path})")
    if errors:
        for error in errors:
            print(f"[SMOKE ERROR] {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
