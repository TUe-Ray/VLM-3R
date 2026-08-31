#!/usr/bin/env python
"""Merge independently run post-SFT MLP sample-efficiency model results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probing.depth_probe_common import read_json, write_csv, write_json
from scripts.probing.run_post_sft_depth_probe_sample_efficiency import full_metrics, load_inventory, write_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--model-output-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    specs = load_inventory(Path(args.inventory).resolve())
    source = Path(args.model_output_root).resolve()
    output = Path(args.output_dir).resolve()
    rows: list[dict[str, Any]] = []
    for spec in specs:
        path = source / spec.label / "raw_results.json"
        payload = read_json(path)
        if not isinstance(payload, list):
            raise TypeError(f"Expected rows list: {path}")
        model_rows = [dict(item) for item in payload if isinstance(item, dict)]
        if not model_rows or any(str(item.get("model")) != spec.label for item in model_rows):
            raise RuntimeError(f"Invalid or missing rows for {spec.label}: {path}")
        rows.extend(model_rows)
    output.mkdir(parents=True, exist_ok=True)
    refs = {spec.label: full_metrics(spec) for spec in specs}
    write_json(output / "raw_results.json", rows)
    write_csv(output / "raw_results.csv", rows)
    write_analysis(output, rows, refs, specs)
    print(f"[DONE] merged rows={len(rows)} output={output}")


if __name__ == "__main__":
    main()
