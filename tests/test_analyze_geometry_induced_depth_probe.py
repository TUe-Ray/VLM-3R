"""CPU test for geometry-induced probe result aggregation."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/probing/analyze_geometry_induced_depth_probe.py"


class AnalyzeGeometryInducedDepthProbeTest(unittest.TestCase):
    def test_writes_joined_csv_for_all_feature_variants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            model = "SS123"
            rms_rows = []
            for feature_type in ("normal", "geometry_off", "geometry_delta"):
                for split in ("train", "val"):
                    rms_rows.append({"model": model, "layer": 0, "feature_type": feature_type, "split": split, "rms": 1.0})
            rms_path = root / "geometry_induced_probe" / model / "feature_rms.json"
            rms_path.parent.mkdir(parents=True)
            rms_path.write_text(json.dumps(rms_rows), encoding="utf-8")
            for suffix in ("", "__geometry_off", "__geometry_delta"):
                metrics_path = root / "probes" / f"{model}{suffix}" / "layer_0" / "metrics.json"
                metrics_path.parent.mkdir(parents=True)
                metrics_path.write_text(json.dumps({"mae": 0.1, "absrel": 0.2, "delta125": 0.9, "probe_seed": 0, "best_epoch": 1, "num_tokens": 10}), encoding="utf-8")
            output = Path(temporary) / "results"
            subprocess.run(
                [sys.executable, str(SCRIPT), "--cache-root", str(root), "--output-dir", str(output), "--models", model, "--layers", "0"],
                check=True,
                cwd=REPO_ROOT,
            )
            with (output / "geometry_induced_depth_probe.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["feature_type"] for row in rows], ["geometry_delta", "geometry_off", "normal"])
            self.assertTrue((output / "summary.md").is_file())


if __name__ == "__main__":
    unittest.main()
