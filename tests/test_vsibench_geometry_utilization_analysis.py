import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/eval/analyze_vsibench_geometry_utilization.py"


class VsiBenchGeometryUtilizationAnalysisTest(unittest.TestCase):
    def _write_run(self, root, model, condition, score, prediction):
        checkpoint = root / "checkpoint"
        checkpoint.mkdir(exist_ok=True)
        (checkpoint / "adapter_model.bin").write_bytes(b"adapter")
        runtime = root / f"runtime_{condition}"
        runtime.mkdir()
        (runtime / "adapter_model.bin").symlink_to(checkpoint / "adapter_model.bin")
        run_dir = root / "runs" / f"{model}_{condition}" / "result"
        run_dir.mkdir(parents=True)
        doc = {
            "id": 7,
            "dataset": "scannet",
            "scene_name": "scene_7",
            "question_type": "object_rel_direction_easy",
            "question": "Which object is left?",
            "ground_truth": "A",
            "options": ["A. first", "B. second"],
            "prediction": prediction,
            "accuracy": score,
        }
        payload = {
            "args": {
                "model_args": (
                    f"pretrained={runtime},model_base=base,spatial_features_subdir=6:dec6;9:dec9,"
                    f"spatialstack_perturbation_mode={condition}"
                )
            },
            "logs": [
                {
                    "doc_id": 0,
                    "doc": doc,
                    "vsibench_score": doc,
                    "target": "A",
                    "doc_hash": "doc",
                    "prompt_hash": "prompt",
                    "target_hash": "target",
                    "filtered_resps": [prediction],
                }
            ],
        }
        (run_dir / "vsibench_local_mp4.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_joins_native_scores_and_requires_baseline_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_run(root, "SS012_new", "none", 1.0, "A")
            self._write_run(root, "SS012_new", "normal", 1.0, "A")
            self._write_run(root, "SS012_new", "geometry_off_all", 0.0, "B")
            output = root / "analysis"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--run-root",
                    str(root / "runs"),
                    "--output-dir",
                    str(output),
                    "--models",
                    "SS012_new",
                    "--baseline-condition",
                    "none",
                    "--require-normal-baseline-match",
                ],
                check=True,
                cwd=REPO_ROOT,
            )
            rows = json.loads((output / "per_example_paired.json").read_text(encoding="utf-8"))
            self.assertEqual(rows[0]["transition"], "correct_to_incorrect")
            self.assertEqual(rows[0]["score_difference"], 1.0)


if __name__ == "__main__":
    unittest.main()
