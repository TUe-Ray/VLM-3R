#!/usr/bin/env python
"""Create a W&B offline run from saved depth-probe histories and metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_probe_dirs(probes_root: Path):
    for model_dir in sorted(probes_root.iterdir()):
        if not model_dir.is_dir():
            continue
        for feature_dir in sorted(model_dir.iterdir()):
            if feature_dir.is_dir():
                yield model_dir.name, feature_dir.name, feature_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="/leonardo_scratch/large/userexternal/shuang00/probing_experiment")
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "vlm3r-depth-probes"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--name", default=os.environ.get("WANDB_NAME", "depth-probes-posthoc"))
    parser.add_argument("--wandb-dir", default=os.environ.get("WANDB_DIR", str(Path(os.environ.get("WORK", "/leonardo_work/EUHPC_D32_006")) / "wandb")))
    parser.add_argument("--mode", default=os.environ.get("WANDB_MODE", "offline"))
    parser.add_argument("--tags", default=os.environ.get("WANDB_TAGS", "depth-probe,posthoc"))
    args = parser.parse_args()

    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise RuntimeError("wandb is not installed in this environment.") from exc

    output_root = Path(args.output_root)
    probes_root = output_root / "probes"
    Path(args.wandb_dir).mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.name,
        dir=args.wandb_dir,
        mode=args.mode,
        tags=[tag.strip() for tag in args.tags.split(",") if tag.strip()],
        config={"output_root": str(output_root), "source": "posthoc_depth_probe_histories"},
    )

    completed = 0
    try:
        for model_label, feature_level, probe_dir in iter_probe_dirs(probes_root):
            history_path = probe_dir / "history.json"
            metrics_path = probe_dir / "metrics.json"
            if history_path.exists():
                history = read_json(history_path)
                for row in history:
                    run.log(
                        {
                            "model_label": model_label,
                            "feature_level": feature_level,
                            "epoch": row.get("epoch"),
                            "train/mae": row.get("train_mae"),
                            "val/mae": row.get("val_mae"),
                            "val/absrel": row.get("val_absrel"),
                            "val/delta125": row.get("val_delta125"),
                            f"{model_label}/{feature_level}/train_mae": row.get("train_mae"),
                            f"{model_label}/{feature_level}/val_mae": row.get("val_mae"),
                            f"{model_label}/{feature_level}/val_absrel": row.get("val_absrel"),
                            f"{model_label}/{feature_level}/val_delta125": row.get("val_delta125"),
                        }
                    )
            if metrics_path.exists():
                metrics = read_json(metrics_path)
                prefix = f"best/{model_label}/{feature_level}"
                run.log(
                    {
                        f"{prefix}/epoch": metrics.get("best_epoch"),
                        f"{prefix}/mae": metrics.get("mae"),
                        f"{prefix}/absrel": metrics.get("absrel"),
                        f"{prefix}/delta125": metrics.get("delta125"),
                        f"{prefix}/num_tokens": metrics.get("num_tokens"),
                    }
                )
                run.summary[f"{prefix}/mae"] = metrics.get("mae")
                run.summary[f"{prefix}/absrel"] = metrics.get("absrel")
                run.summary[f"{prefix}/delta125"] = metrics.get("delta125")
                run.summary[f"{prefix}/epoch"] = metrics.get("best_epoch")
                completed += 1
        results_csv = probes_root / "results.csv"
        if results_csv.exists():
            artifact = wandb.Artifact("depth-probe-results", type="results")
            artifact.add_file(str(results_csv))
            run.log_artifact(artifact)
        run.summary["completed_probes"] = completed
    finally:
        run.finish()


if __name__ == "__main__":
    main()
