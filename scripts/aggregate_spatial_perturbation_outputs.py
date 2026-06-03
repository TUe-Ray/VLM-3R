#!/usr/bin/env python3
"""Merge split spatial perturbation diagnostic outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", required=True, help="Directory containing per-category output dirs.")
    parser.add_argument("--output_dir", required=True, help="Directory for merged CSVs and plots.")
    return parser.parse_args()


def summarize(deltas: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in deltas.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n_samples": int(len(group)),
                "mean_delta_margin": group["delta_margin"].mean(),
                "median_delta_margin": group["delta_margin"].median(),
                "std_delta_margin": group["delta_margin"].std(ddof=1),
                "mean_margin_normal": group["margin_normal"].mean(),
                "mean_margin_perturbed": group["margin_perturbed"].mean(),
                "accuracy_normal": group["is_correct_normal"].mean(),
                "accuracy_perturbed": group["is_correct_perturbed"].mean(),
                "accuracy_delta": group["is_correct_perturbed"].mean()
                - group["is_correct_normal"].mean(),
                "flip_rate": group["answer_flipped"].mean(),
                "correct_to_wrong_flip_rate": group["correct_to_wrong_flip"].mean(),
                "wrong_to_correct_flip_rate": group["wrong_to_correct_flip"].mean(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def save_plots(category_summary: pd.DataFrame, deltas: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def grouped_bar(value: str, filename: str, ylabel: str) -> None:
        pivot = category_summary.pivot(index="category", columns="perturbation", values=value)
        ax = pivot.plot(kind="bar", figsize=(13, 6))
        ax.set_ylabel(ylabel)
        ax.set_xlabel("category")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend(title="perturbation", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(plots_dir / filename, dpi=200)
        plt.close()

    grouped_bar("mean_delta_margin", "category_delta_margin_bar.png", "mean delta margin")
    grouped_bar("flip_rate", "category_flip_rate_bar.png", "flip rate")
    grouped_bar("accuracy_delta", "category_accuracy_delta_bar.png", "accuracy delta")

    plt.figure(figsize=(11, 6))
    deltas.boxplot(column="delta_margin", by="perturbation", rot=35)
    plt.suptitle("")
    plt.title("Delta margin by perturbation")
    plt.ylabel("delta margin")
    plt.tight_layout()
    plt.savefig(plots_dir / "delta_margin_distribution_by_perturbation.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    score_paths = sorted(input_root.glob("*/per_sample_perturbation_scores.csv"))
    delta_paths = sorted(input_root.glob("*/per_sample_perturbation_deltas.csv"))
    if not score_paths or not delta_paths:
        raise SystemExit(f"No per-category CSVs found under {input_root}")

    scores = pd.concat((pd.read_csv(path) for path in score_paths), ignore_index=True)
    deltas = pd.concat((pd.read_csv(path) for path in delta_paths), ignore_index=True)

    scores.to_csv(output_dir / "per_sample_perturbation_scores.csv", index=False)
    deltas.to_csv(output_dir / "per_sample_perturbation_deltas.csv", index=False)

    selected = []
    for path in sorted(input_root.glob("*/selected_samples.json")):
        with path.open("r", encoding="utf-8") as handle:
            selected.extend(json.load(handle))
    with (output_dir / "selected_samples.json").open("w", encoding="utf-8") as handle:
        json.dump(selected, handle, indent=2)

    category_summary = summarize(deltas, ["category", "perturbation"])
    global_summary = summarize(deltas.assign(category="ALL"), ["category", "perturbation"])
    category_summary.to_csv(output_dir / "category_perturbation_summary.csv", index=False)
    global_summary.to_csv(output_dir / "global_perturbation_summary.csv", index=False)

    save_plots(category_summary, deltas, output_dir)

    print(f"Merged {len(score_paths)} category score files")
    print(f"Scores rows: {len(scores)}")
    print(f"Delta rows: {len(deltas)}")
    print(f"Selected samples: {len(selected)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
