#!/usr/bin/env python
"""Compare frozen post-SFT v1 ridge/VF results for two SS012 checkpoints.

The new checkpoint is analysed by the existing ``run_v1`` implementation;
this script only combines that immutable result with the prior post-SFT cache
and performs the requested paired whole-profile tests.  It does not introduce
a new target, probe, alpha selection, VF calculation, or sampling rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probing.analyze_depth_subspace_occupancy import (  # noqa: E402
    DEFAULT_ALPHAS,
    PROBE_POINTS,
    aggregate_rows,
    manifest_by_split,
    plot_profiles,
    profile_permutation_test,
    profile_tensor,
    run_v1,
    stable_seed,
    write_json,
    write_rows,
)


OLD_MODEL = "SS012_old"
NEW_MODEL = "SS012_new"
CONTEXT_MODELS = (OLD_MODEL, NEW_MODEL, "SS123", "SS036")
METRICS = ("linear_r2", "vf_enrich")
LATE_POINTS = ("L9", "L12", "L15", "L18", "L21", "L24", "L27")


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Expected a finite value, got {value!r}")
    return result


def load_selection(path: Path) -> dict[str, list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): [int(index) for index in indices] for key, indices in payload["selected_valid_indices"].items()}


def relabel(rows: list[dict[str, Any]], source: str, target: str) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        clone = dict(row)
        if str(clone["model"]) == source:
            clone["model"] = target
        result.append(clone)
    return result


def pairwise_profile_result(
    rows: list[dict[str, Any]], *, metric: str, points: tuple[str, ...], seed: int
) -> dict[str, Any]:
    values, ids = profile_tensor(rows, metric=metric, models=[OLD_MODEL, NEW_MODEL], points=points)
    common: dict[str, Any] = {
        "metric": metric,
        "models": f"{OLD_MODEL} vs {NEW_MODEL}",
        "profile_points": ";".join(points),
        "video_count": int(values.shape[0]),
    }
    if values.shape[0] < 3:
        return {**common, "stable": False, "reason": "fewer_than_three_complete_videos"}
    return {
        **common,
        "video_ids": ";".join(ids),
        **profile_permutation_test(values, seed=stable_seed(seed, "old_new", metric, *points)),
        "reason": "",
    }


def paired_differences(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["model"] not in (OLD_MODEL, NEW_MODEL) or row["split"] != "dev_eval":
            continue
        lookup.setdefault((str(row["video_id"]), str(row["probe_point"])), {})[str(row["model"])] = row
    result = []
    for (video_id, point), values in sorted(lookup.items()):
        if set(values) != {OLD_MODEL, NEW_MODEL}:
            continue
        old, new = values[OLD_MODEL], values[NEW_MODEL]
        for metric in METRICS:
            result.append(
                {
                    "video_id": video_id,
                    "video_path": old["video_path"],
                    "probe_point": point,
                    "metric": metric,
                    "old_value": as_float(old[metric]),
                    "new_value": as_float(new[metric]),
                    "new_minus_old": as_float(new[metric]) - as_float(old[metric]),
                }
            )
    return result


def plot_paired_difference(rows: list[dict[str, Any]], *, metric: str, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_video: dict[str, dict[str, float]] = {}
    for row in rows:
        if row["metric"] == metric:
            by_video.setdefault(str(row["video_id"]), {})[str(row["probe_point"])] = as_float(row["new_minus_old"])
    figure, axis = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(PROBE_POINTS))
    vectors = np.asarray([[values.get(point, np.nan) for point in PROBE_POINTS] for values in by_video.values()], dtype=float)
    for vector in vectors:
        axis.plot(x, vector, color="0.72", linewidth=0.8, alpha=0.55)
    means = np.nanmean(vectors, axis=0)
    axis.plot(x, means, marker="o", color="C3", linewidth=2, label="SS012-new − SS012-old")
    axis.axhline(0, color="0.35", linewidth=0.9)
    axis.set_xticks(x, PROBE_POINTS, rotation=45, ha="right")
    axis.set_xlabel("probe point")
    axis.set_ylabel("paired held-out R² difference" if metric == "linear_r2" else "paired VF enrichment difference")
    axis.legend(loc="best")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--token-selection", required=True)
    parser.add_argument("--prior-result-dir", required=True)
    parser.add_argument("--new-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-directions", type=int, default=64)
    parser.add_argument("--label-permutations", type=int, default=32)
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory must be new and empty: {output}")
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = Path(args.manifest).resolve()
    by_split = manifest_by_split(json.loads(manifest_path.read_text(encoding="utf-8")))
    expected_counts = {"train": 6, "val": 2, "dev_eval": 12}
    if {key: len(by_split.get(key, [])) for key in expected_counts} != expected_counts:
        raise RuntimeError("Same-architecture VF comparison requires the fixed 6/2/12 manifest")
    selection_path = Path(args.token_selection).resolve()
    selection = load_selection(selection_path)
    train_ids = {str(video["video_sample_id"]) for video in by_split["train"]}
    if set(selection) != train_ids:
        raise RuntimeError("Frozen token selection does not match the manifest train videos")

    new_rows, _profiles, _fitted = run_v1(
        cache_root=Path(args.cache_root).resolve(),
        output_dir=output / "new_checkpoint_v1",
        models=[NEW_MODEL],
        by_split=by_split,
        selection=selection,
        alphas=tuple(DEFAULT_ALPHAS),
        random_count=args.random_directions,
        label_permutations=args.label_permutations,
        seed=args.seed,
    )
    prior_dir = Path(args.prior_result_dir).resolve()
    prior_rows = read_csv(prior_dir / "v1_ridge_vf" / "linear_vf_per_video.csv")
    prior_rows = relabel(prior_rows, "SS012", OLD_MODEL)
    combined = prior_rows + new_rows
    write_rows(output / "combined_linear_vf_per_video.csv", combined)
    dev_rows = [row for row in combined if row["split"] == "dev_eval"]
    aggregates = aggregate_rows(
        dev_rows,
        ["model", "probe_point"],
        ["linear_r2", "linear_mae", "linear_absrel", "vf_depth", "vf_enrich", "random_vf_mean", "random_vf_std"],
    )
    write_rows(output / "combined_linear_vf_aggregate.csv", aggregates)
    paired = paired_differences(combined)
    write_rows(output / "ss012_old_new_paired_differences_per_video.csv", paired)
    write_rows(output / "ss012_old_new_paired_differences_aggregate.csv", aggregate_rows(paired, ["metric", "probe_point"], ["old_value", "new_value", "new_minus_old"]))
    tests = []
    for scope, points in (("full_14_points", PROBE_POINTS), ("late_L9_L27", LATE_POINTS)):
        for metric in METRICS:
            tests.append({"scope": scope, **pairwise_profile_result(dev_rows, metric=metric, points=points, seed=args.seed)})
    write_rows(output / "ss012_old_new_profile_tests.csv", tests)
    plot_profiles(dev_rows, metric="linear_r2", models=list(CONTEXT_MODELS), output=output / "context_linear_r2.png", ylabel="held-out development R²")
    plot_profiles(dev_rows, metric="vf_enrich", models=list(CONTEXT_MODELS), output=output / "context_vf_enrich.png", ylabel="depth VF enrichment")
    for metric in METRICS:
        plot_paired_difference(paired, metric=metric, output=output / f"ss012_old_new_{metric}_difference.png")

    paired_agg = aggregate_rows(paired, ["metric", "probe_point"], ["new_minus_old"])
    late_vf = {row["probe_point"]: as_float(row["new_minus_old_mean"]) for row in paired_agg if row["metric"] == "vf_enrich" and row["probe_point"] in LATE_POINTS}
    late_positive = sum(value > 0 for value in late_vf.values())
    tests_by_key = {(row["scope"], row["metric"]): row for row in tests}
    late_test = tests_by_key[("late_L9_L27", "vf_enrich")]
    full_test = tests_by_key[("full_14_points", "vf_enrich")]
    conclusion = (
        "The new checkpoint has a stable, positive late VF profile difference." if bool(late_test.get("stable")) and late_positive == len(LATE_POINTS)
        else "The frozen v1 comparison does not establish a uniformly positive, stable late VF advantage for the new checkpoint."
    )
    provenance = {
        "schema_version": "post_sft_ss012_checkpoint_vf_v1",
        "analysis": "unchanged post-SFT frozen v1 ridge depth direction and original-coordinate VF enrichment",
        "old_label": OLD_MODEL,
        "old_checkpoint": "/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/cut3r_spatialstack_44323703",
        "new_label": NEW_MODEL,
        "new_checkpoint": str(Path(args.new_checkpoint).resolve()),
        "context_models": list(CONTEXT_MODELS),
        "probe_points": list(PROBE_POINTS),
        "late_points": list(LATE_POINTS),
        "manifest": str(manifest_path),
        "token_selection": str(selection_path),
        "prior_result_dir": str(prior_dir),
        "seed": args.seed,
        "random_directions": args.random_directions,
        "label_permutations": args.label_permutations,
        "paired_null": "joint per-video swap of the complete old/new 14-point profile; exact for 12 videos",
    }
    write_json(output / "analysis_provenance.json", provenance)
    lines = [
        "# Same-architecture SS012 post-SFT frozen VF comparison",
        "",
        "Both checkpoints use SS012 (additive L0/L1/L2), the fixed 6/2/12 manifest, identical token selection, the frozen alpha grid, and the unchanged v1 ridge/VF implementation.",
        "The 12 development-evaluation videos are exploratory; they are not a final confirmation set.",
        "",
        "## Result",
        "",
        conclusion,
        f"Late VF points with positive `SS012-new - SS012-old` mean: {late_positive}/{len(LATE_POINTS)}.",
        f"Full-14 VF profile test: p={full_test.get('p_value')}, LOO={full_test.get('leave_one_video_out_passes')}/{full_test.get('leave_one_video_out_total')}, stable={full_test.get('stable')}.",
        f"Late L9--L27 VF profile test: p={late_test.get('p_value')}, LOO={late_test.get('leave_one_video_out_passes')}/{late_test.get('leave_one_video_out_total')}, stable={late_test.get('stable')}.",
        "",
        "Profile tests use a joint per-video swap of the complete profile, retaining layer correlation. A difference is a representation distinction, not evidence that VF causes the VSI difference.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "status": "complete"}, indent=2))


if __name__ == "__main__":
    main()
