#!/usr/bin/env python
"""Run the unchanged frozen v1 ridge/VF analysis on post-SFT feature caches.

This is intentionally a thin post-processing wrapper around ``run_v1``.  It
does not define a new decoder, target, occupancy calculation, or token mask.
It adds only requested comparisons: full/late omnibus and pairwise profiles,
the SS123-vs-average(SS012,SS036) capability contrast, and matched pre/post
tables on the same independently held-out videos.
"""

from __future__ import annotations

import argparse
import csv
import itertools
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
    profile_permutation_test,
    profile_tensor,
    run_v1,
    stable_seed,
    write_json,
    write_rows,
)


MODELS = ("SS012", "SS123", "SS036")
METRICS = ("linear_r2", "vf_enrich")
LATE_POINTS = ("L9", "L12", "L15", "L18", "L21", "L24", "L27")


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Expected finite value, got {value!r}")
    return result


def load_selection(path: Path) -> dict[str, list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): [int(index) for index in values] for key, values in payload["selected_valid_indices"].items()}


def profile_result(
    rows: list[dict[str, Any]], *, metric: str, models: tuple[str, ...], points: tuple[str, ...], seed: int
) -> dict[str, Any]:
    values, ids = profile_tensor(rows, metric=metric, models=list(models), points=points)
    result: dict[str, Any] = {
        "metric": metric,
        "models": " vs ".join(models),
        "profile_points": ";".join(points),
        "video_count": int(values.shape[0]),
    }
    if values.shape[0] < 3:
        return {**result, "stable": False, "reason": "fewer_than_three_complete_videos"}
    return {
        **result,
        "video_ids": ";".join(ids),
        **profile_permutation_test(values, seed=stable_seed(seed, metric, *models, *points)),
        "reason": "",
    }


def contrast_values(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        lookup.setdefault((str(row["video_id"]), str(row["probe_point"])), {})[str(row["model"])] = to_float(row[metric])
    result = []
    for (video_id, point), values in sorted(lookup.items()):
        if set(values) != set(MODELS):
            continue
        result.append(
            {
                "metric": metric,
                "video_id": video_id,
                "probe_point": point,
                "capability_contrast": values["SS123"] - 0.5 * (values["SS012"] + values["SS036"]),
                "SS123": values["SS123"],
                "SS012": values["SS012"],
                "SS036": values["SS036"],
            }
        )
    return result


def contrast_profile_test(rows: list[dict[str, Any]], *, metric: str, points: tuple[str, ...], seed: int) -> dict[str, Any]:
    by_video: dict[str, dict[str, float]] = {}
    for row in rows:
        if row["metric"] != metric or row["probe_point"] not in points:
            continue
        by_video.setdefault(str(row["video_id"]), {})[str(row["probe_point"])] = to_float(row["capability_contrast"])
    vectors = [values for _, values in sorted(by_video.items()) if all(point in values for point in points)]
    ids = [video_id for video_id, values in sorted(by_video.items()) if all(point in values for point in points)]
    if len(vectors) < 3:
        return {"metric": metric, "profile_points": ";".join(points), "video_count": len(vectors), "stable": False, "reason": "fewer_than_three_complete_videos"}
    values = np.asarray([[vector[point] for point in points] for vector in vectors], dtype=float)
    observed = float(np.square(values.mean(axis=0)).sum())
    sign_options = list(itertools.product((-1.0, 1.0), repeat=values.shape[0]))
    null = np.asarray(
        [np.square((values * np.asarray(signs)[:, None]).mean(axis=0)).sum() for signs in sign_options],
        dtype=float,
    )
    q95 = float(np.quantile(null, 0.95))
    loo = []
    for leave_out in range(values.shape[0]):
        kept = np.delete(values, leave_out, axis=0)
        options = list(itertools.product((-1.0, 1.0), repeat=kept.shape[0]))
        nested = np.asarray(
            [np.square((kept * np.asarray(signs)[:, None]).mean(axis=0)).sum() for signs in options],
            dtype=float,
        )
        loo.append(bool(float(np.square(kept.mean(axis=0)).sum()) > float(np.quantile(nested, 0.95))))
    return {
        "metric": metric,
        "profile_points": ";".join(points),
        "video_count": int(values.shape[0]),
        "video_ids": ";".join(ids),
        "observed_T": observed,
        "null_q95": q95,
        "p_value": float((1 + np.count_nonzero(null >= observed)) / (1 + null.size)),
        "permutation_count": int(null.size),
        "permutation_exact": True,
        "leave_one_video_out_passes": int(sum(loo)),
        "leave_one_video_out_total": int(len(loo)),
        "stable": bool(observed > q95 and sum(loo) >= max(1, values.shape[0] - 1)),
        "test": "paired_video_sign_flip_of_SS123_minus_mean_SS012_SS036",
        "reason": "",
    }


def state_rows(pre_rows: list[dict[str, Any]], post_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for state, rows in (("pre_sft", pre_rows), ("post_sft", post_rows)):
        for row in rows:
            for metric in METRICS:
                result.append({
                    "state": state,
                    "model": row["model"],
                    "probe_point": row["probe_point"],
                    "video_id": row["video_id"],
                    "metric": metric,
                    "value": to_float(row[metric]),
                })
    return result


def pre_post_deltas(pre_rows: list[dict[str, Any]], post_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {}
    for row in pre_rows:
        for metric in METRICS:
            lookup[(row["model"], row["probe_point"], row["video_id"], metric)] = to_float(row[metric])
    result = []
    for row in post_rows:
        for metric in METRICS:
            key = (row["model"], row["probe_point"], row["video_id"], metric)
            if key not in lookup:
                raise RuntimeError(f"Missing matched pre-SFT held-out row: {key}")
            result.append({
                "model": row["model"], "probe_point": row["probe_point"], "video_id": row["video_id"], "metric": metric,
                "pre_sft": lookup[key], "post_sft": to_float(row[metric]),
                "post_minus_pre": to_float(row[metric]) - lookup[key],
            })
    return result


def plot_state_curves(rows: list[dict[str, Any]], *, metric: str, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = {point: index for index, point in enumerate(PROBE_POINTS)}
    figure, axis = plt.subplots(figsize=(12, 4.5))
    for state, linestyle in (("pre_sft", "--"), ("post_sft", "-")):
        for model in MODELS:
            values = {row["probe_point"]: to_float(row["value_mean"]) for row in rows if row["state"] == state and row["model"] == model and row["metric"] == metric}
            axis.plot(range(len(PROBE_POINTS)), [values.get(point, np.nan) for point in PROBE_POINTS], linestyle=linestyle, marker="o", label=f"{model} {state}")
    axis.set_xticks(range(len(PROBE_POINTS)), PROBE_POINTS, rotation=45, ha="right")
    axis.set_xlabel("probe point")
    axis.set_ylabel("held-out R²" if metric == "linear_r2" else "VF enrichment")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_contrast(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    for axis, metric in zip(axes, METRICS):
        for state, linestyle in (("pre_sft", "--"), ("post_sft", "-")):
            values = {row["probe_point"]: to_float(row["capability_contrast_mean"]) for row in rows if row["state"] == state and row["metric"] == metric}
            axis.plot(range(len(PROBE_POINTS)), [values.get(point, np.nan) for point in PROBE_POINTS], linestyle=linestyle, marker="o", label=state)
        axis.axhline(0, color="0.35", linewidth=0.8)
        axis.set_title(metric)
        axis.set_xticks(range(len(PROBE_POINTS)), PROBE_POINTS, rotation=45, ha="right")
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def capability_profile_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for metric in METRICS:
        for scope, points in (("full_14_points", PROBE_POINTS), ("late_L9_L27", LATE_POINTS)):
            profiles = {}
            for state in ("pre_sft", "post_sft"):
                lookup = {
                    row["probe_point"]: to_float(row["capability_contrast_mean"])
                    for row in rows
                    if row["state"] == state and row["metric"] == metric and row["probe_point"] in points
                }
                profiles[state] = np.asarray([lookup[point] for point in points], dtype=float)
            pre, post = profiles["pre_sft"], profiles["post_sft"]
            correlation = float(np.corrcoef(pre, post)[0, 1]) if np.std(pre) > 0 and np.std(post) > 0 else float("nan")
            result.append({
                "metric": metric,
                "scope": scope,
                "point_count": len(points),
                "pearson_profile_correlation": correlation,
            })
    return result


def result_for(rows: list[dict[str, Any]], *, scope: str, metric: str) -> dict[str, Any]:
    matches = [row for row in rows if row.get("scope") == scope and row.get("metric") == metric]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one result for scope={scope}, metric={metric}; got {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--token-selection", required=True)
    parser.add_argument("--pre-sft-result-dir", required=True)
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
    if {key: len(by_split.get(key, [])) for key in ("train", "val", "dev_eval")} != {"train": 6, "val": 2, "dev_eval": 12}:
        raise RuntimeError("Post-SFT frozen VF analysis requires the fixed 6/2/12 manifest")
    selection_path = Path(args.token_selection).resolve()
    selection = load_selection(selection_path)
    expected_ids = {str(video["video_sample_id"]) for video in by_split["train"]}
    if set(selection) != expected_ids:
        raise RuntimeError("Token selection does not match post-SFT frozen train videos")
    write_json(output / "token_selection_reused.json", {
        "source": str(selection_path), "selection": selection,
        "semantics": "exact deterministic selected valid-token indices reused from pre-SFT development",
    })
    v1_rows, _profiles, _fitted = run_v1(
        cache_root=Path(args.cache_root).resolve(), output_dir=output, models=list(MODELS), by_split=by_split,
        selection=selection, alphas=tuple(DEFAULT_ALPHAS), random_count=args.random_directions,
        label_permutations=args.label_permutations, seed=args.seed,
    )
    post_rows = [row for row in v1_rows if row["split"] == "dev_eval"]
    full_late = []
    for scope, points in (("full_14_points", PROBE_POINTS), ("late_L9_L27", LATE_POINTS)):
        for metric in METRICS:
            full_late.append({"scope": scope, **profile_result(post_rows, metric=metric, models=MODELS, points=points, seed=args.seed)})
    write_rows(output / "post_sft_full_late_omnibus_profile_tests.csv", full_late)
    pairwise = []
    for first_index, first in enumerate(MODELS):
        for second in MODELS[first_index + 1:]:
            for scope, points in (("full_14_points", PROBE_POINTS), ("late_L9_L27", LATE_POINTS)):
                for metric in METRICS:
                    pairwise.append({"scope": scope, **profile_result(post_rows, metric=metric, models=(first, second), points=points, seed=args.seed)})
    write_rows(output / "post_sft_pairwise_profile_tests.csv", pairwise)
    contrast = [item for metric in METRICS for item in contrast_values(post_rows, metric)]
    write_rows(output / "post_sft_capability_contrast_per_video.csv", contrast)
    contrast_agg = aggregate_rows(contrast, ["metric", "probe_point"], ["capability_contrast"])
    write_rows(output / "post_sft_capability_contrast_aggregate.csv", contrast_agg)
    contrast_tests = [{"scope": scope, **contrast_profile_test(contrast, metric=metric, points=points, seed=args.seed)} for scope, points in (("full_14_points", PROBE_POINTS), ("late_L9_L27", LATE_POINTS)) for metric in METRICS]
    write_rows(output / "post_sft_capability_contrast_profile_tests.csv", contrast_tests)
    pre_path = Path(args.pre_sft_result_dir).resolve() / "v1_ridge_vf" / "linear_vf_per_video.csv"
    pre_rows = [row for row in read_csv(pre_path) if row["split"] == "dev_eval"]
    post_csv_rows = [
        {key: value for key, value in row.items()}
        for row in post_rows
    ]
    combined = state_rows(pre_rows, post_csv_rows)
    combined_agg = aggregate_rows(combined, ["state", "model", "probe_point", "metric"], ["value"])
    write_rows(output / "pre_post_matched_per_video.csv", combined)
    write_rows(output / "pre_post_matched_aggregate.csv", combined_agg)
    deltas = pre_post_deltas(pre_rows, post_csv_rows)
    write_rows(output / "post_minus_pre_per_video.csv", deltas)
    write_rows(output / "post_minus_pre_aggregate.csv", aggregate_rows(deltas, ["model", "probe_point", "metric"], ["post_minus_pre"]))
    pre_contrast = [item for metric in METRICS for item in contrast_values(pre_rows, metric)]
    post_contrast = contrast
    state_contrast = [{"state": "pre_sft", **row} for row in pre_contrast] + [{"state": "post_sft", **row} for row in post_contrast]
    state_contrast_agg = aggregate_rows(state_contrast, ["state", "metric", "probe_point"], ["capability_contrast"])
    write_rows(output / "pre_post_capability_contrast_per_video.csv", state_contrast)
    write_rows(output / "pre_post_capability_contrast_aggregate.csv", state_contrast_agg)
    correlations = capability_profile_correlations(state_contrast_agg)
    write_rows(output / "pre_post_capability_profile_correlations.csv", correlations)
    plot_state_curves(combined_agg, metric="linear_r2", output=output / "pre_post_linear_r2.png")
    plot_state_curves(combined_agg, metric="vf_enrich", output=output / "pre_post_vf_enrich.png")
    plot_contrast(state_contrast_agg, output / "pre_post_capability_contrast.png")
    provenance = {
        "schema_version": "post_sft_frozen_depth_subspace_vf_v1",
        "metric": "unchanged pre-SFT v1 ridge depth direction and original-coordinate VF enrichment",
        "models": list(MODELS), "probe_points": list(PROBE_POINTS), "late_points": list(LATE_POINTS),
        "manifest": str(manifest_path), "token_selection": str(selection_path),
        "pre_sft_result_dir": str(Path(args.pre_sft_result_dir).resolve()), "seed": args.seed,
        "random_directions": args.random_directions, "label_permutations": args.label_permutations,
        "capability_contrast": "SS123 - 0.5 * (SS012 + SS036)",
        "pairwise_null": "joint per-video architecture-label permutation across the complete profile",
    }
    write_json(output / "analysis_provenance.json", provenance)
    lines = ["# Post-SFT frozen VF analysis", "", "All rows use the fixed 6/2/12 manifest and the unchanged v1 definition.", ""]
    for row in full_late + contrast_tests:
        lines.append(f"- `{row.get('scope')}` `{row['metric']}`: p={row.get('p_value')}, LOO={row.get('leave_one_video_out_passes')}/{row.get('leave_one_video_out_total')}, stable={row.get('stable')}.")
    lines.extend(["", "## Requested capability contrast", ""])
    for metric in METRICS:
        for scope in ("full_14_points", "late_L9_L27"):
            row = result_for(contrast_tests, scope=scope, metric=metric)
            lines.append(
                f"- `SS123 - mean(SS012, SS036)`, `{metric}`, `{scope}`: "
                f"p={row['p_value']}, LOO={row['leave_one_video_out_passes']}/{row['leave_one_video_out_total']}, stable={row['stable']}."
            )
    lines.extend(["", "## Pre-/post-SFT capability-profile agreement", ""])
    for row in correlations:
        lines.append(
            f"- `{row['metric']}`, `{row['scope']}`: Pearson r={row['pearson_profile_correlation']}."
        )
    lines.extend([
        "",
        "A stable contrast is a held-out-video representation distinction, not evidence of VSI causality. "
        "The R² and VF rows are reported side-by-side; this analysis does not optimize either for the known VSI pattern.",
    ])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "status": "complete"}, indent=2))


if __name__ == "__main__":
    main()
