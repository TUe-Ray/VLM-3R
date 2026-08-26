#!/usr/bin/env python
"""Cache-only persistent-profile, pairwise, and pre-LLM sanity follow-up."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probing.analyze_depth_subspace_occupancy import (  # noqa: E402
    PROBE_POINTS,
    profile_permutation_test,
    profile_tensor,
    stable_seed,
    write_rows,
)


MODELS = ("SS012", "SS123", "SS036")
METRICS = ("linear_r2", "vf_enrich")
LATE_POINTS = ("L9", "L12", "L15", "L18", "L21", "L24", "L27")
PRE_LLM_POINTS = ("fusion_output", "projected_features")


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def numeric_rows(rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    converted = []
    for row in rows:
        item = dict(row)
        for metric in metrics:
            item[metric] = as_float(row.get(metric))
        converted.append(item)
    return converted


def profile_result(
    rows: list[dict[str, Any]], *, metric: str, models: tuple[str, ...], points: tuple[str, ...], seed: int
) -> dict[str, Any]:
    values, video_ids = profile_tensor(rows, metric=metric, models=list(models), points=points)
    result: dict[str, Any] = {
        "metric": metric,
        "models": " vs ".join(models),
        "points": ";".join(points),
        "video_count": int(values.shape[0]),
    }
    if values.shape[0] < 3:
        return {**result, "stable": False, "reason": "fewer_than_three_complete_videos"}
    return {
        **result,
        "video_ids": ";".join(video_ids),
        **profile_permutation_test(values, seed=stable_seed(seed, metric, *models, *points)),
        "reason": "",
    }


def torch_load(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor cache at {path}")
    return value


def level_for_point(point: str) -> str:
    return f"layer_{point[1:]}" if point.startswith("L") else point


def pre_llm_sanity(cache_root: Path, manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame_ids = [
        str(frame["frame_sample_id"])
        for video in manifest["videos"]
        for frame in video["frames"]
    ]
    per_frame: list[dict[str, Any]] = []
    for point in PRE_LLM_POINTS:
        level = level_for_point(point)
        for first_index, first in enumerate(MODELS):
            for second in MODELS[first_index + 1 :]:
                for frame_id in frame_ids:
                    first_value = torch_load(cache_root / "features" / first / level / f"frame_{frame_id}.pt").float()
                    second_value = torch_load(cache_root / "features" / second / level / f"frame_{frame_id}.pt").float()
                    if tuple(first_value.shape) != tuple(second_value.shape):
                        raise RuntimeError(f"Pre-LLM cache shape mismatch at {point}/{frame_id}: {first_value.shape} vs {second_value.shape}")
                    delta = first_value - second_value
                    rms_first = torch.sqrt(first_value.square().mean())
                    rms_delta = torch.sqrt(delta.square().mean())
                    per_frame.append(
                        {
                            "probe_point": point,
                            "model_a": first,
                            "model_b": second,
                            "frame_id": frame_id,
                            "exact_equal": bool(torch.equal(first_value, second_value)),
                            "max_abs_difference": float(delta.abs().max()),
                            "mean_abs_difference": float(delta.abs().mean()),
                            "rms_difference": float(rms_delta),
                            "relative_rms_difference": float(rms_delta / rms_first) if rms_first > 0 else float("nan"),
                        }
                    )
    aggregate: list[dict[str, Any]] = []
    for point in PRE_LLM_POINTS:
        for first_index, first in enumerate(MODELS):
            for second in MODELS[first_index + 1 :]:
                group = [row for row in per_frame if row["probe_point"] == point and row["model_a"] == first and row["model_b"] == second]
                aggregate.append(
                    {
                        "probe_point": point,
                        "model_a": first,
                        "model_b": second,
                        "frame_count": len(group),
                        "all_frames_exact_equal": all(bool(row["exact_equal"]) for row in group),
                        "max_abs_difference_max": max(float(row["max_abs_difference"]) for row in group),
                        "mean_abs_difference_mean": float(np.mean([float(row["mean_abs_difference"]) for row in group])),
                        "relative_rms_difference_max": max(float(row["relative_rms_difference"]) for row in group),
                    }
                )
    return per_frame, aggregate


def ordered_curve(rows: list[dict[str, Any]], *, metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    order = {point: index for index, point in enumerate(PROBE_POINTS)}
    result = []
    for row in rows:
        item = {"probe_order": order[row["probe_point"]], **row}
        for metric in metrics:
            item[metric] = as_float(item.get(metric))
        result.append(item)
    return sorted(result, key=lambda item: (str(item["model"]), int(item["probe_order"])))


def write_summary(
    path: Path,
    *,
    late_rows: list[dict[str, Any]],
    pairwise_rows: list[dict[str, Any]],
    sanity_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Cache-only persistent-profile follow-up",
        "",
        "This report reuses the completed development cache only. It neither reloads a model nor accesses the frozen confirmation videos.",
        "",
        "## L9--L27 persistent profile",
        "",
    ]
    for row in late_rows:
        if bool(row.get("stable", False)):
            lines.append(
                f"- `{row['metric']}`: T={float(row['observed_T']):.4g}, null 95th={float(row['null_q95']):.4g}, "
                f"exact p={float(row['p_value']):.4g}, LOO={row['leave_one_video_out_passes']}/{row['leave_one_video_out_total']}."
            )
        else:
            lines.append(f"- `{row['metric']}`: no stable L9--L27 distinction ({row.get('reason', 'threshold not met')}).")
    lines.extend(["", "## Pairwise interpretation", ""])
    for row in pairwise_rows:
        lines.append(
            f"- `{row['metric']}`, {row['models']} ({row['profile_scope']}): "
            f"T={float(row['observed_T']):.4g}, p={float(row['p_value']):.4g}, "
            f"LOO={row['leave_one_video_out_passes']}/{row['leave_one_video_out_total']}."
        )
    lines.extend(
        [
            "",
            "With four paired development videos there are only 2⁴=16 joint label permutations for a pair. "
            "Because the squared profile-separation statistic is invariant to swapping both labels globally, a two-sided exact pairwise p<0.05 is unattainable here. "
            "The pairwise rows are therefore descriptive effect/null comparisons; confirmation or more development videos are needed for pairwise α=0.05 claims.",
            "",
            "## Pre-LLM sanity control",
            "",
        ]
    )
    for row in sanity_rows:
        lines.append(
            f"- `{row['probe_point']}` {row['model_a']} vs {row['model_b']}: "
            f"exact fp16 equality across all frames={row['all_frames_exact_equal']}; "
            f"max |Δ|={float(row['max_abs_difference_max']):.3g}; "
            f"max relative RMS Δ={float(row['relative_rms_difference_max']):.3g}."
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result-dir", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.source_result_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory must be new and empty: {output}")
    output.mkdir(parents=True, exist_ok=False)
    cache_root = Path(args.cache_root).resolve()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    aggregate = ordered_curve(read_csv(source / "v1_ridge_vf" / "linear_vf_aggregate.csv"), metrics=())
    per_video = numeric_rows(read_csv(source / "v1_ridge_vf" / "linear_vf_per_video.csv"), METRICS)
    dev_rows = [row for row in per_video if row["split"] == "dev_eval"]
    write_rows(output / "full_14point_linear_vf_curve.csv", aggregate)

    late_rows = [
        profile_result(dev_rows, metric=metric, models=MODELS, points=LATE_POINTS, seed=args.seed)
        for metric in METRICS
    ]
    write_rows(output / "late_L9_L27_omnibus_profile_tests.csv", late_rows)

    pairwise_rows: list[dict[str, Any]] = []
    for first_index, first in enumerate(MODELS):
        for second in MODELS[first_index + 1 :]:
            for scope, points in (("full_14_points", PROBE_POINTS), ("late_L9_L27", LATE_POINTS)):
                for metric in METRICS:
                    result = profile_result(dev_rows, metric=metric, models=(first, second), points=points, seed=args.seed)
                    pairwise_rows.append({"profile_scope": scope, **result})
    write_rows(output / "pairwise_profile_tests.csv", pairwise_rows)

    on_off = ordered_curve(read_csv(source / "v4_geometry_propagation" / "on_off_aggregate.csv"), metrics=())
    write_rows(output / "full_14point_on_off_curves.csv", on_off)
    write_rows(output / "late_L9_L27_on_off_curves.csv", [row for row in on_off if row["probe_point"] in LATE_POINTS])

    sanity_frames, sanity_aggregate = pre_llm_sanity(cache_root, manifest)
    write_rows(output / "pre_llm_tensor_sanity_per_frame.csv", sanity_frames)
    write_rows(output / "pre_llm_tensor_sanity_aggregate.csv", sanity_aggregate)
    write_summary(output / "followup_summary.md", late_rows=late_rows, pairwise_rows=pairwise_rows, sanity_rows=sanity_aggregate)
    print(json.dumps({"output_dir": str(output), "late_profile_tests": late_rows}, indent=2))


if __name__ == "__main__":
    main()
