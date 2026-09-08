#!/usr/bin/env python
"""Development-only coupling of native SpatialStack ON/OFF deltas to depth.

This diagnostic asks whether the *change caused by geometry injection* occupies
the independently trained visual depth direction.  It is not another decoder:
the continuous depth ridge and alpha are fitted exactly as in v1 on the fixed
6/2/4 development split; deltas are projected afterward.  A random unit-vector
control gives the expected raw-coordinate occupancy baseline.  Text quantities
are projections onto that same visual depth direction and quantify propagation,
not text-token depth supervision.

The program rejects manifests other than the fixed development split and never
opens a confirmation cache.
"""

from __future__ import annotations

import argparse
import json
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
    aggregate_rows,
    fit_ridge,
    manifest_by_split,
    point_data,
    profile_rows,
    selected_training_arrays,
    stable_seed,
    video_id,
    write_json,
    write_rows,
)
from scripts.probing.goal_mode_depth_schedule_mechanisms import (  # noqa: E402
    LATE_POINTS,
    MODELS,
    load_selection,
    selected_alpha_rows,
)


LLM_POINTS = tuple(point for point in PROBE_POINTS if point.startswith("L"))
# L0 is intentionally retained as a mechanics sanity point.  SS123's first
# injection is at L1, however, so an across-schedule profile that includes L0
# contains a structural zero rather than a common post-injection quantity.
POST_ALL_POINTS = tuple(point for point in LLM_POINTS if point != "L0")


def rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value, dtype=np.float64)))) if value.size else float("nan")


def delta_path(delta_root: Path, model: str, video: dict[str, Any]) -> Path:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in video_id(video))
    return delta_root / model / f"video_{safe}.pt"


def load_delta(delta_root: Path, model: str, video: dict[str, Any], point: str) -> tuple[np.ndarray, np.ndarray]:
    path = delta_path(delta_root, model, video)
    if not path.is_file():
        raise FileNotFoundError(f"Missing development ON/OFF delta cache: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != "spatialstack_geometry_on_off_delta_v1":
        raise RuntimeError(f"Unexpected delta schema in {path}")
    frames = [int(item) for item in payload["selected_frames"]]
    visual_by_frame = payload["visual_delta_by_layer"].get(point)
    text = payload["text_delta_by_layer"].get(point)
    if not isinstance(visual_by_frame, dict) or text is None:
        raise RuntimeError(f"Delta cache {path} lacks {point}")
    visual = np.concatenate([np.asarray(visual_by_frame[str(frame)], dtype=np.float64) for frame in frames], axis=0)
    return visual, np.asarray(text, dtype=np.float64)


def fit_directions(
    cache_root: Path,
    by_split: dict[str, list[dict[str, Any]]],
    selection: dict[str, list[int]],
    alphas: dict[tuple[str, str], float],
) -> dict[tuple[str, str], np.ndarray]:
    fitted: dict[tuple[str, str], np.ndarray] = {}
    for model in MODELS:
        for point in LLM_POINTS:
            data = point_data(cache_root, model, point, by_split["train"])
            train_data = {video_id(video): data[video_id(video)] for video in by_split["train"]}
            x_train, y_train = selected_training_arrays(train_data, by_split["train"], selection)
            coefficient = np.asarray(fit_ridge(x_train, y_train, alphas[(model, point)]).coef_, dtype=float).reshape(-1)
            norm = float(np.linalg.norm(coefficient))
            if not np.isfinite(norm) or norm <= 0:
                raise RuntimeError(f"Non-finite ridge direction: {model}/{point}")
            fitted[(model, point)] = coefficient / norm
    return fitted


def direction_fraction(delta: np.ndarray, direction: np.ndarray) -> float:
    denominator = float(np.square(delta, dtype=np.float64).sum())
    if denominator <= 0:
        return float("nan")
    projection = delta @ direction
    return float(np.square(projection, dtype=np.float64).sum() / denominator)


def finite_mean_std(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return float("nan"), float("nan")
    return float(finite.mean()), float(finite.std())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--delta-cache-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--parent-result-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--random-directions", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cache_root = Path(args.cache_root).resolve()
    delta_root = Path(args.delta_cache_root).resolve()
    parent = Path(args.parent_result_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory must be new and empty: {output}")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    by_split = manifest_by_split(manifest)
    if {key: len(by_split.get(key, [])) for key in ("train", "val", "dev_eval")} != {"train": 6, "val": 2, "dev_eval": 4}:
        raise RuntimeError("ON/OFF coupling requires the fixed 6/2/4 development manifest")
    output.mkdir(parents=True, exist_ok=False)
    selection = load_selection(parent / "token_selection.json")
    alphas = selected_alpha_rows(parent / "v1_ridge_vf" / "ridge_selection.csv")
    directions = fit_directions(cache_root, by_split, selection, alphas)
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for point in LLM_POINTS:
            direction = directions[(model, point)]
            dimension = int(direction.size)
            rng = np.random.default_rng(stable_seed(args.seed, "on_off_coupling", model, point))
            random_directions = rng.normal(size=(int(args.random_directions), dimension))
            random_directions /= np.linalg.norm(random_directions, axis=1, keepdims=True)
            for video in by_split["dev_eval"]:
                visual, text = load_delta(delta_root, model, video, point)
                # The extracted selected visual grids must share the cache's
                # token ordering/length; check this rather than assume it.
                cache_video = point_data(cache_root, model, point, [video])[video_id(video)]
                if visual.shape != cache_video.x_all.shape:
                    raise RuntimeError(f"Visual delta/cache mismatch {model}/{point}/{video_id(video)}: {visual.shape} vs {cache_video.x_all.shape}")
                visual_fraction = direction_fraction(visual, direction)
                visual_projection_rms = rms(visual @ direction)
                text_projection_rms = rms(text @ direction)
                random_fractions = np.asarray([direction_fraction(visual, random) for random in random_directions], dtype=float)
                random_mean, random_std = finite_mean_std(random_fractions)
                rows.append(
                    {
                        "model": model,
                        "probe_point": point,
                        "video_id": video_id(video),
                        "split": "dev_eval",
                        "hidden_dim": dimension,
                        "visual_delta_rms": rms(visual),
                        "text_delta_rms": rms(text),
                        "depth_aligned_visual_delta_fraction": visual_fraction,
                        "depth_aligned_visual_delta_enrich": dimension * visual_fraction,
                        "random_visual_delta_fraction_mean": random_mean,
                        "random_visual_delta_fraction_std": random_std,
                        "random_visual_delta_enrich_mean": float(dimension * random_mean),
                        "random_visual_delta_enrich_std": float(dimension * random_std),
                        "depth_aligned_visual_delta_rms": visual_projection_rms,
                        "depth_aligned_text_delta_rms": text_projection_rms,
                        "depth_aligned_text_visual_transfer_ratio": text_projection_rms / visual_projection_rms if visual_projection_rms > 0 else float("nan"),
                        "semantics": "post-hoc ON-OFF delta projected onto a visual ridge depth direction; text is propagation into that same coordinate",
                    }
                )
    write_rows(output / "on_off_depth_coupling_per_video.csv", rows)
    metrics = [
        "visual_delta_rms", "text_delta_rms", "depth_aligned_visual_delta_fraction", "depth_aligned_visual_delta_enrich",
        "random_visual_delta_enrich_mean", "depth_aligned_visual_delta_rms", "depth_aligned_text_delta_rms",
        "depth_aligned_text_visual_transfer_ratio",
    ]
    write_rows(output / "on_off_depth_coupling_aggregate.csv", aggregate_rows(rows, ["model", "probe_point"], metrics))
    profile_metrics = ["depth_aligned_visual_delta_enrich", "depth_aligned_text_visual_transfer_ratio"]
    profiles = [
        {"scope": "post_injection_L1_L27", **row}
        for row in profile_rows(
            [row for row in rows if row["probe_point"] in POST_ALL_POINTS],
            metrics=profile_metrics,
            models=list(MODELS),
            seed=args.seed,
            points=POST_ALL_POINTS,
        )
    ]
    late_rows = [row for row in rows if row["probe_point"] in LATE_POINTS]
    for row in profile_rows(late_rows, metrics=profile_metrics, models=list(MODELS), seed=stable_seed(args.seed, "late_on_off_coupling"), points=LATE_POINTS):
        profiles.append({"scope": "late_L9_L27", **row})
    write_rows(output / "profile_discrimination.csv", profiles)
    write_json(output / "analysis_provenance.json", {
        "analysis_name": "goal_mode_on_off_depth_coupling_v1",
        "manifest": str(Path(args.manifest).resolve()),
        "cache_root": str(cache_root),
        "delta_cache_root": str(delta_root),
        "models": list(MODELS),
        "probe_points": list(LLM_POINTS),
        "common_post_injection_profile_points": list(POST_ALL_POINTS),
        "confirmation_accessed": False,
        "direction_definition": "continuous visual depth ridge fit on selected train tokens with v1 validation-selected alpha",
        "metric_definition": "sum((Delta H b_hat)^2) / ||Delta H||_F^2 in original representation coordinates",
        "random_control": f"{args.random_directions} fixed-seed random unit directions per model/probe point",
        "permutation": "complete within-video architecture trajectories are permuted together",
        "seed": args.seed,
    })
    print(json.dumps({"output_dir": str(output), "status": "complete"}, indent=2))


if __name__ == "__main__":
    main()
