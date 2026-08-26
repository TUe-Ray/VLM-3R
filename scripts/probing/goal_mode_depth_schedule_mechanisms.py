#!/usr/bin/env python
"""Development-only mechanistic analysis for C1 SpatialStack schedules.

Hypotheses are recorded in ``analysis_provenance.json`` before numerical work:

* Coordinate organization: frozen linear probes transfer poorly when schedules
  encode similarly readable depth in different residual-stream coordinates.
* Compactness/orientation: fixed-rank supervised depth subspaces differ in
  recovery, energy, and alignment when depth is distributed differently.
* Dynamics: late subspace drift differs when schedules retain distinct
  geometry trajectories after their final injection.
* Spectral/global control: covariance spectra and linear CKA distinguish a
  depth-specific occupancy effect from generic whole-representation change.

This program never reads confirmation videos and never runs a model forward.
"""

from __future__ import annotations

import argparse
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
    DEFAULT_RANKS,
    PROBE_POINTS,
    aggregate_rows,
    fit_ridge,
    load_video_point,
    manifest_by_split,
    point_data,
    profile_rows,
    regression_metrics,
    run_v2,
    run_v3,
    selected_training_arrays,
    stable_seed,
    video_id,
    write_json,
    write_rows,
)


MODELS = ("SS012", "SS123", "SS036")
LATE_POINTS = ("L9", "L12", "L15", "L18", "L21", "L24", "L27")


def load_selection(path: Path) -> dict[str, list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): [int(item) for item in value] for key, value in payload["selected_valid_indices"].items()}


def selected_alpha_rows(path: Path) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for row in __import__("csv").DictReader(path.open(newline="", encoding="utf-8")):
        if row.get("selected") == "True":
            result[(str(row["model"]), str(row["probe_point"]))] = float(row["ridge_alpha"])
    if len(result) != len(MODELS) * len(PROBE_POINTS):
        raise RuntimeError(f"Expected 42 selected development ridge alphas, found {len(result)}")
    return result


def fit_linear_cache(
    cache_root: Path,
    by_split: dict[str, list[dict[str, Any]]],
    selection: dict[str, list[int]],
    alphas: dict[tuple[str, str], float],
) -> dict[tuple[str, str], dict[str, Any]]:
    all_videos = by_split["train"] + by_split["val"] + by_split["dev_eval"]
    fitted: dict[tuple[str, str], dict[str, Any]] = {}
    for model in MODELS:
        for point in PROBE_POINTS:
            data = point_data(cache_root, model, point, all_videos)
            train_data = {video_id(video): data[video_id(video)] for video in by_split["train"]}
            x_train, y_train = selected_training_arrays(train_data, by_split["train"], selection)
            alpha = alphas[(model, point)]
            probe = fit_ridge(x_train, y_train, alpha)
            coefficient = np.asarray(probe.coef_, dtype=float).reshape(-1)
            norm = float(np.linalg.norm(coefficient))
            if not norm > 0 or not np.isfinite(norm):
                raise RuntimeError(f"Non-finite linear depth direction: {model}/{point}")
            fitted[(model, point)] = {
                "probe": probe,
                "direction": coefficient / norm,
                "data": data,
                "alpha": alpha,
            }
    return fitted


def assert_paired_targets(fitted: dict[tuple[str, str], dict[str, Any]], videos: list[dict[str, Any]]) -> None:
    for point in PROBE_POINTS:
        for video in videos:
            reference = fitted[(MODELS[0], point)]["data"][video_id(video)].y_valid
            for model in MODELS[1:]:
                candidate = fitted[(model, point)]["data"][video_id(video)].y_valid
                if not np.array_equal(reference, candidate):
                    raise RuntimeError(f"Cross-model target mismatch: {point}/{video_id(video)}/{model}")


def run_linear_transfer(
    fitted: dict[tuple[str, str], dict[str, Any]],
    by_split: dict[str, list[dict[str, Any]]],
    output: Path,
    seed: int,
) -> list[dict[str, Any]]:
    """Apply every frozen source ridge to every target architecture unchanged."""
    assert_paired_targets(fitted, by_split["dev_eval"])
    rows: list[dict[str, Any]] = []
    for point in PROBE_POINTS:
        for video in by_split["dev_eval"]:
            current_id = video_id(video)
            own_metrics: dict[str, dict[str, float]] = {}
            for model in MODELS:
                own = fitted[(model, point)]
                current = own["data"][current_id]
                own_metrics[model] = regression_metrics(current.y_valid, own["probe"].predict(current.x_valid))
            for source in MODELS:
                probe = fitted[(source, point)]["probe"]
                source_in_domain = own_metrics[source]
                for target in MODELS:
                    target_data = fitted[(target, point)]["data"][current_id]
                    metrics = regression_metrics(target_data.y_valid, probe.predict(target_data.x_valid))
                    target_own = own_metrics[target]
                    rows.append(
                        {
                            "source_model": source,
                            "target_model": target,
                            "probe_point": point,
                            "video_id": current_id,
                            "video_path": target_data.video_path,
                            "split": "dev_eval",
                            "transfer_r2": metrics["r2"],
                            "transfer_mae": metrics["mae"],
                            "source_in_domain_r2": source_in_domain["r2"],
                            "target_own_r2": target_own["r2"],
                            "degradation_vs_source_r2": source_in_domain["r2"] - metrics["r2"],
                            "degradation_vs_target_own_r2": target_own["r2"] - metrics["r2"],
                        }
                    )
    write_rows(output / "linear_probe_transfer_per_video.csv", rows)
    write_rows(
        output / "linear_probe_transfer_aggregate.csv",
        aggregate_rows(
            rows,
            ["source_model", "target_model", "probe_point"],
            ["transfer_r2", "transfer_mae", "degradation_vs_source_r2", "degradation_vs_target_own_r2"],
        ),
    )
    # Target model receives the two foreign coordinate systems.  Its average
    # own-vs-foreign degradation is a three-model profile, not a token test.
    degradation: list[dict[str, Any]] = []
    for target in MODELS:
        for point in PROBE_POINTS:
            for video in by_split["dev_eval"]:
                current_id = video_id(video)
                foreign = [
                    float(row["degradation_vs_target_own_r2"])
                    for row in rows
                    if row["target_model"] == target
                    and row["source_model"] != target
                    and row["probe_point"] == point
                    and row["video_id"] == current_id
                ]
                degradation.append(
                    {
                        "model": target,
                        "probe_point": point,
                        "video_id": current_id,
                        "foreign_probe_degradation_r2": float(np.mean(foreign)),
                    }
                )
    write_rows(output / "linear_foreign_probe_degradation_per_video.csv", degradation)
    profiles = []
    profiles.extend(profile_rows(degradation, metrics=["foreign_probe_degradation_r2"], models=list(MODELS), seed=seed))
    late = [row for row in degradation if row["probe_point"] in LATE_POINTS]
    for row in profile_rows(late, metrics=["foreign_probe_degradation_r2"], models=list(MODELS), seed=stable_seed(seed, "late_transfer"), points=LATE_POINTS):
        profiles.append({"scope": "late_L9_L27", **row})
    write_rows(output / "linear_transfer_profile_tests.csv", profiles)
    return rows


def run_direction_alignment(
    fitted: dict[tuple[str, str], dict[str, Any]],
    output: Path,
) -> None:
    cross_rows: list[dict[str, Any]] = []
    for point in PROBE_POINTS:
        dimension = int(fitted[(MODELS[0], point)]["direction"].size)
        for index, first in enumerate(MODELS):
            for second in MODELS[index + 1 :]:
                cosine = float(np.dot(fitted[(first, point)]["direction"], fitted[(second, point)]["direction"]))
                cross_rows.append(
                    {
                        "probe_point": point,
                        "model_a": first,
                        "model_b": second,
                        "absolute_cosine": abs(cosine),
                        "squared_cosine": cosine * cosine,
                        "random_squared_cosine_expectation": 1.0 / dimension,
                        "squared_cosine_enrichment": dimension * cosine * cosine,
                        "hidden_dim": dimension,
                    }
                )
    write_rows(output / "linear_depth_direction_cross_model_alignment.csv", cross_rows)
    drift_rows: list[dict[str, Any]] = []
    for model in MODELS:
        anchor = fitted[(model, "L9")]["direction"]
        for previous, current in zip(LATE_POINTS[:-1], LATE_POINTS[1:]):
            first = fitted[(model, previous)]["direction"]
            second = fitted[(model, current)]["direction"]
            drift_rows.append(
                {
                    "model": model,
                    "comparison": f"{previous}->{current}",
                    "comparison_type": "adjacent",
                    "absolute_cosine": abs(float(np.dot(first, second))),
                }
            )
        for point in LATE_POINTS[1:]:
            drift_rows.append(
                {
                    "model": model,
                    "comparison": f"L9->{point}",
                    "comparison_type": "anchor_L9",
                    "absolute_cosine": abs(float(np.dot(anchor, fitted[(model, point)]["direction"]))),
                }
            )
    write_rows(output / "linear_depth_direction_late_dynamics.csv", drift_rows)


def normalized_projector_overlap(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.square(first.T @ second).sum() / first.shape[1])


def run_subspace_postprocessing(
    fitted: dict[tuple[str, str], dict[str, Any]],
    output: Path,
) -> None:
    spectra: list[dict[str, Any]] = []
    alignment: list[dict[str, Any]] = []
    dynamics: list[dict[str, Any]] = []
    for point in PROBE_POINTS:
        dimension = int(fitted[(MODELS[0], point)]["basis"].shape[0])
        for model in MODELS:
            singular = np.asarray(fitted[(model, point)]["singular_values"], dtype=float)
            energy = np.square(singular)
            cumulative = np.cumsum(energy) / energy.sum()
            effective_rank = float(np.square(energy.sum()) / np.square(energy).sum())
            for index, value in enumerate(singular, start=1):
                spectra.append(
                    {
                        "model": model,
                        "probe_point": point,
                        "component": index,
                        "singular_value": float(value),
                        "energy_fraction": float(energy[index - 1] / energy.sum()),
                        "cumulative_energy": float(cumulative[index - 1]),
                        "effective_probe_weight_rank": effective_rank,
                    }
                )
        for rank in DEFAULT_RANKS:
            if rank > fitted[(MODELS[0], point)]["basis"].shape[1]:
                continue
            expected = rank / dimension
            for index, first_model in enumerate(MODELS):
                for second_model in MODELS[index + 1 :]:
                    first = fitted[(first_model, point)]["basis"][:, :rank]
                    second = fitted[(second_model, point)]["basis"][:, :rank]
                    overlap = normalized_projector_overlap(first, second)
                    alignment.append(
                        {
                            "probe_point": point,
                            "rank_k": rank,
                            "model_a": first_model,
                            "model_b": second_model,
                            "normalized_projector_overlap": overlap,
                            "random_overlap_expectation": expected,
                            "overlap_enrichment": overlap / expected,
                        }
                    )
    for model in MODELS:
        for rank in DEFAULT_RANKS:
            if rank > fitted[(model, "L9")]["basis"].shape[1]:
                continue
            anchor = fitted[(model, "L9")]["basis"][:, :rank]
            for previous, current in zip(LATE_POINTS[:-1], LATE_POINTS[1:]):
                overlap = normalized_projector_overlap(fitted[(model, previous)]["basis"][:, :rank], fitted[(model, current)]["basis"][:, :rank])
                dynamics.append({"model": model, "rank_k": rank, "comparison": f"{previous}->{current}", "comparison_type": "adjacent", "normalized_projector_overlap": overlap})
            for point in LATE_POINTS[1:]:
                overlap = normalized_projector_overlap(anchor, fitted[(model, point)]["basis"][:, :rank])
                dynamics.append({"model": model, "rank_k": rank, "comparison": f"L9->{point}", "comparison_type": "anchor_L9", "normalized_projector_overlap": overlap})
    write_rows(output / "depth_subspace_singular_spectra.csv", spectra)
    write_rows(output / "depth_subspace_cross_model_alignment.csv", alignment)
    write_rows(output / "depth_subspace_late_dynamics.csv", dynamics)


def linear_cka(first: np.ndarray, second: np.ndarray) -> float:
    first_centered = first.astype(np.float64, copy=False) - first.mean(axis=0, keepdims=True)
    second_centered = second.astype(np.float64, copy=False) - second.mean(axis=0, keepdims=True)
    first_gram = first_centered @ first_centered.T
    second_gram = second_centered @ second_centered.T
    numerator = float((first_gram * second_gram).sum())
    denominator = math.sqrt(float(np.square(first_gram).sum()) * float(np.square(second_gram).sum()))
    return numerator / denominator if denominator > 0 else float("nan")


def spectral_metrics(x: np.ndarray, direction: np.ndarray) -> dict[str, float]:
    centered = x.astype(np.float64, copy=False) - x.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    values, vectors = np.linalg.eigh(gram)
    values = np.clip(values, 0, None)[::-1]
    vectors = vectors[:, ::-1]
    positive = values > np.finfo(float).eps
    values, vectors = values[positive], vectors[:, positive]
    total = float(values.sum())
    if not total:
        return {"effective_rank": float("nan"), "top_pc_fraction": float("nan"), "top10_pc_fraction": float("nan"), "depth_direction_top10_pc_overlap": float("nan")}
    effective = float(total * total / np.square(values).sum())
    top10 = min(10, values.size)
    # Right singular axes r_i = X^T u_i / sqrt(lambda_i).  The squared cosine
    # sum is coordinate alignment of the supervised depth direction with the
    # dominant activation-PC subspace, not another depth decoder.
    right = centered.T @ vectors[:, :top10] / np.sqrt(values[:top10])[None, :]
    overlap = float(np.square(right.T @ direction).sum())
    return {
        "effective_rank": effective,
        "top_pc_fraction": float(values[0] / total),
        "top10_pc_fraction": float(values[:top10].sum() / total),
        "depth_direction_top10_pc_overlap": overlap,
    }


def run_spectral_and_global_controls(
    fitted: dict[tuple[str, str], dict[str, Any]],
    by_split: dict[str, list[dict[str, Any]]],
    output: Path,
    seed: int,
) -> None:
    spectral_rows: list[dict[str, Any]] = []
    cka_rows: list[dict[str, Any]] = []
    for point in PROBE_POINTS:
        for video in by_split["dev_eval"]:
            current_id = video_id(video)
            for model in MODELS:
                current = fitted[(model, point)]["data"][current_id]
                spectral_rows.append(
                    {
                        "model": model,
                        "probe_point": point,
                        "video_id": current_id,
                        **spectral_metrics(current.x_all, fitted[(model, point)]["direction"]),
                    }
                )
            for index, first in enumerate(MODELS):
                for second in MODELS[index + 1 :]:
                    first_x = fitted[(first, point)]["data"][current_id].x_all
                    second_x = fitted[(second, point)]["data"][current_id].x_all
                    if first_x.shape[0] != second_x.shape[0]:
                        raise RuntimeError(f"CKA token pairing mismatch: {point}/{current_id}")
                    cka_rows.append({"model_pair": f"{first} vs {second}", "model_a": first, "model_b": second, "probe_point": point, "video_id": current_id, "linear_cka": linear_cka(first_x, second_x)})
    write_rows(output / "activation_spectral_per_video.csv", spectral_rows)
    write_rows(output / "activation_spectral_aggregate.csv", aggregate_rows(spectral_rows, ["model", "probe_point"], ["effective_rank", "top_pc_fraction", "top10_pc_fraction", "depth_direction_top10_pc_overlap"]))
    profiles = profile_rows(spectral_rows, metrics=["effective_rank", "top_pc_fraction", "top10_pc_fraction", "depth_direction_top10_pc_overlap"], models=list(MODELS), seed=seed)
    late = [row for row in spectral_rows if row["probe_point"] in LATE_POINTS]
    for row in profile_rows(late, metrics=["effective_rank", "top_pc_fraction", "top10_pc_fraction", "depth_direction_top10_pc_overlap"], models=list(MODELS), seed=stable_seed(seed, "late_spectrum"), points=LATE_POINTS):
        profiles.append({"scope": "late_L9_L27", **row})
    write_rows(output / "activation_spectral_profile_tests.csv", profiles)
    write_rows(output / "global_linear_cka_per_video.csv", cka_rows)
    write_rows(output / "global_linear_cka_aggregate.csv", aggregate_rows(cka_rows, ["model_pair", "probe_point"], ["linear_cka"]))


def write_summary(output: Path) -> None:
    (output / "summary.md").write_text(
        "# Goal-mode development mechanism analysis\n\n"
        "This directory uses only the fixed 6/2/4 development manifest and the existing hidden-state cache. "
        "It does not access confirmation videos. Results are exploratory video-level evidence; all architecture-label "
        "permutations preserve each video's full layer trajectory.\n\n"
        "- `linear_transfer/`: frozen continuous-ridge source→target transfer and one-dimensional coordinate alignment.\n"
        "- `multidimensional_depth/`: 16-bin multi-output ridge, fixed ranks 1/2/4/8/15, SVD compactness, subspace orientation, and late dynamics.\n"
        "- `spectral_global_control/`: per-video activation spectra, depth-to-dominant-PC alignment, and paired linear CKA.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--parent-result-dir", required=True, help="Completed development_v1_v4 result containing frozen token selection/alphabetas.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-directions", type=int, default=64)
    args = parser.parse_args()
    cache_root = Path(args.cache_root).resolve()
    parent = Path(args.parent_result_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory must be new and empty: {output}")
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    by_split = manifest_by_split(manifest)
    if {key: len(by_split.get(key, [])) for key in ("train", "val", "dev_eval")} != {"train": 6, "val": 2, "dev_eval": 4}:
        raise RuntimeError("Goal-mode exploration requires the fixed 6/2/4 development manifest")
    selection = load_selection(parent / "token_selection.json")
    alphas = selected_alpha_rows(parent / "v1_ridge_vf" / "ridge_selection.csv")
    provenance = {
        "analysis_name": "goal_mode_depth_schedule_mechanisms_v1",
        "parent_result": str(parent),
        "manifest": str(Path(args.manifest).resolve()),
        "cache_root": str(cache_root),
        "models": list(MODELS),
        "probe_points": list(PROBE_POINTS),
        "late_points": list(LATE_POINTS),
        "seed": args.seed,
        "confirmation_accessed": False,
        "hypotheses": [
            "If similarly readable depth uses different residual-stream coordinates, unchanged source ridge probes will degrade on target models.",
            "If depth is distributed with different compactness, fixed-rank supervised depth subspaces will differ in recovery, energy, and orientation.",
            "If schedules leave persistent geometry trajectories, adjacent and L9-anchor supervised-subspace alignment will differ after L9.",
            "If VF is depth-specific rather than a generic representation shift, spectral depth-to-PC measures will differ beyond paired global CKA.",
        ],
        "null_controls": {
            "profile_permutation": "Permute complete within-video architecture layer trajectories only.",
            "subspace_alignment": "Random rank-k overlap expectation k/D.",
            "direction_alignment": "Random squared cosine expectation 1/D.",
            "global_control": "Paired linear CKA across matching visual tokens.",
        },
    }
    write_json(output / "analysis_provenance.json", provenance)
    write_summary(output)
    fitted_linear = fit_linear_cache(cache_root, by_split, selection, alphas)
    linear_dir = output / "linear_transfer"
    linear_dir.mkdir()
    run_linear_transfer(fitted_linear, by_split, linear_dir, args.seed)
    run_direction_alignment(fitted_linear, linear_dir)
    multidim_dir = output / "multidimensional_depth"
    multidim_dir.mkdir()
    _, _, fitted_subspace = run_v2(
        cache_root=cache_root,
        output_dir=multidim_dir,
        models=list(MODELS),
        by_split=by_split,
        selection=selection,
        alphas=tuple(DEFAULT_ALPHAS),
        random_count=args.random_directions,
        seed=args.seed,
        requested_bins=16,
    )
    run_v3(output_dir=multidim_dir, models=list(MODELS), by_split=by_split, fitted=fitted_subspace)
    run_subspace_postprocessing(fitted_subspace, multidim_dir)
    spectral_dir = output / "spectral_global_control"
    spectral_dir.mkdir()
    run_spectral_and_global_controls(fitted_linear, by_split, spectral_dir, args.seed)
    print(json.dumps({"output_dir": str(output), "status": "complete"}, indent=2))


if __name__ == "__main__":
    main()
