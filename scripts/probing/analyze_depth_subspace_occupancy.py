#!/usr/bin/env python
"""Cache-only depth-subspace analysis for C1 SpatialStack schedules.

The program deliberately separates model extraction from all probe fitting and
analysis.  It uses development-evaluation videos to navigate the predefined
refinement ladder; confirmation is a later frozen re-run, never a selector.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.linear_model import Ridge


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probing.depth_probe_common import write_csv, write_json  # noqa: E402


PROBE_POINTS = (
    "fusion_output",
    "projected_features",
    "L0",
    "L1",
    "L2",
    "L3",
    "L6",
    "L9",
    "L12",
    "L15",
    "L18",
    "L21",
    "L24",
    "L27",
)
DEFAULT_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
DEFAULT_RANKS = (1, 2, 4, 8, 15)


@dataclass(frozen=True)
class VideoPoint:
    video_id: str
    video_path: str
    split: str
    x_all: np.ndarray
    x_valid: np.ndarray
    y_valid: np.ndarray


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("\0".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def video_id(video: dict[str, Any]) -> str:
    return str(video.get("video_sample_id", video.get("video_path")))


def feature_path(cache_root: Path, model: str, point: str, frame_id: str) -> Path:
    level = f"layer_{point[1:]}" if point.startswith("L") else point
    return cache_root / "features" / model / level / f"frame_{frame_id}.pt"


def load_video_point(cache_root: Path, model: str, point: str, video: dict[str, Any]) -> VideoPoint:
    all_features: list[np.ndarray] = []
    valid_features: list[np.ndarray] = []
    valid_depths: list[np.ndarray] = []
    for frame in video["frames"]:
        frame_id = str(frame["frame_sample_id"])
        feature = load_torch(feature_path(cache_root, model, point, frame_id))
        if not isinstance(feature, torch.Tensor):
            raise TypeError(f"Feature is not a tensor: {model}/{point}/{frame_id}")
        gt = load_torch(cache_root / "gt_depth" / f"frame_{frame_id}.pt")
        meta = load_torch(cache_root / "metadata" / f"frame_{frame_id}.pt")
        if not isinstance(gt, torch.Tensor) or not isinstance(meta, dict):
            raise TypeError(f"Invalid cached target payload for {frame_id}")
        valid = meta.get("gt_valid_mask", torch.isfinite(gt) & (gt > 0))
        if not isinstance(valid, torch.Tensor):
            raise TypeError(f"Invalid gt_valid_mask for {frame_id}")
        x = feature.reshape(-1, feature.shape[-1]).float().numpy()
        y = gt.reshape(-1).float().numpy()
        mask = valid.reshape(-1).bool().numpy() & np.isfinite(y) & (y > 0)
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Feature/depth token mismatch for {frame_id}: {x.shape[0]} != {y.shape[0]}")
        all_features.append(x)
        valid_features.append(x[mask])
        valid_depths.append(y[mask])
    x_all = np.concatenate(all_features, axis=0)
    x_valid = np.concatenate(valid_features, axis=0)
    y_valid = np.concatenate(valid_depths, axis=0)
    return VideoPoint(
        video_id=video_id(video),
        video_path=str(video["video_path"]),
        split=str(video["split"]),
        x_all=x_all,
        x_valid=x_valid,
        y_valid=y_valid,
    )


def manifest_by_split(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for video in payload.get("videos", []):
        result.setdefault(str(video.get("split", "")), []).append(video)
    return result


def point_data(cache_root: Path, model: str, point: str, videos: Iterable[dict[str, Any]]) -> dict[str, VideoPoint]:
    return {video_id(video): load_video_point(cache_root, model, point, video) for video in videos}


def save_token_selection(
    *,
    output_dir: Path,
    train_videos: list[dict[str, Any]],
    cache_root: Path,
    reference_model: str,
    reference_point: str,
    max_tokens_per_video: int,
    seed: int,
) -> dict[str, list[int]]:
    path = output_dir / "token_selection.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {str(key): [int(value) for value in values] for key, values in payload["selected_valid_indices"].items()}
    selected: dict[str, list[int]] = {}
    for video in train_videos:
        data = load_video_point(cache_root, reference_model, reference_point, video)
        count = min(int(max_tokens_per_video), int(data.y_valid.shape[0]))
        rng = np.random.default_rng(stable_seed(seed, "tokens", data.video_id))
        selected[data.video_id] = sorted(rng.choice(data.y_valid.shape[0], size=count, replace=False).tolist())
    write_json(
        path,
        {
            "schema_version": "depth_subspace_shared_token_selection_v1",
            "reference_model": reference_model,
            "reference_point": reference_point,
            "max_tokens_per_video": int(max_tokens_per_video),
            "seed": int(seed),
            "selected_valid_indices": selected,
        },
    )
    return selected


def selected_training_arrays(data: dict[str, VideoPoint], train_videos: list[dict[str, Any]], selection: dict[str, list[int]]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for video in train_videos:
        current = data[video_id(video)]
        indices = np.asarray(selection[current.video_id], dtype=np.int64)
        xs.append(current.x_valid[indices])
        ys.append(current.y_valid[indices])
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(y) & np.isfinite(pred) & (y > 0)
    if not finite.any():
        return {"r2": float("nan"), "mae": float("nan"), "absrel": float("nan"), "num_tokens": 0.0}
    y, pred = y[finite], pred[finite]
    denominator = float(np.square(y - y.mean()).sum())
    r2 = float(1.0 - np.square(y - pred).sum() / denominator) if denominator > 0 else float("nan")
    return {
        "r2": r2,
        "mae": float(np.abs(y - pred).mean()),
        "absrel": float((np.abs(y - pred) / np.maximum(y, 1e-6)).mean()),
        "num_tokens": float(y.size),
    }


def variance_fraction(x: np.ndarray, basis: np.ndarray) -> float:
    centered = x.astype(np.float64, copy=False) - x.mean(axis=0, keepdims=True)
    denominator = float(np.square(centered).sum())
    if denominator <= 0:
        return float("nan")
    projected = centered @ basis
    return float(np.square(projected).sum() / denominator)


def variance_fractions_for_bases(x: np.ndarray, bases: np.ndarray) -> np.ndarray:
    """Compute raw-coordinate VF for many orthonormal bases after one centering.

    ``bases`` is ``[basis_count, hidden_dim, rank]``.  This is algebraically
    identical to calling :func:`variance_fraction` once per basis, but avoids
    repeating the required within-video centering for every random control.
    """
    if bases.ndim != 3:
        raise ValueError(f"Expected [basis_count, hidden_dim, rank], got {bases.shape}")
    centered = x.astype(np.float64, copy=False) - x.mean(axis=0, keepdims=True)
    denominator = float(np.square(centered).sum())
    if denominator <= 0:
        return np.full(bases.shape[0], np.nan, dtype=float)
    basis_count, hidden_dim, rank = bases.shape
    if centered.shape[1] != hidden_dim:
        raise ValueError(f"Feature/basis hidden-dimension mismatch: {centered.shape[1]} != {hidden_dim}")
    # One GEMM also preserves each direction/subspace's original-coordinate
    # normalization; no hidden-dimension standardization is introduced.
    flattened_basis = bases.astype(np.float64, copy=False).transpose(0, 2, 1).reshape(basis_count * rank, hidden_dim)
    projected = centered @ flattened_basis.T
    squared_by_component = np.square(projected).sum(axis=0).reshape(basis_count, rank)
    return squared_by_component.sum(axis=1) / denominator


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> Ridge:
    model = Ridge(alpha=float(alpha), fit_intercept=True, solver="lsqr", tol=1e-5)
    model.fit(x, y)
    return model


def choose_ridge_alpha(
    x_train: np.ndarray,
    y_train: np.ndarray,
    val_data: dict[str, VideoPoint],
    alphas: tuple[float, ...],
) -> tuple[float, Ridge, list[dict[str, Any]]]:
    candidates: list[tuple[float, float, Ridge]] = []
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        model = fit_ridge(x_train, y_train, alpha)
        values = []
        for current in val_data.values():
            values.append(regression_metrics(current.y_valid, model.predict(current.x_valid))["r2"])
        macro = float(np.nanmean(values))
        rows.append({"ridge_alpha": float(alpha), "validation_macro_r2": macro})
        candidates.append((macro, float(alpha), model))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1], candidates[0][2], rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        write_csv(path, rows)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n", encoding="utf-8")


def aggregate_rows(rows: list[dict[str, Any]], keys: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    result = []
    for group_key, group_rows in sorted(groups.items()):
        out = dict(zip(keys, group_key))
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group_rows if row.get(metric) is not None], dtype=float)
            values = values[np.isfinite(values)]
            out[f"{metric}_mean"] = float(values.mean()) if values.size else float("nan")
            out[f"{metric}_std"] = float(values.std(ddof=0)) if values.size else float("nan")
        out["video_count"] = len(group_rows)
        result.append(out)
    return result


def profile_tensor(rows: list[dict[str, Any]], *, metric: str, models: list[str], points: tuple[str, ...]) -> tuple[np.ndarray, list[str]]:
    by_video: dict[str, dict[tuple[str, str], float]] = {}
    for row in rows:
        value = row.get(metric)
        if value is None or not np.isfinite(float(value)):
            continue
        by_video.setdefault(str(row["video_id"]), {})[(str(row["model"]), str(row["probe_point"]))] = float(value)
    complete = []
    ids = []
    for current_id, values in sorted(by_video.items()):
        if all((model, point) in values for model in models for point in points):
            complete.append([[values[(model, point)] for point in points] for model in models])
            ids.append(current_id)
    if not complete:
        return np.empty((0, len(models), len(points))), ids
    return np.asarray(complete, dtype=float), ids


def profile_statistic(values: np.ndarray) -> float:
    means = values.mean(axis=0)
    grand = means.mean(axis=0, keepdims=True)
    return float(np.square(means - grand).sum())


def profile_permutation_test(
    values: np.ndarray,
    *,
    seed: int,
    max_permutations: int = 10000,
    compute_leave_one_video_out: bool = True,
) -> dict[str, Any]:
    """Jointly permute each video's full architecture profile."""
    video_count, model_count, _ = values.shape
    if model_count < 2:
        raise ValueError("Profile permutation requires at least two architectures")
    observed = profile_statistic(values)
    options = list(itertools.permutations(range(model_count)))
    total = len(options) ** video_count
    if total <= max_permutations:
        assignments = list(itertools.product(options, repeat=video_count))
        exact = True
    else:
        rng = np.random.default_rng(seed)
        assignments = [tuple(options[int(rng.integers(len(options)))] for _ in range(video_count)) for _ in range(max_permutations)]
        exact = False
    null = np.empty(len(assignments), dtype=float)
    for index, assignment in enumerate(assignments):
        permuted = np.stack([values[video_index, list(permutation)] for video_index, permutation in enumerate(assignment)])
        null[index] = profile_statistic(permuted)
    threshold = float(np.quantile(null, 0.95))
    p_value = float((1 + np.count_nonzero(null >= observed)) / (1 + null.size))
    loo = []
    if compute_leave_one_video_out and video_count >= 3:
        for leave_out in range(video_count):
            kept = np.delete(values, leave_out, axis=0)
            # A leave-one-video-out check needs the statistic/null for that
            # one reduced dataset only.  Recursing into LOO-of-LOO datasets
            # changes no reported value, but grows exponentially for the
            # independently held-out 8--16 video confirmation set.
            nested = profile_permutation_test(
                kept,
                seed=stable_seed(seed, "loo", leave_out),
                max_permutations=max_permutations,
                compute_leave_one_video_out=False,
            )
            loo.append(bool(nested["observed_T"] > nested["null_q95"]))
    return {
        "observed_T": observed,
        "null_q95": threshold,
        "p_value": p_value,
        "permutation_count": int(null.size),
        "permutation_exact": exact,
        "leave_one_video_out_passes": int(sum(loo)),
        "leave_one_video_out_total": int(len(loo)),
        "stable": bool(observed > threshold and (not loo or sum(loo) >= max(1, video_count - 1))),
    }


def profile_rows(
    rows: list[dict[str, Any]],
    *,
    metrics: list[str],
    models: list[str],
    seed: int,
    points: tuple[str, ...] = PROBE_POINTS,
) -> list[dict[str, Any]]:
    """Evaluate the frozen whole-profile statistic for each metric.

    Text/query influence is undefined at pre-LLM points, so callers can pass
    the LLM-only subset.  A single-model smoke has no architecture-label null
    to permute; retain the output row, but never call it discriminative.
    """
    result = []
    for metric in metrics:
        values, ids = profile_tensor(rows, metric=metric, models=models, points=points)
        common = {
            "metric": metric,
            "profile_points": ";".join(points),
            "video_count": int(values.shape[0]),
        }
        if len(models) != 3:
            result.append(
                {
                    **common,
                    "stable": False,
                    "reason": "architecture_profile_permutation_requires_three_models",
                }
            )
            continue
        if values.shape[0] < 3:
            result.append({**common, "stable": False, "reason": "fewer_than_three_complete_videos"})
            continue
        result.append(
            {
                **common,
                "video_ids": ";".join(ids),
                **profile_permutation_test(values, seed=stable_seed(seed, metric)),
            }
        )
    return result


def plot_profiles(rows: list[dict[str, Any]], *, metric: str, models: list[str], output: Path, ylabel: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to render analysis plots") from exc
    figure, axis = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(PROBE_POINTS))
    for model in models:
        model_rows = [row for row in rows if row.get("model") == model]
        by_video: dict[str, dict[str, float]] = {}
        for row in model_rows:
            value = row.get(metric)
            if value is not None and np.isfinite(float(value)):
                by_video.setdefault(str(row["video_id"]), {})[str(row["probe_point"])] = float(value)
        vectors = []
        for point_values in by_video.values():
            vectors.append([point_values.get(point, np.nan) for point in PROBE_POINTS])
        if not vectors:
            continue
        matrix = np.asarray(vectors, dtype=float)
        for vector in matrix:
            axis.plot(x, vector, color="0.75", linewidth=0.8, alpha=0.45)
        means = np.asarray(
            [
                float(np.nanmean(matrix[:, index])) if np.isfinite(matrix[:, index]).any() else float("nan")
                for index in range(matrix.shape[1])
            ]
        )
        axis.plot(x, means, marker="o", linewidth=2, label=model)
    axis.set_xticks(x, PROBE_POINTS, rotation=45, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_xlabel("probe point")
    axis.legend(loc="best")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def run_v1(
    *,
    cache_root: Path,
    output_dir: Path,
    models: list[str],
    by_split: dict[str, list[dict[str, Any]]],
    selection: dict[str, list[int]],
    alphas: tuple[float, ...],
    random_count: int,
    label_permutations: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    result_dir = output_dir / "v1_ridge_vf"
    result_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    label_null_rows: list[dict[str, Any]] = []
    fitted: dict[tuple[str, str], dict[str, Any]] = {}
    all_videos = by_split.get("train", []) + by_split.get("val", []) + by_split.get("dev_eval", [])
    for model in models:
        for point in PROBE_POINTS:
            data = point_data(cache_root, model, point, all_videos)
            train_data = {video_id(video): data[video_id(video)] for video in by_split["train"]}
            val_data = {video_id(video): data[video_id(video)] for video in by_split["val"]}
            x_train, y_train = selected_training_arrays(train_data, by_split["train"], selection)
            alpha, probe, choices = choose_ridge_alpha(x_train, y_train, val_data, alphas)
            for choice in choices:
                alpha_rows.append({"model": model, "probe_point": point, **choice, "selected": choice["ridge_alpha"] == alpha})
            coefficient = np.asarray(probe.coef_, dtype=float).reshape(-1)
            norm = float(np.linalg.norm(coefficient))
            direction = coefficient / norm if norm > 0 else np.full_like(coefficient, np.nan)
            random_rng = np.random.default_rng(stable_seed(seed, "random_direction", model, point))
            random_directions = random_rng.standard_normal((int(random_count), direction.size))
            random_directions /= np.linalg.norm(random_directions, axis=1, keepdims=True)
            null_rng = np.random.default_rng(stable_seed(seed, "depth_label_null", model, point))
            for null_index in range(int(label_permutations)):
                null_targets = []
                for video in by_split["train"]:
                    current = train_data[video_id(video)]
                    selected_y = current.y_valid[np.asarray(selection[current.video_id], dtype=np.int64)].copy()
                    null_rng.shuffle(selected_y)
                    null_targets.append(selected_y)
                null_probe = fit_ridge(x_train, np.concatenate(null_targets), alpha)
                null_dev = [
                    regression_metrics(data[video_id(video)].y_valid, null_probe.predict(data[video_id(video)].x_valid))["r2"]
                    for video in by_split["dev_eval"]
                ]
                label_null_rows.append({"model": model, "probe_point": point, "permutation": null_index, "ridge_alpha": alpha, "dev_eval_macro_r2": float(np.nanmean(null_dev))})
            for split in ("train", "val", "dev_eval"):
                for video in by_split.get(split, []):
                    current = data[video_id(video)]
                    pred = probe.predict(current.x_valid)
                    metrics = regression_metrics(current.y_valid, pred)
                    vf = variance_fraction(current.x_all, direction[:, None])
                    random_vf = variance_fractions_for_bases(
                        current.x_all,
                        random_directions[:, :, None],
                    )
                    rows.append(
                        {
                            "model": model,
                            "probe_point": point,
                            "video_id": current.video_id,
                            "video_path": current.video_path,
                            "split": split,
                            "ridge_alpha": alpha,
                            "linear_r2": metrics["r2"],
                            "linear_mae": metrics["mae"],
                            "linear_absrel": metrics["absrel"],
                            "num_valid_tokens": int(metrics["num_tokens"]),
                            "vf_depth": vf,
                            "vf_enrich": float(current.x_all.shape[1] * vf),
                            "random_vf_mean": float(np.nanmean(random_vf)),
                            "random_vf_std": float(np.nanstd(random_vf)),
                            "random_vf_enrich_mean": float(current.x_all.shape[1] * np.nanmean(random_vf)),
                            "hidden_dim": int(current.x_all.shape[1]),
                        }
                    )
            fitted[(model, point)] = {"probe": probe, "direction": direction, "alpha": alpha, "data": data, "x_train": x_train, "y_train": y_train}
    dev_rows = [row for row in rows if row["split"] == "dev_eval"]
    aggregates = aggregate_rows(
        dev_rows,
        ["model", "probe_point"],
        ["linear_r2", "linear_mae", "linear_absrel", "vf_depth", "vf_enrich", "random_vf_mean", "random_vf_std"],
    )
    write_rows(result_dir / "linear_vf_per_video.csv", rows)
    write_rows(result_dir / "linear_vf_aggregate.csv", aggregates)
    write_rows(result_dir / "ridge_selection.csv", alpha_rows)
    write_rows(result_dir / "linear_depth_label_null.csv", label_null_rows)
    profiles = profile_rows(dev_rows, metrics=["linear_r2", "vf_enrich"], models=models, seed=seed)
    write_rows(result_dir / "profile_discrimination.csv", profiles)
    plot_profiles(dev_rows, metric="linear_r2", models=models, output=result_dir / "linear_r2.png", ylabel="held-out development R²")
    plot_profiles(dev_rows, metric="vf_enrich", models=models, output=result_dir / "vf_enrich.png", ylabel="depth VF enrichment")
    return rows, profiles, fitted


def depth_bins(y: np.ndarray, requested_bins: int) -> tuple[np.ndarray, np.ndarray]:
    boundaries = np.unique(np.quantile(y, np.linspace(0.0, 1.0, int(requested_bins) + 1)))
    if boundaries.size < 3:
        raise RuntimeError("Depth training targets have insufficient variation for a multidimensional subspace")
    labels = np.digitize(y, boundaries[1:-1], right=False)
    medians = np.asarray([np.median(y[labels == index]) for index in range(boundaries.size - 1)], dtype=float)
    return boundaries, medians


def bin_labels(y: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    return np.digitize(y, boundaries[1:-1], right=False).astype(np.int64)


def run_v2(
    *,
    cache_root: Path,
    output_dir: Path,
    models: list[str],
    by_split: dict[str, list[dict[str, Any]]],
    selection: dict[str, list[int]],
    alphas: tuple[float, ...],
    random_count: int,
    seed: int,
    requested_bins: int,
    ranks_to_report: tuple[int, ...] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    result_dir = output_dir / "v2_rankk_depth_subspace"
    result_dir.mkdir(parents=True, exist_ok=True)
    reference = point_data(cache_root, models[0], "projected_features", by_split["train"])
    reference_y = np.concatenate([reference[video_id(video)].y_valid[np.asarray(selection[video_id(video)], dtype=np.int64)] for video in by_split["train"]])
    boundaries, medians = depth_bins(reference_y, requested_bins)
    bin_count = int(medians.size)
    available_ranks = tuple(rank for rank in DEFAULT_RANKS if rank <= bin_count - 1)
    if ranks_to_report is None:
        ranks = available_ranks
    else:
        ranks = tuple(int(rank) for rank in ranks_to_report)
        invalid = [rank for rank in ranks if rank not in available_ranks]
        if invalid:
            raise ValueError(f"Requested rank(s) not available for {bin_count} depth bins: {invalid}")
    write_json(result_dir / "depth_bins.json", {"boundaries": boundaries.tolist(), "bin_medians": medians.tolist(), "ranks": list(ranks)})
    rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    fitted: dict[tuple[str, str], dict[str, Any]] = {}
    all_videos = by_split.get("train", []) + by_split.get("val", []) + by_split.get("dev_eval", [])
    for model in models:
        for point in PROBE_POINTS:
            data = point_data(cache_root, model, point, all_videos)
            train_data = {video_id(video): data[video_id(video)] for video in by_split["train"]}
            val_data = {video_id(video): data[video_id(video)] for video in by_split["val"]}
            x_train, y_train = selected_training_arrays(train_data, by_split["train"], selection)
            labels_train = bin_labels(y_train, boundaries)
            y_one_hot = np.eye(bin_count, dtype=float)[labels_train]
            candidates: list[tuple[float, float, Ridge]] = []
            for alpha in alphas:
                probe = fit_ridge(x_train, y_one_hot, alpha)
                values = []
                for current in val_data.values():
                    labels = bin_labels(current.y_valid, boundaries)
                    target = np.eye(bin_count)[labels]
                    values.append(float(np.square(probe.predict(current.x_valid) - target).mean()))
                macro = float(np.mean(values))
                selection_rows.append({"model": model, "probe_point": point, "ridge_alpha": float(alpha), "validation_indicator_mse": macro, "selected": False})
                candidates.append((macro, float(alpha), probe))
            candidates.sort(key=lambda item: (item[0], item[1]))
            _, alpha, probe = candidates[0]
            for row in selection_rows[-len(alphas):]:
                row["selected"] = row["ridge_alpha"] == alpha
            coefficient = np.asarray(probe.coef_, dtype=float).T
            basis, singular_values, _ = np.linalg.svd(coefficient, full_matrices=False)
            fitted[(model, point)] = {"probe": probe, "basis": basis, "singular_values": singular_values, "data": data, "alpha": alpha, "boundaries": boundaries, "medians": medians}
            singular_total = float(np.square(singular_values).sum())
            for rank in ranks:
                subspace = basis[:, :rank]
                random_rng = np.random.default_rng(stable_seed(seed, "random_subspace", model, point, rank))
                random_bases = []
                for _ in range(int(random_count)):
                    raw = random_rng.standard_normal((subspace.shape[0], rank))
                    random_bases.append(np.linalg.qr(raw, mode="reduced")[0])
                for split in ("train", "val", "dev_eval"):
                    for video in by_split.get(split, []):
                        current = data[video_id(video)]
                        x_projected = current.x_valid @ subspace @ subspace.T
                        scores = probe.predict(x_projected)
                        prediction = scores.argmax(axis=1)
                        labels = bin_labels(current.y_valid, boundaries)
                        one_hot = np.eye(bin_count)[labels]
                        discrete_depth = medians[prediction]
                        vf = variance_fraction(current.x_all, subspace)
                        random_vf = variance_fractions_for_bases(current.x_all, np.stack(random_bases, axis=0))
                        rows.append(
                            {
                                "model": model,
                                "probe_point": point,
                                "rank_k": rank,
                                "video_id": current.video_id,
                                "video_path": current.video_path,
                                "split": split,
                                "ridge_alpha": alpha,
                                "bin_accuracy": float((prediction == labels).mean()),
                                "bin_indicator_mse": float(np.square(scores - one_hot).mean()),
                                "continuous_bin_median_mae": float(np.abs(discrete_depth - current.y_valid).mean()),
                                "vf_depth_subspace": vf,
                                "vf_enrich": float(current.x_all.shape[1] * vf / rank),
                                "random_vf_mean": float(np.mean(random_vf)),
                                "random_vf_std": float(np.std(random_vf)),
                                "singular_energy_recovered": float(np.square(singular_values[:rank]).sum() / singular_total) if singular_total else float("nan"),
                                "hidden_dim": int(current.x_all.shape[1]),
                            }
                        )
    dev_rows = [row for row in rows if row["split"] == "dev_eval"]
    aggregates = aggregate_rows(dev_rows, ["model", "probe_point", "rank_k"], ["bin_accuracy", "bin_indicator_mse", "continuous_bin_median_mae", "vf_depth_subspace", "vf_enrich", "singular_energy_recovered"])
    write_rows(result_dir / "rankk_per_video.csv", rows)
    write_rows(result_dir / "rankk_aggregate.csv", aggregates)
    write_rows(result_dir / "ridge_selection.csv", selection_rows)
    profile_results = []
    for rank in ranks:
        rank_rows = [row for row in dev_rows if row["rank_k"] == rank]
        for result in profile_rows(rank_rows, metrics=["bin_accuracy", "vf_enrich"], models=models, seed=stable_seed(seed, rank)):
            profile_results.append({"rank_k": rank, **result})
    write_rows(result_dir / "profile_discrimination.csv", profile_results)
    for rank in ranks:
        plot_profiles([row for row in dev_rows if row["rank_k"] == rank], metric="bin_accuracy", models=models, output=result_dir / f"bin_accuracy_k{rank}.png", ylabel=f"depth-bin accuracy (k={rank})")
    return rows, profile_results, fitted


def run_v3(
    *,
    output_dir: Path,
    models: list[str],
    by_split: dict[str, list[dict[str, Any]]],
    fitted: dict[tuple[str, str], dict[str, Any]],
    ranks_to_report: tuple[int, ...] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    result_dir = output_dir / "v3_alignment_transfer"
    result_dir.mkdir(parents=True, exist_ok=True)
    alignment_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    pairs = list(itertools.combinations(models, 2))
    for point in PROBE_POINTS:
        available_ranks = [rank for rank in DEFAULT_RANKS if rank <= fitted[(models[0], point)]["basis"].shape[1]]
        ranks = available_ranks if ranks_to_report is None else [int(rank) for rank in ranks_to_report]
        invalid = [rank for rank in ranks if rank not in available_ranks]
        if invalid:
            raise ValueError(f"Requested rank(s) not available at {point}: {invalid}")
        for rank in ranks:
            for first, second in pairs:
                first_basis = fitted[(first, point)]["basis"][:, :rank]
                second_basis = fitted[(second, point)]["basis"][:, :rank]
                cosines = np.linalg.svd(first_basis.T @ second_basis, compute_uv=False)
                alignment_rows.append({"probe_point": point, "rank_k": rank, "model_a": first, "model_b": second, "mean_squared_canonical_correlation": float(np.square(cosines).mean()), "mean_principal_angle_degrees": float(np.degrees(np.arccos(np.clip(cosines, -1, 1))).mean())})
            for source in models:
                source_fit = fitted[(source, point)]
                source_basis = source_fit["basis"][:, :rank]
                source_probe: Ridge = source_fit["probe"]
                medians = source_fit["medians"]
                boundaries = source_fit["boundaries"]
                for target in models:
                    target_data = fitted[(target, point)]["data"]
                    for video in by_split.get("dev_eval", []):
                        current = target_data[video_id(video)]
                        scores = source_probe.predict(current.x_valid @ source_basis @ source_basis.T)
                        pred = scores.argmax(axis=1)
                        labels = bin_labels(current.y_valid, boundaries)
                        transfer_rows.append({"source_model": source, "target_model": target, "probe_point": point, "rank_k": rank, "video_id": current.video_id, "video_path": current.video_path, "split": "dev_eval", "bin_accuracy": float((pred == labels).mean()), "continuous_bin_median_mae": float(np.abs(medians[pred] - current.y_valid).mean())})
    write_rows(result_dir / "subspace_alignment.csv", alignment_rows)
    write_rows(result_dir / "cross_model_transfer_per_video.csv", transfer_rows)
    write_rows(result_dir / "cross_model_transfer_aggregate.csv", aggregate_rows(transfer_rows, ["source_model", "target_model", "probe_point", "rank_k"], ["bin_accuracy", "continuous_bin_median_mae"]))
    # Each target architecture receives the two foreign probes.  The
    # per-video loss from its in-domain rank-k probe is a compact,
    # architecture-specific transfer-organization profile.
    in_domain: dict[tuple[str, str, int, str], float] = {}
    degradation_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, int, str], list[float]] = {}
    for row in transfer_rows:
        if row["source_model"] != row["target_model"]:
            grouped.setdefault((row["target_model"], row["probe_point"], row["rank_k"], row["video_id"]), []).append(float(row["bin_accuracy"]))
        else:
            in_domain[(row["target_model"], row["probe_point"], row["rank_k"], row["video_id"])] = float(row["bin_accuracy"])
    for (target, point, rank, current_video), foreign in grouped.items():
        own = in_domain[(target, point, rank, current_video)]
        degradation_rows.append({"model": target, "probe_point": point, "rank_k": rank, "video_id": current_video, "transfer_accuracy_degradation": own - float(np.mean(foreign))})
    profile_results = []
    for rank in sorted({int(row["rank_k"]) for row in degradation_rows}):
        rank_rows = [row for row in degradation_rows if int(row["rank_k"]) == rank]
        for result in profile_rows(rank_rows, metrics=["transfer_accuracy_degradation"], models=models, seed=stable_seed("transfer", rank)):
            profile_results.append({"rank_k": rank, **result})
    write_rows(result_dir / "transfer_degradation_per_video.csv", degradation_rows)
    write_rows(result_dir / "profile_discrimination.csv", profile_results)
    return alignment_rows, transfer_rows, profile_results


def run_v4(*, cache_root: Path, output_dir: Path, models: list[str], by_split: dict[str, list[dict[str, Any]]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result_dir = output_dir / "v4_geometry_propagation"
    result_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for model in models:
        for video in by_split.get("dev_eval", []):
            current_id = video_id(video)
            safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in current_id)
            path = cache_root / "geometry_on_off" / model / f"video_{safe}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing ON/OFF result for {model}/{current_id}: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            for row in payload:
                rows.append({"model": model, **row})
    write_rows(result_dir / "on_off_per_video.csv", rows)
    write_rows(result_dir / "on_off_aggregate.csv", aggregate_rows(rows, ["model", "probe_point"], ["I_visual", "I_text", "text_visual_transfer_ratio"]))
    profiles = profile_rows(rows, metrics=["I_visual"], models=models, seed=seed)
    profiles.extend(
        profile_rows(
            rows,
            metrics=["I_text"],
            models=models,
            seed=seed,
            points=PROBE_POINTS[2:],
        )
    )
    # SS123 has no visual delta at L0, so its L0 transfer ratio is the
    # correctly undefined 0/0 rather than an invented zero.  Compare the
    # common L1--L27 support jointly and retain the L0 N/A in the raw table.
    profiles.extend(
        profile_rows(
            rows,
            metrics=["text_visual_transfer_ratio"],
            models=models,
            seed=seed,
            points=PROBE_POINTS[3:],
        )
    )
    write_rows(result_dir / "profile_discrimination.csv", profiles)
    plot_profiles(rows, metric="I_visual", models=models, output=result_dir / "i_visual.png", ylabel="I visual")
    plot_profiles(rows, metric="I_text", models=models, output=result_dir / "i_text.png", ylabel="I text")
    plot_profiles(rows, metric="text_visual_transfer_ratio", models=models, output=result_dir / "transfer_ratio.png", ylabel="text/visual transfer")
    return rows, profiles


def any_stable(rows: list[dict[str, Any]]) -> bool:
    return any(bool(row.get("stable", False)) for row in rows)


def write_summary(output_dir: Path, stage_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = ["# Depth-subspace occupancy analysis", "", "## Frozen protocol", "", f"- Manifest: `{args.manifest}`", f"- Models: `{', '.join(args.models)}`", f"- Development evaluation, not test, determines the refinement ladder.", f"- Seed: `{args.seed}`; max training tokens/video: `{args.max_tokens_per_video}`.", "", "## Stage decisions", ""]
    for row in stage_rows:
        lines.append(f"- {row['stage']}: {'stable development distinction' if row['stable'] else 'advance / no stable distinction'}")
    lines.extend(["", "## Interpretation boundary", "", "All reported signals are representational diagnostics. They are not optimized for nor claimed to establish VSI causality.", ""])
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_confirmation_selection(
    *,
    path: Path,
    args: argparse.Namespace,
    output_dir: Path,
    profiles_by_stage: dict[str, list[dict[str, Any]]],
) -> None:
    """Freeze one development-selected analysis before confirmation is read.

    The frozen artifact is deliberately explicit about the version, metric,
    rank (if applicable), and scientific rationale.  The confirmation runner
    re-fits only on the recorded train/validation videos and never re-enters
    the refinement ladder.
    """
    required = (args.confirmation_manifest, args.frozen_stage, args.frozen_metric, args.selection_rationale)
    if not all(required):
        raise ValueError(
            "--freeze-selection requires --confirmation-manifest, --frozen-stage, "
            "--frozen-metric, and --selection-rationale"
        )
    stage_rows = profiles_by_stage.get(str(args.frozen_stage), [])
    selected = [
        row
        for row in stage_rows
        if str(row.get("metric")) == str(args.frozen_metric)
        and (args.frozen_rank is None or int(row.get("rank_k", -1)) == int(args.frozen_rank))
    ]
    if len(selected) != 1:
        raise ValueError(
            "Frozen stage/metric/rank did not identify exactly one development profile result: "
            f"stage={args.frozen_stage}, metric={args.frozen_metric}, rank={args.frozen_rank}"
        )
    if not bool(selected[0].get("stable", False)):
        raise ValueError("Only a stable development profile can be frozen for independent confirmation")
    development_manifest = Path(args.manifest).resolve()
    confirmation_manifest = Path(args.confirmation_manifest).resolve()
    selection_path = output_dir / "token_selection.json"
    provenance_path = output_dir / "analysis_provenance.json"
    if not confirmation_manifest.is_file() or not selection_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError("Cannot freeze selection: confirmation manifest, token selection, or provenance is missing")
    write_json(
        path,
        {
            "schema_version": "depth_subspace_frozen_confirmation_selection_v1",
            "development_output_dir": str(output_dir),
            "development_manifest": str(development_manifest),
            "development_manifest_sha256": file_sha256(development_manifest),
            "confirmation_manifest": str(confirmation_manifest),
            "confirmation_manifest_sha256": file_sha256(confirmation_manifest),
            "token_selection": str(selection_path),
            "token_selection_sha256": file_sha256(selection_path),
            "analysis_provenance": str(provenance_path),
            "analysis_provenance_sha256": file_sha256(provenance_path),
            "cache_root": str(Path(args.cache_root).resolve()),
            "models": list(args.models),
            "stage": str(args.frozen_stage),
            "metric": str(args.frozen_metric),
            "rank_k": int(args.frozen_rank) if args.frozen_rank is not None else None,
            "scientific_rationale": str(args.selection_rationale),
            "frozen_development_profile": selected[0],
            "frozen_analysis_config": {
                "alphas": [float(value) for value in args.alphas],
                "max_tokens_per_video": int(args.max_tokens_per_video),
                "random_directions": int(args.random_directions),
                "label_permutations": int(args.label_permutations),
                "depth_bins": int(args.depth_bins),
                "seed": int(args.seed),
            },
            "confirmation_rule": (
                "Apply this exact stage/metric/rank once to the pre-frozen confirmation videos. "
                "Do not use confirmation results to choose a replacement metric, rank, or stage."
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="+", default=["SS012", "SS123", "SS036"])
    parser.add_argument("--stages", choices=["auto", "v1", "v1,v2", "v1,v2,v3", "all"], default="auto")
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--max-tokens-per-video", type=int, default=128)
    parser.add_argument("--random-directions", type=int, default=64)
    parser.add_argument("--label-permutations", type=int, default=32)
    parser.add_argument("--depth-bins", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-selection", default=None, help="Write a development-selected, immutable confirmation spec here.")
    parser.add_argument("--confirmation-manifest", default=None, help="Pre-frozen confirmation manifest recorded by --freeze-selection.")
    parser.add_argument("--frozen-stage", choices=["v1_ridge_vf", "v2_rankk_depth_subspace", "v3_alignment_transfer", "v4_geometry_propagation"], default=None)
    parser.add_argument("--frozen-metric", default=None)
    parser.add_argument("--frozen-rank", type=int, default=None)
    parser.add_argument("--selection-rationale", default=None, help="Scientific reason for the development-stage selection; required when freezing.")
    args = parser.parse_args()
    if len(args.models) not in {1, 3}:
        parser.error("--models must list one smoke model or all three SS012 SS123 SS036 labels")
    cache_root = Path(args.cache_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    by_split = manifest_by_split(manifest)
    for split, minimum in (("train", 1), ("val", 1), ("dev_eval", 1)):
        if len(by_split.get(split, [])) < minimum:
            raise RuntimeError(f"Manifest requires at least {minimum} {split} video(s)")
    selection = save_token_selection(output_dir=output_dir, train_videos=by_split["train"], cache_root=cache_root, reference_model=args.models[0], reference_point="projected_features", max_tokens_per_video=args.max_tokens_per_video, seed=args.seed)
    stage_decisions: list[dict[str, Any]] = []
    v1_rows, v1_profiles, _ = run_v1(cache_root=cache_root, output_dir=output_dir, models=args.models, by_split=by_split, selection=selection, alphas=tuple(args.alphas), random_count=args.random_directions, label_permutations=args.label_permutations, seed=args.seed)
    v1_stable = any_stable(v1_profiles)
    stage_decisions.append({"stage": "v1_ridge_vf", "stable": v1_stable})
    run_v2_stage = args.stages in {"auto", "v1,v2", "v1,v2,v3", "all"} and (args.stages != "auto" or not v1_stable)
    v2_fitted: dict[tuple[str, str], dict[str, Any]] = {}
    v2_stable = False
    v2_profiles: list[dict[str, Any]] = []
    if run_v2_stage:
        _, v2_profiles, v2_fitted = run_v2(cache_root=cache_root, output_dir=output_dir, models=args.models, by_split=by_split, selection=selection, alphas=tuple(args.alphas), random_count=args.random_directions, seed=args.seed, requested_bins=args.depth_bins)
        v2_stable = any_stable(v2_profiles)
        stage_decisions.append({"stage": "v2_rankk_depth_subspace", "stable": v2_stable})
    run_v3_stage = len(args.models) == 3 and (
        args.stages in {"v1,v2,v3", "all"} or (args.stages == "auto" and run_v2_stage and not v2_stable)
    )
    v3_stable = False
    v3_profiles: list[dict[str, Any]] = []
    if run_v3_stage:
        if not v2_fitted:
            raise RuntimeError("v3 requires v2 fitted subspaces")
        _, transfer_rows, v3_profiles = run_v3(output_dir=output_dir, models=args.models, by_split=by_split, fitted=v2_fitted)
        v3_stable = any_stable(v3_profiles)
        stage_decisions.append({"stage": "v3_alignment_transfer", "stable": v3_stable})
    # ON/OFF is a primary training-free diagnostic, not a refinement selected
    # because an earlier metric happened to fail.  Run it for the three-model
    # development analysis even when v1 already supplies a stable profile.
    run_v4_stage = args.stages == "all" or (args.stages == "auto" and len(args.models) == 3)
    v4_profiles: list[dict[str, Any]] = []
    if run_v4_stage:
        _, v4_profiles = run_v4(cache_root=cache_root, output_dir=output_dir, models=args.models, by_split=by_split, seed=args.seed)
        stage_decisions.append({"stage": "v4_geometry_propagation", "stable": any_stable(v4_profiles)})
    write_json(output_dir / "analysis_provenance.json", {"schema_version": "depth_subspace_analysis_v1", "manifest": str(Path(args.manifest).resolve()), "cache_root": str(cache_root), "models": args.models, "probe_points": list(PROBE_POINTS), "ridge_alphas": args.alphas, "rank_grid": list(DEFAULT_RANKS), "random_directions": args.random_directions, "label_permutations": args.label_permutations, "seed": args.seed, "stage_decisions": stage_decisions})
    write_summary(output_dir, stage_decisions, args)
    if args.freeze_selection:
        freeze_confirmation_selection(
            path=Path(args.freeze_selection).resolve(),
            args=args,
            output_dir=output_dir,
            profiles_by_stage={
                "v1_ridge_vf": v1_profiles,
                "v2_rankk_depth_subspace": v2_profiles,
                "v3_alignment_transfer": v3_profiles,
                "v4_geometry_propagation": v4_profiles,
            },
        )
    print(json.dumps({"output_dir": str(output_dir), "stage_decisions": stage_decisions}, indent=2))


if __name__ == "__main__":
    main()
