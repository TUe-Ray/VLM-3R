#!/usr/bin/env python
"""Cache-only, streaming regression LogME for the formal pre-SFT C1 study.

The primary score is deliberately predeclared as the mean of the ten decoder
layers in :data:`PRIMARY_LAYERS`.  This script never loads a VLM, never creates
an optimizer, and refuses to use a cache whose provenance does not explicitly
say that it was extracted with fresh pre-SFT fusion.

The evidence calculation is exact for the materialized feature matrix.  It
streams frames into F.T @ F, F.T @ y, and y.T @ y, then eigendecomposes only
the D x D Gram matrix.  ``--smoke-only`` additionally compares this sufficient
statistics route to a materialized calculation on a deterministic small video
subset.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import resource
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probing.depth_probe_common import load_frame_records, stable_int_seed
from scripts.probing.train_depth_probes import load_feature_tensor


PRIMARY_LAYERS = (1, 3, 6, 9, 12, 15, 18, 21, 24, 27)
SPLIT_SHA256 = "d478cb684958dfc25066821ec83d5216469577c9e282e33bdf87d3c88b200d8e"
DEFAULT_SAMPLE_INDICES = Path(
    "/home/shaoruei/probe_provenance/scannet_baseline_L6/"
    "scannet_baseline_L6_depth_provenance/splits/"
    "semantic_probe_scannet_final_usable_sample_indices.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "pre_sft_logme_proxy_v1"
EPS = 1e-5


@dataclass(frozen=True)
class Candidate:
    label: str
    display_name: str
    root: Path
    vsi_name: str


CANDIDATES = (
    Candidate("c1_vlm3r", "C1 VLM3R Baseline", Path("/home/shaoruei/probe_cache/c1_vlm3r_v1/full"), "Baseline"),
    Candidate("c1_spatialstack_add", "SpatialStack additive 0/1/2", Path("/home/shaoruei/probe_cache/c1_additive_v1/full"), "Spatial Stack to Layer 0/1/2"),
    Candidate("c1_spatialstack_add_123", "SpatialStack additive 1/2/3", Path("/home/shaoruei/probe_cache/c1_ss_add_123/full"), "Spatial Stack to Layer 1/2/3"),
    Candidate("c1_spatialstack_add_036", "SpatialStack additive 0/3/6", Path("/home/shaoruei/probe_cache/c1_ss_add_036/full"), "Spatial Stack to Layer 0/3/6"),
    Candidate("c1_spatialstack_cross_attn_v1", "SpatialStack cross-attention 0/1/2", Path("/home/shaoruei/probe_cache/c1_ss_cross_attn_v1/full"), "Spatial Stack- Cross Attn"),
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance_path(candidate: Candidate) -> Path:
    return candidate.root / "features" / candidate.label / "extraction_provenance.json"


def load_and_validate_provenance(candidate: Candidate, sample_indices: Path) -> dict[str, Any]:
    path = provenance_path(candidate)
    if not path.is_file():
        raise RuntimeError(f"{candidate.label}: missing extraction provenance: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{candidate.label}: provenance is not an object")
    if payload.get("model_label") != candidate.label:
        raise RuntimeError(f"{candidate.label}: provenance model label mismatch")
    if payload.get("model_loading_mode") != "pre_sft_fusion":
        raise RuntimeError(f"{candidate.label}: not a pre-SFT fusion cache")
    if payload.get("no_vlm3r_sft_adapter_loaded") is not True:
        raise RuntimeError(f"{candidate.label}: provenance does not prove no post-SFT adapter was loaded")
    if payload.get("sample_indices_sha256") != SPLIT_SHA256:
        raise RuntimeError(f"{candidate.label}: unexpected sample split hash")
    if sha256_file(sample_indices) != SPLIT_SHA256:
        raise RuntimeError(f"Sample-index hash differs from the formal split: {sample_indices}")
    if payload.get("sample_indices") != str(sample_indices):
        raise RuntimeError(f"{candidate.label}: provenance references a different sample-index path")
    return payload


def candidate_missing_layers(candidate: Candidate) -> list[int]:
    root = candidate.root / "features" / candidate.label
    return [layer for layer in PRIMARY_LAYERS if not (root / f"layer_{layer}").is_dir()]


def select_videos(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """Return records for a deterministic video-level subset, preserving order."""
    if limit is None:
        return records
    video_ids = list(dict.fromkeys(str(record["video_sample_id"]) for record in records))
    if limit <= 0 or limit > len(video_ids):
        raise ValueError(f"--video-limit must be in [1, {len(video_ids)}], got {limit}")
    selected_indices = sorted(
        range(len(video_ids)), key=lambda index: stable_int_seed("pre_sft_logme_proxy_v1", limit, video_ids[index])
    )[:limit]
    selected = {video_ids[index] for index in selected_indices}
    return [record for record in records if str(record["video_sample_id"]) in selected]


def target_and_feature(
    candidate: Candidate, layer: int, record: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, bytes]:
    """Load exactly the target/mask treatment used by CachedFrameDepthDataset."""
    frame_id = str(record["frame_sample_id"])
    feature = load_feature_tensor(candidate.root, candidate.label, f"layer_{layer}", frame_id)
    target = torch.load(candidate.root / "gt_depth" / f"frame_{frame_id}.pt", map_location="cpu")
    metadata = torch.load(candidate.root / "metadata" / f"frame_{frame_id}.pt", map_location="cpu")
    valid = metadata.get("gt_valid_mask", torch.isfinite(target) & (target > 0)).reshape(-1).bool()
    x = feature.reshape(-1, feature.shape[-1])
    y = target.reshape(-1)
    if x.shape[0] != y.shape[0] or y.shape[0] != valid.shape[0]:
        raise RuntimeError(
            f"{candidate.label}/layer_{layer}/{frame_id}: feature/target alignment mismatch "
            f"{tuple(x.shape)}, {tuple(y.shape)}, {tuple(valid.shape)}"
        )
    valid = valid & torch.isfinite(y) & (y > 0)
    if not torch.isfinite(x[valid]).all():
        raise RuntimeError(f"{candidate.label}/layer_{layer}/{frame_id}: non-finite valid feature")
    # This signature proves both target values and their valid-token positions.
    signature = valid.numpy().tobytes() + y.to(dtype=torch.float32).numpy().tobytes()
    return x[valid].to(dtype=torch.float64), y[valid].to(dtype=torch.float64), signature


def accumulate_statistics(
    candidate: Candidate, layer: int, records: Iterable[dict[str, Any]], *, materialize: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, str, list[torch.Tensor] | None, list[torch.Tensor] | None]:
    gram: torch.Tensor | None = None
    cross: torch.Tensor | None = None
    yy = torch.zeros((), dtype=torch.float64)
    count = 0
    frame_count = 0
    target_hash = hashlib.sha256()
    xs: list[torch.Tensor] | None = [] if materialize else None
    ys: list[torch.Tensor] | None = [] if materialize else None
    for record in records:
        x, y, signature = target_and_feature(candidate, layer, record)
        target_hash.update(str(record["frame_sample_id"]).encode("utf-8"))
        target_hash.update(signature)
        if gram is None:
            dim = int(x.shape[1])
            gram = torch.zeros((dim, dim), dtype=torch.float64)
            cross = torch.zeros(dim, dtype=torch.float64)
        if x.shape[1] != gram.shape[0]:
            raise RuntimeError(f"{candidate.label}/layer_{layer}: feature dimension changed within cache")
        gram.addmm_(x.T, x, beta=1.0, alpha=1.0)
        cross.addmv_(x.T, y, beta=1.0, alpha=1.0)
        yy += torch.dot(y, y)
        count += int(y.numel())
        frame_count += 1
        if materialize:
            assert xs is not None and ys is not None
            xs.append(x)
            ys.append(y)
    if gram is None or cross is None or count == 0:
        raise RuntimeError(f"{candidate.label}/layer_{layer}: no valid tokens")
    return gram, cross, yy, count, frame_count, target_hash.hexdigest(), xs, ys


def logme_from_statistics(gram: torch.Tensor, cross: torch.Tensor, yy: torch.Tensor, count: int) -> dict[str, float | int]:
    """Original Bayesian linear-regression fixed-point evidence maximization."""
    gram = (gram + gram.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    largest = max(1.0, float(eigenvalues[-1].abs().item()))
    min_eigenvalue = float(eigenvalues[0].item())
    # A Gram matrix is PSD.  Only tolerate round-off sized negative values.
    negative_tolerance = 1e-8 * largest
    if min_eigenvalue < -negative_tolerance:
        raise RuntimeError(
            f"Materially non-PSD Gram matrix: min={min_eigenvalue:.6e}, max={largest:.6e}, "
            f"tolerance={negative_tolerance:.6e}"
        )
    eigenvalues = eigenvalues.clamp_min(0.0)
    projected_cross = eigenvectors.T @ cross
    alpha = 1.0
    beta = 1.0
    converged = False
    m_norm = float("nan")
    residual_sq = float("nan")
    gamma = float("nan")
    for iteration in range(1, 101):
        denominator = alpha + beta * eigenvalues
        if not torch.isfinite(denominator).all() or torch.any(denominator <= 0):
            raise RuntimeError("Invalid LogME precision denominator")
        coeff = beta * projected_cross / denominator
        m_norm_tensor = torch.dot(coeff, coeff)
        m_cross = torch.dot(coeff, projected_cross)
        m_gram_m = torch.dot(eigenvalues * coeff, coeff)
        gamma_tensor = torch.sum(beta * eigenvalues / denominator)
        residual_tensor = yy - 2.0 * m_cross + m_gram_m
        # Floating error may push this sum of squares infinitesimally negative.
        if float(residual_tensor.item()) < -EPS:
            raise RuntimeError(f"Materially negative residual square: {float(residual_tensor.item()):.6e}")
        residual_tensor = residual_tensor.clamp_min(0.0)
        new_alpha = float((gamma_tensor / (m_norm_tensor + EPS)).item())
        new_beta = float(((count - gamma_tensor) / (residual_tensor + EPS)).item())
        if not (math.isfinite(new_alpha) and math.isfinite(new_beta) and new_alpha > 0 and new_beta > 0):
            raise RuntimeError(f"Non-finite LogME alpha/beta: alpha={new_alpha}, beta={new_beta}")
        m_norm = float(m_norm_tensor.item())
        residual_sq = float(residual_tensor.item())
        gamma = float(gamma_tensor.item())
        if max(abs(new_alpha - alpha) / (abs(alpha) + EPS), abs(new_beta - beta) / (abs(beta) + EPS)) < 1e-6:
            alpha, beta, converged = new_alpha, new_beta, True
            break
        alpha, beta = new_alpha, new_beta
    denominator = alpha + beta * eigenvalues
    coeff = beta * projected_cross / denominator
    m_norm_tensor = torch.dot(coeff, coeff)
    residual_tensor = (yy - 2.0 * torch.dot(coeff, projected_cross) + torch.dot(eigenvalues * coeff, coeff)).clamp_min(0.0)
    log_evidence = 0.5 * (
        gram.shape[0] * math.log(alpha)
        + count * math.log(beta)
        - float(torch.log(denominator).sum().item())
        - beta * float(residual_tensor.item())
        - alpha * float(m_norm_tensor.item())
        - count * math.log(2.0 * math.pi)
    )
    score = log_evidence / count
    if not math.isfinite(score):
        raise RuntimeError("Non-finite normalized LogME")
    return {
        "logme": score,
        "alpha": alpha,
        "beta": beta,
        "iterations": iteration,
        "converged": converged,
        "gamma": gamma,
        "residual_sq": float(residual_tensor.item()),
        "m_norm_sq": float(m_norm_tensor.item()),
        "minimum_eigenvalue": min_eigenvalue,
    }


def materialized_logme(xs: list[torch.Tensor], ys: list[torch.Tensor]) -> tuple[dict[str, float | int], int]:
    features = torch.cat(xs, dim=0)
    targets = torch.cat(ys, dim=0)
    stats = logme_from_statistics(features.T @ features, features.T @ targets, torch.dot(targets, targets), int(targets.numel()))
    return stats, int(targets.numel())


def average_ranks(values: list[float], *, higher_is_better: bool) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda idx: -values[idx] if higher_is_better else values[idx])
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and math.isclose(values[ordered[cursor]], values[ordered[end]], rel_tol=1e-12, abs_tol=1e-12):
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for index in ordered[cursor:end]:
            result[index] = rank
        cursor = end
    return result


def pearson(first: list[float], second: list[float]) -> float:
    first_mean = sum(first) / len(first)
    second_mean = sum(second) / len(second)
    numerator = sum((x - first_mean) * (y - second_mean) for x, y in zip(first, second))
    first_norm = math.sqrt(sum((x - first_mean) ** 2 for x in first))
    second_norm = math.sqrt(sum((y - second_mean) ** 2 for y in second))
    return numerator / (first_norm * second_norm) if first_norm and second_norm else float("nan")


def spearman(first: list[float], second: list[float]) -> float:
    return pearson(average_ranks(first, higher_is_better=True), average_ranks(second, higher_is_better=True))


def kendall_tau_b(first: list[float], second: list[float]) -> float:
    concordant = discordant = ties_first = ties_second = 0
    for left in range(len(first)):
        for right in range(left + 1, len(first)):
            a = (first[left] > first[right]) - (first[left] < first[right])
            b = (second[left] > second[right]) - (second[left] < second[right])
            if a == 0 and b == 0:
                continue
            if a == 0:
                ties_first += 1
            elif b == 0:
                ties_second += 1
            elif a == b:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_first) * (concordant + discordant + ties_second))
    return (concordant - discordant) / denom if denom else float("nan")


def load_vsi_scores(path: Path) -> dict[str, float]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {str(row["Model"]): float(row["Avg."]) for row in rows}


def completed_row_is_compatible(row: dict[str, Any], candidate: Candidate, layer: int, protocol: dict[str, Any]) -> bool:
    return (
        row.get("status") == "complete"
        and row.get("architecture") == candidate.label
        and int(row.get("layer", -1)) == layer
        and row.get("protocol_sha256") == protocol["protocol_sha256"]
        and math.isfinite(float(row.get("logme", float("nan"))))
    )


def draw_figures(output: Path, scores: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    made: list[str] = []
    complete = [row for row in scores if row.get("status") == "complete"]
    figure, axis = plt.subplots(figsize=(7, 5))
    if complete:
        axis.scatter([row["vsi_score"] for row in complete], [row["logme_primary"] for row in complete])
        for row in complete:
            axis.annotate(row["architecture"], (row["vsi_score"], row["logme_primary"]), fontsize=7)
        axis.set_xlabel("Post-SFT VSI-Bench Avg.")
        axis.set_ylabel("Mean 10-layer LogME")
    else:
        axis.text(0.5, 0.5, "No candidate has all predeclared LogME layers", ha="center", va="center")
        axis.set_axis_off()
    figure.tight_layout(); figure.savefig(output / "logme_vs_vsibench.png", dpi=180); plt.close(figure)
    made.append("logme_vs_vsibench.png")
    figure, axis = plt.subplots(figsize=(10, 3.8))
    labels = [candidate.label for candidate in CANDIDATES]
    matrix = []
    for label in labels:
        matrix.append([float(next((row.get("logme", float("nan")) for row in rows if row["architecture"] == label and row["layer"] == layer and row["status"] == "complete"), float("nan"))) for layer in PRIMARY_LAYERS])
    image = axis.imshow(torch.tensor(matrix).numpy(), aspect="auto")
    figure.colorbar(image, ax=axis, label="LogME")
    axis.set_xticks(range(len(PRIMARY_LAYERS)), [str(layer) for layer in PRIMARY_LAYERS])
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Decoder layer"); axis.set_title("Pre-SFT LogME layer profile (NaN = unavailable cache)")
    figure.tight_layout(); figure.savefig(output / "logme_layer_heatmap.png", dpi=180); plt.close(figure)
    made.append("logme_layer_heatmap.png")
    figure, axis = plt.subplots(figsize=(7, 3.5))
    axis.text(0.5, 0.58, "Existing finalized pre-SFT depth-probe architecture correlation was not found.", ha="center", va="center", wrap=True)
    axis.text(0.5, 0.38, "LogME comparison is emitted only when the fixed ten-layer score is eligible.", ha="center", va="center", wrap=True)
    axis.set_axis_off(); figure.tight_layout(); figure.savefig(output / "proxy_correlation_comparison.png", dpi=180); plt.close(figure)
    made.append("proxy_correlation_comparison.png")
    return made


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-indices", type=Path, default=DEFAULT_SAMPLE_INDICES)
    parser.add_argument("--vsi-results", type=Path, default=REPO_ROOT / "VSI result.csv")
    parser.add_argument("--video-limit", type=int, default=None, help="Deterministic future video-level subset; default is all training videos.")
    parser.add_argument("--smoke-videos", type=int, default=1)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute completed compatible architecture/layer rows.")
    parser.add_argument("--cpu-threads", type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args()
    run_started = time.perf_counter()
    torch.set_num_threads(args.cpu_threads)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sample_indices = args.sample_indices.resolve()
    all_train_records = load_frame_records(sample_indices, split="train")
    records = select_videos(all_train_records, args.video_limit)
    video_count = len({str(record["video_sample_id"]) for record in records})
    if len(all_train_records) != 2012 or len({str(record["video_sample_id"]) for record in all_train_records}) != 1006:
        raise RuntimeError("The formal split must contain 1,006 training videos / 2,012 selected frames")
    protocol = {
        "schema_version": "pre_sft_logme_proxy_v1",
        "definition": "Bayesian linear regression maximum evidence, normalized log p(y|F)/N",
        "dtype": "float64", "alpha_init": 1.0, "beta_init": 1.0, "eps": EPS,
        "primary_layers": list(PRIMARY_LAYERS), "sample_indices": str(sample_indices),
        "sample_indices_sha256": sha256_file(sample_indices), "split": "train",
        "training_videos": video_count, "training_frames": len(records), "video_limit": args.video_limit,
        "no_vlm_forward": True, "no_optimizer": True,
    }
    protocol["protocol_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode("utf-8")).hexdigest()
    write_json(output / "protocol.json", protocol)
    provenances = {candidate.label: load_and_validate_provenance(candidate, sample_indices) for candidate in CANDIDATES}
    missing = {candidate.label: candidate_missing_layers(candidate) for candidate in CANDIDATES}
    smoke_candidate = next(candidate for candidate in CANDIDATES if 1 not in missing[candidate.label])
    smoke_records = select_videos(all_train_records, args.smoke_videos)
    smoke_started = time.perf_counter()
    smoke_stream = accumulate_statistics(smoke_candidate, 1, smoke_records, materialize=False)
    smoke_materialized = accumulate_statistics(smoke_candidate, 1, smoke_records, materialize=True)
    stream_result = logme_from_statistics(*smoke_stream[:4])
    materialized_result, materialized_count = materialized_logme(smoke_materialized[6] or [], smoke_materialized[7] or [])
    if smoke_stream[3] != materialized_count or smoke_stream[5] != smoke_materialized[5]:
        raise RuntimeError("Streaming/materialized LogME smoke used different target samples")
    difference = abs(float(stream_result["logme"]) - float(materialized_result["logme"]))
    if difference > 1e-10:
        raise RuntimeError(f"Streaming/materialized LogME disagreement: {difference:.3e}")
    smoke = {
        "status": "PASS", "architecture": smoke_candidate.label, "layer": 1,
        "videos": args.smoke_videos, "frames": smoke_stream[4], "valid_tokens": smoke_stream[3],
        "target_signature": smoke_stream[5], "streaming": stream_result,
        "materialized": materialized_result, "absolute_logme_difference": difference,
        "runtime_seconds": time.perf_counter() - smoke_started,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "provenance_path": str(provenance_path(smoke_candidate)),
    }
    write_json(output / "logme_smoke.json", smoke)
    print(f"[PASS] streaming/materialized smoke: N={smoke_stream[3]} LogME={stream_result['logme']:.12f} diff={difference:.3e}", flush=True)
    if args.smoke_only:
        return
    existing: list[dict[str, Any]] = []
    per_layer_path = output / "logme_per_layer.csv"
    if per_layer_path.is_file():
        with per_layer_path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    expected_signature: str | None = None
    for candidate in CANDIDATES:
        # The protocol forbids a partial-layer architecture score.  Avoid an
        # expensive seven-layer diagnostic when this candidate can never enter
        # the declared ten-layer comparison; the smoke above already proves
        # the numerical implementation on a valid cached representation.
        candidate_is_incomplete = bool(missing[candidate.label])
        for layer in PRIMARY_LAYERS:
            base = {
                "architecture": candidate.label, "display_name": candidate.display_name, "layer": layer,
                "cache_root": str(candidate.root), "cache_provenance": str(provenance_path(candidate)),
                "protocol_sha256": protocol["protocol_sha256"], "training_videos": video_count,
                "training_frames": len(records), "feature_level": f"layer_{layer}",
            }
            if candidate_is_incomplete:
                rows.append({
                    **base,
                    "status": "unavailable_incomplete_pre_sft_fixed_layer_set",
                    "reason": (
                        "Candidate lacks required pre-SFT layers "
                        + ",".join(f"layer_{value}" for value in missing[candidate.label])
                        + "; no partial LogME score was computed."
                    ),
                })
                continue
            prior = next((row for row in existing if completed_row_is_compatible(row, candidate, layer, protocol)), None)
            if prior is not None and not args.force:
                rows.append(prior)
                continue
            if layer in missing[candidate.label]:
                rows.append({**base, "status": "unavailable_missing_pre_sft_feature", "reason": f"Missing layer_{layer}; required fixed layer set cannot be completed from this pre-SFT cache."})
                continue
            started = time.perf_counter()
            gram, cross, yy, count, frames, target_signature, _, _ = accumulate_statistics(candidate, layer, records)
            if expected_signature is None:
                expected_signature = target_signature
            elif target_signature != expected_signature:
                raise RuntimeError(f"{candidate.label}/layer_{layer}: target-valid samples differ from the formal comparison set")
            result = logme_from_statistics(gram, cross, yy, count)
            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            rows.append({
                **base, "status": "complete", "logme": result["logme"], "alpha": result["alpha"], "beta": result["beta"],
                "iterations": result["iterations"], "converged": result["converged"], "gamma": result["gamma"],
                "residual_sq": result["residual_sq"], "minimum_eigenvalue": result["minimum_eigenvalue"],
                "valid_tokens": count, "feature_dimension": int(gram.shape[0]), "frames_loaded": frames,
                "target_signature": target_signature, "runtime_seconds": time.perf_counter() - started,
                "peak_process_rss_bytes": peak_rss,
            })
            write_csv(per_layer_path, rows)
            del gram, cross, yy
            gc.collect()
    write_csv(per_layer_path, rows)
    vsi = load_vsi_scores(args.vsi_results.resolve())
    score_rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        candidate_rows = [row for row in rows if row["architecture"] == candidate.label]
        complete = [row for row in candidate_rows if row.get("status") == "complete"]
        missing_layers = missing[candidate.label]
        score_rows.append({
            "architecture": candidate.label, "display_name": candidate.display_name,
            "status": "complete" if len(complete) == len(PRIMARY_LAYERS) else "unavailable_incomplete_fixed_layer_set",
            "logme_primary": sum(float(row["logme"]) for row in complete) / len(complete) if len(complete) == len(PRIMARY_LAYERS) else None,
            "available_layer_count": len(PRIMARY_LAYERS) - len(missing_layers), "required_layer_count": len(PRIMARY_LAYERS),
            "missing_layers": ",".join(map(str, missing_layers)), "vsi_score": vsi[candidate.vsi_name],
        })
    completed_scores = [row for row in score_rows if row["status"] == "complete"]
    if completed_scores:
        ranked = average_ranks([float(row["logme_primary"]) for row in completed_scores], higher_is_better=True)
        vsi_ranked = average_ranks([float(row["vsi_score"]) for row in completed_scores], higher_is_better=True)
        for row, rank, vsi_rank in zip(completed_scores, ranked, vsi_ranked):
            row["logme_rank"] = rank; row["vsi_rank"] = vsi_rank
        correlations = {
            "spearman_rho": spearman([float(row["logme_primary"]) for row in completed_scores], [float(row["vsi_score"]) for row in completed_scores]),
            "kendall_tau": kendall_tau_b([float(row["logme_primary"]) for row in completed_scores], [float(row["vsi_score"]) for row in completed_scores]),
        }
    else:
        correlations = {"spearman_rho": None, "kendall_tau": None}
    layer_correlations: dict[str, float | None] = {}
    for layer in PRIMARY_LAYERS:
        items = [row for row in rows if row["layer"] == layer and row.get("status") == "complete"]
        layer_correlations[str(layer)] = spearman([float(row["logme"]) for row in items], [vsi[next(candidate.vsi_name for candidate in CANDIDATES if candidate.label == row["architecture"])] for row in items]) if len(items) >= 2 else None
    write_csv(output / "logme_architecture_scores.csv", score_rows)
    figures = draw_figures(output, score_rows, rows)
    summary = {
        "protocol": protocol, "completed_at": datetime.now(timezone.utc).isoformat(), "smoke": smoke,
        "correlations": correlations, "per_layer_spearman": layer_correlations,
        "architectures": score_rows, "unavailable": {label: layers for label, layers in missing.items() if layers},
        "figures": figures, "existing_depth_probe_correlation": None,
        "existing_depth_probe_note": "No finalized pre-SFT architecture-level depth-probe correlation artifact was found; depth-probe results were not modified.",
        "runtime_memory": {
            "overall_wall_seconds": time.perf_counter() - run_started,
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "completed_architecture_layer_runs": sum(row.get("status") == "complete" for row in rows),
        },
    }
    write_json(output / "logme_summary.json", summary)
    lines = [
        "# Pre-SFT regression LogME architecture proxy", "",
        "This cache-only run uses the existing camera-Z depth tensors, validity masks, and token alignment from the formal depth probe. It uses float64 streaming sufficient statistics and never loads a VLM or post-SFT checkpoint.", "",
        "## Result", "",
        "The declared primary score is the unweighted mean over layers 1, 3, 6, 9, 12, 15, 18, 21, 24, and 27. All formal C1 caches are missing layers 12, 18, and 24, so no primary LogME score or VSI correlation is reported. No post-SFT cache was substituted.", "",
        "## Cache inventory", "",
        "| Architecture | Available fixed layers | Missing fixed layers | VSI Avg. | Status |", "|---|---:|---|---:|---|",
    ]
    for row in score_rows:
        lines.append(f"| {row['display_name']} | {row['available_layer_count']}/10 | {row['missing_layers']} | {row['vsi_score']:.1f} | {row['status']} |")
    lines.extend(["", "## Correlation", "", "| Proxy | Spearman vs VSI | Kendall vs VSI |", "|---|---:|---:|", "| Existing pre-SFT depth probe | unavailable finalized architecture score | unavailable |", "| LogME | unavailable: incomplete fixed layer set | unavailable |", ""])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] eligible_architectures={len(completed_scores)}/{len(CANDIDATES)} Spearman={correlations['spearman_rho']} output={output}", flush=True)


if __name__ == "__main__":
    main()
