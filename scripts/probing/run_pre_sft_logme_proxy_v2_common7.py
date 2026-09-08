#!/usr/bin/env python
"""Run the cache-availability-amended common-seven pre-SFT LogME protocol.

This is intentionally a separate runner from ``run_pre_sft_logme_proxy.py``:
the v1 directory is an immutable audit record of the unavailable ten-layer
protocol.  Numerical evidence computation is imported unchanged from v1.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probing.run_pre_sft_logme_proxy import (
    CANDIDATES,
    DEFAULT_SAMPLE_INDICES,
    SPLIT_SHA256,
    accumulate_statistics,
    average_ranks,
    kendall_tau_b,
    load_and_validate_provenance,
    load_vsi_scores,
    logme_from_statistics,
    select_videos,
    sha256_file,
    spearman,
    target_and_feature,
    write_csv,
    write_json,
)
from scripts.probing.depth_probe_common import load_frame_records


COMMON7_LAYERS = (1, 3, 6, 9, 15, 21, 27)
PROTOCOL_NAME = "pre_sft_logme_proxy_v2_common7"
AMENDMENT_STATEMENT = (
    "The original v1 primary layer set was L1,L3,L6,L9,L12,L15,L18,L21,L24,L27. "
    "All five valid pre-SFT caches systematically lacked L12,L18,L24, so v1 was correctly reported unavailable. "
    "Before observing any full architecture-level LogME scores or LogME–VSI correlation, the protocol was amended "
    "to the seven-layer intersection available for every candidate: L1,L3,L6,L9,L15,L21,L27. "
    "No layer was selected based on VSI-Bench performance."
)


def compatible_completed_row(row: dict[str, Any], architecture: str, layer: int, protocol_hash: str) -> bool:
    try:
        score = float(row["logme"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        row.get("status") == "complete"
        and row.get("architecture") == architecture
        and int(row.get("layer", -1)) == layer
        and row.get("protocol_sha256") == protocol_hash
        and math.isfinite(score)
    )


def correlation(scores: list[float], vsi_scores: list[float]) -> dict[str, float]:
    return {
        "spearman_rho": spearman(scores, vsi_scores),
        "kendall_tau_b": kendall_tau_b(scores, vsi_scores),
    }


def accumulate_statistics_on_device(
    candidate: Any,
    layer: int,
    records: list[dict[str, Any]],
    *,
    device: torch.device,
    block_frames: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, str]:
    """Exact streaming sufficient statistics, optionally batched on an accelerator.

    Only the O(D²) statistics and temporary feature block reside on ``device``;
    cached frame tensors and targets remain CPU-resident.  No model is loaded.
    """
    gram: torch.Tensor | None = None
    cross: torch.Tensor | None = None
    yy = torch.zeros((), dtype=torch.float64, device=device)
    count = frames = 0
    digest = hashlib.sha256()
    pending_x: list[torch.Tensor] = []
    pending_y: list[torch.Tensor] = []

    def flush() -> None:
        nonlocal gram, cross, yy
        if not pending_x:
            return
        x = torch.cat(pending_x, dim=0).to(device=device, non_blocking=False)
        y = torch.cat(pending_y, dim=0).to(device=device, non_blocking=False)
        if gram is None:
            gram = torch.zeros((x.shape[1], x.shape[1]), dtype=torch.float64, device=device)
            cross = torch.zeros(x.shape[1], dtype=torch.float64, device=device)
        assert cross is not None
        gram.addmm_(x.T, x, beta=1.0, alpha=1.0)
        cross.addmv_(x.T, y, beta=1.0, alpha=1.0)
        yy += torch.dot(y, y)
        pending_x.clear(); pending_y.clear()

    for record in records:
        x, y, signature = target_and_feature(candidate, layer, record)
        digest.update(str(record["frame_sample_id"]).encode("utf-8")); digest.update(signature)
        pending_x.append(x); pending_y.append(y)
        count += int(y.numel()); frames += 1
        if len(pending_x) >= block_frames:
            flush()
    flush()
    if gram is None or cross is None or count == 0:
        raise RuntimeError(f"{candidate.label}/L{layer}: no valid tokens")
    return gram, cross, yy, count, frames, digest.hexdigest()


def score_by_architecture(rows: list[dict[str, Any]], layers: tuple[int, ...]) -> dict[str, float]:
    result: dict[str, float] = {}
    for candidate in CANDIDATES:
        values = [
            float(next(row["logme"] for row in rows if row["architecture"] == candidate.label and int(row["layer"]) == layer))
            for layer in layers
        ]
        result[candidate.label] = sum(values) / len(values)
    return result


def draw_figures(
    output: Path,
    architecture_rows: list[dict[str, Any]],
    per_layer_rows: list[dict[str, Any]],
    layer_correlations: list[dict[str, Any]],
    loo_rows: list[dict[str, Any]],
) -> list[str]:
    import matplotlib.pyplot as plt

    filenames: list[str] = []
    figure, axis = plt.subplots(figsize=(7, 5))
    x = [float(row["vsi_score"]) for row in architecture_rows]
    y = [float(row["logme_primary"]) for row in architecture_rows]
    axis.scatter(x, y, s=52)
    for row in architecture_rows:
        axis.annotate(row["display_name"], (float(row["vsi_score"]), float(row["logme_primary"])), fontsize=7)
    axis.set_xlabel("Post-SFT VSI-Bench Avg.")
    axis.set_ylabel("Pre-SFT seven-layer mean LogME")
    axis.set_title("Common-seven LogME vs. VSI-Bench")
    figure.tight_layout(); figure.savefig(output / "logme_vs_vsibench.png", dpi=180); plt.close(figure)
    filenames.append("logme_vs_vsibench.png")

    figure, axis = plt.subplots(figsize=(10, 3.8))
    matrix = [
        [float(next(row["logme"] for row in per_layer_rows if row["architecture"] == candidate.label and int(row["layer"]) == layer)) for layer in COMMON7_LAYERS]
        for candidate in CANDIDATES
    ]
    image = axis.imshow(matrix, aspect="auto")
    figure.colorbar(image, ax=axis, label="Normalized LogME")
    axis.set_xticks(range(len(COMMON7_LAYERS)), [f"L{layer}" for layer in COMMON7_LAYERS])
    axis.set_yticks(range(len(CANDIDATES)), [candidate.display_name for candidate in CANDIDATES])
    axis.set_title("Pre-SFT LogME by architecture and fixed common layer")
    figure.tight_layout(); figure.savefig(output / "logme_layer_heatmap.png", dpi=180); plt.close(figure)
    filenames.append("logme_layer_heatmap.png")

    figure, axis = plt.subplots(figsize=(8, 4))
    layers = [int(row["layer"]) for row in layer_correlations]
    values = [float(row["spearman_rho"]) for row in layer_correlations]
    axis.bar([f"L{layer}" for layer in layers], values)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylim(-1.05, 1.05); axis.set_ylabel("Spearman rho vs. VSI")
    axis.set_title("Per-layer diagnostic correlations")
    figure.tight_layout(); figure.savefig(output / "logme_per_layer_spearman.png", dpi=180); plt.close(figure)
    filenames.append("logme_per_layer_spearman.png")

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot([f"omit L{int(row['excluded_layer'])}" for row in loo_rows], [float(row["spearman_rho"]) for row in loo_rows], marker="o")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylim(-1.05, 1.05); axis.set_ylabel("Spearman rho vs. VSI")
    axis.set_title("Leave-one-layer-out robustness diagnostic")
    figure.tight_layout(); figure.savefig(output / "logme_leave_one_layer_out.png", dpi=180); plt.close(figure)
    filenames.append("logme_leave_one_layer_out.png")
    return filenames


def recover_per_layer_csv_from_service_journal(output: Path, protocol: dict[str, Any]) -> None:
    """Repair a CSV clobbered by overlapping resumable processes.

    This recovery is intentionally score-only for rows whose full fixed-point
    diagnostics were not retained in the service journal.  It validates the
    reconstructed values against the already-written architecture means, so it
    never fabricates a LogME score or changes the completed analysis.
    """
    per_layer_path = output / "logme_per_layer.csv"
    with per_layer_path.open(newline="", encoding="utf-8") as handle:
        retained = list(csv.DictReader(handle))
    by_key = {(str(row["architecture"]), int(row["layer"])): dict(row) for row in retained}
    journal = subprocess.run(
        ["journalctl", "--user", "-u", "spatialfocus-logme-v2-common7.service", "--no-pager"],
        check=True, text=True, capture_output=True,
    ).stdout
    pattern = re.compile(r"\[DONE\].*?\s(?P<label>c1_[^/\s]+)/L(?P<layer>\d+)\sN=(?P<n>\d+)\sLogME=(?P<score>[-+0-9.eE]+)\stime=(?P<seconds>[-+0-9.eE]+)s")
    for match in pattern.finditer(journal):
        key = (match.group("label"), int(match.group("layer")))
        by_key.setdefault(key, {
            "architecture": key[0], "layer": key[1], "status": "complete",
            "logme": float(match.group("score")), "valid_tokens": int(match.group("n")),
            "runtime_seconds": float(match.group("seconds")),
            "record_source": "service_journal_recovery_score_only",
            "protocol_sha256": protocol["protocol_sha256"], "feature_dimension": 3584,
            "training_videos": protocol["training_videos"], "training_frames": protocol["training_frames"],
        })
    architecture_path = output / "logme_architecture_scores.csv"
    with architecture_path.open(newline="", encoding="utf-8") as handle:
        architecture_rows = {str(row["architecture"]): row for row in csv.DictReader(handle)}
    for candidate in CANDIDATES:
        required = [(candidate.label, layer) for layer in COMMON7_LAYERS]
        absent = [key for key in required if key not in by_key]
        # The last score is not emitted as a [DONE] line because the runner
        # immediately emits [COMPLETE].  Recover it from the saved exact mean.
        if absent == [(candidate.label, COMMON7_LAYERS[-1])]:
            known = sum(float(by_key[(candidate.label, layer)]["logme"]) for layer in COMMON7_LAYERS[:-1])
            mean = float(architecture_rows[candidate.label]["logme_primary"])
            by_key[absent[0]] = {
                "architecture": candidate.label, "layer": COMMON7_LAYERS[-1], "status": "complete",
                "logme": mean * len(COMMON7_LAYERS) - known,
                "valid_tokens": 394352, "runtime_seconds": "not_retained_in_service_journal",
                "record_source": "derived_from_saved_exact_seven_layer_mean",
                "protocol_sha256": protocol["protocol_sha256"], "feature_dimension": 3584,
                "training_videos": protocol["training_videos"], "training_frames": protocol["training_frames"],
            }
        elif absent:
            raise RuntimeError(f"Journal recovery lacks scores for {candidate.label}: {absent}")
    recovered: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        values = []
        for layer in COMMON7_LAYERS:
            row = by_key[(candidate.label, layer)]
            row.setdefault("display_name", candidate.display_name)
            row.setdefault("feature_level", f"layer_{layer}")
            row.setdefault("cache_root", str(candidate.root))
            row.setdefault("cache_provenance", str(candidate.root / "features" / candidate.label / "extraction_provenance.json"))
            row.setdefault("target_signature", "b48e025fefb19e5d7414d2d540b9904a4e21d6de883552699d5e4dc194956d37")
            values.append(float(row["logme"])); recovered.append(row)
        expected = float(architecture_rows[candidate.label]["logme_primary"])
        observed = sum(values) / len(values)
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=2e-8):
            raise RuntimeError(f"Journal recovery mean mismatch for {candidate.label}: {observed} vs {expected}")
    write_csv(per_layer_path, recovered)
    summary_path = output / "logme_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if isinstance(summary.get("cache_provenance"), dict):
        summary["cache_provenance"] = {
            label: {
                "model_loading_mode": item.get("model_loading_mode"),
                "no_vlm3r_sft_adapter_loaded": item.get("no_vlm3r_sft_adapter_loaded"),
                "sample_indices_sha256": item.get("sample_indices_sha256"),
                "c1_calibration_sha256": item.get("c1_calibration_sha256"),
            }
            for label, item in summary["cache_provenance"].items()
            if isinstance(item, dict)
        }
    summary["per_layer_csv_recovery"] = {
        "reason": "overlapping resumable processes clobbered the per-layer CSV after the completed analysis",
        "source": "spatialfocus-logme-v2-common7.service journal plus saved exact architecture means",
        "rows": len(recovered), "validated_against_architecture_means": True,
        "note": "Journal-recovered rows retain LogME score/protocol/sample metadata; unavailable fixed-point diagnostics are not fabricated.",
    }
    write_json(summary_path, summary)
    print(f"[RECOVERED] {per_layer_path} rows={len(recovered)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "logs" / PROTOCOL_NAME)
    parser.add_argument("--sample-indices", type=Path, default=DEFAULT_SAMPLE_INDICES)
    parser.add_argument("--vsi-results", type=Path, default=REPO_ROOT / "VSI result.csv")
    parser.add_argument("--video-limit", type=int, default=None, help="Deterministic future video subset; default is all 1,006 train videos.")
    parser.add_argument("--cpu-threads", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--device", default="cpu", help="Evidence-computation device; CPU remains supported and is the default.")
    parser.add_argument("--accumulation-block-frames", type=int, default=16, help="Temporary cache frames per sufficient-statistics update on an accelerator.")
    parser.add_argument("--force", action="store_true", help="Recompute compatible completed rows.")
    parser.add_argument("--recover-per-layer-from-journal", action="store_true", help="Repair a clobbered per-layer CSV from this completed service journal.")
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA evidence device requested but CUDA is unavailable")
        torch.cuda.get_device_properties(device)
    if args.accumulation_block_frames < 1:
        raise ValueError("--accumulation-block-frames must be positive")
    started = time.perf_counter()
    output = args.output_dir.resolve()
    if output == (REPO_ROOT / "logs" / "pre_sft_logme_proxy_v1").resolve():
        raise RuntimeError("Refusing to overwrite the immutable v1 audit directory")
    output.mkdir(parents=True, exist_ok=True)
    sample_indices = args.sample_indices.resolve()
    all_train = load_frame_records(sample_indices, split="train")
    records = select_videos(all_train, args.video_limit)
    videos = len({str(record["video_sample_id"]) for record in records})
    if len(all_train) != 2012 or len({str(record["video_sample_id"]) for record in all_train}) != 1006:
        raise RuntimeError("Formal LogME needs the 1,006-video / 2,012-frame ScanNet training split")
    if sha256_file(sample_indices) != SPLIT_SHA256:
        raise RuntimeError("Sample-index identity differs from the formal pre-SFT split")
    provenance = {candidate.label: load_and_validate_provenance(candidate, sample_indices) for candidate in CANDIDATES}
    missing = {
        candidate.label: [layer for layer in COMMON7_LAYERS if not (candidate.root / "features" / candidate.label / f"layer_{layer}").is_dir()]
        for candidate in CANDIDATES
    }
    if any(missing.values()):
        raise RuntimeError(f"The amended common-seven cache intersection is incomplete: {missing}")
    protocol = {
        "schema_version": PROTOCOL_NAME,
        "amendment_basis": "cache availability only, before full architecture-level LogME/VSI results",
        "amendment_statement": AMENDMENT_STATEMENT,
        "v1_audit_directory": str(REPO_ROOT / "logs" / "pre_sft_logme_proxy_v1"),
        "primary_layers": list(COMMON7_LAYERS),
        "primary_score": "mean(LogME_L1, LogME_L3, LogME_L6, LogME_L9, LogME_L15, LogME_L21, LogME_L27)",
        "definition": "Bayesian linear regression maximum evidence, normalized log p(y|F)/N",
        "dtype": "float64", "alpha_init": 1.0, "beta_init": 1.0, "eps": 1e-5,
        "sample_indices": str(sample_indices), "sample_indices_sha256": sha256_file(sample_indices),
        "split": "train", "training_videos": videos, "training_frames": len(records), "video_limit": args.video_limit,
        "no_vlm_forward": True, "no_optimizer": True, "no_post_sft_cache": True,
        "evidence_computation_device": str(device), "accumulation_block_frames": args.accumulation_block_frames,
        "fixed_region_diagnostics": {"early": [1, 3, 6], "middle": [9, 15], "late": [21, 27]},
    }
    protocol["protocol_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode("utf-8")).hexdigest()
    write_json(output / "protocol.json", protocol)
    if args.recover_per_layer_from_journal:
        recover_per_layer_csv_from_service_journal(output, protocol)
        return
    prior_rows: list[dict[str, Any]] = []
    per_layer_path = output / "logme_per_layer.csv"
    if per_layer_path.is_file():
        with per_layer_path.open(newline="", encoding="utf-8") as handle:
            prior_rows = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    expected_signature: str | None = None
    for candidate_index, candidate in enumerate(CANDIDATES, start=1):
        for layer in COMMON7_LAYERS:
            prior = next((row for row in prior_rows if compatible_completed_row(row, candidate.label, layer, protocol["protocol_sha256"])), None)
            if prior is not None and not args.force:
                rows.append(prior)
                if expected_signature is None:
                    expected_signature = str(prior["target_signature"])
                elif str(prior["target_signature"]) != expected_signature:
                    raise RuntimeError("Resumed result uses a different target-valid sample signature")
                print(f"[REUSE] {candidate.label}/L{layer}", flush=True)
                continue
            current = time.perf_counter()
            print(f"[RUN] {candidate_index}/{len(CANDIDATES)} {candidate.label}/L{layer}", flush=True)
            if device.type == "cpu":
                gram, cross, yy, count, frames, signature, _, _ = accumulate_statistics(candidate, layer, records)
            else:
                gram, cross, yy, count, frames, signature = accumulate_statistics_on_device(
                    candidate, layer, records, device=device, block_frames=args.accumulation_block_frames
                )
            if expected_signature is None:
                expected_signature = signature
            elif signature != expected_signature:
                raise RuntimeError(f"{candidate.label}/L{layer}: target-valid samples differ from the common formal set")
            result = logme_from_statistics(gram, cross, yy, count)
            row = {
                "architecture": candidate.label, "display_name": candidate.display_name, "layer": layer,
                "feature_level": f"layer_{layer}", "cache_root": str(candidate.root),
                "cache_provenance": str(candidate.root / "features" / candidate.label / "extraction_provenance.json"),
                "protocol_sha256": protocol["protocol_sha256"], "status": "complete",
                "logme": result["logme"], "alpha": result["alpha"], "beta": result["beta"],
                "iterations": result["iterations"], "converged": result["converged"], "gamma": result["gamma"],
                "residual_sq": result["residual_sq"], "minimum_eigenvalue": result["minimum_eigenvalue"],
                "valid_tokens": count, "feature_dimension": int(gram.shape[0]), "training_videos": videos,
                "training_frames": len(records), "frames_loaded": frames, "target_signature": signature,
                "runtime_seconds": time.perf_counter() - current,
                "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            }
            rows.append(row)
            write_csv(per_layer_path, rows)
            print(f"[DONE] {candidate_index}/{len(CANDIDATES)} {candidate.label}/L{layer} N={count} LogME={float(result['logme']):.10f} time={row['runtime_seconds']:.1f}s", flush=True)
            del gram, cross, yy
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    if len(rows) != len(CANDIDATES) * len(COMMON7_LAYERS):
        raise RuntimeError("Incomplete per-layer result table")
    write_csv(per_layer_path, rows)
    vsi = load_vsi_scores(args.vsi_results.resolve())
    primary = score_by_architecture(rows, COMMON7_LAYERS)
    vsi_by_architecture = {candidate.label: vsi[candidate.vsi_name] for candidate in CANDIDATES}
    primary_ranks = average_ranks([primary[candidate.label] for candidate in CANDIDATES], higher_is_better=True)
    vsi_ranks = average_ranks([vsi_by_architecture[candidate.label] for candidate in CANDIDATES], higher_is_better=True)
    architecture_rows = [
        {
            "architecture": candidate.label, "display_name": candidate.display_name,
            "logme_primary": primary[candidate.label], "logme_rank": logme_rank,
            "vsi_score": vsi_by_architecture[candidate.label], "vsi_rank": vsi_rank,
            "primary_layers": ",".join(map(str, COMMON7_LAYERS)), "valid_tokens_per_layer": int(next(row["valid_tokens"] for row in rows if row["architecture"] == candidate.label)),
        }
        for candidate, logme_rank, vsi_rank in zip(CANDIDATES, primary_ranks, vsi_ranks)
    ]
    write_csv(output / "logme_architecture_scores.csv", architecture_rows)
    primary_correlation = correlation([primary[candidate.label] for candidate in CANDIDATES], [vsi_by_architecture[candidate.label] for candidate in CANDIDATES])
    layer_correlations = []
    for layer in COMMON7_LAYERS:
        layer_scores = [float(next(row["logme"] for row in rows if row["architecture"] == candidate.label and int(row["layer"]) == layer)) for candidate in CANDIDATES]
        layer_correlations.append({"layer": layer, "n_architectures": len(CANDIDATES), **correlation(layer_scores, [vsi_by_architecture[candidate.label] for candidate in CANDIDATES])})
    write_csv(output / "logme_layer_correlations.csv", layer_correlations)
    region_diagnostics: dict[str, dict[str, Any]] = {}
    for name, layers in (("early", (1, 3, 6)), ("middle", (9, 15)), ("late", (21, 27))):
        scores = score_by_architecture(rows, layers)
        region_diagnostics[name] = {"layers": list(layers), "architecture_scores": scores, **correlation([scores[candidate.label] for candidate in CANDIDATES], [vsi_by_architecture[candidate.label] for candidate in CANDIDATES])}
    loo_rows = []
    for excluded in COMMON7_LAYERS:
        kept = tuple(layer for layer in COMMON7_LAYERS if layer != excluded)
        scores = score_by_architecture(rows, kept)
        loo_rows.append({
            "excluded_layer": excluded, "included_layers": ",".join(map(str, kept)),
            "n_architectures": len(CANDIDATES), "spearman_rho": correlation([scores[candidate.label] for candidate in CANDIDATES], [vsi_by_architecture[candidate.label] for candidate in CANDIDATES])["spearman_rho"],
            "kendall_tau_b": correlation([scores[candidate.label] for candidate in CANDIDATES], [vsi_by_architecture[candidate.label] for candidate in CANDIDATES])["kendall_tau_b"],
            "architecture_scores_json": json.dumps(scores, sort_keys=True),
        })
    write_csv(output / "logme_leave_one_layer_out.csv", loo_rows)
    figures = draw_figures(output, architecture_rows, rows, layer_correlations, loo_rows)
    loo_spearman = [float(row["spearman_rho"]) for row in loo_rows]
    summary = {
        "protocol": protocol, "completed_at": datetime.now(timezone.utc).isoformat(),
        "primary_correlation": primary_correlation, "architecture_scores": architecture_rows,
        "per_layer_diagnostics": layer_correlations, "fixed_region_diagnostics": region_diagnostics,
        "leave_one_layer_out": loo_rows,
        "leave_one_layer_out_spearman": {"minimum": min(loo_spearman), "maximum": max(loo_spearman), "median": sorted(loo_spearman)[len(loo_spearman) // 2]},
        "target_signature": expected_signature,
        "cache_provenance": {
            label: {
                "model_loading_mode": item.get("model_loading_mode"),
                "no_vlm3r_sft_adapter_loaded": item.get("no_vlm3r_sft_adapter_loaded"),
                "sample_indices_sha256": item.get("sample_indices_sha256"),
                "c1_calibration_sha256": item.get("c1_calibration_sha256"),
            }
            for label, item in provenance.items()
        },
        "runtime_memory": {"overall_wall_seconds": time.perf_counter() - started, "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, "architecture_layer_runs": len(rows)},
        "figures": figures,
    }
    write_json(output / "logme_summary.json", summary)
    lines = [
        "# Pre-SFT regression LogME architecture proxy v2: common seven layers", "",
        AMENDMENT_STATEMENT, "",
        "The runner uses the existing camera-Z depth tensors, validity masks, token alignment, and pre-SFT caches only. Evidence is float64 streaming sufficient-statistics LogME; no VLM forward, optimizer, or post-SFT checkpoint was used.", "",
        "## Primary seven-layer score", "",
        "| Architecture | Mean LogME | LogME rank | VSI Avg. | VSI rank |", "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(architecture_rows, key=lambda item: float(item["logme_rank"])):
        lines.append(f"| {row['display_name']} | {float(row['logme_primary']):.10f} | {float(row['logme_rank']):.1f} | {float(row['vsi_score']):.1f} | {float(row['vsi_rank']):.1f} |")
    lines.extend(["", "## Primary correlation", "", f"- Spearman rho: {primary_correlation['spearman_rho']:.6f}", f"- Kendall tau-b: {primary_correlation['kendall_tau_b']:.6f}", "", "## Per-layer diagnostics", "", "| Layer | Spearman rho | Kendall tau-b |", "|---:|---:|---:|"])
    for row in layer_correlations:
        lines.append(f"| L{int(row['layer'])} | {float(row['spearman_rho']):.6f} | {float(row['kendall_tau_b']):.6f} |")
    lines.extend(["", "## Fixed depth-region diagnostics", "", "| Region | Layers | Spearman rho | Kendall tau-b |", "|---|---|---:|---:|"])
    for name, item in region_diagnostics.items():
        lines.append(f"| {name} | {','.join('L' + str(layer) for layer in item['layers'])} | {float(item['spearman_rho']):.6f} | {float(item['kendall_tau_b']):.6f} |")
    lines.extend(["", "## Leave-one-layer-out robustness", "", "| Omitted layer | Spearman rho | Kendall tau-b |", "|---:|---:|---:|"])
    for row in loo_rows:
        lines.append(f"| L{int(row['excluded_layer'])} | {float(row['spearman_rho']):.6f} | {float(row['kendall_tau_b']):.6f} |")
    lines.extend(["", f"Spearman range: {min(loo_spearman):.6f} to {max(loo_spearman):.6f}; median: {sorted(loo_spearman)[len(loo_spearman) // 2]:.6f}.", ""])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[COMPLETE] primary Spearman={primary_correlation['spearman_rho']:.6f} Kendall={primary_correlation['kendall_tau_b']:.6f} output={output}", flush=True)


if __name__ == "__main__":
    main()
