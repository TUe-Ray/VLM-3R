#!/usr/bin/env python
"""One-A100 primary pre-SFT trainable-scope proxy for the audited C1 roster.

This is deliberately narrower than the historical fusion-only evaluator.  It
scores exactly fresh LoRA + the candidate C1 fusion module + mm_projector,
using a single ordinary supervised QA CE backward and no optimizer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PROBING_DIR = REPO_ROOT / "scripts" / "probing"
if str(PROBING_DIR) not in sys.path:
    sys.path.insert(0, str(PROBING_DIR))

import evaluate_pre_sft_zero_cost_proxies as common  # noqa: E402
from depth_probe_common import write_json  # noqa: E402
from extract_depth_probe_features import build_dataset  # noqa: E402
from local_depth_probe_cache import install_forward_frame_loader  # noqa: E402
from scripts.diagnose_layerwise_spatial_hidden_scan import load_model  # noqa: E402
from llava.model.c1_structured_isometry import apply_c1_calibration_artifact  # noqa: E402


SCHEMA_VERSION = "pre_sft_trainable_proxy_a100_v1"
BASELINE_ID = "c1_vlm3r_native"
ALL_IDS = tuple(item.identifier for item in common.CANDIDATES)
EXPECTED_BASELINE_COUNTS = {
    "lora": 322_961_408,
    "fusion_block": 9_747_456,
    "mm_projector": 16_980_992,
    "total": 349_689_856,
}
EXPECTED_C1_SHA256 = {
    "c1_ss_add_012": "29a54005d90ea083d14418dca67e41aaf95947f2575c14910287ce1cf2fc80dc",
    "c1_ss_add_036": "f51409a74fd0735e9b782ccd7fe67da8265b8d80ca55f69035a090145a7e42d9",
    "c1_ss_add_123": "8ff199bfb49cd4fbcfae0a49fd48b4ef3c4b6a36796b25833a38fb8c18b2a150",
    "c1_ss_cross_attn_012": "d1080b6f8a9f0b983aae36867ed560c84fc02c81a321a7aa8158fdbbc675a520",
    "c1_vlm3r_native": "edb6ab3c255d0875e37cf6a18de078511fcfb61a37e3b102541e3f6219548f9c",
}
C1_FILENAMES = {
    "c1_ss_add_012": ("c1_additive_v1", "spatialstack_add.json"),
    "c1_ss_add_036": ("c1_ss_add_036", "spatialstack_add.json"),
    "c1_ss_add_123": ("c1_ss_add_123", "spatialstack_add.json"),
    "c1_ss_cross_attn_012": ("c1_ss_cross_attn_v1", "spatialstack_cross_attn_v1.json"),
    "c1_vlm3r_native": ("c1_vlm3r_v1", "vlm3r.json"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, (Path, torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def candidate_specs(c1_root: Path) -> dict[str, common.CandidateSpec]:
    output: dict[str, common.CandidateSpec] = {}
    for candidate in common.CANDIDATES:
        unit, filename = C1_FILENAMES[candidate.identifier]
        output[candidate.identifier] = replace(
            candidate, calibration_artifact=c1_root / unit / "official" / filename
        )
    return output


def parse_ids(value: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(values).difference(ALL_IDS))
    if unknown:
        raise ValueError(f"Unknown candidate IDs: {unknown}; expected {list(ALL_IDS)}")
    if len(values) != len(set(values)):
        raise ValueError("Candidate IDs must not be repeated")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "smoke", "formal"), required=True)
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--siglip-model", type=Path, required=True)
    parser.add_argument("--forward-frames-root", type=Path, required=True)
    parser.add_argument("--probe-targets-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--sample-indices", type=Path, required=True)
    parser.add_argument("--c1-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    # ``make_load_args`` is shared with the historical evaluator and reads
    # this legacy placement field before ``run_candidate`` asserts the direct
    # one-A100 policy.  It is deliberately fixed to the validated placement;
    # this runner never accepts the historical auto/CPU-offload modes.
    parser.add_argument("--device-map", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--dtype", choices=("float16",), default="float16")
    parser.add_argument("--attn-implementation", default=None)
    # Retained solely because the shared load-argument constructor carries
    # the historical TITAN-V placement fields.  This A100 runner forces
    # ``device_map=cuda:0`` and never reads these values for dispatch.
    parser.add_argument("--pre-sft-gpu-weight-budget", default=None)
    parser.add_argument("--pre-sft-gpu-weight-budgets", default=None)
    parser.add_argument("--pre-sft-cpu-offload-budget", default=None)
    # ``load_calibration_records`` is also shared with the historical
    # evaluator.  The Baseline protocol fixes this at one minibatch.
    parser.add_argument("--calibration-batches", type=int, choices=(1,), default=1)
    parser.add_argument("--rng-seed", type=int, default=42)
    args = parser.parse_args()
    if args.candidates is None:
        args.candidates = BASELINE_ID if args.mode == "smoke" else ",".join(ALL_IDS)
    args.candidate_ids = parse_ids(args.candidates)
    if args.mode == "smoke" and args.candidate_ids != [BASELINE_ID]:
        raise ValueError("The migration smoke must be exactly the C1 VLM3R Baseline")
    if args.mode == "formal" and set(args.candidate_ids) != set(ALL_IDS):
        raise ValueError("The formal run must contain exactly the audited five-candidate roster")
    return args


def required_sidecars(candidate: common.CandidateSpec, feature_root: Path) -> list[Path]:
    scene = "scene0384_00.pt"
    if candidate.fusion_variant == "c1_vlm3r":
        return [feature_root / "scannet" / "spatial_features" / scene]
    return [
        feature_root / "scannet" / "spatial_features_dec_6" / scene,
        feature_root / "scannet" / "spatial_features_dec_9" / scene,
        feature_root / "scannet" / "spatial_features" / scene,
    ]


def validate(args: argparse.Namespace, specs: dict[str, common.CandidateSpec]) -> dict[str, Any]:
    required = (
        args.base_model / "config.json", args.siglip_model / "config.json", args.forward_frames_root,
        args.probe_targets_root, args.feature_root, args.data_yaml, args.sample_indices,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required proxy inputs: {missing}")
    forbidden = [
        path for name in ("adapter_config.json", "adapter_model.bin", "non_lora_trainables.bin")
        for path in args.base_model.rglob(name)
    ]
    if forbidden:
        raise RuntimeError(f"Base model contains forbidden adapter/checkpoint artifacts: {[str(x) for x in forbidden]}")
    artifacts: dict[str, dict[str, str]] = {}
    for identifier in args.candidate_ids:
        candidate = specs[identifier]
        paths = [candidate.calibration_artifact, *required_sidecars(candidate, args.feature_root)]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{identifier} is missing C1/sidecar inputs: {missing}")
        actual = sha256(candidate.calibration_artifact)
        if actual != EXPECTED_C1_SHA256[identifier]:
            raise RuntimeError(f"{identifier} C1 hash mismatch: {actual} != {EXPECTED_C1_SHA256[identifier]}")
        payload = json.loads(candidate.calibration_artifact.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "c1_calibration_v1" or payload.get("no_training") is not True:
            raise RuntimeError(f"{identifier} is not a verified no-training C1 calibration artifact")
        artifacts[identifier] = {"path": str(candidate.calibration_artifact), "sha256": actual}
    if args.mode != "preflight" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for an A100 proxy execution")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output root: {args.output_root}")
    return {
        "base_model": str(args.base_model),
        "base_model_config_sha256": sha256(args.base_model / "config.json"),
        "base_adapter_artifacts": [],
        "c1_artifacts": artifacts,
        "sample_indices": str(args.sample_indices),
        "sample_indices_sha256": sha256(args.sample_indices),
    }


def runtime_metadata(model: torch.nn.Module) -> dict[str, Any]:
    gpu = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        gpu.append({"logical_index": index, "name": properties.name, "total_memory_bytes": int(properties.total_memory)})
    config = getattr(model, "config", None)
    return {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": gpu,
        "dtype": "float16",
        "tf32": {
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
        "attention_implementation": getattr(config, "_attn_implementation", None),
        "gradient_checkpointing": bool(getattr(model, "is_gradient_checkpointing", False)),
        "hf_device_map": jsonable(getattr(model, "hf_device_map", None)),
        "cpu_or_meta_offload": False,
    }


def selected_residency(groups: dict[str, list[torch.nn.Parameter]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, parameters in groups.items():
        devices = sorted({"meta" if parameter.is_meta else str(parameter.device) for parameter in parameters})
        invalid = [parameter for parameter in parameters if parameter.is_meta or parameter.device.type != "cuda"]
        result[name] = {
            "devices": devices,
            "parameter_elements": common.parameter_count(parameters),
            "invalid_parameter_elements": common.parameter_count(invalid),
        }
    return result


def finite_scores(scope: dict[str, Any]) -> bool:
    groups = scope.get("proxy_groups", {})
    return all(math.isfinite(float(groups[name][metric])) for name in groups for metric in ("gradnorm", "snip", "fisher"))


def run_candidate(args: argparse.Namespace, candidate: common.CandidateSpec, provenance: dict[str, Any]) -> dict[str, Any]:
    cut3r_layers, llm_layers, artifact = common.candidate_schedule(candidate)
    load_args = common.make_load_args(args, candidate, cut3r_layers, llm_layers)
    # Direct one-A100 placement.  Never activate the TITAN-V auto-map or CPU
    # offload/deferred dispatch path for this protocol.
    if load_args.device_map != "cuda:0":
        raise RuntimeError(f"A100 proxy requires direct cuda:0 placement, got {load_args.device_map!r}")
    load_args.pre_sft_defer_dispatch = False
    device = torch.device(args.device)
    common.reset_proxy_rng(args.rng_seed)
    tokenizer, model, image_processor = load_model(load_args, device, torch.float16)
    try:
        existing_lora = [name for name, _ in model.named_parameters() if "lora_" in name]
        if existing_lora:
            raise RuntimeError("Plain base unexpectedly contains existing LoRA parameters")
        # The shared loader constructs the candidate fusion module after the
        # pretrained checkpoint dispatch, so this *new* C1 module otherwise
        # retains PyTorch's CPU construction device.  Move precisely that
        # intended SFT-trainable module to the direct A100 placement before
        # C1 calibration; do not alter any pretrained model weights or the
        # frozen-backbone placement.
        common.fusion_module(model).to(device=device, dtype=torch.float16)
        apply_c1_calibration_artifact(model, artifact)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model, lora_recipe = common.attach_intended_sft_lora(model, seed=args.rng_seed)
        lora_initialization = common.lora_initialization_summary(model)
        groups = common.intended_sft_trainable_groups(model)
        selected = common.configure_intended_sft_trainable_parameters(model, groups)
        residency = selected_residency(groups)
        if any(info["invalid_parameter_elements"] for info in residency.values()):
            raise RuntimeError(f"Intended trainable parameters are not fully CUDA resident: {residency}")
        if {id(parameter) for parameter in selected} != {id(parameter) for values in groups.values() for parameter in values}:
            raise RuntimeError("Selected scope does not equal disjoint intended SFT group union")
        if candidate.identifier == BASELINE_ID:
            actual_counts = {name: common.parameter_count(parameters) for name, parameters in groups.items()}
            actual_counts["total"] = common.parameter_count(selected)
            if actual_counts != EXPECTED_BASELINE_COUNTS:
                raise RuntimeError(f"Baseline trainable count gate failed: {actual_counts} != {EXPECTED_BASELINE_COUNTS}")
        install_forward_frame_loader(args.forward_frames_root)
        dataset, collator, by_video = build_dataset(load_args, tokenizer, image_processor)
        records = common.load_calibration_records(args, by_video)
        if len(records) != 1:
            raise RuntimeError(f"Expected exactly one calibration minibatch, got {len(records)}")
        record = records[0]
        batch = common.prepare_batch(dataset, collator, by_video[str(record["video_path"])], model, device, torch.float16)
        batch_info = common.batch_metadata(batch, record)
        if batch_info.get("scene_id") != "scene0384_00" or batch_info.get("input_ids_shape") != [1, 424] or batch_info.get("supervised_label_tokens") != 13:
            raise RuntimeError(f"Fixed calibration contract failed: {batch_info}")
        started = time.perf_counter()
        scope = common.run_grouped_backward_scope(model, batch, "sft_trainable", groups, rng_seed=args.rng_seed)
        wall_seconds = time.perf_counter() - started
        scope["requires_grad_flags_restored"] = bool(getattr(model, "_last_proxy_requires_grad_restored", False))
        scope["selected_parameter_elements"] = common.parameter_count(selected)
        if scope["status"] != "PASS":
            raise RuntimeError(f"{candidate.identifier} primary scope did not pass: {scope['status']}")
        if not finite_scores(scope):
            raise RuntimeError(f"{candidate.identifier} produced non-finite proxy scores")
        runtime = runtime_metadata(model)
        runtime["cpu_or_meta_offload"] = any(info["invalid_parameter_elements"] for info in residency.values())
        return {
            "schema_version": SCHEMA_VERSION,
            "candidate": {
                **asdict(candidate),
                "calibration_artifact": str(candidate.calibration_artifact),
                "c1_artifact_sha256": provenance["c1_artifacts"][candidate.identifier]["sha256"],
                "post_sft_weights_loaded": False,
            },
            "provenance": provenance,
            "runtime": runtime,
            "calibration": {
                **batch_info,
                "sample_indices": str(args.sample_indices),
                "sample_indices_sha256": provenance["sample_indices_sha256"],
                "calibration_minibatches": 1,
                "loss_definition": "L_proxy = L_QA: ordinary supervised causal-LM next-token cross entropy",
            },
            "lora_initialization": {
                "peft_version": package_version("peft"),
                "recipe": jsonable(lora_recipe),
                "actual_state": lora_initialization,
            },
            "scope_definition": "fresh LoRA + candidate-specific C1 fusion + mm_projector",
            "selected_residency": residency,
            "primary_scope": scope,
            "wall_clock_seconds": wall_seconds,
            "no_training": {"optimizer_constructed": False, "optimizer_step_called": False, "parameter_updates": False},
        }
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def rankdata(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    rank = np.empty(array.size, dtype=np.float64)
    rank[order] = np.arange(array.size, dtype=np.float64)
    _, inverse, counts = np.unique(array, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size, dtype=np.float64)
    np.add.at(sums, inverse, rank)
    return sums[inverse] / counts[inverse]


def spearman(left: list[float], right: list[float]) -> float:
    if len(left) < 3 or len(set(left)) < 2 or len(set(right)) < 2:
        return float("nan")
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def kendall_tau_b(left: list[float], right: list[float]) -> float:
    concordant = discordant = ties_left = ties_right = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            dx, dy = left[first] - left[second], right[first] - right[second]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_left += 1
            elif dy == 0:
                ties_right += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt((concordant + discordant + ties_left) * (concordant + discordant + ties_right))
    return float("nan") if denominator == 0 else (concordant - discordant) / denominator


def pairwise_accuracy(left: list[float], right: list[float]) -> float:
    correct = total = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            dx, dy = left[first] - left[second], right[first] - right[second]
            if dx == 0 or dy == 0:
                continue
            total += 1
            correct += int((dx > 0) == (dy > 0))
    return float("nan") if total == 0 else correct / total


def flattened(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        groups = result["primary_scope"]["proxy_groups"]
        total = groups["total"]
        peak = max((item["peak_allocated_bytes"] for item in result["primary_scope"]["peak_gpu_memory"]), default=0)
        row = {
            "candidate": result["candidate"]["identifier"],
            "display_name": result["candidate"]["vsi_model"],
            "vsi_avg": result["candidate"]["vsi_avg"],
            "selected_params": total["parameter_elements"],
            "gradient_covered_params": total["parameters_with_gradient"],
            "ce_loss": result["primary_scope"]["loss"],
            "gradnorm": total["gradnorm"], "snip": total["snip"], "fisher": total["fisher"],
            "peak_vram_bytes": peak, "runtime_seconds": result["wall_clock_seconds"],
        }
        for group, values in groups.items():
            for metric in ("parameter_elements", "parameters_with_gradient", "gradnorm", "snip", "fisher"):
                row[f"{group}_{metric}"] = values[metric]
        rows.append(row)
    return rows


def analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vsi = [float(row["vsi_avg"]) for row in rows]
    metrics: dict[str, Any] = {}
    for name in ("gradnorm", "snip", "fisher"):
        values = [float(row[name]) for row in rows]
        metrics[name] = {
            "spearman_rho": spearman(values, vsi), "kendall_tau_b": kendall_tau_b(values, vsi),
            "pairwise_ordering_accuracy": pairwise_accuracy(values, vsi),
            "proxy_ranking_descending": [row["candidate"] for row in sorted(rows, key=lambda row: row[name], reverse=True)],
        }
    return {
        "label": "preliminary/pilot statistics; n=5",
        "vsi_ranking_descending": [row["candidate"] for row in sorted(rows, key=lambda row: row["vsi_avg"], reverse=True)],
        "metrics": metrics,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, Any]], ranking: dict[str, Any] | None) -> None:
    lines = ["# A100 pre-SFT primary trainable-scope proxy", "", "One fixed QA minibatch, one forward/backward per candidate, and no optimizer/update.", "", "| Candidate | Selected params | Gradient-covered params | CE loss | GradNorm | SNIP | Fisher | Peak VRAM bytes | Runtime s |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['display_name']} | {row['selected_params']} | {row['gradient_covered_params']} | {row['ce_loss']:.8g} | {row['gradnorm']:.8g} | {row['snip']:.8g} | {row['fisher']:.8g} | {row['peak_vram_bytes']} | {row['runtime_seconds']:.3f} |")
    if ranking is not None:
        lines.extend(["", "## Preliminary/pilot ranking analysis (n=5)", "", f"VSI ranking: `{', '.join(ranking['vsi_ranking_descending'])}`", "", "| Proxy | Spearman rho | Kendall tau-b | Pairwise ordering accuracy | Proxy ranking |", "|---|---:|---:|---:|---|"])
        for name, values in ranking["metrics"].items():
            lines.append(f"| {name} | {values['spearman_rho']:.6g} | {values['kendall_tau_b']:.6g} | {values['pairwise_ordering_accuracy']:.6g} | {', '.join(values['proxy_ranking_descending'])} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    # The migration gate preserves the audited fp16/TF32-off numerical path.
    # Set these before any CUDA model construction and record them per run.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    specs = candidate_specs(args.c1_root)
    provenance = validate(args, specs)
    provenance.update({
        "git_commit": git_commit(), "rng_seed": args.rng_seed,
        "environment": {"python": sys.version, "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "transformers": package_version("transformers"), "peft": package_version("peft"), "accelerate": package_version("accelerate")},
        "requested": {"mode": args.mode, "candidate_ids": args.candidate_ids, "device": args.device, "dtype": args.dtype, "attn_implementation": args.attn_implementation},
    })
    args.output_root.mkdir(parents=True, exist_ok=False)
    if args.mode == "preflight":
        write_json(args.output_root / "preflight.json", {"schema_version": SCHEMA_VERSION, "status": "PASS", "provenance": provenance})
        print(json.dumps({"status": "PASS", "output_root": str(args.output_root)})); return
    results: list[dict[str, Any]] = []
    for identifier in args.candidate_ids:
        result = run_candidate(args, specs[identifier], provenance)
        results.append(result)
        candidate_dir = args.output_root / "per_candidate" / identifier
        candidate_dir.mkdir(parents=True, exist_ok=False)
        write_json(candidate_dir / "provenance.json", result)
    rows = flattened(results)
    ranking = analysis(rows) if args.mode == "formal" else None
    payload = {"schema_version": SCHEMA_VERSION, "status": "PASS", "provenance": provenance, "results": results, "ranking_analysis": ranking}
    write_json(args.output_root / "results.json", payload)
    write_csv(args.output_root / "results.csv", rows)
    write_summary(args.output_root / "summary.md", rows, ranking)
    print(json.dumps({"status": "PASS", "output_root": str(args.output_root), "candidates": args.candidate_ids}))


if __name__ == "__main__":
    main()
