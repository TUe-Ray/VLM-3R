#!/usr/bin/env python
"""Read-only zero-cost proxy suite for the fixed post-SFT 3D model roster.

The roster is intentionally the one named in ``VSI result.csv``: SS+depth,
Spatial Stack, SS cross-attention, Baseline+depth, Baseline, Extra Object
token, selective fusion, and 0 spatial.  It uses each checkpoint's normal
adapter construction and supervised forward loss.  In particular, depth and
point-map candidates receive the migrated *32-frame* point-map sidecars; the
two-frame probe target bundles are never passed to a model forward.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
PROBING_DIR = REPO_ROOT / "scripts" / "probing"
for path in (REPO_ROOT, PROBING_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_pre_sft_zero_cost_proxies as common  # noqa: E402
from extract_depth_probe_features import build_dataset, load_eomt_consumer_cache  # noqa: E402
from local_depth_probe_cache import install_forward_frame_loader  # noqa: E402
from scripts.diagnose_layerwise_spatial_hidden_scan import load_model, move_to_device  # noqa: E402
from scripts.probing.post_sft_geometry_probe_specs import (  # noqa: E402
    MODEL_SPECS,
    validate_checkpoint_signature,
)


SCHEMA_VERSION = "post_sft_3d_zero_cost_proxy_v1"
BASE_MODEL = Path("/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2")
SIGLIP_MODEL = Path("/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384")
VLM3R_ROOT = Path("/mnt/DATA_SSD/shaoruei/models/vlm3r_runs")
FEATURE_ROOT = Path("/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features")
POINT_MAP_ROOT = Path("/mnt/DATA_SSD/shaoruei/probing_data/cut3r_point_maps_32_v1")
FORWARD_ROOT = Path("/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1")
TARGET_ROOT = Path("/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1")
DATA_YAML = REPO_ROOT / "scripts" / "probing" / "scannet_depth_probe_local_data.yaml"
SAMPLE_INDICES = Path(
    "/home/shaoruei/probe_provenance/scannet_baseline_L6/"
    "scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json"
)
EOMT_CACHE = Path("/home/shaoruei/probe_cache/eomt_consumer_grid_v2")
EOMT_RUNTIME = REPO_ROOT / ".offline_runtime" / "post_sft_geometry_probes"
VSI_CSV = REPO_ROOT / "VSI result.csv"


@dataclass(frozen=True)
class Candidate:
    identifier: str
    display_name: str
    label: str
    checkpoint: Path
    vsi_row: str
    feature_subdir: str = "spatial_features"
    feature_preset: str = "original"
    auxiliary_loss: str | None = None
    post_sft_architecture: str | None = None
    eomt_spec: str | None = None


CANDIDATES = (
    Candidate(
        "ss_depth", "SS + depth", "cut3r_spatialstack_d2_pointmap_45457911",
        VLM3R_ROOT / "cut3r_spatialstack_d2_pointmap_45457911", "SS+depth",
        "6:spatial_features_dec_6,9:spatial_features_dec_9,12:spatial_features", auxiliary_loss="pointmap",
    ),
    Candidate(
        "spatial_stack", "Spatial Stack", "cut3r_spatialstack_44323703",
        VLM3R_ROOT / "cut3r_spatialstack_44323703", "Spatial Stack to Layer 0/1/2",
        "6:spatial_features_dec_6,9:spatial_features_dec_9,12:spatial_features",
    ),
    Candidate(
        "ss_cross_attn", "SS + cross attn", "cut3r_spatialstack_cross_attn_45303862",
        VLM3R_ROOT / "cut3r_spatialstack_cross_attn_45303862", "Spatial Stack- Cross Attn",
        "6:spatial_features_dec_6,9:spatial_features_dec_9,12:spatial_features",
    ),
    Candidate(
        "baseline_depth", "Baseline + depth", "cut3r_depth_loss_43817021",
        VLM3R_ROOT / "cut3r_depth_loss_43817021", "Baseline+depth", auxiliary_loss="depth",
    ),
    Candidate(
        "baseline", "Baseline", "vlm3r_baseline", VLM3R_ROOT / "Reproduction_2", "Baseline",
    ),
    Candidate(
        "extra_object_token", "Extra Object token", "eomt_object",
        EOMT_RUNTIME / "eomt_obj_text_phrase_100p_40403422", "Extra Object token",
        post_sft_architecture="eomt_object", eomt_spec="eomt_object",
    ),
    Candidate(
        "selective_fusion", "selective fusion", "eomt_selective",
        EOMT_RUNTIME / "cut3r_eomt_sel3dr2_wmzero_40416881", "selective fusion",
        post_sft_architecture="eomt_selective", eomt_spec="eomt_selective",
    ),
    Candidate(
        "zero_spatial", "0 spatial", "zero_spatial", VLM3R_ROOT / "zero_spatial_features", "0 spatial",
        feature_preset="zero_spatial",
    ),
)
BY_ID = {candidate.identifier: candidate for candidate in CANDIDATES}


def load_vsi_scores() -> dict[str, float]:
    with VSI_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {str(row["Model"]): float(row["Avg."]) for row in rows}
    missing = [candidate.vsi_row for candidate in CANDIDATES if candidate.vsi_row not in result]
    if missing:
        raise RuntimeError(f"VSI reference is missing requested rows: {missing}")
    return result


def load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"Expected a tensor state dict at {path}")
    return value


def state_numel(state: dict[str, torch.Tensor]) -> int:
    return sum(int(value.numel()) for value in state.values() if isinstance(value, torch.Tensor))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_checkpoint(candidate: Candidate) -> Path:
    if candidate.eomt_spec is None:
        return candidate.checkpoint
    return MODEL_SPECS[candidate.eomt_spec].checkpoint


def validate_candidate(candidate: Candidate) -> None:
    checkpoint = source_checkpoint(candidate)
    required = ("adapter_model.bin", "non_lora_trainables.bin", "adapter_config.json", "config.json", "generation_config.json")
    if candidate.eomt_spec:
        signature = validate_checkpoint_signature(MODEL_SPECS[candidate.eomt_spec])
        if not candidate.checkpoint.is_dir():
            raise FileNotFoundError(f"Missing checkpoint-exact EoMT runtime overlay: {candidate.checkpoint}")
        if not (candidate.checkpoint / "post_sft_probe_reconstruction.json").is_file():
            raise RuntimeError(f"EoMT runtime overlay lacks reconstruction provenance: {candidate.checkpoint}")
        if signature["checkpoint"] != str(checkpoint):
            raise RuntimeError(f"EoMT checkpoint signature did not resolve to {checkpoint}")
    missing = [name for name in required if not (candidate.checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{candidate.identifier} lacks required checkpoint files: {missing}")
    if candidate.auxiliary_loss and not (POINT_MAP_ROOT / "scannet" / "spatial_features_points").is_dir():
        raise FileNotFoundError("Exact depth/point-map proxy requires full 32-frame point maps, but none were found")
    if candidate.post_sft_architecture and not (EOMT_CACHE / "validation.json").is_file():
        raise FileNotFoundError("EoMT proxy requires the validated consumer-grid cache")


def make_args(args: argparse.Namespace, candidate: Candidate) -> SimpleNamespace:
    return SimpleNamespace(
        model_label=candidate.label,
        model_path=str(candidate.checkpoint),
        model_base=str(BASE_MODEL),
        model_name="vlm-3r-llava-qwen2-lora",
        model_loading_mode="adapter",
        runtime_root=str(args.output_root / "runtime" / candidate.identifier),
        siglip_path=str(SIGLIP_MODEL), cut3r_weights=None,
        attn_implementation="sdpa",
        device_map=args.device_map, skip_spatial_tower_load=True, zero_spatial_features=False,
        mm_spatial_pool_stride=2, pool_mode="bilinear", dtype=args.dtype, cache_dtype=args.dtype,
        train_data_json=str(DATA_YAML), data_yaml=str(DATA_YAML), feature_root=str(FEATURE_ROOT),
        spatial_feature_dir=str(FEATURE_ROOT), spatial_features_subdir=candidate.feature_subdir,
        image_folder=str(FORWARD_ROOT), video_folder=str(FORWARD_ROOT), frames_upbound=32,
        add_time_instruction=None, seed=0, model_loading_info=None, architecture=None,
        feature_preset=candidate.feature_preset, post_sft_architecture=candidate.post_sft_architecture,
        geometry_spatial_features_root=(str(POINT_MAP_ROOT) if candidate.auxiliary_loss else None),
        geometry_spatial_features_subdir=("spatial_features_points" if candidate.auxiliary_loss else None),
        geometry_point_map_key=("point_maps_cam" if candidate.auxiliary_loss == "depth" else "point_maps_ref"),
        eomt_consumer_cache_root=str(EOMT_CACHE), eomt_cache_validation=str(EOMT_CACHE / "validation.json"),
        verify_eomt_file_checksum=False, eomt_selective_kv_gate=False,
    )


def move_value(value: Any, target: torch.device, dtype: torch.dtype | None = None) -> Any:
    if torch.is_tensor(value):
        target_dtype = dtype if dtype is not None and value.is_floating_point() else None
        return value.to(device=target, dtype=target_dtype, non_blocking=True)
    if isinstance(value, list):
        return [move_value(item, target, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(move_value(item, target, dtype) for item in value)
    if isinstance(value, dict):
        return {key: move_value(item, target, dtype) for key, item in value.items()}
    return value


def module_device(module: Any, fallback: torch.device) -> torch.device:
    declared = getattr(module, "device", None)
    if isinstance(declared, torch.device) and declared.type != "meta":
        return declared
    if isinstance(module, nn.Module):
        for parameter in module.parameters():
            if not parameter.is_meta:
                return parameter.device
    return fallback


def prepare_batch(dataset: Any, collator: Any, index: int, model: nn.Module, device: torch.device, dtype: torch.dtype,
                  candidate: Candidate, record: dict[str, Any], loader_args: SimpleNamespace) -> dict[str, Any]:
    batch = move_to_device(collator([dataset[index]]), device, dtype)
    vision = model.get_vision_tower()
    if vision is None:
        raise RuntimeError("Adapter model has no vision tower")
    if "images" in batch:
        batch["images"] = move_value(batch["images"], module_device(vision, device), getattr(vision, "dtype", dtype))
    spatial = model.get_spatial_tower()
    if spatial is not None:
        spatial_device = module_device(spatial, device)
        if "spatial_features" in batch:
            batch["spatial_features"] = move_value(batch["spatial_features"], spatial_device)
    # The depth/point-map losses consume the full-frame geometry sidecar after
    # the language forward.  The existing extractor routes it to cuda:0 under
    # the local sharded dispatch, so preserve that same contract.
    if "geometry_spatial_features" in batch:
        batch["geometry_spatial_features"] = move_value(batch["geometry_spatial_features"], device)
    if candidate.post_sft_architecture:
        payload = load_eomt_consumer_cache(loader_args, record)
        if payload is None:
            raise RuntimeError(f"{candidate.identifier} did not resolve its required EoMT cache payload")
        # The consumer receives its validated FP32 cache payload in the same
        # format as the post-SFT extractor; it performs its own placement.
        batch["eomt_cached_outputs"] = [payload]
    return batch


def match_non_lora_parameters(model: nn.Module, state: dict[str, torch.Tensor]) -> list[nn.Parameter]:
    named = dict(model.named_parameters())
    selected: list[nn.Parameter] = []
    missing: list[str] = []
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        choices = [key]
        for prefix in ("base_model.model.", "base_model.model.model."):
            if key.startswith(prefix):
                choices.append(key[len(prefix):])
        matches = [name for name in choices if name in named]
        if not matches:
            suffix = key.split(".model.", 1)[-1]
            matches = [name for name in named if name.endswith(suffix)]
        if len(matches) != 1:
            missing.append(f"{key} -> {matches[:3]}")
            continue
        parameter = named[matches[0]]
        if tuple(parameter.shape) != tuple(tensor.shape):
            missing.append(f"{key} shape={tuple(tensor.shape)} != {matches[0]} shape={tuple(parameter.shape)}")
            continue
        selected.append(parameter)
    if missing:
        raise RuntimeError("Could not map checkpoint-specific trainables to loaded model: " + "; ".join(missing[:5]))
    unique = list({id(parameter): parameter for parameter in selected}.values())
    if common.parameter_count(unique) != state_numel(state):
        raise RuntimeError("Candidate-specific parameter count does not match non_lora_trainables.bin")
    return unique


def candidate_modules(model: nn.Module, candidate_parameters: list[nn.Parameter]) -> list[nn.Module]:
    chosen = {id(parameter) for parameter in candidate_parameters}
    modules: list[nn.Module] = []
    for name, module in model.named_modules():
        if not name:
            continue
        direct = list(module.parameters(recurse=False))
        if any(id(parameter) in chosen for parameter in direct):
            modules.append(module)
    # Count a direct parameter's parent leaf module and no broad parent. This
    # gives exact linear/normalization FLOPs without attributing shared base
    # modules to the candidate scope.
    if not modules:
        raise RuntimeError("No modules own the checkpoint-specific parameter scope")
    return modules


def profile_flops(model: nn.Module, batch: dict[str, Any], modules: list[nn.Module]) -> tuple[int, int, float]:
    lm_head = getattr(model, "lm_head", None)
    if not isinstance(lm_head, nn.Linear):
        raise RuntimeError("Expected nn.Linear lm_head")
    counter = common.ForwardFlopCounter(modules, lm_head)
    previous_mode = model.training
    model.train(True)  # Includes any checkpoint-configured auxiliary SFT loss.
    common.reset_cuda_peaks()
    counter.install(model)
    started = time.perf_counter()
    try:
        with torch.no_grad(), common.proxy_supervised_logits_only(model):
            output = model(**batch, use_cache=False, return_dict=True)
        if getattr(output, "loss", None) is None:
            raise RuntimeError("Post-SFT proxy forward returned no supervised loss")
        common.synchronize_cuda()
    finally:
        counter.close()
        model.train(previous_mode)
    counter.restore_full_sft_lm_head_flops()
    return counter.total, counter.fusion, time.perf_counter() - started


def structural(model: nn.Module, candidate: Candidate, non_lora: dict[str, torch.Tensor], adapter: dict[str, torch.Tensor],
               candidate_parameters: list[nn.Parameter], vsi_scores: dict[str, float]) -> dict[str, Any]:
    return {
        "candidate": candidate.identifier, "display_name": candidate.display_name, "model_label": candidate.label,
        "checkpoint": str(candidate.checkpoint), "source_checkpoint": str(source_checkpoint(candidate)),
        "checkpoint_config_sha256": sha256(candidate.checkpoint / "config.json"), "vsi_model": candidate.vsi_row,
        "vsi_avg": vsi_scores[candidate.vsi_row], "total_params": common.parameter_count(model.parameters()),
        "runtime_trainable_params": common.parameter_count(p for p in model.parameters() if p.requires_grad),
        "sft_lora_params": state_numel(adapter), "candidate_specific_params": common.parameter_count(candidate_parameters),
        "sft_trainable_params": state_numel(adapter) + state_numel(non_lora),
        "trainable_param_definition": "checkpoint adapter_model.bin plus non_lora_trainables.bin; loaded proxy runtime is frozen",
        "auxiliary_loss": candidate.auxiliary_loss, "post_sft_architecture": candidate.post_sft_architecture,
    }


def flatten(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        row = dict(result["structural"])
        for scope in ("whole_model", "candidate_specific"):
            summary = result.get("proxy_scopes", {}).get(scope, {})
            row[f"{scope}_status"] = summary.get("status", "NOT_ATTEMPTED")
            for metric in ("gradnorm", "snip", "fisher", "loss", "runtime_seconds", "parameters_with_gradient", "gradient_elements"):
                row[f"{scope}_{metric}"] = summary.get(metric)
            samples = summary.get("minibatches", [])
            peaks = [entry for sample in samples if sample.get("status") == "PASS" for entry in sample.get("peak_gpu_memory", [])]
            row[f"{scope}_peak_gpu_allocated_bytes"] = max((int(entry["peak_allocated_bytes"]) for entry in peaks), default=None)
            row[f"{scope}_peak_gpu_reserved_bytes"] = max((int(entry["peak_reserved_bytes"]) for entry in peaks), default=None)
        rows.append(row)
    return rows


def correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = (("total_params", "structural", "lower_is_better"), ("sft_trainable_params", "structural", "lower_is_better"),
             ("forward_dense_matmul_flops", "structural", "lower_is_better"),
             ("whole_model_gradnorm", "whole_model", "higher_is_better"), ("whole_model_snip", "whole_model", "higher_is_better"),
             ("whole_model_fisher", "whole_model", "higher_is_better"),
             ("candidate_specific_gradnorm", "candidate_specific", "higher_is_better"),
             ("candidate_specific_snip", "candidate_specific", "higher_is_better"),
             ("candidate_specific_fisher", "candidate_specific", "higher_is_better"))
    output = []
    for metric, scope, orientation in specs:
        valid = [row for row in rows if row.get(metric) is not None and math.isfinite(float(row[metric]))]
        raw = [float(row[metric]) for row in valid]
        reference = [float(row["vsi_avg"]) for row in valid]
        oriented = [-value for value in raw] if orientation == "lower_is_better" else raw
        output.append({"proxy": metric, "scope": scope, "expected_orientation": orientation,
                       "n_architectures": len(valid), "spearman_vs_vsi_avg": common.spearman(oriented, reference),
                       "raw_spearman_without_cost_flip": common.spearman(raw, reference),
                       "candidate_order": ",".join(row["candidate"] for row in valid)})
    return output


def write_summary(output_root: Path, rows: list[dict[str, Any]], correlation_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = ["# Post-SFT 3D model proxy suite", "", "Each model uses its exact checkpoint forward/loss path on the same fixed 32-frame supervised minibatch. No optimizer was constructed and no parameter was updated.", "",
             "| Model | VSI | Params | SFT trainable | FLOPs | Specific GradNorm | Specific SNIP | Specific Fisher |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        show = lambda value: "—" if value is None else common.human_count(float(value))
        lines.append(f"| {row['display_name']} | {row['vsi_avg']:.1f} | {show(row['total_params'])} | {show(row['sft_trainable_params'])} | {show(row.get('forward_dense_matmul_flops'))} | {show(row.get('candidate_specific_gradnorm'))} | {show(row.get('candidate_specific_snip'))} | {show(row.get('candidate_specific_fisher'))} |")
    lines += ["", "## Spearman correlation with VSI-Bench", "", "| Proxy | Scope | n | Spearman |", "|---|---|---:|---:|"]
    for row in correlation_rows:
        value = row["spearman_vs_vsi_avg"]
        score = f"{value:.4f}" if math.isfinite(value) else "NaN"
        lines.append(f"| {row['proxy']} | {row['scope']} | {row['n_architectures']} | {score} |")
    lines += ["", "GradNorm = sum of per-tensor gradient L2 norms; SNIP = sum |p·dL/dp|; Fisher = sum (dL/dp)^2.",
              "Depth and point-map models consume full 32-frame CUT3R sidecars, never compact two-frame targets.",
              f"Calibration minibatches: {args.calibration_batches}; fixed manifest: `{args.sample_indices}`."]
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def score_candidate(args: argparse.Namespace, candidate: Candidate, attempt_whole_model: bool, vsi_scores: dict[str, float]) -> dict[str, Any]:
    validate_candidate(candidate)
    loader_args = make_args(args, candidate)
    device = torch.device(args.device)
    dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]
    non_lora = load_state(source_checkpoint(candidate) / "non_lora_trainables.bin")
    adapter = load_state(source_checkpoint(candidate) / "adapter_model.bin")
    tokenizer, model, image_processor = load_model(loader_args, device, dtype)
    try:
        # Match the activation-saving SFT contract.  After LoRA is merged the
        # normal PEFT hook is absent, so explicitly retain the frozen input
        # embedding gradient needed for re-entrant activation checkpointing.
        # Neither call changes architecture, loss, or parameter values.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        install_forward_frame_loader(FORWARD_ROOT)
        dataset, collator, by_video = build_dataset(loader_args, tokenizer, image_processor)
        records = common.load_calibration_records(args, by_video)
        specific_parameters = match_non_lora_parameters(model, non_lora)
        modules = candidate_modules(model, specific_parameters)
        whole_parameters = [parameter for parameter in model.parameters() if not parameter.is_meta]
        result: dict[str, Any] = {"schema_version": SCHEMA_VERSION,
            "structural": structural(model, candidate, non_lora, adapter, specific_parameters, vsi_scores),
            "calibration": {"sample_indices": str(args.sample_indices), "sample_indices_sha256": sha256(args.sample_indices),
                            "requested_batches": args.calibration_batches, "minibatches": [],
                            "loss_definition": "checkpoint's ordinary supervised causal-LM CE plus enabled depth/point-map auxiliary loss",
                            "forward_contract": "32 decoded RGB frames, exact CUT3R token sidecars, and full 32-frame point maps when configured"},
            "proxy_definitions": {"gradnorm": "sum_p ||dL/dp||_2", "snip": "sum_p |p*dL/dp|", "fisher": "sum_p (dL/dp)^2"},
            "proxy_scope_definition": "candidate_specific = tensors in exact checkpoint non_lora_trainables.bin (fusion/projector/architecture heads); whole_model = all materialized parameters",
            "no_training": {"optimizer_constructed": False, "parameter_updates": False}, "proxy_scopes": {}}
        aggregates: dict[str, list[dict[str, Any]]] = {"candidate_specific": []}
        if attempt_whole_model:
            aggregates["whole_model"] = []
        totals: list[int] = []; specific_flops: list[int] = []; runtimes: list[float] = []
        whole_oom = False
        for record in records:
            batch = prepare_batch(dataset, collator, by_video[str(record["video_path"])], model, device, dtype, candidate, record, loader_args)
            info = common.batch_metadata(batch, record)
            total, scoped, runtime = profile_flops(model, batch, modules)
            info.update({"forward_dense_matmul_flops": total, "candidate_specific_dense_matmul_flops": scoped, "flop_forward_runtime_seconds": runtime})
            score = common.run_backward_scope(model, batch, "candidate_specific", specific_parameters)
            info["candidate_specific"] = score; aggregates["candidate_specific"].append(score)
            if attempt_whole_model and not whole_oom:
                score = common.run_backward_scope(model, batch, "whole_model", whole_parameters)
                info["whole_model"] = score; aggregates["whole_model"].append(score); whole_oom = score["status"] == "OOM"
            result["calibration"]["minibatches"].append(info); totals.append(total); specific_flops.append(scoped); runtimes.append(runtime)
        result["structural"].update({"forward_dense_matmul_flops": sum(totals) / len(totals), "candidate_specific_dense_matmul_flops": sum(specific_flops) / len(specific_flops), "flop_forward_runtime_seconds": sum(runtimes) / len(runtimes)})
        for scope, samples in aggregates.items():
            passed = [sample for sample in samples if sample["status"] == "PASS"]
            summary: dict[str, Any] = {"status": "PASS" if len(passed) == len(samples) else samples[0]["status"], "minibatches": samples}
            if passed:
                summary.update({"calibration_batches_scored": len(passed)})
                for metric in ("gradnorm", "snip", "fisher", "loss", "runtime_seconds", "parameters_with_gradient", "gradient_elements"):
                    summary[metric] = sum(float(sample[metric]) for sample in passed) / len(passed)
            result["proxy_scopes"][scope] = summary
        return result
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--candidates", default=",".join(candidate.identifier for candidate in CANDIDATES))
    parser.add_argument("--calibration-batches", type=int, default=1)
    parser.add_argument("--attempt-whole-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--prior-results",
        type=Path,
        help="Optional successful prior results.json to merge with newly retried candidates.",
    )
    parser.add_argument("--sample-indices", type=Path, default=SAMPLE_INDICES)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    return parser.parse_args()


def load_prior_results(path: Path) -> list[dict[str, Any]]:
    payload = common.read_json(path)
    if not isinstance(payload, list):
        raise TypeError(f"Expected a list of candidate results in {path}")
    seen: set[str] = set()
    for result in payload:
        if not isinstance(result, dict):
            raise TypeError(f"Prior results contains a non-object entry in {path}")
        candidate = result.get("structural", {}).get("candidate")
        if candidate not in BY_ID:
            raise RuntimeError(f"Prior results contains unknown candidate {candidate!r}")
        if candidate in seen:
            raise RuntimeError(f"Prior results contains duplicate candidate {candidate!r}")
        seen.add(candidate)
    return payload


def main() -> None:
    args = parse_args()
    requested = [item.strip() for item in args.candidates.split(",") if item.strip()]
    unknown = sorted(set(requested).difference(BY_ID))
    if unknown:
        raise ValueError(f"Unknown candidates: {unknown}; valid={sorted(BY_ID)}")
    candidates = [BY_ID[item] for item in requested]
    if args.calibration_batches < 1:
        raise ValueError("--calibration-batches must be positive")
    if args.mode == "smoke" and (len(candidates) != 1 or args.calibration_batches != 1):
        raise ValueError("Smoke requires exactly one candidate and one minibatch")
    args.output_root.mkdir(parents=True, exist_ok=True)
    prior_results = load_prior_results(args.prior_results) if args.prior_results else []
    prior_by_candidate = {
        result["structural"]["candidate"]: result
        for result in prior_results
    }
    vsi_scores = load_vsi_scores(); started = time.time(); results: list[dict[str, Any]] = []; failures = []
    whole_safe = bool(args.attempt_whole_model)
    print(json.dumps({"mode": args.mode, "candidates": requested, "output_root": str(args.output_root), "whole_model_initial": whole_safe}), flush=True)
    for candidate in candidates:
        print(f"[RUN] {candidate.identifier}", flush=True)
        try:
            result = score_candidate(args, candidate, whole_safe, vsi_scores); results.append(result)
            if result.get("proxy_scopes", {}).get("whole_model", {}).get("status") == "OOM":
                whole_safe = False; print("[SAFE MODE] whole-model OOM; skipping remaining whole-model scopes.", flush=True)
        except Exception as exc:
            failure = {"candidate": candidate.identifier, "exception_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
            failures.append(failure); print("[ERROR] " + json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
            if args.mode == "smoke":
                raise
    merged_by_candidate = dict(prior_by_candidate)
    merged_by_candidate.update({result["structural"]["candidate"]: result for result in results})
    requested_or_prior = set(requested).union(prior_by_candidate)
    combined_ids = [candidate.identifier for candidate in CANDIDATES if candidate.identifier in requested_or_prior]
    combined_results = [merged_by_candidate[candidate_id] for candidate_id in combined_ids if candidate_id in merged_by_candidate]
    missing_after_retry = [candidate_id for candidate_id in combined_ids if candidate_id not in merged_by_candidate]
    rows = flatten(combined_results); correlation_rows = correlations(rows)
    metadata = {"schema_version": SCHEMA_VERSION, "mode": args.mode, "elapsed_seconds": time.time() - started,
                "candidates_requested": combined_ids, "candidates_retried": requested,
                "candidates_completed": [result["structural"]["candidate"] for result in combined_results],
                "calibration_batches": args.calibration_batches, "whole_model_scope_attempted_initially": bool(args.attempt_whole_model),
                "whole_model_scope_safe_after_smoke": whole_safe, "no_optimizer_constructed": True, "no_weight_updates": True,
                "prior_results": str(args.prior_results) if args.prior_results else None,
                "prior_candidates_reused": sorted(prior_by_candidate), "missing_after_retry": missing_after_retry,
                "failures": failures}
    common.write_json(args.output_root / "results.json", combined_results); common.write_json(args.output_root / "metadata.json", metadata)
    common.write_csv(args.output_root / "proxy_scores.csv", rows); common.write_csv(args.output_root / "spearman_correlations.csv", correlation_rows)
    common.write_json(args.output_root / "spearman_correlations.json", correlation_rows); write_summary(args.output_root, rows, correlation_rows, args)
    final_pass = not failures and not missing_after_retry
    print(json.dumps({"status": "PASS" if final_pass else "PARTIAL", "completed": len(combined_results), "failures": len(failures), "output_root": str(args.output_root)}), flush=True)
    if not final_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
