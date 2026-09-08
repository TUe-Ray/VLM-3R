#!/usr/bin/env python
"""Exact chunked-backward pre-SFT Baseline zero-cost proxy smoke.

The full intended SFT scope is fresh LoRA + C1 fusion + mm_projector.  A
CPU-offloaded decoder layer does not retain LoRA ``.grad`` under Accelerate's
inference-style hooks, so this runner reconstructs the same fixed pre-SFT
state for each disjoint LoRA chunk and maps only that chunk's decoder layers to
GPU.  Additive proxy reductions are then summed exactly across chunks.

This runner is deliberately limited to one fixed QA minibatch and one Baseline
candidate.  It never creates an optimizer or updates a parameter.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PROBING_DIR = REPO_ROOT / "scripts" / "probing"
if str(PROBING_DIR) not in sys.path:
    sys.path.insert(0, str(PROBING_DIR))

import evaluate_pre_sft_zero_cost_proxies as common  # noqa: E402
from depth_probe_common import write_json  # noqa: E402
from extract_depth_probe_features import build_dataset  # noqa: E402
from local_depth_probe_cache import install_forward_frame_loader  # noqa: E402
from scripts.diagnose_layerwise_spatial_hidden_scan import (  # noqa: E402
    dispatch_deferred_pre_sft_model,
    load_model,
)
from llava.model.c1_structured_isometry import apply_c1_calibration_artifact  # noqa: E402


SCHEMA_VERSION = "pre_sft_chunked_exact_proxy_smoke_v1"
BASELINE_ID = "c1_vlm3r_native"
LAYER_PATTERN = re.compile(r"\.model\.layers\.(\d+)\.")
DEFAULT_ROOT = Path("/home/shaoruei/probe_outputs/pre_sft_zero_cost_proxies_v2")
DEFAULT_VALIDATION = DEFAULT_ROOT / "validate_baseline_chunked_exact_v3_mask_replay"
DEFAULT_SMOKE = DEFAULT_ROOT / "smoke_baseline_chunked_exact_v3_mask_replay"
LOSS_ATOL = 1e-5
LOSS_RTOL = 1e-6
PROXY_ATOL = 1e-5
# Chunk passes retain only their target decoder layer on GPU. The remaining
# frozen path may cross CPU/GPU boundaries differently, so fp16 GEMM rounding
# accumulates at roughly 1e-4 relative scale even when CE loss and replayed
# LoRA masks are identical. This narrow bound is a validation gate.
PROXY_RTOL = 2e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "smoke"), required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validation-results", type=Path)
    parser.add_argument("--base-model", type=Path, default=common.DEFAULT_BASE_MODEL)
    parser.add_argument("--siglip-model", type=Path, default=common.DEFAULT_SIGLIP_MODEL)
    parser.add_argument("--forward-frames-root", type=Path, default=common.DEFAULT_FORWARD_ROOT)
    parser.add_argument("--probe-targets-root", type=Path, default=common.DEFAULT_TARGET_ROOT)
    parser.add_argument("--feature-root", type=Path, default=common.DEFAULT_FEATURE_ROOT)
    parser.add_argument("--data-yaml", type=Path, default=common.DEFAULT_DATA_YAML)
    parser.add_argument("--sample-indices", type=Path, default=common.DEFAULT_SAMPLE_INDICES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", choices=("auto",), default="auto")
    parser.add_argument("--dtype", choices=("float16",), default="float16")
    parser.add_argument("--pre-sft-gpu-weight-budgets", default="4GiB,6GiB")
    parser.add_argument("--pre-sft-cpu-offload-budget", default="45GiB")
    parser.add_argument("--resident-decoder-device", type=int, default=1)
    parser.add_argument("--rng-seed", type=int, default=42)
    args = parser.parse_args()
    args.calibration_batches = 1
    args.pre_sft_gpu_weight_budget = "5GiB"  # Required fallback for make_load_args.
    args.attn_implementation = None
    if args.output_root is None:
        args.output_root = DEFAULT_VALIDATION if args.mode == "validate" else DEFAULT_SMOKE
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite chunked-proxy output: {args.output_root}")
    if args.mode == "smoke":
        if args.validation_results is None:
            raise ValueError("--validation-results is required for --mode smoke")
        if not args.validation_results.is_file():
            raise FileNotFoundError(f"Chunked validation result not found: {args.validation_results}")
    return args


def decoder_layer_index(name: str) -> int:
    match = LAYER_PATTERN.search(name)
    if match is None:
        raise RuntimeError(f"LoRA parameter is not owned by a decoder layer: {name}")
    return int(match.group(1))


def lora_named_parameters(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    pairs = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    ]
    if not pairs:
        raise RuntimeError("Fresh pre-SFT construction has no LoRA parameters")
    return pairs


def group_scalar(group: dict[str, Any], metric: str) -> float:
    value = group.get(metric)
    if value is None:
        raise RuntimeError(f"Missing {metric} from a chunk score")
    return float(value)


def add_groups(groups: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    items = list(groups)
    if not items:
        raise RuntimeError("Cannot sum an empty collection of proxy groups")
    return {
        "parameter_elements": sum(int(item["parameter_elements"]) for item in items),
        "parameters_with_gradient": sum(int(item["parameters_with_gradient"]) for item in items),
        "gradient_elements": sum(int(item["gradient_elements"]) for item in items),
        "gradnorm": sum(group_scalar(item, "gradnorm") for item in items),
        "snip": sum(group_scalar(item, "snip") for item in items),
        "fisher": sum(group_scalar(item, "fisher") for item in items),
    }


def same_float(left: float, right: float, *, atol: float, rtol: float) -> bool:
    return math.isclose(left, right, abs_tol=atol, rel_tol=rtol)


def compare_proxy_groups(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    passed = True
    for metric in ("gradnorm", "snip", "fisher"):
        reference_value = group_scalar(reference, metric)
        candidate_value = group_scalar(candidate, metric)
        agrees = same_float(reference_value, candidate_value, atol=PROXY_ATOL, rtol=PROXY_RTOL)
        metrics[metric] = {
            "simultaneous": reference_value,
            "chunked": candidate_value,
            "absolute_difference": abs(reference_value - candidate_value),
            "passed": agrees,
        }
        passed = passed and agrees
    for field in ("parameter_elements", "parameters_with_gradient", "gradient_elements"):
        agrees = int(reference[field]) == int(candidate[field])
        metrics[field] = {"simultaneous": int(reference[field]), "chunked": int(candidate[field]), "passed": agrees}
        passed = passed and agrees
    return {"passed": passed, "atol": PROXY_ATOL, "rtol": PROXY_RTOL, "metrics": metrics}


def check_loss_consistency(losses: list[float]) -> dict[str, Any]:
    if not losses:
        raise RuntimeError("Cannot check an empty loss trace")
    reference = float(losses[0])
    deviations = [abs(float(loss) - reference) for loss in losses]
    return {
        "reference": reference,
        "losses": [float(loss) for loss in losses],
        "maximum_absolute_difference": max(deviations),
        "atol": LOSS_ATOL,
        "rtol": LOSS_RTOL,
        "passed": all(same_float(reference, float(loss), atol=LOSS_ATOL, rtol=LOSS_RTOL) for loss in losses),
    }


def max_peaks(passes: Iterable[dict[str, Any]]) -> list[dict[str, int]]:
    values: dict[int, dict[str, int]] = {}
    for entry in passes:
        for peak in entry["score"]["peak_gpu_memory"]:
            index = int(peak["logical_gpu"])
            current = values.setdefault(index, {"logical_gpu": index, "peak_allocated_bytes": 0, "peak_reserved_bytes": 0})
            current["peak_allocated_bytes"] = max(current["peak_allocated_bytes"], int(peak["peak_allocated_bytes"]))
            current["peak_reserved_bytes"] = max(current["peak_reserved_bytes"], int(peak["peak_reserved_bytes"]))
    return [values[index] for index in sorted(values)]


def load_fixed_validation(path: Path) -> dict[str, Any]:
    payload = common.read_json(path)
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("mode") != "validate":
        raise RuntimeError(f"Not a compatible chunked validation artifact: {path}")
    validation = payload.get("validation", {})
    if validation.get("status") == "PASS":
        payload["validation_review"] = "recorded_PASS"
        return payload
    # A completed validation artifact can predate a documented tolerance-only
    # update. Re-evaluate the stored values, without another model execution.
    agreement = validation.get("proxy_agreement", {}).get("metrics", {})
    proxy_passed = all(
        same_float(
            float(item["simultaneous"]),
            float(item["chunked"]),
            atol=PROXY_ATOL,
            rtol=PROXY_RTOL,
        )
        for metric in ("gradnorm", "snip", "fisher")
        if (item := agreement.get(metric)) is not None
    )
    coverage = validation.get("coverage", {})
    coverage_passed = bool(coverage.get("matches_simultaneous_names") and coverage.get("disjoint"))
    if validation.get("loss_consistency", {}).get("passed") and proxy_passed and coverage_passed:
        payload["validation_review"] = "PASS_REEVALUATED_WITH_CURRENT_FP16_TOLERANCE"
        return payload
    if validation.get("status") != "PASS":
        raise RuntimeError(f"Chunked validation did not pass: {path}")
    return payload


class LoRADropoutMaskRecorder:
    """Capture the canonical training-mode mask for each PEFT LoRA dropout."""

    def __init__(self, model: torch.nn.Module):
        self.masks: dict[str, torch.Tensor] = {}
        self._handles: list[Any] = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Dropout) and ".lora_dropout." in name:
                self._handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name: str):
        def capture(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if len(inputs) != 1 or not torch.is_tensor(inputs[0]) or not torch.is_tensor(output):
                raise RuntimeError(f"Unexpected LoRA dropout signature for {name}")
            # For nonzero input values, this exactly identifies Dropout's
            # Bernoulli decision.  A zero input remains zero under either
            # decision, so its stored value cannot affect the replayed output.
            self.masks[name] = output.detach().ne(0).to(device="cpu", dtype=torch.bool).contiguous()
        return capture

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def dropout_module_name(parameter_name: str) -> str:
    for factor in ("lora_A", "lora_B"):
        needle = f".{factor}.default.weight"
        if needle in parameter_name:
            return parameter_name.replace(needle, ".lora_dropout.default")
    raise RuntimeError(f"Cannot locate LoRA dropout module for parameter {parameter_name}")


@contextmanager
def replay_lora_dropout_masks(
    model: torch.nn.Module,
    target_parameter_names: Iterable[str],
    canonical_masks: dict[str, torch.Tensor],
):
    """Replay canonical masks for the LoRA modules whose gradients are scored.

    The model remains in training mode.  We replay only the target chunk's
    masks because a freshly initialized LoRA has B=0: frozen non-target LoRA
    branches contribute exactly zero to the forward and to the target's
    derivative.  This neutralizes device-dependent CUDA RNG consumption while
    preserving the target modules' exact canonical stochastic condition.
    """
    modules = dict(model.named_modules())
    handles: list[Any] = []
    names = sorted({dropout_module_name(name) for name in target_parameter_names})
    try:
        for name in names:
            module = modules.get(name)
            mask = canonical_masks.get(name)
            if not isinstance(module, torch.nn.Dropout) or mask is None:
                raise RuntimeError(f"Missing canonical LoRA dropout mask/module: {name}")
            probability = float(module.p)
            if probability <= 0.0 or probability >= 1.0:
                raise RuntimeError(f"Unexpected LoRA dropout probability for {name}: {probability}")

            def replay(
                _module: torch.nn.Module,
                inputs: tuple[torch.Tensor, ...],
                _output: torch.Tensor,
                *,
                saved_mask: torch.Tensor = mask,
                dropout_name: str = name,
                p: float = probability,
            ) -> torch.Tensor:
                if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
                    raise RuntimeError(f"Unexpected LoRA dropout signature for {dropout_name}")
                input_tensor = inputs[0]
                if tuple(input_tensor.shape) != tuple(saved_mask.shape):
                    raise RuntimeError(
                        f"Canonical mask shape mismatch for {dropout_name}: "
                        f"{tuple(saved_mask.shape)} versus {tuple(input_tensor.shape)}"
                    )
                return input_tensor * saved_mask.to(device=input_tensor.device, dtype=input_tensor.dtype) / (1.0 - p)

            handles.append(module.register_forward_hook(replay))
        yield
    finally:
        for handle in handles:
            handle.remove()


def capture_canonical_dropout_reference(args: argparse.Namespace) -> dict[str, Any]:
    """Run the unmodified canonical forward once and retain only LoRA masks."""
    candidate = common.BY_IDENTIFIER[BASELINE_ID]
    cut3r_layers, llm_layers, artifact = common.candidate_schedule(candidate)
    load_args = common.make_load_args(args, candidate, cut3r_layers, llm_layers)
    load_args.pre_sft_defer_dispatch = True
    common.reset_proxy_rng(args.rng_seed)
    tokenizer, model, image_processor = load_model(load_args, torch.device(args.device), torch.float16)
    try:
        apply_c1_calibration_artifact(model, artifact)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model, lora_recipe = common.attach_intended_sft_lora(model, seed=args.rng_seed)
        lora_initialization = common.lora_initialization_summary(model)
        model = dispatch_deferred_pre_sft_model(model, load_args, torch.device(args.device))
        all_lora_names = [name for name, _ in lora_named_parameters(model)]
        install_forward_frame_loader(args.forward_frames_root)
        dataset, collator, by_video = build_dataset(load_args, tokenizer, image_processor)
        records = common.load_calibration_records(args, by_video)
        if len(records) != 1:
            raise RuntimeError(f"Canonical reference requires one minibatch, got {len(records)}")
        record = records[0]
        batch = common.prepare_batch(
            dataset, collator, by_video[str(record["video_path"])], model, torch.device(args.device), torch.float16
        )
        model.train(True)
        model.config.use_cache = False
        recorder = LoRADropoutMaskRecorder(model)
        try:
            rng_reset = common.reset_proxy_rng(args.rng_seed)
            with torch.no_grad(), common.proxy_supervised_logits_only(model):
                output = model(**batch, use_cache=False, return_dict=True)
            loss = getattr(output, "loss", None)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(f"Canonical dropout-reference forward returned invalid loss: {loss}")
        finally:
            recorder.close()
        expected_dropout_names = {dropout_module_name(name) for name in all_lora_names}
        if set(recorder.masks) != expected_dropout_names:
            missing = sorted(expected_dropout_names - set(recorder.masks))
            extra = sorted(set(recorder.masks) - expected_dropout_names)
            raise RuntimeError(f"Canonical LoRA dropout capture mismatch; missing={missing[:3]}, extra={extra[:3]}")
        return {
            "loss": float(loss.detach().float().item()),
            "rng_reset": rng_reset,
            "masks": recorder.masks,
            "mask_count": len(recorder.masks),
            "lora_recipe": lora_recipe,
            "lora_initialization": lora_initialization,
            "calibration": common.batch_metadata(batch, record),
        }
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def score_chunk(
    args: argparse.Namespace,
    layers: list[int],
    *,
    include_auxiliary: bool,
    label: str,
    canonical_masks: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Reconstruct the fixed model state, then score one resident LoRA chunk."""
    candidate = common.BY_IDENTIFIER[BASELINE_ID]
    cut3r_layers, llm_layers, artifact = common.candidate_schedule(candidate)
    load_args = common.make_load_args(args, candidate, cut3r_layers, llm_layers)
    load_args.pre_sft_defer_dispatch = True
    load_args.pre_sft_resident_decoder_layers = list(layers)
    load_args.pre_sft_resident_decoder_device = int(args.resident_decoder_device)

    common.reset_proxy_rng(args.rng_seed)
    tokenizer, model, image_processor = load_model(load_args, torch.device(args.device), torch.float16)
    try:
        preexisting_lora = [name for name, _ in model.named_parameters() if "lora_" in name]
        if preexisting_lora:
            raise RuntimeError("Base model unexpectedly already has LoRA parameters")
        apply_c1_calibration_artifact(model, artifact)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model, lora_recipe = common.attach_intended_sft_lora(model, seed=args.rng_seed)
        lora_initialization = common.lora_initialization_summary(model)
        model = dispatch_deferred_pre_sft_model(model, load_args, torch.device(args.device))
        all_groups = common.intended_sft_trainable_groups(model)
        lora_pairs = lora_named_parameters(model)
        all_lora_names = [name for name, _ in lora_pairs]
        target_pairs = [(name, parameter) for name, parameter in lora_pairs if decoder_layer_index(name) in set(layers)]
        if not target_pairs:
            raise RuntimeError(f"{label} selected no LoRA parameters")
        target_names = [name for name, _ in target_pairs]
        target_parameters = [parameter for _, parameter in target_pairs]
        target_devices = sorted({str(parameter.device) for parameter in target_parameters})
        expected_device = f"cuda:{args.resident_decoder_device}"
        if target_devices != [expected_device]:
            raise RuntimeError(
                f"{label} target LoRA is not fully resident on {expected_device}: {target_devices}"
            )
        device_map = dict(getattr(model, "_pre_sft_deferred_dispatch_device_map", {}))
        target_layer_map = {
            str(index): next(
                (value for key, value in device_map.items() if key.endswith(f".model.layers.{index}")),
                None,
            )
            for index in layers
        }
        if any(str(value) != str(args.resident_decoder_device) for value in target_layer_map.values()):
            raise RuntimeError(f"{label} target decoder placement differs from requested GPU: {target_layer_map}")
        score_groups: dict[str, list[torch.nn.Parameter]] = {"lora": target_parameters}
        if include_auxiliary:
            score_groups["fusion_block"] = all_groups["fusion_block"]
            score_groups["mm_projector"] = all_groups["mm_projector"]
        selected_ids = {id(parameter) for values in score_groups.values() for parameter in values}
        if len(selected_ids) != sum(len(values) for values in score_groups.values()):
            raise RuntimeError(f"{label} has overlapping score groups")
        common.configure_intended_sft_trainable_parameters(model, all_groups)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        install_forward_frame_loader(args.forward_frames_root)
        dataset, collator, by_video = build_dataset(load_args, tokenizer, image_processor)
        records = common.load_calibration_records(args, by_video)
        if len(records) != 1:
            raise RuntimeError(f"{label} requires exactly one fixed minibatch, got {len(records)}")
        record = records[0]
        batch = common.prepare_batch(
            dataset, collator, by_video[str(record["video_path"])], model, torch.device(args.device), torch.float16
        )
        with replay_lora_dropout_masks(model, target_names, canonical_masks):
            score = common.run_grouped_backward_scope(model, batch, label, score_groups, rng_seed=args.rng_seed)
        if score["status"] != "PASS":
            raise RuntimeError(f"{label} did not obtain complete target gradients: {score['status']}")
        lora_score = score["proxy_groups"]["lora"]
        expected_elements = common.parameter_count(target_parameters)
        if int(lora_score["parameters_with_gradient"]) != expected_elements:
            raise RuntimeError(f"{label} LoRA gradients are incomplete despite PASS status")
        return {
            "label": label,
            "layers": list(layers),
            "include_fusion_block": include_auxiliary,
            "include_mm_projector": include_auxiliary,
            "target_lora_parameter_names": target_names,
            "target_lora_parameter_elements": expected_elements,
            "target_lora_devices_before_forward": target_devices,
            "target_layer_device_map": target_layer_map,
            "all_lora_parameter_names": all_lora_names,
            "all_lora_parameter_elements": common.parameter_count(parameter for _, parameter in lora_pairs),
            "lora_initialization": lora_initialization,
            "lora_recipe": lora_recipe,
            "calibration": common.batch_metadata(batch, record),
            "score": score,
        }
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def validate(args: argparse.Namespace) -> dict[str, Any]:
    layers = [0, 1, 2]
    canonical = capture_canonical_dropout_reference(args)
    simultaneous = score_chunk(
        args, layers, include_auxiliary=False, label="simultaneous_layers_0_2", canonical_masks=canonical["masks"]
    )
    chunks = [
        score_chunk(args, [layer], include_auxiliary=False, label=f"chunk_layer_{layer}", canonical_masks=canonical["masks"])
        for layer in layers
    ]
    simultaneous_group = simultaneous["score"]["proxy_groups"]["lora"]
    chunked_group = add_groups(chunk["score"]["proxy_groups"]["lora"] for chunk in chunks)
    simultaneous_names = set(simultaneous["target_lora_parameter_names"])
    chunk_names = [name for chunk in chunks for name in chunk["target_lora_parameter_names"]]
    coverage = {
        "simultaneous_parameter_elements": int(simultaneous_group["parameter_elements"]),
        "chunked_parameter_elements": sum(chunk["target_lora_parameter_elements"] for chunk in chunks),
        "unique_chunk_names": len(set(chunk_names)),
        "chunk_name_count": len(chunk_names),
        "matches_simultaneous_names": set(chunk_names) == simultaneous_names,
        "disjoint": len(chunk_names) == len(set(chunk_names)),
    }
    losses = check_loss_consistency(
        [float(canonical["loss"]), float(simultaneous["score"]["loss"])]
        + [float(chunk["score"]["loss"]) for chunk in chunks]
    )
    agreement = compare_proxy_groups(simultaneous_group, chunked_group)
    passed = bool(losses["passed"] and agreement["passed"] and coverage["matches_simultaneous_names"] and coverage["disjoint"])
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "validate",
        "candidate": BASELINE_ID,
        "post_sft_weights_loaded": False,
        "no_optimizer_constructed": True,
        "no_parameter_updates": True,
        "no_structural_metrics_recomputed": True,
        "validation": {
            "status": "PASS" if passed else "FAIL",
            "method": "simultaneous layers 0-2 versus three one-layer chunked backwards",
            "canonical_dropout_reference": {
                key: value for key, value in canonical.items() if key != "masks"
            },
            "simultaneous": simultaneous,
            "chunks": chunks,
            "coverage": coverage,
            "loss_consistency": losses,
            "proxy_agreement": agreement,
        },
    }


def smoke(args: argparse.Namespace) -> dict[str, Any]:
    validation_payload = load_fixed_validation(args.validation_results)
    canonical = capture_canonical_dropout_reference(args)
    chunks: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    for layer in range(28):
        chunks.append(
            score_chunk(
                args,
                [layer],
                include_auxiliary=layer == 0,
                label=f"chunk_layer_{layer}" + ("_with_fusion_projector" if layer == 0 else ""),
                canonical_masks=canonical["masks"],
            )
        )
    wall_seconds = time.perf_counter() - total_started
    all_names = set(chunks[0]["all_lora_parameter_names"])
    chunk_names = [name for chunk in chunks for name in chunk["target_lora_parameter_names"]]
    lora_groups = [chunk["score"]["proxy_groups"]["lora"] for chunk in chunks]
    lora_total = add_groups(lora_groups)
    first_groups = chunks[0]["score"]["proxy_groups"]
    fusion = first_groups["fusion_block"]
    projector = first_groups["mm_projector"]
    total = add_groups([lora_total, fusion, projector])
    losses = check_loss_consistency([float(canonical["loss"])] + [float(chunk["score"]["loss"]) for chunk in chunks])
    coverage = {
        "expected_lora_parameter_elements": chunks[0]["all_lora_parameter_elements"],
        "observed_lora_parameter_elements": int(lora_total["parameter_elements"]),
        "observed_lora_gradient_elements": int(lora_total["gradient_elements"]),
        "expected_lora_parameter_names": len(all_names),
        "chunk_name_count": len(chunk_names),
        "unique_chunk_names": len(set(chunk_names)),
        "lora_union_is_exact": set(chunk_names) == all_names,
        "lora_chunks_are_disjoint": len(chunk_names) == len(set(chunk_names)),
        "fusion_counted_once": sum(chunk["include_fusion_block"] for chunk in chunks) == 1,
        "mm_projector_counted_once": sum(chunk["include_mm_projector"] for chunk in chunks) == 1,
    }
    coverage["all_intended_trainable_parameters_counted_once"] = bool(
        coverage["lora_union_is_exact"]
        and coverage["lora_chunks_are_disjoint"]
        and coverage["fusion_counted_once"]
        and coverage["mm_projector_counted_once"]
        and int(lora_total["gradient_elements"]) == int(coverage["expected_lora_parameter_elements"])
    )
    all_pass = all(chunk["score"]["status"] == "PASS" for chunk in chunks)
    status = "PASS" if all_pass and losses["passed"] and coverage["all_intended_trainable_parameters_counted_once"] else "FAIL"
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "smoke",
        "status": status,
        "candidate": BASELINE_ID,
        "post_sft_weights_loaded": False,
        "no_optimizer_constructed": True,
        "no_parameter_updates": True,
        "no_structural_metrics_recomputed": True,
        "validation_source": str(args.validation_results),
        "validation_status": validation_payload["validation"]["status"],
        "validation_review": validation_payload["validation_review"],
        "rng_seed": args.rng_seed,
        "canonical_dropout_reference": {key: value for key, value in canonical.items() if key != "masks"},
        "loss_definition": "ordinary supervised causal-LM next-token cross entropy",
        "proxy_definitions": {
            "gradnorm": "sum_p ||dL/dp||_2",
            "snip": "sum_{p,i} |p_i*g_i|",
            "fisher": "sum_{p,i} g_i^2; empirical diagonal Fisher on this minibatch",
            "aggregation": "sum disjoint LoRA chunk contributions; fusion and mm_projector included only in chunk 0",
        },
        "chunks": chunks,
        "loss_consistency": losses,
        "coverage": coverage,
        "final_sft_trainable_groups": {
            "lora": lora_total,
            "fusion_block": fusion,
            "mm_projector": projector,
            "total": total,
        },
        "runtime": {
            "wall_seconds_including_model_reconstruction": wall_seconds,
            "sum_shared_forward_backward_seconds": sum(
                float(chunk["score"]["shared_forward_backward_runtime_seconds"]) for chunk in chunks
            ),
            "max_per_chunk_peak_gpu_memory": max_peaks(chunks),
        },
    }


def markdown(payload: dict[str, Any]) -> str:
    if payload["mode"] == "validate":
        validation = payload["validation"]
        return "\n".join(
            [
                "# Chunked exact-backward validation",
                "",
                f"Status: **{validation['status']}**",
                "",
                f"Loss consistency: `{json.dumps(validation['loss_consistency'])}`",
                "",
                f"Proxy agreement: `{json.dumps(validation['proxy_agreement'])}`",
                "",
            ]
        )
    groups = payload["final_sft_trainable_groups"]
    lines = [
        "# Pre-SFT Baseline chunked exact-backward proxy smoke",
        "",
        f"Status: **{payload['status']}**",
        "",
        "| Group | Parameters | GradNorm | SNIP | Fisher |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("lora", "fusion_block", "mm_projector", "total"):
        group = groups[name]
        lines.append(
            f"| {name} | {group['parameter_elements']} | {group['gradnorm']:.9g} | "
            f"{group['snip']:.9g} | {group['fisher']:.9g} |"
        )
    lines.extend(["", f"Coverage: `{json.dumps(payload['coverage'])}`", "", f"Losses: `{json.dumps(payload['loss_consistency'])}`", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This chunked proxy smoke requires CUDA")
    args.output_root.mkdir(parents=True, exist_ok=False)
    payload = validate(args) if args.mode == "validate" else smoke(args)
    write_json(args.output_root / "results.json", payload)
    (args.output_root / "summary.md").write_text(markdown(payload), encoding="utf-8")
    print(json.dumps({"mode": args.mode, "status": payload.get("status", payload.get("validation", {}).get("status")), "output_root": str(args.output_root)}))


if __name__ == "__main__":
    main()
