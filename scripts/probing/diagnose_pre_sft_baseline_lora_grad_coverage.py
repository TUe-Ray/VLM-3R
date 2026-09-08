#!/usr/bin/env python
"""Read-only, per-LoRA gradient-coverage diagnosis for the pre-SFT Baseline.

This is deliberately not a proxy run: it performs one fixed-minibatch
training-mode CE backward solely because ``Parameter.grad`` is transient and
the prior smoke has already released its model.  It neither reduces proxy
scores nor constructs an optimizer nor changes a parameter value.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

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


BASELINE_ID = "c1_vlm3r_native"
DEFAULT_OUTPUT = Path(
    "/home/shaoruei/probe_outputs/pre_sft_zero_cost_proxies_v2/"
    "diagnose_baseline_retry6_lora_gradient_coverage.json"
)
LAYER_PATTERN = re.compile(r"\.model\.layers\.(\d+)\.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
    parser.add_argument("--rng-seed", type=int, default=42)
    args = parser.parse_args()
    args.calibration_batches = 1
    # make_load_args also accepts this uniform fallback, although the explicit
    # asymmetric budgets above are authoritative for this diagnosis.
    args.pre_sft_gpu_weight_budget = "5GiB"
    args.attn_implementation = None
    # The shared loader records a candidate-specific runtime root even though
    # this diagnostic itself writes only the inventory JSON below.
    args.output_root = args.output.parent / "runtime_lora_gradient_coverage"
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {args.output}")
    return args


def layer_index(name: str) -> int:
    match = LAYER_PATTERN.search(name)
    if match is None:
        raise RuntimeError(f"LoRA parameter is not under a decoder layer: {name}")
    return int(match.group(1))


def parent_layer_map(device_map: dict[str, str], index: int) -> str | None:
    suffix = f".model.layers.{index}"
    matches = [value for key, value in device_map.items() if key.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one device-map entry for decoder layer {index}, got {matches}")
    return matches[0]


def inventory(
    model: torch.nn.Module,
    device_map: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        if ".lora_A." not in name and ".lora_B." not in name:
            continue
        index = layer_index(name)
        mapped_device = parent_layer_map(device_map, index)
        gradient = parameter.grad
        rows.append(
            {
                "parameter_name": name,
                "decoder_layer_index": index,
                "parameter_device_after_backward": str(parameter.device),
                "parent_layer_device_map": mapped_device,
                "parent_layer_cpu_offloaded": mapped_device == "cpu",
                "requires_grad_after_backward": bool(parameter.requires_grad),
                "grad_present_after_backward": gradient is not None,
                "grad_device": str(gradient.device) if gradient is not None else None,
                "parameter_elements": int(parameter.numel()),
                "gradient_elements": int(gradient.numel()) if gradient is not None else 0,
            }
        )
    if not rows:
        raise RuntimeError("No LoRA parameters found after construction")
    return rows


def count(rows: list[dict[str, Any]], predicate) -> dict[str, int]:
    selected = [row for row in rows if predicate(row)]
    return {"parameter_tensors": len(selected), "parameter_elements": sum(row["parameter_elements"] for row in selected)}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    candidate = common.BY_IDENTIFIER[BASELINE_ID]
    cut3r_layers, llm_layers, artifact = common.candidate_schedule(candidate)
    device = torch.device(args.device)
    load_args = common.make_load_args(args, candidate, cut3r_layers, llm_layers)
    load_args.pre_sft_defer_dispatch = True

    common.reset_proxy_rng(args.rng_seed)
    tokenizer, model, image_processor = load_model(load_args, device, torch.float16)
    try:
        apply_c1_calibration_artifact(model, artifact)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model, lora_recipe = common.attach_intended_sft_lora(model, seed=args.rng_seed)
        model = dispatch_deferred_pre_sft_model(model, load_args, device)
        groups = common.intended_sft_trainable_groups(model)
        selected = common.configure_intended_sft_trainable_parameters(model, groups)
        selected_versions = {id(parameter): int(parameter._version) for parameter in selected}
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        install_forward_frame_loader(args.forward_frames_root)
        dataset, collator, by_video = build_dataset(load_args, tokenizer, image_processor)
        records = common.load_calibration_records(args, by_video)
        if len(records) != 1:
            raise RuntimeError(f"Diagnostic requires exactly one minibatch, got {len(records)}")
        record = records[0]
        batch = common.prepare_batch(dataset, collator, by_video[str(record["video_path"])], model, device, torch.float16)

        common.reset_cuda_peaks()
        model.train(True)
        model.config.use_cache = False
        rng_reset = common.reset_proxy_rng(args.rng_seed)
        common.synchronize_cuda()
        started = time.perf_counter()
        with common.proxy_supervised_logits_only(model):
            output = model(**batch, use_cache=False, return_dict=True)
        loss = getattr(output, "loss", None)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"Diagnostic forward returned invalid CE loss: {loss}")
        loss.backward()
        common.synchronize_cuda()
        elapsed = time.perf_counter() - started

        device_map = dict(getattr(model, "_pre_sft_deferred_dispatch_device_map", {}))
        rows = inventory(model, device_map)
        versions_unchanged = selected_versions == {
            id(parameter): int(parameter._version) for parameter in selected
        }
        missing = lambda row: not row["grad_present_after_backward"]
        resident = lambda row: not row["parent_layer_cpu_offloaded"]
        offloaded = lambda row: row["parent_layer_cpu_offloaded"]
        output_payload = {
            "schema_version": "pre_sft_lora_gradient_coverage_diagnostic_v1",
            "diagnostic_only": True,
            "no_proxy_metrics_computed": True,
            "no_optimizer_constructed": True,
            "no_optimizer_step": True,
            "no_parameter_updates": True,
            "parameter_versions_unchanged": versions_unchanged,
            "candidate": candidate.identifier,
            "post_sft_weights_loaded": False,
            "lora_recipe": lora_recipe,
            "calibration": common.batch_metadata(batch, record),
            "rng_reset": rng_reset,
            "dispatch": {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu_weight_budgets": args.pre_sft_gpu_weight_budgets,
                "cpu_offload_budget": args.pre_sft_cpu_offload_budget,
                "effective_device_map": device_map,
            },
            "backward": {
                "loss": float(loss.detach().float().item()),
                "runtime_seconds": elapsed,
                "peak_gpu_memory": common.cuda_peaks(),
            },
            "lora_parameters": rows,
            "summary": {
                "all_lora": count(rows, lambda _row: True),
                "with_gradient": count(rows, lambda row: not missing(row)),
                "missing_gradient": count(rows, missing),
                "gpu_resident": count(rows, resident),
                "gpu_resident_missing_gradient": count(rows, lambda row: resident(row) and missing(row)),
                "cpu_offloaded": count(rows, offloaded),
                "cpu_offloaded_with_gradient": count(rows, lambda row: offloaded(row) and not missing(row)),
                "cpu_offloaded_missing_gradient": count(rows, lambda row: offloaded(row) and missing(row)),
            },
        }
        write_json(args.output, output_payload)
        print(f"Wrote diagnostic: {args.output}")
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
