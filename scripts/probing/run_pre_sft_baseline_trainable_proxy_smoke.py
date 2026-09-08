#!/usr/bin/env python
"""One-minibatch, read-only pre-SFT Baseline proxy smoke.

This intentionally does not run structural metrics or the multi-architecture
sweep.  Its primary score is the exact intended SFT trainable partition
(fresh LoRA + C1 fusion block + mm_projector); its control is all materialized
parameters, attempted only if the primary scope passes.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PROBING_DIR = REPO_ROOT / "scripts" / "probing"
if str(PROBING_DIR) not in sys.path:
    sys.path.insert(0, str(PROBING_DIR))

import evaluate_pre_sft_zero_cost_proxies as common  # noqa: E402
from depth_probe_common import write_csv, write_json  # noqa: E402
from extract_depth_probe_features import build_dataset  # noqa: E402
from local_depth_probe_cache import install_forward_frame_loader  # noqa: E402
from scripts.diagnose_layerwise_spatial_hidden_scan import (  # noqa: E402
    dispatch_deferred_pre_sft_model,
    load_model,
)
from llava.model.c1_structured_isometry import apply_c1_calibration_artifact  # noqa: E402


SCHEMA_VERSION = "pre_sft_baseline_sft_trainable_proxy_smoke_v1"
BASELINE_ID = "c1_vlm3r_native"
DEFAULT_OUTPUT = Path("/home/shaoruei/probe_outputs/pre_sft_zero_cost_proxies_v2/smoke_baseline")
DEFAULT_POST_SFT_ARTIFACT = Path(
    "/home/shaoruei/probe_outputs/post_sft_3d_zero_cost_proxies_v1/complete/results.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-model", type=Path, default=common.DEFAULT_BASE_MODEL)
    parser.add_argument("--siglip-model", type=Path, default=common.DEFAULT_SIGLIP_MODEL)
    parser.add_argument("--forward-frames-root", type=Path, default=common.DEFAULT_FORWARD_ROOT)
    parser.add_argument("--probe-targets-root", type=Path, default=common.DEFAULT_TARGET_ROOT)
    parser.add_argument("--feature-root", type=Path, default=common.DEFAULT_FEATURE_ROOT)
    parser.add_argument("--data-yaml", type=Path, default=common.DEFAULT_DATA_YAML)
    parser.add_argument("--sample-indices", type=Path, default=common.DEFAULT_SAMPLE_INDICES)
    parser.add_argument("--post-sft-artifact", type=Path, default=DEFAULT_POST_SFT_ARTIFACT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", choices=("auto", "cuda:0", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--pre-sft-gpu-weight-budget", default="5GiB")
    parser.add_argument(
        "--pre-sft-gpu-weight-budgets",
        default=None,
        help="Optional comma-separated per-visible-GPU weight budgets, e.g. 4GiB,6GiB.",
    )
    parser.add_argument("--pre-sft-cpu-offload-budget", default="45GiB")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--rng-seed", type=int, default=42)
    args = parser.parse_args()
    # The fixed-manifest helper is shared with the multi-candidate evaluator;
    # this dedicated smoke deliberately has exactly one supervised minibatch.
    args.calibration_batches = 1
    return args


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(jsonable(item) for item in value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def validate(args: argparse.Namespace) -> None:
    required = (
        args.base_model,
        args.siglip_model,
        args.forward_frames_root,
        args.probe_targets_root,
        args.feature_root,
        args.data_yaml,
        args.sample_indices,
        args.post_sft_artifact,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required smoke inputs: {missing}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for this smoke but is unavailable")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing smoke output: {args.output_root}")


def post_sft_cost_reference(path: Path) -> dict[str, Any]:
    payload = common.read_json(path)
    if not isinstance(payload, list):
        raise TypeError(f"Expected list-valued post-SFT results: {path}")
    baseline = next(
        (item for item in payload if item.get("structural", {}).get("candidate") == "baseline"),
        None,
    )
    if baseline is None:
        raise RuntimeError("Post-SFT reference lacks the Baseline candidate")
    minibatch = baseline["calibration"]["minibatches"][0]
    candidate_scope = minibatch["candidate_specific"]
    return {
        "source": str(path),
        "source_sha256": common.sha256_file(path),
        "batch": {
            "input_ids_shape": minibatch["input_ids_shape"],
            "supervised_label_tokens": minibatch["supervised_label_tokens"],
            "frames_upbound": minibatch["frames_upbound"],
            "scene_id": minibatch["scene_id"],
        },
        "successful_candidate_specific": {
            "status": candidate_scope["status"],
            "included_parameter_elements": candidate_scope["parameters_with_gradient"],
            "shared_backward_runtime_seconds": candidate_scope["runtime_seconds"],
            "peak_gpu_memory": candidate_scope["peak_gpu_memory"],
            "shared_backward_pass": candidate_scope.get("shared_backward_pass"),
        },
        "whole_model": "not attempted in completed post-SFT sweep; an earlier baseline smoke attempted it and OOMed",
        "trainable_parameter_scope": "not attempted; post-SFT score used only checkpoint-specific non-LoRA parameters",
        "path_difference": (
            "The smoke below uses fresh pre-SFT C1 VLM3R and fresh LoRA; post-SFT values "
            "are cost evidence only, not proxy-score comparators."
        ),
    }


def scope_row(scope: str, payload: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scope": scope,
        "status": payload["status"],
        "shared_forward_backward_runtime_seconds": payload.get("shared_forward_backward_runtime_seconds"),
        "peak_gpu_allocated_bytes": max(
            (int(item["peak_allocated_bytes"]) for item in payload.get("peak_gpu_memory", [])), default=None
        ),
        "peak_gpu_reserved_bytes": max(
            (int(item["peak_reserved_bytes"]) for item in payload.get("peak_gpu_memory", [])), default=None
        ),
    }
    total = payload.get("proxy_groups", {}).get("total", {})
    row.update(
        {
            "parameter_elements": total.get("parameter_elements", payload.get("selected_parameter_elements")),
            "parameters_with_gradient": total.get("parameters_with_gradient"),
            "gradnorm": total.get("gradnorm"),
            "snip": total.get("snip"),
            "fisher": total.get("fisher"),
            "gradnorm_reduction_seconds": payload.get("metric_reduction_runtime_seconds", {}).get("gradnorm"),
            "snip_reduction_seconds": payload.get("metric_reduction_runtime_seconds", {}).get("snip"),
            "fisher_reduction_seconds": payload.get("metric_reduction_runtime_seconds", {}).get("fisher"),
        }
    )
    return row


def markdown(result: dict[str, Any]) -> str:
    def scalar(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    lines = [
        "# Pre-SFT Baseline SFT-trainable proxy smoke",
        "",
        "One fixed supervised QA minibatch; no optimizer was constructed and no parameter was updated.",
        "No Params, Trainable Params, or FLOPs were recomputed in this smoke.",
        "",
        "## Fixed construction",
        "",
        f"- Candidate: `{result['candidate']['identifier']}`",
        f"- C1 artifact SHA-256: `{result['candidate']['c1_artifact_sha256']}`",
        f"- PEFT version: `{result['lora_initialization']['peft_version']}`",
        "- PEFT `init_lora_weights` was not passed explicitly; the installed PEFT default was used.",
        "- Primary scope: LoRA + C1 fusion block + mm_projector.",
        "",
        "## Proxy definitions",
        "",
        "- GradNorm = `sum_p ||dL/dp||_2`",
        "- SNIP = `sum_{p,i} |p_i * g_i|`",
        "- Fisher = `sum_{p,i} g_i^2` (empirical diagonal Fisher on this minibatch)",
        "",
        "Each scope uses one shared training-mode forward/backward; the three reductions are then computed from the same gradients.",
        "",
        "## Scope results",
        "",
        "| Scope | Status | Params | GradNorm | SNIP | Fisher | Shared fwd/bwd s | Peak allocated | Peak reserved |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["scope_rows"]:
        lines.append(
            "| {scope} | {status} | {params} | {gradnorm} | {snip} | {fisher} | {runtime} | {allocated} | {reserved} |".format(
                scope=row["scope"], status=row["status"], params=scalar(row.get("parameter_elements")),
                gradnorm=scalar(row.get("gradnorm")), snip=scalar(row.get("snip")), fisher=scalar(row.get("fisher")),
                runtime=scalar(row.get("shared_forward_backward_runtime_seconds")),
                allocated=common.human_count(row["peak_gpu_allocated_bytes"]) if row.get("peak_gpu_allocated_bytes") else "—",
                reserved=common.human_count(row["peak_gpu_reserved_bytes"]) if row.get("peak_gpu_reserved_bytes") else "—",
            )
        )
    primary = result["scopes"]["sft_trainable"]
    if primary["status"] == "PASS":
        lines.extend(
            [
                "",
                "## Primary SFT-trainable breakdown",
                "",
                "| Group | # parameters | GradNorm | SNIP | Fisher |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        labels = (("lora", "LoRA"), ("fusion_block", "fusion block"), ("mm_projector", "mm_projector"), ("total", "total SFT-trainable"))
        for key, label in labels:
            values = primary["proxy_groups"][key]
            lines.append(
                f"| {label} | {values['parameter_elements']} | {scalar(values['gradnorm'])} | "
                f"{scalar(values['snip'])} | {scalar(values['fisher'])} |"
            )
        lines.extend(
            [
                "",
                "Fresh standard LoRA commonly has random A and zero B. Therefore its initial SNIP contribution can be exactly zero; this is expected rather than a proxy failure.",
            ]
        )
    lines.extend(
        [
            "",
            "## RNG and stopping rule",
            "",
            f"RNG seed reset immediately before each attempted proxy forward: `{result['rng_seed']}`.",
            "Full-model was attempted only after a passing primary scope. A primary OOM skips full-model by design.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    validate(args)
    candidate = common.BY_IDENTIFIER[BASELINE_ID]
    cut3r_layers, llm_layers, artifact = common.candidate_schedule(candidate)
    args.output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]
    load_args = common.make_load_args(args, candidate, cut3r_layers, llm_layers)
    # Fresh adapters must be inserted before Accelerate offload dispatch; see
    # dispatch_deferred_pre_sft_model for the meta-device safety rationale.
    load_args.pre_sft_defer_dispatch = True
    common.reset_proxy_rng(args.rng_seed)
    tokenizer, model, image_processor = load_model(load_args, device, dtype)
    try:
        preexisting_lora = [name for name, _ in model.named_parameters() if "lora_" in name]
        if preexisting_lora:
            raise RuntimeError("Pre-SFT smoke base unexpectedly contains preexisting LoRA parameters")
        apply_c1_calibration_artifact(model, artifact)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model, lora_recipe = common.attach_intended_sft_lora(model, seed=args.rng_seed)
        # Capture this before CPU-offload dispatch replaces inactive adapter
        # tensors with meta placeholders.
        lora_actual_state = common.lora_initialization_summary(model)
        model = dispatch_deferred_pre_sft_model(model, load_args, device)
        groups = common.intended_sft_trainable_groups(model)
        sft_trainable = common.configure_intended_sft_trainable_parameters(model, groups)
        group_ids = {id(parameter) for parameters in groups.values() for parameter in parameters}
        if group_ids != {id(parameter) for parameter in sft_trainable}:
            raise RuntimeError("Primary scope differs from LoRA + fusion block + mm_projector union")
        intended_scope_elements = {
            "sft_trainable": common.parameter_count(sft_trainable),
            "full_model": common.parameter_count(parameter for parameter in model.parameters() if not parameter.is_meta),
        }
        # Accelerate dispatch wraps modules after the initial training-style
        # hook installation. Reinstall those hooks on the PEFT wrapper so the
        # first re-entrant checkpoint receives a differentiable embedding
        # output while all base weights remain frozen.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        install_forward_frame_loader(args.forward_frames_root)
        dataset, collator, by_video = build_dataset(load_args, tokenizer, image_processor)
        records = common.load_calibration_records(args, by_video)
        if len(records) != 1:
            raise RuntimeError(f"Smoke requires exactly one minibatch, got {len(records)}")
        record = records[0]
        batch = common.prepare_batch(dataset, collator, by_video[str(record["video_path"])], model, device, dtype)
        primary = common.run_grouped_backward_scope(
            model, batch, "sft_trainable", groups, rng_seed=args.rng_seed
        )
        primary["requires_grad_flags_restored"] = bool(
            getattr(model, "_last_proxy_requires_grad_restored", False)
        )
        primary["selected_parameter_elements"] = intended_scope_elements["sft_trainable"]
        scopes: dict[str, dict[str, Any]] = {"sft_trainable": primary}
        if primary["status"] == "PASS":
            whole = [parameter for parameter in model.parameters() if not parameter.is_meta]
            scopes["full_model"] = common.run_grouped_backward_scope(
                model,
                batch,
                "full_model",
                {"all_model_parameters": whole},
                rng_seed=args.rng_seed,
            )
            scopes["full_model"]["requires_grad_flags_restored"] = bool(
                getattr(model, "_last_proxy_requires_grad_restored", False)
            )
            scopes["full_model"]["selected_parameter_elements"] = intended_scope_elements["full_model"]
        else:
            scopes["full_model"] = {
                "status": "SKIPPED_PRIMARY_NOT_PASS",
                "scope": "full_model",
                "reason": "full-model scope is strictly larger than a non-passing primary SFT-trainable scope",
                "selected_parameter_elements": intended_scope_elements["full_model"],
                "no_optimizer_constructed": True,
                "no_weight_update": True,
            }
        try:
            peft_version = importlib.metadata.version("peft")
        except importlib.metadata.PackageNotFoundError:
            peft_version = "unknown"
        lora_init = {
            "peft_version": peft_version,
            "recipe": jsonable(lora_recipe),
            "actual_state": lora_actual_state,
        }
        candidate_info = {
            "identifier": candidate.identifier,
            "display_name": "pre-SFT Baseline",
            "base_model": str(args.base_model),
            "base_model_config_sha256": common.sha256_file(args.base_model / "config.json"),
            "c1_artifact": str(candidate.calibration_artifact),
            "c1_artifact_sha256": common.sha256_file(candidate.calibration_artifact),
            "post_sft_weights_loaded": False,
        }
        dispatch_info = {
            "requested_device_map": args.device_map,
            "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_weight_budgets": args.pre_sft_gpu_weight_budgets or args.pre_sft_gpu_weight_budget,
            "cpu_offload_budget": args.pre_sft_cpu_offload_budget,
            "effective_device_map": getattr(model, "_pre_sft_deferred_dispatch_device_map", None),
        }
        result = {
            "schema_version": SCHEMA_VERSION,
            "candidate": candidate_info,
            "dispatch": dispatch_info,
            "calibration": {
                **common.batch_metadata(batch, record),
                "sample_indices": str(args.sample_indices),
                "sample_indices_sha256": common.sha256_file(args.sample_indices),
                "loss_definition": "ordinary supervised causal-LM next-token cross entropy",
                "calibration_minibatches": 1,
            },
            "rng_seed": int(args.rng_seed),
            "lora_initialization": lora_init,
            "scope_definition": {
                "sft_trainable": "fresh LoRA + C1 fusion block + mm_projector",
                "full_model": "all materialized model parameters",
            },
            "proxy_definitions": {
                "gradnorm": "sum_p ||dL/dp||_2",
                "snip": "sum_{p,i} |p_i*g_i|",
                "fisher": "sum_{p,i} g_i^2; empirical diagonal Fisher on this minibatch",
                "execution": "one shared backward pass per scope; reductions use the same gradients",
            },
            "no_training": {"optimizer_constructed": False, "optimizer_step_called": False, "parameter_updates": False},
            "post_sft_cost_reference": post_sft_cost_reference(args.post_sft_artifact),
            "scopes": scopes,
        }
        result["scope_rows"] = [scope_row(name, payload) for name, payload in scopes.items()]
        write_json(args.output_root / "results.json", result)
        write_csv(args.output_root / "proxy_scores.csv", result["scope_rows"])
        (args.output_root / "summary.md").write_text(markdown(result), encoding="utf-8")
        write_json(
            args.output_root / "metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "PASS" if primary["status"] == "PASS" else primary["status"],
                "candidate": candidate.identifier,
                "calibration_minibatches": 1,
                "full_model_attempted": primary["status"] == "PASS",
                "no_structural_metrics_recomputed": True,
                "no_optimizer_constructed": True,
                "no_weight_updates": True,
            },
        )
        print(json.dumps({"status": result["scopes"]["sft_trainable"]["status"], "output_root": str(args.output_root)}))
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
