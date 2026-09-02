#!/usr/bin/env python
"""Evaluate structural and gradient zero-cost proxies for the C1 pre-SFT study.

This is deliberately a *read-only* experiment.  It recreates the five
architectures used by the current pre-SFT C1 depth-probe study from their
official calibration artifacts, evaluates the ordinary supervised causal-LM
loss on a fixed calibration minibatch, and never constructs an optimizer or
updates a parameter.

The backward proxies use these conventional scalar definitions for a loss L:

* GradNorm = sum_p ||dL/dp||_2
* SNIP = sum_p |p * dL/dp|
* Fisher = sum_p (dL/dp)^2  (empirical diagonal Fisher, one minibatch)

``whole_model`` is the standard all-parameter score.  It can be too large for
the 2x12-GiB local setup, so an OOM in that optional scope is recorded rather
than retried for every candidate.  ``candidate_fusion`` always means only the
fresh fusion parameters that differ between candidates.  Those parameters are
temporarily made differentiable solely for autograd; their values are not
changed and their original ``requires_grad`` flags are restored afterwards.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PROBING_DIR = REPO_ROOT / "scripts" / "probing"
if str(PROBING_DIR) not in sys.path:
    sys.path.insert(0, str(PROBING_DIR))

from depth_probe_common import read_json, write_csv, write_json  # noqa: E402
from extract_depth_probe_features import build_dataset  # noqa: E402
from local_depth_probe_cache import install_forward_frame_loader  # noqa: E402
from scripts.diagnose_layerwise_spatial_hidden_scan import load_model, move_to_device  # noqa: E402
from llava.model.c1_structured_isometry import apply_c1_calibration_artifact  # noqa: E402


SCHEMA_VERSION = "pre_sft_zero_cost_proxy_v1"
DEFAULT_BASE_MODEL = Path("/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2")
DEFAULT_SIGLIP_MODEL = Path("/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384")
DEFAULT_FORWARD_ROOT = Path("/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1")
DEFAULT_TARGET_ROOT = Path("/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1")
DEFAULT_FEATURE_ROOT = Path("/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features")
DEFAULT_DATA_YAML = REPO_ROOT / "scripts" / "probing" / "scannet_depth_probe_local_data.yaml"
DEFAULT_SAMPLE_INDICES = Path(
    "/home/shaoruei/probe_provenance/scannet_baseline_L6/"
    "scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json"
)
DEFAULT_OUTPUT_ROOT = Path("/home/shaoruei/probe_outputs/pre_sft_zero_cost_proxies_v1")


@dataclass(frozen=True)
class CandidateSpec:
    """One architecture exactly represented in the C1 pre-SFT study."""

    identifier: str
    probe_label: str
    fusion_variant: str
    calibration_artifact: Path
    vsi_model: str
    vsi_avg: float


# The schedule is loaded from each immutable official C1 artifact rather than
# duplicated here.  This protects against accidentally scoring a same-shaped
# but differently injected topology.
CANDIDATES = (
    CandidateSpec(
        "c1_ss_add_012",
        "c1_spatialstack_add",
        "c1_ss_add",
        Path("/home/shaoruei/probe_outputs/c1_additive_v1/official/spatialstack_add.json"),
        "Spatial Stack to Layer 0/1/2",
        61.2,
    ),
    CandidateSpec(
        "c1_ss_add_036",
        "c1_spatialstack_add_036",
        "c1_ss_add",
        Path("/home/shaoruei/probe_outputs/c1_ss_add_036/official/spatialstack_add.json"),
        "Spatial Stack to Layer 0/3/6",
        61.2,
    ),
    CandidateSpec(
        "c1_ss_add_123",
        "c1_spatialstack_add_123",
        "c1_ss_add",
        Path("/home/shaoruei/probe_outputs/c1_ss_add_123/official/spatialstack_add.json"),
        "Spatial Stack to Layer 1/2/3",
        62.2,
    ),
    CandidateSpec(
        "c1_ss_cross_attn_012",
        "c1_spatialstack_cross_attn_v1",
        "c1_ss_cross_attn_v1",
        Path("/home/shaoruei/probe_outputs/c1_ss_cross_attn_v1/official/spatialstack_cross_attn_v1.json"),
        "Spatial Stack- Cross Attn",
        60.6,
    ),
    CandidateSpec(
        "c1_vlm3r_native",
        "c1_vlm3r",
        "c1_vlm3r",
        Path("/home/shaoruei/probe_outputs/c1_vlm3r_v1/official/vlm3r.json"),
        "Baseline",
        59.3,
    ),
)
BY_IDENTIFIER = {item.identifier: item for item in CANDIDATES}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def bool_config(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def human_count(value: int | float) -> str:
    value = float(value)
    for suffix, divisor in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(value) >= divisor:
            return f"{value / divisor:.3f}{suffix}"
    return f"{value:.0f}"


def module_device(module: Any, fallback: torch.device) -> torch.device:
    declared = getattr(module, "device", None)
    if isinstance(declared, torch.device) and declared.type != "meta":
        return declared
    if module is not None:
        for parameter in module.parameters():
            if not parameter.is_meta:
                return parameter.device
    return fallback


def move_value(value: Any, target: torch.device, dtype: torch.dtype | None = None) -> Any:
    if torch.is_tensor(value):
        requested_dtype = dtype if dtype is not None and value.is_floating_point() else None
        return value.to(device=target, dtype=requested_dtype, non_blocking=True)
    if isinstance(value, list):
        return [move_value(item, target, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(move_value(item, target, dtype) for item in value)
    if isinstance(value, dict):
        return {key: move_value(item, target, dtype) for key, item in value.items()}
    return value


def candidate_schedule(candidate: CandidateSpec) -> tuple[str, str, dict[str, Any]]:
    """Read and validate the exact source/injection schedule from C1 evidence."""
    if not candidate.calibration_artifact.is_file():
        raise FileNotFoundError(f"Missing official C1 artifact for {candidate.identifier}: {candidate.calibration_artifact}")
    artifact = read_json(candidate.calibration_artifact)
    architecture = str(artifact.get("architecture", ""))
    expected_architecture = {
        "c1_ss_add": "spatialstack_add",
        "c1_ss_cross_attn_v1": "spatialstack_cross_attn_v1",
        "c1_vlm3r": "vlm3r",
    }[candidate.fusion_variant]
    if architecture != expected_architecture:
        raise RuntimeError(
            f"{candidate.identifier} artifact architecture mismatch: expected {expected_architecture}, got {architecture}"
        )
    spatialstack = artifact.get("spatialstack")
    if not isinstance(spatialstack, dict):
        raise RuntimeError(f"{candidate.identifier} C1 artifact has no spatialstack schedule")
    source_layers = spatialstack.get("cut3r_source_layers")
    llm_layers = spatialstack.get("llm_injection_layers")
    if not isinstance(source_layers, list) or not isinstance(llm_layers, list):
        raise RuntimeError(f"{candidate.identifier} C1 artifact lacks source/LLM layer lists")
    if candidate.fusion_variant != "c1_vlm3r" and len(source_layers) != len(llm_layers):
        raise RuntimeError(f"{candidate.identifier} C1 source/LLM schedule lengths differ")
    return ",".join(str(int(value)) for value in source_layers), ",".join(str(int(value)) for value in llm_layers), artifact


def make_load_args(args: argparse.Namespace, candidate: CandidateSpec, cut3r_layers: str, llm_layers: str) -> SimpleNamespace:
    """Arguments consumed by the existing pre-SFT model loader, unchanged in meaning."""
    # SpatialStack consumes three decoder-layer sidecars; native VLM3R instead
    # consumes the historical final-layer sidecar directly (one camera token
    # plus 729 patches per frame).  Do not wrap that latter payload in the
    # SpatialStack ``cut3r_dec_layers`` schema.
    spatial_features_subdir = (
        "spatial_features"
        if candidate.fusion_variant == "c1_vlm3r"
        else "6:spatial_features_dec_6,9:spatial_features_dec_9,12:spatial_features"
    )
    return SimpleNamespace(
        model_label=candidate.probe_label,
        model_path=str(args.base_model),
        model_base=None,
        model_name="vlm-3r-llava-qwen2-lora",
        model_loading_mode="pre_sft_fusion",
        pre_sft_fusion_variant=candidate.fusion_variant,
        fusion_init_seed=0,
        common_model_init_seed=0,
        spatialstack_cut3r_layers=cut3r_layers,
        spatialstack_llm_layers=llm_layers,
        siglip_path=str(args.siglip_model),
        cut3r_weights=None,
        runtime_root=str(args.output_root / "runtime" / candidate.identifier),
        device_map=args.device_map,
        pre_sft_gpu_weight_budget=args.pre_sft_gpu_weight_budget,
        pre_sft_cpu_offload_budget=args.pre_sft_cpu_offload_budget,
        attn_implementation=args.attn_implementation,
        skip_spatial_tower_load=None,
        zero_spatial_features=False,
        mm_spatial_pool_stride=2,
        pool_mode="bilinear",
        dtype=args.dtype,
        cache_dtype=args.dtype,
        train_data_json=str(args.data_yaml),
        data_yaml=str(args.data_yaml),
        feature_root=str(args.feature_root),
        spatial_features_subdir=spatial_features_subdir,
        spatial_feature_dir=str(args.feature_root),
        image_folder=str(args.forward_frames_root),
        video_folder=str(args.forward_frames_root),
        frames_upbound=32,
        add_time_instruction=None,
        seed=0,
        model_loading_info=None,
        feature_preset="original",
        geometry_spatial_features_root=None,
        geometry_spatial_features_subdir=None,
        geometry_point_map_key="point_maps_ref",
        post_sft_architecture=None,
    )


def fusion_module(model: nn.Module) -> nn.Module:
    # A PEFT causal-LM wrapper retains the original LLaVA object as its base
    # model.  Resolve that object first so the pre-SFT C1 fusion modules are
    # found identically before and after LoRA construction.
    get_base_model = getattr(model, "get_base_model", None)
    base_model = (
        get_base_model()
        if hasattr(model, "peft_config") and callable(get_base_model)
        else model
    )
    base = base_model.get_model()
    merger = getattr(base, "get_cut3r_spatialstack_merger", lambda: None)()
    module = merger if merger is not None else getattr(base, "get_fusion_block", lambda: None)()
    if module is None:
        raise RuntimeError("Candidate model has neither a SpatialStack merger nor a fusion block")
    return module


def parameter_count(parameters: Iterable[nn.Parameter]) -> int:
    return sum(int(parameter.numel()) for parameter in parameters)


def historical_lora_target_linear_names(model: nn.Module) -> list[str]:
    """Exact local copy of ``llava.train.train.find_all_linear_names``.

    Importing the trainer only for this helper imports DeepSpeed, whose CUDA
    extension discovery requires CUDA_HOME on this host.  Keeping this short,
    source-identical selection rule here makes the structural count depend on
    the same SFT recipe without making the proxy experiment depend on a
    training-only launcher dependency.
    """
    multimodal_keywords = (
        "mm_projector",
        "vision_tower",
        "vision_resampler",
        "spatial_tower",
        "fusion_block",
        "geometry_aware_projection",
        "cut3r_spatialstack",
        "cut3r_camera_token_projector",
        "bev_head",
        "depth_head",
        "pointmap_head",
        "spatial_bridge_tokens",
    )
    names = {
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and not any(keyword in name for keyword in multimodal_keywords)
    }
    names.discard("lm_head")
    return list(names)


def reset_proxy_rng(seed: int) -> dict[str, Any]:
    """Reset every RNG participating in the training-mode proxy forward."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_seeded = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cuda_seeded = True
    return {
        "python": int(seed),
        "numpy": int(seed),
        "torch_cpu": int(seed),
        "torch_cuda_all": int(seed) if cuda_seeded else None,
    }


def attach_intended_sft_lora(
    model: nn.Module,
    *,
    seed: int,
    rank: int = 128,
    alpha: int = 256,
    dropout: float = 0.05,
    bias: str = "none",
) -> tuple[nn.Module, dict[str, Any]]:
    """Attach fresh LoRA using the current SFT recipe's PEFT defaults.

    The training path calls ``LoraConfig`` without ``init_lora_weights``.  Do
    the same here: the installed PEFT version owns the initialization default,
    and the resulting configuration/state are recorded by the smoke runner.
    """
    from peft import LoraConfig, get_peft_model

    reset_proxy_rng(seed)
    targets = sorted(historical_lora_target_linear_names(model))
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=targets,
        lora_dropout=dropout,
        bias=bias,
        task_type="CAUSAL_LM",
    )
    wrapped = get_peft_model(model, config)
    return wrapped, {
        "construction_seed": int(seed),
        "rank": int(rank),
        "alpha": int(alpha),
        "dropout": float(dropout),
        "bias": bias,
        "target_modules": targets,
        "target_module_count": len(targets),
        "init_lora_weights_explicitly_passed": False,
        "peft_config": config.to_dict(),
    }


def intended_sft_trainable_groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    """Return the exact Baseline SFT groups: LoRA, fusion, projector."""
    lora = [
        parameter
        for name, parameter in model.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    ]
    if not lora:
        raise RuntimeError("Fresh SFT LoRA construction produced no lora_A/lora_B parameters")
    fusion = list(fusion_module(model).parameters())
    get_base_model = getattr(model, "get_base_model", None)
    base_model = (
        get_base_model()
        if hasattr(model, "peft_config") and callable(get_base_model)
        else model
    )
    projector = getattr(base_model.get_model(), "mm_projector", None)
    if not isinstance(projector, nn.Module):
        raise RuntimeError("Baseline SFT recipe requires a materialized mm_projector")
    mm_projector = list(projector.parameters())
    groups = {"lora": lora, "fusion_block": fusion, "mm_projector": mm_projector}
    identities = [id(parameter) for parameters in groups.values() for parameter in parameters]
    if len(identities) != len(set(identities)):
        raise RuntimeError("LoRA, fusion_block, and mm_projector scopes must be disjoint")
    if not all(parameters for parameters in groups.values()):
        empty = [name for name, parameters in groups.items() if not parameters]
        raise RuntimeError(f"Intended SFT trainable group is empty: {empty}")
    return groups


def configure_intended_sft_trainable_parameters(
    model: nn.Module,
    groups: Mapping[str, list[nn.Parameter]],
) -> list[nn.Parameter]:
    """Apply the training recipe's intended frozen/trainable partition."""
    selected = [parameter for parameters in groups.values() for parameter in parameters]
    selected_ids = {id(parameter) for parameter in selected}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in selected_ids)
    actual_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if actual_ids != selected_ids:
        raise RuntimeError("Runtime requires_grad partition differs from intended SFT trainable scope")
    return selected


def lora_initialization_summary(model: nn.Module) -> dict[str, Any]:
    """Lightweight evidence for the installed PEFT initialization outcome."""
    groups = {"lora_A": [], "lora_B": []}
    for name, parameter in model.named_parameters():
        if ".lora_A." in name:
            groups["lora_A"].append(parameter)
        elif ".lora_B." in name:
            groups["lora_B"].append(parameter)
    output: dict[str, Any] = {}
    for name, parameters in groups.items():
        elements = 0
        zeros = 0
        squared_norm = 0.0
        max_abs = 0.0
        for parameter in parameters:
            value = parameter.detach()
            elements += int(value.numel())
            zeros += int(value.numel() - torch.count_nonzero(value).item())
            squared_norm += float(value.float().square().sum().item())
            max_abs = max(max_abs, float(value.detach().abs().max().item()))
        output[name] = {
            "parameter_tensors": len(parameters),
            "parameter_elements": elements,
            "zero_elements": zeros,
            "nonzero_elements": elements - zeros,
            "l2_norm": math.sqrt(squared_norm),
            "max_abs": max_abs,
        }
    return output


def lora_parameter_count(model: nn.Module, rank: int) -> tuple[int, list[str]]:
    """Count exactly the rank-r LoRA matrices used by the historical wrapper.

    The train wrapper calls ``find_all_linear_names`` before attaching PEFT.
    Replicating the count algebraically avoids allocating roughly a half-billion
    LoRA parameters merely to report a structural baseline.
    """
    targets = sorted(historical_lora_target_linear_names(model))
    modules = dict(model.named_modules())
    total = 0
    for name in targets:
        module = modules.get(name)
        if not isinstance(module, nn.Linear):
            raise RuntimeError(f"LoRA target is not nn.Linear: {name} -> {type(module).__name__}")
        total += int(rank) * (int(module.in_features) + int(module.out_features))
    return total, targets


def structural_row(model: nn.Module, candidate: CandidateSpec, artifact: dict[str, Any], lora_rank: int) -> dict[str, Any]:
    fusion = fusion_module(model)
    total = parameter_count(model.parameters())
    fusion_total = parameter_count(fusion.parameters())
    runtime_trainable = parameter_count(parameter for parameter in model.parameters() if parameter.requires_grad)
    lora_total, lora_targets = lora_parameter_count(model, lora_rank)
    return {
        "candidate": candidate.identifier,
        "probe_label": candidate.probe_label,
        "fusion_variant": candidate.fusion_variant,
        "vsi_model": candidate.vsi_model,
        "vsi_avg": candidate.vsi_avg,
        "c1_artifact": str(candidate.calibration_artifact),
        "c1_artifact_sha256": sha256_file(candidate.calibration_artifact),
        "c1_architecture": artifact["architecture"],
        "total_params": total,
        "fusion_params": fusion_total,
        "runtime_trainable_params": runtime_trainable,
        "sft_lora_rank": lora_rank,
        "sft_lora_params": lora_total,
        "sft_lora_target_linear_modules": len(lora_targets),
        "sft_trainable_params": lora_total + fusion_total,
        "trainable_param_definition": "historical SFT: rank-128 LoRA target modules plus tuned candidate fusion; C1 runtime itself is frozen",
    }


def load_calibration_records(args: argparse.Namespace, by_video: dict[str, int]) -> list[dict[str, Any]]:
    payload = read_json(args.sample_indices)
    records = payload.get("videos", []) if isinstance(payload, dict) else []
    if not isinstance(records, list):
        raise TypeError(f"Expected a videos list in {args.sample_indices}")
    selected = []
    missing = []
    for record in records:
        if not isinstance(record, dict):
            continue
        video = str(record.get("video_path", ""))
        if video in by_video:
            selected.append(record)
        else:
            missing.append(video)
        if len(selected) == args.calibration_batches:
            break
    if len(selected) != args.calibration_batches:
        raise RuntimeError(
            f"Only found {len(selected)} fixed-manifest records in the supervised dataset; "
            f"requested {args.calibration_batches}. First missing={missing[:3]}"
        )
    return selected


def prepare_batch(
    dataset: Any,
    collator: Any,
    dataset_index: int,
    model: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Use the existing supervised dataset/collator and its normal SFT inputs."""
    batch = collator([dataset[dataset_index]])
    batch = move_to_device(batch, device, dtype)
    vision = model.get_vision_tower()
    if vision is None:
        raise RuntimeError("C1 pre-SFT model has no materialized SigLIP vision tower")
    if "images" in batch:
        vision_dtype = getattr(vision, "dtype", dtype)
        batch["images"] = move_value(batch["images"], module_device(vision, device), vision_dtype)
    spatial_tower = model.get_spatial_tower()
    if spatial_tower is not None:
        spatial_device = module_device(spatial_tower, device)
        for key in ("spatial_features", "point_maps"):
            if key in batch:
                batch[key] = move_value(batch[key], spatial_device)
    return batch


def batch_metadata(batch: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    labels = batch.get("labels")
    valid_labels = int((labels != -100).sum().item()) if isinstance(labels, torch.Tensor) else None
    return {
        "video_path": str(record.get("video_path")),
        "scene_id": record.get("scene_id"),
        "split": record.get("split"),
        "input_ids_shape": list(batch["input_ids"].shape) if isinstance(batch.get("input_ids"), torch.Tensor) else None,
        "supervised_label_tokens": valid_labels,
        "frames_upbound": 32,
    }


class ForwardFlopCounter:
    """Dynamic analytical FLOP counter for the actual calibration forward.

    It counts 2 FLOPs per multiply-accumulate.  Linear layers are measured
    from their runtime output shape.  Attention QK^T/AV products are added for
    Qwen/SigLIP/CUT3R attention blocks; projection linears are already covered
    by the Linear hooks.  This intentionally excludes non-matmul operations
    such as norms, GELU, softmax, interpolation, and CE reduction, which makes
    the reported value a standard dense-matmul FLOP proxy rather than a device
    kernel timing estimate.
    """

    def __init__(self, fusion: nn.Module | Iterable[nn.Module], lm_head: nn.Linear):
        self.total = 0
        self.fusion = 0
        self._handles: list[Any] = []
        fusion_roots = [fusion] if isinstance(fusion, nn.Module) else list(fusion)
        if not fusion_roots:
            raise ValueError("ForwardFlopCounter requires at least one candidate-specific module")
        self._fusion_ids = {
            id(module)
            for root in fusion_roots
            for module in root.modules()
        }
        self._lm_head = lm_head
        self._observed_lm_head_flops = 0
        self._expanded_hidden_elements: int | None = None

    def _add(self, module: nn.Module, flops: int) -> None:
        flops = int(flops)
        self.total += flops
        if id(module) in self._fusion_ids:
            self.fusion += flops

    @staticmethod
    def _first_tensor(value: Any) -> torch.Tensor | None:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                tensor = ForwardFlopCounter._first_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    def _linear_hook(self, module: nn.Linear, _inputs: tuple[Any, ...], output: Any) -> None:
        tensor = self._first_tensor(output)
        if tensor is None:
            return
        flops = 2 * int(tensor.numel()) * int(module.in_features)
        self._add(module, flops)
        if module is self._lm_head:
            self._observed_lm_head_flops += flops

    def _base_model_hook(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        hidden = getattr(output, "last_hidden_state", None)
        if not isinstance(hidden, torch.Tensor):
            hidden = self._first_tensor(output)
        if isinstance(hidden, torch.Tensor):
            self._expanded_hidden_elements = int(hidden.numel())

    def _mha_hook(self, module: nn.MultiheadAttention, inputs: tuple[Any, ...], output: Any) -> None:
        if len(inputs) < 3:
            return
        query, key, value = inputs[:3]
        if not all(isinstance(item, torch.Tensor) and item.ndim == 3 for item in (query, key, value)):
            return
        if module.batch_first:
            batch, q_tokens, embed = map(int, query.shape)
            key_tokens = int(key.shape[1])
        else:
            q_tokens, batch, embed = map(int, query.shape)
            key_tokens = int(key.shape[0])
        heads = int(module.num_heads)
        head_dim = int(module.head_dim)
        # F.linear projections are implemented inside MultiheadAttention, so
        # they are not seen by the Linear hooks.
        projections = 2 * batch * (q_tokens + 2 * key_tokens) * embed * embed
        attention = 4 * batch * heads * q_tokens * key_tokens * head_dim
        output_projection = 2 * batch * q_tokens * embed * embed
        self._add(module, projections + attention + output_projection)

    def _attention_hook(self, module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
        hidden = self._first_tensor(inputs)
        if hidden is None or hidden.ndim not in {2, 3}:
            return
        name = module.__class__.__name__
        if name.startswith("Qwen2") and "Attention" in name:
            batch, tokens, _ = map(int, hidden.shape)
            heads = int(getattr(module, "num_heads", 0) or 0)
            head_dim = int(getattr(module, "head_dim", 0) or 0)
            if heads and head_dim:
                self._add(module, 4 * batch * heads * tokens * tokens * head_dim)
        elif name == "SiglipAttention":
            batch, tokens, _ = map(int, hidden.shape)
            heads = int(getattr(module, "num_heads", 0) or 0)
            head_dim = int(getattr(module, "head_dim", 0) or 0)
            if heads and head_dim:
                self._add(module, 4 * batch * heads * tokens * tokens * head_dim)
        elif name == "Cut3RSpatialStackCrossAttentionBlock" and len(inputs) >= 2:
            geometry = inputs[1]
            if isinstance(geometry, torch.Tensor) and geometry.ndim in {2, 3}:
                if hidden.ndim == 2:
                    batch, q_tokens = 1, int(hidden.shape[0])
                else:
                    batch, q_tokens, _ = map(int, hidden.shape)
                key_tokens = int(geometry.shape[0] if geometry.ndim == 2 else geometry.shape[1])
                heads = int(getattr(module, "num_heads", 0) or 0)
                head_dim = int(getattr(module, "head_dim", 0) or 0)
                if heads and head_dim:
                    self._add(module, 4 * batch * heads * q_tokens * key_tokens * head_dim)

    def install(self, model: nn.Module) -> None:
        self._handles.append(model.get_model().register_forward_hook(self._base_model_hook))
        for module in model.modules():
            if isinstance(module, nn.Linear):
                self._handles.append(module.register_forward_hook(self._linear_hook))
            elif isinstance(module, nn.MultiheadAttention):
                self._handles.append(module.register_forward_hook(self._mha_hook))
            else:
                name = module.__class__.__name__
                if (
                    (name.startswith("Qwen2") and "Attention" in name)
                    or name == "SiglipAttention"
                    or name == "Cut3RSpatialStackCrossAttentionBlock"
                ):
                    self._handles.append(module.register_forward_hook(self._attention_hook))

    def restore_full_sft_lm_head_flops(self) -> None:
        """Replace compact proxy-logit work with ordinary full-logit FLOPs."""
        if self._expanded_hidden_elements is None:
            return
        expected = 2 * self._expanded_hidden_elements * int(self._lm_head.out_features)
        if expected > self._observed_lm_head_flops:
            self.total += expected - self._observed_lm_head_flops

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def profile_forward_flops(model: nn.Module, batch: dict[str, Any]) -> tuple[int, int, float]:
    fusion = fusion_module(model)
    lm_head = getattr(model, "lm_head", None)
    if not isinstance(lm_head, nn.Linear):
        raise RuntimeError(f"Expected an nn.Linear LM head, got {type(lm_head).__name__}")
    counter = ForwardFlopCounter(fusion, lm_head)
    previous_mode = model.training
    model.eval()
    counter.install(model)
    start = time.perf_counter()
    try:
        with torch.no_grad(), proxy_supervised_logits_only(model):
            output = model(**batch, use_cache=False, return_dict=True)
        if getattr(output, "loss", None) is None:
            raise RuntimeError("Structural FLOP forward did not produce the normal supervised CE loss")
        synchronize_cuda()
    finally:
        counter.close()
        model.train(previous_mode)
    counter.restore_full_sft_lm_head_flops()
    return counter.total, counter.fusion, time.perf_counter() - start


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)


def reset_cuda_peaks() -> None:
    if not torch.cuda.is_available():
        return
    for index in range(torch.cuda.device_count()):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(index)


def cuda_peaks() -> list[dict[str, int]]:
    if not torch.cuda.is_available():
        return []
    return [
        {
            "logical_gpu": index,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
        }
        for index in range(torch.cuda.device_count())
    ]


@contextmanager
def temporary_grad_scope(model: nn.Module, parameters: list[nn.Parameter]):
    """Make only ``parameters`` differentiable and restore every original flag."""
    original = {id(parameter): bool(parameter.requires_grad) for parameter in model.parameters()}
    selected = {id(parameter) for parameter in parameters}
    try:
        for parameter in model.parameters():
            parameter.requires_grad_(id(parameter) in selected)
        yield
    finally:
        for parameter in model.parameters():
            # Accelerate may temporarily replace CPU-offloaded Parameter
            # objects during a forward.  Those newly materialized objects
            # were frozen before the proxy and must remain frozen afterwards.
            parameter.requires_grad_(original.get(id(parameter), False))


@contextmanager
def proxy_supervised_logits_only(model: nn.Module):
    """Temporarily activate the exact compact-label CE path for proxy memory."""
    original = bool(getattr(model.config, "proxy_supervised_logits_only", False))
    model.config.proxy_supervised_logits_only = True
    try:
        yield
    finally:
        model.config.proxy_supervised_logits_only = original


def proxy_scores(parameters: list[nn.Parameter], gradients: Iterable[torch.Tensor | None]) -> dict[str, float | int]:
    grad_norm = 0.0
    snip = 0.0
    fisher = 0.0
    parameter_count_with_gradient = 0
    gradient_elements = 0
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            continue
        grad = gradient.detach().float()
        parameter_value = parameter.detach().float()
        grad_norm += float(torch.linalg.vector_norm(grad).item())
        snip += float((parameter_value * grad).abs().sum().item())
        fisher += float(grad.square().sum().item())
        parameter_count_with_gradient += int(parameter.numel())
        gradient_elements += int(grad.numel())
    return {
        "gradnorm": grad_norm,
        "snip": snip,
        "fisher": fisher,
        "parameters_with_gradient": parameter_count_with_gradient,
        "gradient_elements": gradient_elements,
    }


def grouped_proxy_scores(
    groups: Mapping[str, list[nn.Parameter]],
    gradients_by_parameter: Mapping[int, torch.Tensor | None],
) -> tuple[dict[str, dict[str, float | int]], dict[str, float]]:
    """Reduce standard proxies by disjoint parameter group and their union.

    A metric is reduced in one pass over the union, so the reported metric
    reduction timing is not inflated by separately traversing LoRA, fusion,
    projector, and total.  ``total`` is the exact sum of the named groups.
    """
    groups = {name: list(parameters) for name, parameters in groups.items()}
    identities = [id(parameter) for parameters in groups.values() for parameter in parameters]
    if len(identities) != len(set(identities)):
        raise RuntimeError("Grouped proxy reduction requires disjoint parameter groups")
    result: dict[str, dict[str, float | int]] = {
        name: {
            "parameter_elements": parameter_count(parameters),
            "parameters_with_gradient": 0,
            "gradient_elements": 0,
            "gradnorm": 0.0,
            "snip": 0.0,
            "fisher": 0.0,
        }
        for name, parameters in groups.items()
    }
    result["total"] = {
        "parameter_elements": sum(int(item["parameter_elements"]) for item in result.values()),
        "parameters_with_gradient": 0,
        "gradient_elements": 0,
        "gradnorm": 0.0,
        "snip": 0.0,
        "fisher": 0.0,
    }
    reduction_seconds: dict[str, float] = {}
    for metric in ("gradnorm", "snip", "fisher"):
        synchronize_cuda()
        started = time.perf_counter()
        for name, parameters in groups.items():
            accumulator = 0.0
            for parameter in parameters:
                gradient = gradients_by_parameter.get(id(parameter))
                if gradient is None:
                    continue
                grad = gradient.detach().float()
                if metric == "gradnorm":
                    accumulator += float(torch.linalg.vector_norm(grad).item())
                elif metric == "snip":
                    accumulator += float((parameter.detach().float() * grad).abs().sum().item())
                else:
                    accumulator += float(grad.square().sum().item())
            result[name][metric] = accumulator
            result["total"][metric] = float(result["total"][metric]) + accumulator
        synchronize_cuda()
        reduction_seconds[metric] = time.perf_counter() - started
    for name, parameters in groups.items():
        with_gradient = [parameter for parameter in parameters if gradients_by_parameter.get(id(parameter)) is not None]
        result[name]["parameters_with_gradient"] = parameter_count(with_gradient)
        result[name]["gradient_elements"] = parameter_count(with_gradient)
        result["total"]["parameters_with_gradient"] = int(result["total"]["parameters_with_gradient"]) + parameter_count(with_gradient)
        result["total"]["gradient_elements"] = int(result["total"]["gradient_elements"]) + parameter_count(with_gradient)
    return result, reduction_seconds


def run_grouped_backward_scope(
    model: nn.Module,
    batch: dict[str, Any],
    scope_name: str,
    groups: Mapping[str, list[nn.Parameter]],
    *,
    rng_seed: int,
) -> dict[str, Any]:
    """One read-only backward pass, with deterministic RNG and group scores."""
    selected = [parameter for parameters in groups.values() for parameter in parameters]
    if not selected:
        raise RuntimeError(f"{scope_name} selected no parameters")
    selected_ids = {id(parameter) for parameter in selected}
    if len(selected) != len(selected_ids):
        raise RuntimeError(f"{scope_name} contains duplicate selected parameters")
    versions_before = {id(parameter): int(parameter._version) for parameter in selected}
    group_parameter_elements = {name: parameter_count(parameters) for name, parameters in groups.items()}
    flags_before = {id(parameter): bool(parameter.requires_grad) for parameter in model.parameters()}
    reset_cuda_peaks()
    previous_mode = model.training
    model.train(True)
    model.config.use_cache = False
    try:
        with temporary_grad_scope(model, selected):
            rng_provenance = reset_proxy_rng(rng_seed)
            synchronize_cuda()
            started = time.perf_counter()
            with proxy_supervised_logits_only(model):
                output = model(**batch, use_cache=False, return_dict=True)
            loss = getattr(output, "loss", None)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(f"{scope_name} SFT forward returned an invalid CE loss: {loss}")
            loss.backward()
            gradients = {id(parameter): parameter.grad for parameter in selected}
            synchronize_cuda()
            shared_seconds = time.perf_counter() - started
            scores, reduction_seconds = grouped_proxy_scores(groups, gradients)
            versions_after = {id(parameter): int(parameter._version) for parameter in selected}
            complete_gradient = (
                int(scores["total"]["parameters_with_gradient"])
                == int(scores["total"]["parameter_elements"])
            )
            return {
                "status": "PASS" if complete_gradient else "INCOMPLETE_GRADIENT",
                "scope": scope_name,
                "loss": float(loss.detach().float().item()),
                "shared_forward_backward_runtime_seconds": shared_seconds,
                "metric_reduction_runtime_seconds": reduction_seconds,
                "peak_gpu_memory": cuda_peaks(),
                "rng_reset": rng_provenance,
                "proxy_groups": scores,
                "all_selected_parameters_received_gradients": complete_gradient,
                "parameter_versions_unchanged": versions_before == versions_after,
                "shared_backward_pass": True,
                "no_optimizer_constructed": True,
                "no_weight_update": True,
            }
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        synchronize_cuda()
        return {
            "status": "OOM",
            "scope": scope_name,
            "error": str(exc),
            "shared_forward_backward_runtime_seconds": None,
            "selected_parameter_elements": parameter_count(selected),
            "group_parameter_elements": group_parameter_elements,
            "peak_gpu_memory": cuda_peaks(),
            "rng_reset": {"requested_seed": int(rng_seed)},
            "no_optimizer_constructed": True,
            "no_weight_update": True,
        }
    finally:
        model.train(previous_mode)
        for parameter in model.parameters():
            parameter.grad = None
        flags_after = {id(parameter): bool(parameter.requires_grad) for parameter in model.parameters()}
        # CPU-offload hooks can replace inactive Parameter objects with fresh
        # meta placeholders, so identity-based equality is advisory here. The
        # context manager still restores every currently materialized flag;
        # callers record any mismatch instead of masking a completed proxy.
        model._last_proxy_requires_grad_restored = flags_after == flags_before
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_backward_scope(model: nn.Module, batch: dict[str, Any], scope_name: str, parameters: list[nn.Parameter]) -> dict[str, Any]:
    """One untouched loss/gradient evaluation for all three proxy definitions."""
    if not parameters:
        raise RuntimeError(f"{scope_name} selected no parameters")
    reset_cuda_peaks()
    previous_mode = model.training
    model.train(True)  # Existing SFT loss path, including normal training-mode behavior.
    model.config.use_cache = False
    try:
        with temporary_grad_scope(model, parameters):
            synchronize_cuda()
            started = time.perf_counter()
            with proxy_supervised_logits_only(model):
                output = model(**batch, use_cache=False, return_dict=True)
            loss = getattr(output, "loss", None)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(f"{scope_name} SFT forward returned an invalid CE loss: {loss}")
            # Re-entrant HF gradient checkpointing (used by the historical
            # SFT recipe) is deliberately compatible with ``backward`` but
            # rejects ``autograd.grad``.  No optimizer exists in this script;
            # collecting .grad is therefore observational only.
            loss.backward()
            gradients = [parameter.grad for parameter in parameters]
            synchronize_cuda()
            elapsed = time.perf_counter() - started
            scores = proxy_scores(parameters, gradients)
            return {
                "status": "PASS",
                "scope": scope_name,
                "loss": float(loss.detach().float().item()),
                "runtime_seconds": elapsed,
                "peak_gpu_memory": cuda_peaks(),
                "shared_backward_pass": True,
                "no_optimizer_constructed": True,
                "no_weight_update": True,
                **scores,
            }
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        synchronize_cuda()
        return {
            "status": "OOM",
            "scope": scope_name,
            "error": str(exc),
            "runtime_seconds": None,
            "peak_gpu_memory": cuda_peaks(),
            "no_optimizer_constructed": True,
            "no_weight_update": True,
        }
    finally:
        model.train(previous_mode)
        # Clear transient backward gradients. This cannot alter a parameter
        # value and keeps the next read-only proxy scope independent.
        for parameter in model.parameters():
            parameter.grad = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def score_candidate(
    args: argparse.Namespace,
    candidate: CandidateSpec,
    *,
    attempt_whole_model: bool,
) -> dict[str, Any]:
    cut3r_layers, llm_layers, artifact = candidate_schedule(candidate)
    load_args = make_load_args(args, candidate, cut3r_layers, llm_layers)
    native_vlm3r = candidate.fusion_variant == "c1_vlm3r"
    # The calibration JSON deliberately retains the common C1 reference
    # SpatialStack schedule, but VLM3R's actual historical topology is a
    # single final CUT3R sidecar fused before the visual projector.  Keep the
    # result record about the executed topology, rather than that reference.
    executed_source_layers = [12] if native_vlm3r else [int(value) for value in cut3r_layers.split(",")]
    executed_injection_layers = [] if native_vlm3r else [int(value) for value in llm_layers.split(",")]
    forward_contract = (
        "32 decoded RGB frames plus final CUT3R camera-token/patch-token sidecar"
        if native_vlm3r
        else "32 decoded RGB frames plus CUT3R decoder 6/9/12 token sidecars"
    )
    device = torch.device(args.device)
    dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]
    tokenizer, model, image_processor = load_model(load_args, device, dtype)
    try:
        apply_c1_calibration_artifact(model, artifact)
        # This mirrors the historical SFT wrapper's gradient-checkpointing
        # setting. It changes activation storage only, never the architecture
        # or loss, and is required to keep a 32-frame loss backward feasible.
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        install_forward_frame_loader(args.forward_frames_root)
        dataset, collator, by_video = build_dataset(load_args, tokenizer, image_processor)
        records = load_calibration_records(args, by_video)
        fusion = fusion_module(model)
        fusion_parameters = list(fusion.parameters())
        # Accelerate normally materializes every checkpoint parameter on CPU or
        # GPU, but exclude a defensive meta placeholder from autograd.  It is
        # still included in structural counts above.
        whole_parameters = [parameter for parameter in model.parameters() if not parameter.is_meta]
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "structural": structural_row(model, candidate, artifact, args.lora_rank),
            "candidate_schedule": {
                "cut3r_source_layers": executed_source_layers,
                "llm_injection_layers": executed_injection_layers,
                "fusion_location": "pre-projector native VLM3R fusion" if native_vlm3r else "LLM residual injection",
            },
            "calibration": {
                "sample_indices": str(args.sample_indices),
                "sample_indices_sha256": sha256_file(args.sample_indices),
                "requested_batches": args.calibration_batches,
                "minibatches": [],
                "loss_definition": "existing LlavaQwenForCausalLM supervised next-token cross entropy",
                "forward_contract": forward_contract,
            },
            "proxy_definitions": {
                "gradnorm": "sum_p ||dL/dp||_2",
                "snip": "sum_p |p * dL/dp|",
                "fisher": "sum_p (dL/dp)^2; empirical diagonal Fisher on each calibration minibatch",
            },
            "flop_definition": "dynamic dense-matmul FLOPs: 2 per MAC; linear projections plus attention QK^T/AV; excludes norms/activations/softmax/interpolation/CE",
            "proxy_scopes": {},
            "no_training": {
                "optimizer_constructed": False,
                "parameter_updates": False,
                "c1_calibration_artifact_applied": True,
            },
        }
        aggregate: dict[str, list[dict[str, Any]]] = {"candidate_fusion": []}
        if attempt_whole_model:
            aggregate["whole_model"] = []
        whole_oom = False
        flops_total: list[int] = []
        flops_fusion: list[int] = []
        flops_runtime: list[float] = []
        for record in records:
            batch = prepare_batch(dataset, collator, by_video[str(record["video_path"])], model, device, dtype)
            batch_info = batch_metadata(batch, record)
            total_flops, fusion_flops, flops_seconds = profile_forward_flops(model, batch)
            batch_info.update(
                {
                    "forward_dense_matmul_flops": total_flops,
                    "fusion_dense_matmul_flops": fusion_flops,
                    "flop_forward_runtime_seconds": flops_seconds,
                }
            )
            fusion_score = run_backward_scope(model, batch, "candidate_fusion", fusion_parameters)
            batch_info["candidate_fusion"] = fusion_score
            aggregate["candidate_fusion"].append(fusion_score)
            if attempt_whole_model and not whole_oom:
                whole_score = run_backward_scope(model, batch, "whole_model", whole_parameters)
                batch_info["whole_model"] = whole_score
                aggregate["whole_model"].append(whole_score)
                whole_oom = whole_score["status"] == "OOM"
            elif attempt_whole_model:
                batch_info["whole_model"] = {
                    "status": "SKIPPED_AFTER_OOM",
                    "scope": "whole_model",
                    "no_optimizer_constructed": True,
                    "no_weight_update": True,
                }
            result["calibration"]["minibatches"].append(batch_info)
            flops_total.append(total_flops)
            flops_fusion.append(fusion_flops)
            flops_runtime.append(flops_seconds)
        result["structural"].update(
            {
                "forward_dense_matmul_flops": float(np.mean(flops_total)),
                "fusion_dense_matmul_flops": float(np.mean(flops_fusion)),
                "flop_forward_runtime_seconds": float(np.mean(flops_runtime)),
            }
        )
        for scope, rows in aggregate.items():
            passed = [row for row in rows if row.get("status") == "PASS"]
            if not passed:
                result["proxy_scopes"][scope] = {
                    "status": rows[0].get("status", "UNAVAILABLE") if rows else "UNAVAILABLE",
                    "minibatches": rows,
                }
                continue
            summary: dict[str, Any] = {
                "status": "PASS" if len(passed) == len(rows) else "PARTIAL",
                "calibration_batches_scored": len(passed),
                "minibatches": rows,
            }
            for key in ("gradnorm", "snip", "fisher", "loss", "runtime_seconds", "parameters_with_gradient", "gradient_elements"):
                values = [float(row[key]) for row in passed]
                summary[key] = float(np.mean(values))
            result["proxy_scopes"][scope] = summary
        return result
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def rankdata(values: list[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    order = np.argsort(values_array, kind="mergesort")
    ranks = np.empty(values_array.size, dtype=np.float64)
    ranks[order] = np.arange(values_array.size, dtype=np.float64)
    unique, inverse, counts = np.unique(values_array, return_inverse=True, return_counts=True)
    del unique
    sums = np.zeros(counts.size, dtype=np.float64)
    np.add.at(sums, inverse, ranks)
    return sums[inverse] / counts[inverse]


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def flattened_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        structural = dict(result["structural"])
        row: dict[str, Any] = dict(structural)
        for scope in ("whole_model", "candidate_fusion"):
            summary = result.get("proxy_scopes", {}).get(scope, {})
            row[f"{scope}_status"] = summary.get("status", "NOT_ATTEMPTED")
            for metric in ("gradnorm", "snip", "fisher", "loss", "runtime_seconds", "parameters_with_gradient", "gradient_elements"):
                row[f"{scope}_{metric}"] = summary.get(metric)
            rows_for_scope = summary.get("minibatches", [])
            peaks = [peak for item in rows_for_scope if item.get("status") == "PASS" for peak in item.get("peak_gpu_memory", [])]
            if peaks:
                row[f"{scope}_peak_gpu_allocated_bytes"] = max(int(peak["peak_allocated_bytes"]) for peak in peaks)
                row[f"{scope}_peak_gpu_reserved_bytes"] = max(int(peak["peak_reserved_bytes"]) for peak in peaks)
            else:
                row[f"{scope}_peak_gpu_allocated_bytes"] = None
                row[f"{scope}_peak_gpu_reserved_bytes"] = None
        rows.append(row)
    return rows


def correlation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Structural quantities are costs, so their rank is negated.  Gradient
    # proxy scores use their conventional higher-is-better orientation.
    specs = [
        ("total_params", "structural", "lower_is_better"),
        ("sft_trainable_params", "structural", "lower_is_better"),
        ("forward_dense_matmul_flops", "structural", "lower_is_better"),
        ("whole_model_gradnorm", "whole_model", "higher_is_better"),
        ("whole_model_snip", "whole_model", "higher_is_better"),
        ("whole_model_fisher", "whole_model", "higher_is_better"),
        ("candidate_fusion_gradnorm", "candidate_fusion", "higher_is_better"),
        ("candidate_fusion_snip", "candidate_fusion", "higher_is_better"),
        ("candidate_fusion_fisher", "candidate_fusion", "higher_is_better"),
    ]
    output = []
    for metric, scope, orientation in specs:
        valid = [row for row in rows if row.get(metric) is not None and math.isfinite(float(row[metric]))]
        raw_values = [float(row[metric]) for row in valid]
        reference = [float(row["vsi_avg"]) for row in valid]
        oriented = [-value for value in raw_values] if orientation == "lower_is_better" else raw_values
        output.append(
            {
                "proxy": metric,
                "scope": scope,
                "expected_orientation": orientation,
                "n_architectures": len(valid),
                "spearman_vs_vsi_avg": spearman(oriented, reference),
                "raw_spearman_without_cost_flip": spearman(raw_values, reference),
                "candidate_order": ",".join(str(row["candidate"]) for row in valid),
            }
        )
    return output


def write_markdown(output: Path, rows: list[dict[str, Any]], correlations: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# Pre-SFT 3D-injection zero-cost proxies",
        "",
        "The models use the exact C1 artifacts and fixed 32-frame supervised calibration minibatch(es). No optimizer was constructed and no weights were updated.",
        "",
        "## Architecture scores",
        "",
        "| Candidate | VSI-Bench Avg. | Total params | SFT trainable params | Forward FLOPs | Fusion GradNorm | Fusion SNIP | Fusion Fisher | Whole GradNorm | Whole SNIP | Whole Fisher |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def displayed(value: Any) -> str:
            return "—" if value is None else human_count(float(value))

        lines.append(
            "| {candidate} | {vsi:.1f} | {total} | {trainable} | {flops} | {fg} | {fs} | {ff} | {wg} | {ws} | {wf} |".format(
                candidate=row["candidate"],
                vsi=float(row["vsi_avg"]),
                total=displayed(row["total_params"]),
                trainable=displayed(row["sft_trainable_params"]),
                flops=displayed(row["forward_dense_matmul_flops"]),
                fg=displayed(row.get("candidate_fusion_gradnorm")),
                fs=displayed(row.get("candidate_fusion_snip")),
                ff=displayed(row.get("candidate_fusion_fisher")),
                wg=displayed(row.get("whole_model_gradnorm")),
                ws=displayed(row.get("whole_model_snip")),
                wf=displayed(row.get("whole_model_fisher")),
            )
        )
    lines.extend(
        [
            "",
            "## Spearman correlation with post-SFT VSI-Bench architecture ranking",
            "",
            "Cost proxies are oriented as lower-is-better; standard gradient scores are higher-is-better. `NaN` is expected for a constant proxy or a scope unavailable on the local memory budget.",
            "",
            "| Proxy | Scope | n | Orientation | Spearman |",
            "|---|---|---:|---|---:|",
        ]
    )
    for row in correlations:
        score = row["spearman_vs_vsi_avg"]
        lines.append(
            f"| {row['proxy']} | {row['scope']} | {row['n_architectures']} | {row['expected_orientation']} | "
            f"{score:.4f} |" if math.isfinite(score) else
            f"| {row['proxy']} | {row['scope']} | {row['n_architectures']} | {row['expected_orientation']} | NaN |"
        )
    lines.extend(
        [
            "",
            "## Definitions",
            "",
            "- GradNorm = `sum_p ||dL/dp||_2`",
            "- SNIP = `sum_p |p * dL/dp|`",
            "- Fisher = `sum_p (dL/dp)^2` (empirical diagonal Fisher on each minibatch)",
            "- Whole-model backward scores request all materialized model parameters. Candidate-fusion scores use only the freshly attached SpatialStack merger or VLM3R fusion block.",
            "- FLOPs count dense linear and attention matmuls dynamically on the actual forward; they exclude elementwise and normalization kernels.",
            "",
            f"Calibration minibatches: {args.calibration_batches}; fixed manifest: `{args.sample_indices}`.",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--candidates", default=",".join(item.identifier for item in CANDIDATES))
    parser.add_argument("--calibration-batches", type=int, default=1)
    parser.add_argument("--attempt-whole-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--siglip-model", type=Path, default=DEFAULT_SIGLIP_MODEL)
    parser.add_argument("--forward-frames-root", type=Path, default=DEFAULT_FORWARD_ROOT)
    parser.add_argument("--probe-targets-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--data-yaml", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument("--sample-indices", type=Path, default=DEFAULT_SAMPLE_INDICES)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--device-map", choices=("auto", "cuda:0", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--pre-sft-gpu-weight-budget", default="5GiB")
    parser.add_argument("--pre-sft-cpu-offload-budget", default="45GiB")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--lora-rank", type=int, default=128)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> list[CandidateSpec]:
    if args.calibration_batches < 1:
        raise ValueError("--calibration-batches must be positive")
    if args.mode == "smoke" and args.calibration_batches != 1:
        raise ValueError("Smoke must use exactly one architecture and one minibatch; pass --calibration-batches 1")
    requested = parse_csv(args.candidates)
    unknown = sorted(set(requested).difference(BY_IDENTIFIER))
    if unknown:
        raise ValueError(f"Unknown candidate identifiers: {unknown}; valid={sorted(BY_IDENTIFIER)}")
    candidates = [BY_IDENTIFIER[item] for item in requested]
    if args.mode == "smoke" and len(candidates) != 1:
        raise ValueError("Smoke must use exactly one architecture; pass one --candidates identifier")
    required_paths = [
        args.base_model,
        args.siglip_model,
        args.forward_frames_root,
        args.probe_targets_root,
        args.feature_root,
        args.data_yaml,
        args.sample_indices,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required local inputs are missing: {missing}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested but PyTorch cannot see CUDA")
    return candidates


def main() -> None:
    args = parse_args()
    candidates = validate_args(args)
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(
        json.dumps(
            {
                "mode": args.mode,
                "candidates": [item.identifier for item in candidates],
                "calibration_batches": args.calibration_batches,
                "device": args.device,
                "device_map": args.device_map,
                "output_root": str(args.output_root),
                "command": [sys.executable, *sys.argv],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    results: list[dict[str, Any]] = []
    whole_model_safe = bool(args.attempt_whole_model)
    failures = []
    for candidate in candidates:
        print(f"[RUN] {candidate.identifier}", flush=True)
        try:
            result = score_candidate(args, candidate, attempt_whole_model=whole_model_safe)
            results.append(result)
            whole_status = result.get("proxy_scopes", {}).get("whole_model", {}).get("status")
            if whole_status == "OOM":
                whole_model_safe = False
                print("[SAFE MODE] whole-model autograd OOM; skipping that scope for remaining candidates.", flush=True)
        except Exception as exc:
            failure = {
                "candidate": candidate.identifier,
                "exception_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            print("[ERROR] " + json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
            if args.mode == "smoke":
                raise
    rows = flattened_rows(results)
    correlations = correlation_rows(rows)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "completed_at_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "candidates_requested": [candidate.identifier for candidate in candidates],
        "candidates_completed": [result["structural"]["candidate"] for result in results],
        "calibration_batches": args.calibration_batches,
        "whole_model_scope_attempted_initially": bool(args.attempt_whole_model),
        "whole_model_scope_safe_after_smoke": whole_model_safe,
        "no_optimizer_constructed": True,
        "no_weight_updates": True,
        "failures": failures,
    }
    write_json(args.output_root / "results.json", results)
    write_json(args.output_root / "metadata.json", metadata)
    write_json(args.output_root / "spearman_correlations.json", correlations)
    write_csv(args.output_root / "proxy_scores.csv", rows)
    write_csv(args.output_root / "spearman_correlations.csv", correlations)
    write_markdown(args.output_root, rows, correlations, args)
    print(
        json.dumps(
            {
                "status": "PASS" if not failures else "PARTIAL",
                "completed": len(results),
                "failures": len(failures),
                "output_root": str(args.output_root),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
