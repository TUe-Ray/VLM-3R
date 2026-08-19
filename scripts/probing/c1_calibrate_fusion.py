#!/usr/bin/env python
"""Forward-only C1 calibration for zero-training pre-SFT fusion probing.

The script deliberately opens no probe targets and never constructs labels for
the model forward.  It reuses the ordinary cached-frame dataset/collator and
its real per-video human prompt, replacing the assistant response with an
empty turn solely to preserve the normal chat-template input construction.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llava.model.c1_structured_isometry import (  # noqa: E402
    SCHEME_VERSION,
    apply_spatialstack_c1,
    apply_vlm3r_c1,
    matrix_scheme_metadata,
)
from scripts.diagnose_layerwise_spatial_hidden_scan import load_model, make_data_args, move_to_device  # noqa: E402
from scripts.probing.depth_probe_common import (  # noqa: E402
    DEFAULT_DATA_YAML,
    DEFAULT_FAST_FEATURE_ROOT,
    DEFAULT_SPATIAL_FEATURES_SUBDIR,
    torch_dtype_from_name,
)
from scripts.probing.extract_depth_probe_features import base_module_device, base_move_value  # noqa: E402
from scripts.probing.local_depth_probe_cache import install_forward_frame_loader  # noqa: E402


C1_ARCHITECTURES = ("base", "spatialstack_add", "spatialstack_cross_attn_v1", "vlm3r")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rms(value: torch.Tensor) -> float:
    value = value.detach().float()
    if value.numel() == 0:
        raise RuntimeError("C1 RMS received an empty tensor.")
    result = float(value.square().mean().sqrt().item())
    if not math.isfinite(result):
        raise RuntimeError(f"C1 RMS is non-finite: {result}")
    return result


def finite_positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"{name} must be finite and positive, got {value}.")
    return value


def summary(values: Iterable[float]) -> dict[str, float | int]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    if tensor.numel() == 0:
        raise RuntimeError("C1 statistic has no values.")
    if not torch.isfinite(tensor).all():
        raise RuntimeError("C1 statistic contains non-finite values.")
    return {
        "count": int(tensor.numel()),
        "median": float(tensor.median().item()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def moments_std(moments: list[dict[str, float | int]], name: str) -> float:
    count = sum(int(item["count"]) for item in moments)
    if count <= 1:
        raise RuntimeError(f"C1 {name} has insufficient values for a standard deviation.")
    total = sum(float(item["sum"]) for item in moments)
    total_sq = sum(float(item["sum_sq"]) for item in moments)
    variance = max(total_sq / count - (total / count) ** 2, 0.0)
    return finite_positive(math.sqrt(variance), name)


def moments_rms(moments: dict[str, float | int], name: str) -> float:
    return finite_positive(math.sqrt(float(moments["sum_sq"]) / int(moments["count"])), name)


def module_device(module: Any, fallback: torch.device) -> torch.device:
    declared = getattr(module, "device", None)
    if isinstance(declared, torch.device) and declared.type != "meta":
        return declared
    for parameter in module.parameters():
        if not parameter.is_meta:
            return parameter.device
    return fallback


def move_value(value: Any, target: torch.device, dtype: torch.dtype | None = None) -> Any:
    if torch.is_tensor(value):
        return value.to(device=target, dtype=dtype if dtype is not None and value.is_floating_point() else None)
    if isinstance(value, list):
        return [move_value(item, target, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(move_value(item, target, dtype) for item in value)
    return value


def build_dataset(args: argparse.Namespace, tokenizer: Any, image_processor: Any):
    from llava.train.train import DataCollatorForSupervisedDataset, LazySupervisedDataset

    data_args = make_data_args(args, image_processor)
    data_args.deterministic_data_order = True
    data_args.train_data_shuffle = False
    if args.architecture == "base":
        data_args.spatial_features_root = None
        data_args.spatial_features_subdir = None
        data_args.spatial_tower_type = None
        data_args.require_spatial_features = False
    else:
        data_args.spatial_features_root = args.feature_root
        data_args.spatial_features_subdir = args.spatial_features_subdir
        data_args.spatial_tower_type = "cut3r"
        data_args.require_spatial_features = True
    data_args.zero_spatial_features = False
    dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=args.train_data_json, data_args=data_args)
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    by_video: dict[str, int] = {}
    for index, item in enumerate(dataset.list_data_dict):
        video = item.get("video")
        if video is not None and str(video) not in by_video:
            by_video[str(video)] = index
    return dataset, collator, by_video


def prompt_only_item(dataset: Any, index: int) -> dict[str, Any]:
    """Use the ordinary selected user prompt without reading the SFT answer."""
    original = dataset.list_data_dict[index]
    replacement = copy.deepcopy(original)
    conversations = replacement.get("conversations") or []
    human = next(
        (copy.deepcopy(turn) for turn in conversations if str(turn.get("from", "")).lower() in {"human", "user"}),
        None,
    )
    if human is None or not str(human.get("value", "")).strip():
        raise RuntimeError(f"Calibration sample {index} has no usable human prompt.")
    replacement["conversations"] = [human, {"from": "gpt", "value": ""}]
    dataset.list_data_dict[index] = replacement
    try:
        # _get_item avoids dataset retry/fallback selecting a different video.
        return dataset._get_item(index)
    finally:
        dataset.list_data_dict[index] = original


def prepare_input(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    collator: Any,
    dataset: Any,
    dataset_index: int,
    device: torch.device,
    model_dtype: torch.dtype,
) -> tuple[dict[str, Any], dict[str, Any], dict[int, Any] | None]:
    item = prompt_only_item(dataset, dataset_index)
    batch = collator([item])
    if args.architecture == "base":
        forbidden = [name for name in ("spatial_features", "point_maps", "geometry_spatial_features") if name in batch]
        if forbidden:
            raise RuntimeError(f"Base r0 calibration unexpectedly received spatial inputs: {forbidden}")
    batch = move_to_device(batch, device, model_dtype)
    vision_tower = model.get_vision_tower()
    if vision_tower is not None and "images" in batch:
        vision_dtype = getattr(vision_tower, "dtype", model_dtype)
        batch["images"] = move_value(batch["images"], module_device(vision_tower, device), vision_dtype)
    spatial_tower = model.get_spatial_tower()
    if spatial_tower is not None:
        for name in ("spatial_features", "point_maps"):
            if name in batch:
                batch[name] = move_value(batch[name], module_device(spatial_tower, device))
    if args.architecture == "base" and vision_tower is not None and "images" in batch:
        batch["images"] = base_move_value(
            batch["images"], base_module_device(vision_tower, device), getattr(vision_tower, "dtype", model_dtype)
        )
    prepare_fn = model.prepare_inputs_labels_for_multimodal
    if "return_visual_metadata" not in inspect.signature(prepare_fn).parameters:
        raise RuntimeError("C1 needs visual_token_indices metadata, but this model does not expose it.")
    with torch.no_grad():
        prepared = prepare_fn(
            input_ids=batch["input_ids"],
            position_ids=None,
            attention_mask=batch["attention_mask"],
            past_key_values=None,
            labels=None,
            images=batch["images"],
            spatial_features=None if args.architecture == "base" else batch.get("spatial_features"),
            point_maps=None,
            modalities=batch.get("modalities"),
            image_sizes=batch.get("image_sizes"),
            return_visual_metadata=True,
        )
    input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, _labels, visual_metadata = prepared
    metadata = visual_metadata[0]
    indices = metadata.get("visual_token_indices")
    if not isinstance(indices, torch.Tensor) or indices.numel() == 0:
        raise RuntimeError("C1 cannot identify visual token positions from visual_token_indices.")
    payload = None
    if args.architecture.startswith("spatialstack"):
        merger = model.get_model().get_cut3r_spatialstack_merger()
        payload = merger(
            batch.get("spatial_features"),
            visual_metadata,
            seq_len=int(inputs_embeds.shape[1]),
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": past_key_values,
        "inputs_embeds": inputs_embeds,
    }, metadata, payload


def run_decoder(model: torch.nn.Module, prepared: dict[str, Any], payload: dict[int, Any] | None, architecture: str):
    kwargs = dict(
        **prepared,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=True,
    )
    if architecture == "spatialstack_add":
        kwargs["spatialstack_residuals_by_layer"] = payload
    elif architecture == "spatialstack_cross_attn_v1":
        kwargs["spatialstack_cross_attn_inputs_by_layer"] = payload
    with torch.no_grad():
        result = model.model(**kwargs)
    if result.hidden_states is None:
        raise RuntimeError("C1 model forward did not return hidden states.")
    return result.hidden_states


def visual_hidden(hidden: torch.Tensor, metadata: dict[str, Any]) -> torch.Tensor:
    indices = metadata["visual_token_indices"].to(hidden.device)
    return hidden[:, indices, :]


def load_manifest(path: Path, limit: int | None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != "c1_calibration_manifest_v1":
        raise RuntimeError(f"Unexpected C1 calibration manifest schema: {payload.get('schema_version')!r}")
    videos = list(payload.get("videos", []))
    if limit is not None:
        videos = videos[:limit]
    if not videos:
        raise RuntimeError("C1 calibration manifest is empty.")
    return videos


def base_calibration(
    *, model: torch.nn.Module, samples: list[tuple[int, dict[str, Any]]], args: argparse.Namespace,
    dataset: Any, collator: Any, device: torch.device, dtype: torch.dtype,
) -> tuple[float, dict[str, Any]]:
    ratios: dict[int, list[float]] = {0: [], 1: [], 2: []}
    for dataset_index, _sample in samples:
        prepared, metadata, _payload = prepare_input(
            args=args, model=model, collator=collator, dataset=dataset, dataset_index=dataset_index, device=device, model_dtype=dtype
        )
        states = run_decoder(model, prepared, None, "base")
        for layer in ratios:
            before = visual_hidden(states[layer], metadata)
            after = visual_hidden(states[layer + 1], metadata)
            ratios[layer].append(finite_positive(rms(after - before) / rms(before), f"base r(sample,L{layer})"))
    layer_stats = {str(layer): summary(values) for layer, values in ratios.items()}
    r0 = finite_positive(torch.tensor([value for values in ratios.values() for value in values]).median().item(), "global r0")
    for layer, values in ratios.items():
        print(f"[C1 base] L{layer}: median={summary(values)['median']:.6g} mean={summary(values)['mean']:.6g} std={summary(values)['std']:.6g}")
    print(f"[C1 base] global r0={r0:.6g} (per-injection-site residual budget)")
    return r0, {"per_layer": layer_stats, "global_r0": r0, "statistic": "median_per_sample_per_layer_rms_update_over_rms_before"}


def calibrate_spatialstack(
    *, model: torch.nn.Module, samples: list[tuple[int, dict[str, Any]]], args: argparse.Namespace,
    dataset: Any, collator: Any, device: torch.device, dtype: torch.dtype, r0: float,
) -> dict[str, Any]:
    """Sequentially calibrate C1 SpatialStack with median per-sample ratios."""
    merger = model.get_model().get_cut3r_spatialstack_merger()
    layers = [0, 1, 2]
    qk_raw_std_by_layer: dict[int, float] = {}
    if args.architecture == "spatialstack_add":
        # For each sample, c1_components is applied on the exact aligned
        # branch token inputs during merger.forward. Hooks collect z_pre only.
        captured: dict[int, list[dict[str, float | int]]] = {layer: [] for layer in layers}
        hooks = []
        for layer in layers:
            branch = merger.branches[str(merger.layer_map[layer])]
            hooks.append(branch.proj_in.register_forward_hook(
                lambda _m, _i, output, layer=layer: captured[layer].append({
                    "count": int(output.numel()),
                    "sum_sq": float(output.detach().float().square().sum().item()),
                })
            ))
        try:
            for dataset_index, _sample in samples:
                prepared, _metadata, _payload = prepare_input(args=args, model=model, collator=collator, dataset=dataset, dataset_index=dataset_index, device=device, model_dtype=dtype)
        finally:
            for handle in hooks:
                handle.remove()
        for layer in layers:
            total_count = sum(int(item["count"]) for item in captured[layer])
            total_sq = sum(float(item["sum_sq"]) for item in captured[layer])
            measured = finite_positive(math.sqrt(total_sq / total_count), f"additive z_pre L{layer}")
            branch = merger.branches[str(merger.layer_map[layer])]
            branch.set_c1_state(pre_gelu_scale=1.0 / measured, residual_gain=0.0)
    else:
        for layer in layers:
            block = merger.cross_attn_blocks[str(layer)]
            block.set_c1_state(qk_scale=1.0, residual_gain=0.0, collect_diagnostics=True)
            logit_moments: list[dict[str, Any]] = []
            for dataset_index, _sample in samples:
                prepared, _metadata, payload = prepare_input(args=args, model=model, collator=collator, dataset=dataset, dataset_index=dataset_index, device=device, model_dtype=dtype)
                run_decoder(model, prepared, payload, args.architecture)
                diag = block._c1_last_diagnostics
                if not isinstance(diag, dict):
                    raise RuntimeError(f"C1 SS-CA V1 L{layer} did not produce diagnostics.")
                logit_moments.append(diag["raw_logits"])
            raw_std = moments_std(logit_moments, f"SS-CA raw QK logit std L{layer}")
            qk_raw_std_by_layer[layer] = raw_std
            block.set_c1_state(qk_scale=1.0 / math.sqrt(raw_std), residual_gain=0.0, collect_diagnostics=True)

    result: dict[str, Any] = {}
    # Native injection order is 0,1,2. Earlier calibrated gains remain active
    # before later layers are measured; later gains are still zero.
    for layer in layers:
        if args.architecture == "spatialstack_add":
            module = merger.branches[str(merger.layer_map[layer])]
        else:
            module = merger.cross_attn_blocks[str(layer)]
        if args.architecture == "spatialstack_add":
            module.set_c1_state(residual_gain=1.0)
        else:
            module.set_c1_state(residual_gain=1.0, collect_diagnostics=True)
        ratios: list[float] = []
        raw_delta_rms: list[float] = []
        for dataset_index, _sample in samples:
            prepared, metadata, payload = prepare_input(args=args, model=model, collator=collator, dataset=dataset, dataset_index=dataset_index, device=device, model_dtype=dtype)
            states = run_decoder(model, prepared, payload, args.architecture)
            h_rms = rms(visual_hidden(states[layer], metadata))
            if args.architecture == "spatialstack_add":
                raw_delta = payload[layer][:, metadata["visual_token_indices"].to(payload[layer].device), :]
                delta_rms = rms(raw_delta)  # current gain is one
            else:
                diag = module._c1_last_diagnostics
                delta_rms = moments_rms(diag["delta_raw"], f"SS-CA raw delta L{layer}")
            raw_delta_rms.append(delta_rms)
            ratios.append(finite_positive(delta_rms / h_rms, f"SpatialStack raw delta/H L{layer}"))
        raw_ratio = finite_positive(summary(ratios)["median"], f"SpatialStack raw median delta/H L{layer}")
        gain = r0 / raw_ratio
        module.set_c1_state(residual_gain=gain)
        result[str(layer)] = {
            "raw_delta_rms_per_sample": summary(raw_delta_rms),
            "raw_delta_over_h": summary(ratios),
            "residual_gain": gain,
        }
        if args.architecture == "spatialstack_add":
            result[str(layer)]["s_pre"] = float(module.c1_pre_gelu_scale.item())
            result[str(layer)]["pre_gelu_raw_rms"] = 1.0 / float(module.c1_pre_gelu_scale.item())
            result[str(layer)]["calibrated_pre_gelu_rms"] = (
                result[str(layer)]["s_pre"] * result[str(layer)]["pre_gelu_raw_rms"]
            )
        else:
            result[str(layer)]["s_qk"] = float(module.c1_qk_scale.item())
        print(f"[C1 {args.architecture}] L{layer}: raw median delta/H={raw_ratio:.6g} gain={gain:.6g}")

    # Final independent validation with every native site enabled; it confirms
    # each site separately targets r0, not an architecture-wide divided budget.
    for layer in layers:
        ratios: list[float] = []
        calibration_diags: list[dict[str, Any]] = []
        for dataset_index, _sample in samples:
            prepared, metadata, payload = prepare_input(args=args, model=model, collator=collator, dataset=dataset, dataset_index=dataset_index, device=device, model_dtype=dtype)
            states = run_decoder(model, prepared, payload, args.architecture)
            h_rms = rms(visual_hidden(states[layer], metadata))
            if args.architecture == "spatialstack_add":
                delta = payload[layer][:, metadata["visual_token_indices"].to(payload[layer].device), :]
                delta_rms = rms(delta)
            else:
                diag = merger.cross_attn_blocks[str(layer)]._c1_last_diagnostics
                delta_rms = moments_rms(diag["delta"], f"SS-CA calibrated delta L{layer}")
                calibration_diags.append(diag)
            ratios.append(finite_positive(delta_rms / h_rms, f"SpatialStack calibrated delta/H L{layer}"))
        result[str(layer)]["calibrated_delta_over_h"] = summary(ratios)
        if args.architecture == "spatialstack_add":
            values = result[str(layer)]
            print(
                f"[C1 additive] L{layer}: pre_rms={values['pre_gelu_raw_rms']:.6g} "
                f"s_pre={values['s_pre']:.6g} raw_delta/H={values['raw_delta_over_h']['median']:.6g} "
                f"gain={values['residual_gain']:.6g} calibrated_delta/H={values['calibrated_delta_over_h']['median']:.6g}"
            )
        else:
            # Aggregate diagnostics across the same calibration samples, not
            # only the last forward.
            if not calibration_diags:
                raise RuntimeError(f"C1 SS-CA L{layer} recorded no final diagnostics.")
            result[str(layer)].update({
                "q_rms": moments_rms({
                    "count": sum(int(diag["q"]["count"]) for diag in calibration_diags),
                    "sum_sq": sum(float(diag["q"]["sum_sq"]) for diag in calibration_diags),
                }, f"Q RMS L{layer}"),
                "k_rms": moments_rms({
                    "count": sum(int(diag["k"]["count"]) for diag in calibration_diags),
                    "sum_sq": sum(float(diag["k"]["sum_sq"]) for diag in calibration_diags),
                }, f"K RMS L{layer}"),
                "v_rms": moments_rms({
                    "count": sum(int(diag["v"]["count"]) for diag in calibration_diags),
                    "sum_sq": sum(float(diag["v"]["sum_sq"]) for diag in calibration_diags),
                }, f"V RMS L{layer}"),
                "raw_qk_logit_std": qk_raw_std_by_layer[layer],
                "calibrated_qk_logit_std": moments_std(
                    [diag["calibrated_logits"] for diag in calibration_diags], f"calibrated QK logit std L{layer}"
                ),
                "q_shape": calibration_diags[-1]["q_shape"],
                "k_shape": calibration_diags[-1]["k_shape"],
                "v_shape": calibration_diags[-1]["v_shape"],
            })
            values = result[str(layer)]
            print(
                f"[C1 SS-CA] L{layer}: Q={values['q_rms']:.6g} K={values['k_rms']:.6g} V={values['v_rms']:.6g} "
                f"raw_logit_std={values['raw_qk_logit_std']:.6g} s_qk={values['s_qk']:.6g} "
                f"cal_logit_std={values['calibrated_qk_logit_std']:.6g} "
                f"raw_delta/H={values['raw_delta_over_h']['median']:.6g} gain={values['residual_gain']:.6g} "
                f"calibrated_delta/H={values['calibrated_delta_over_h']['median']:.6g}"
            )
    return result


def calibrate_vlm3r(
    *, model: torch.nn.Module, samples: list[tuple[int, dict[str, Any]]], args: argparse.Namespace,
    dataset: Any, collator: Any, device: torch.device, dtype: torch.dtype, r0: float,
) -> dict[str, Any]:
    fusion = model.get_model().get_fusion_block()
    fusion.set_c1_state(qk_scale=1.0, residual_gain=0.0, collect_diagnostics=True, capture_branch=True)
    raw_moments: list[dict[str, Any]] = []
    for dataset_index, _sample in samples:
        prepare_input(args=args, model=model, collator=collator, dataset=dataset, dataset_index=dataset_index, device=device, model_dtype=dtype)
        if not isinstance(fusion._c1_last_diagnostics, dict):
            raise RuntimeError("C1 VLM3R did not produce internal attention diagnostics.")
        raw_moments.append(fusion._c1_last_diagnostics["raw_logits"])
    raw_std = moments_std(raw_moments, "VLM3R raw QK logit std")
    s_qk = 1.0 / math.sqrt(raw_std)
    fusion.set_c1_state(qk_scale=s_qk, residual_gain=0.0, collect_diagnostics=True, capture_branch=True)

    # Evaluate an exact median-per-sample effective residual through the
    # frozen native projector. The scalar solve is deterministic bisection,
    # not a sweep or fitting procedure.
    cached: list[tuple[torch.Tensor, torch.Tensor]] = []
    calibrated_diags: list[dict[str, Any]] = []
    for dataset_index, _sample in samples:
        prepare_input(args=args, model=model, collator=collator, dataset=dataset, dataset_index=dataset_index, device=device, model_dtype=dtype)
        clip = fusion._c1_last_clip_features
        branch = fusion._c1_last_branch
        if clip is None or branch is None:
            raise RuntimeError("C1 VLM3R failed to capture visual features and fusion branch.")
        cached.append((clip.detach().cpu(), branch.detach().cpu()))
        calibrated_diags.append(dict(fusion._c1_last_diagnostics))
    projector = model.get_model().mm_projector
    projector_device = module_device(projector, device)
    projector_dtype = next(parameter.dtype for parameter in projector.parameters() if not parameter.is_meta)

    def projected_ratios(lam: float) -> list[float]:
        ratios: list[float] = []
        with torch.no_grad():
            for clip_cpu, branch_cpu in cached:
                clip = clip_cpu.to(device=projector_device, dtype=projector_dtype)
                branch = branch_cpu.to(device=projector_device, dtype=projector_dtype)
                e_base = projector(clip)
                e_fused = projector(clip + float(lam) * branch)
                ratios.append(finite_positive(rms(e_fused - e_base) / rms(e_base), "VLM3R projected delta/E_base"))
        return ratios

    upper = 1.0
    observed = []
    while True:
        value = summary(projected_ratios(upper))["median"]
        observed.append((upper, value))
        if value >= r0:
            break
        upper *= 2.0
        if upper > 2.0**20:
            raise RuntimeError(f"VLM3R lambda could not bracket r0={r0}; observed={observed}")
    lower = 0.0
    for _ in range(32):
        middle = (lower + upper) / 2.0
        value = summary(projected_ratios(middle))["median"]
        if value < r0:
            lower = middle
        else:
            upper = middle
    lam = (lower + upper) / 2.0
    final_ratios = projected_ratios(lam)
    fusion.set_c1_state(residual_gain=lam, capture_branch=False)
    fusion._c1_last_clip_features = None
    fusion._c1_last_branch = None
    if not calibrated_diags:
        raise RuntimeError("C1 VLM3R recorded no calibrated diagnostics.")
    def aggregate_rms(name: str) -> float:
        return moments_rms(
            {
                "count": sum(int(diag[name]["count"]) for diag in calibrated_diags),
                "sum_sq": sum(float(diag[name]["sum_sq"]) for diag in calibrated_diags),
            },
            f"VLM3R {name} RMS",
        )
    result = {
        "s_qk": s_qk,
        "raw_qk_logit_std": raw_std,
        "calibrated_qk_logit_std": moments_std(
            [diag["calibrated_logits"] for diag in calibrated_diags], "VLM3R calibrated logit std"
        ),
        "q_rms": aggregate_rms("q"),
        "k_rms": aggregate_rms("k"),
        "v_rms": aggregate_rms("v"),
        "lambda": lam,
        "effective_projected_delta_over_e_base": summary(final_ratios),
        "lambda_bracket": observed,
        "internal_mha": "identity_qkv_and_identity_out_proj",
    }
    print(
        f"[C1 vlm3r] Q={result['q_rms']:.6g} K={result['k_rms']:.6g} V={result['v_rms']:.6g} "
        f"raw_logit_std={raw_std:.6g} s_qk={s_qk:.6g} "
        f"cal_logit_std={result['calibrated_qk_logit_std']:.6g} lambda={lam:.6g} "
        f"median_effective_ratio={result['effective_projected_delta_over_e_base']['median']:.6g}"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=C1_ARCHITECTURES, required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-calibration", default=None, help="Required for non-base architecture calibration.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--siglip-path", required=True)
    parser.add_argument("--train-data-json", default=str(DEFAULT_DATA_YAML))
    parser.add_argument("--feature-root", default=str(DEFAULT_FAST_FEATURE_ROOT))
    parser.add_argument("--spatial-features-subdir", default=DEFAULT_SPATIAL_FEATURES_SUBDIR)
    parser.add_argument("--forward-frames-root", required=True)
    parser.add_argument("--frames-upbound", type=int, default=32)
    parser.add_argument("--image-folder", default=str(DEFAULT_FAST_FEATURE_ROOT))
    parser.add_argument("--video-folder", default=str(DEFAULT_FAST_FEATURE_ROOT))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--device-map", choices=("auto", "cuda:0", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--pre-sft-gpu-weight-budget", default="5GiB")
    parser.add_argument("--pre-sft-cpu-offload-budget", default="45GiB")
    parser.add_argument("--runtime-root", default=str(REPO_ROOT / ".offline_runtime"))
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--common-model-init-seed", type=int, default=0)
    parser.add_argument("--qk-basis-mode", choices=("shared_canonical", "role_offset"), default="shared_canonical")
    parser.add_argument("--max-samples", type=int, default=None, help="Smoke only; official C1 uses all 32 manifest samples.")
    args = parser.parse_args()
    if args.architecture != "base" and not args.base_calibration:
        parser.error("Non-base C1 calibration requires --base-calibration to obtain the frozen r0.")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    args.model_loading_mode = "pre_sft_base_vlm" if args.architecture == "base" else "pre_sft_fusion"
    args.pre_sft_fusion_variant = {
        "spatialstack_add": "c1_ss_add",
        "spatialstack_cross_attn_v1": "c1_ss_cross_attn_v1",
        "vlm3r": "c1_vlm3r",
    }.get(args.architecture)
    args.fusion_init_seed = 0
    args.skip_spatial_tower_load = True
    args.seed = 42
    args.add_time_instruction = None
    args.spatial_feature_dir = args.feature_root
    return args


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.calibration_manifest).resolve()
    manifest = load_manifest(manifest_path, args.max_samples)
    install_forward_frame_loader(Path(args.forward_frames_root).resolve())
    device = torch.device(args.device)
    dtype = torch_dtype_from_name(args.dtype)
    tokenizer, model, image_processor = load_model(args, device, dtype)
    # The official C1 default is shared_canonical.  This explicit second
    # application only enables the documented seedless role_offset option;
    # neither option reads RNG state or a trained fusion checkpoint.
    if args.architecture.startswith("spatialstack"):
        apply_spatialstack_c1(model.get_model().get_cut3r_spatialstack_merger(), qk_basis_mode=args.qk_basis_mode)
    elif args.architecture == "vlm3r":
        apply_vlm3r_c1(model.get_model().get_fusion_block(), qk_basis_mode=args.qk_basis_mode)
    model.eval()
    dataset, collator, by_video = build_dataset(args, tokenizer, image_processor)
    samples: list[tuple[int, dict[str, Any]]] = []
    for entry in manifest:
        video = str(entry["video_path"])
        if video not in by_video:
            raise RuntimeError(f"Calibration manifest video is absent from the SFT dataset prompt mapping: {video}")
        samples.append((by_video[video], entry))
    if args.architecture == "base":
        r0, base_stats = base_calibration(model=model, samples=samples, args=args, dataset=dataset, collator=collator, device=device, dtype=dtype)
        artifact: dict[str, Any] = {"architecture": "base", "r0": r0, "base_block_updates": base_stats}
    else:
        base_path = Path(args.base_calibration).resolve()
        with base_path.open("r", encoding="utf-8") as handle:
            base_artifact = json.load(handle)
        r0 = finite_positive(base_artifact.get("r0"), "base artifact r0")
        artifact = {
            "architecture": args.architecture,
            "r0": r0,
            "base_calibration": str(base_path),
            "base_calibration_sha256": sha256_file(base_path),
            "base_block_updates": base_artifact.get("base_block_updates"),
        }
        if args.architecture.startswith("spatialstack"):
            artifact["layers"] = calibrate_spatialstack(model=model, samples=samples, args=args, dataset=dataset, collator=collator, device=device, dtype=dtype, r0=r0)
        else:
            artifact["vlm3r"] = calibrate_vlm3r(model=model, samples=samples, args=args, dataset=dataset, collator=collator, device=device, dtype=dtype, r0=r0)
    artifact.update({
        "schema_version": "c1_calibration_v1",
        "canonicalization_scheme_version": SCHEME_VERSION,
        "canonical_matrix_scheme": matrix_scheme_metadata(dimensions=(768, 1152, 3584), qk_basis_mode=args.qk_basis_mode),
        "qk_basis_mode": args.qk_basis_mode,
        "calibration_manifest": str(manifest_path),
        "calibration_manifest_sha256": sha256_file(manifest_path),
        "num_calibration_samples": len(samples),
        "calibration_input": "cached_32_frame_probe_preprocessing_with_real_sft_human_prompt_and_empty_assistant_turn",
        "residual_statistic": "median_over_samples_of_rms_delta_over_rms_hidden_per_injection_site",
        "residual_budget_interpretation": "r0 is per injection site; SpatialStack L0/L1/L2 are each calibrated to r0 and are not divided",
        "feature_dims": {"cut3r": 768, "vlm3r_visual": 1152, "llm_hidden": 3584},
        "spatialstack": {"cut3r_source_layers": [6, 9, 12], "llm_injection_layers": [0, 1, 2], "cross_attn_heads": 28, "head_dim": 128},
        "no_training": True,
    })
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[C1] wrote calibration artifact: {output}")


if __name__ == "__main__":
    main()
