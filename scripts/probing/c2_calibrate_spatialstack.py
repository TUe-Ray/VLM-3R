#!/usr/bin/env python
"""Sequential, label-free C2 CCA-QK calibration for SpatialStack V1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llava.model.c1_structured_isometry import apply_spatialstack_c1  # noqa: E402
from llava.model.c2_cca_qk import (  # noqa: E402
    ARTIFACT_SCHEMA,
    PairedQKObserver,
    c1_stat_contract,
    compose_c2_qk,
)
from scripts.diagnose_layerwise_spatial_hidden_scan import load_model  # noqa: E402
from scripts.probing.c1_calibrate_fusion import (  # noqa: E402
    build_dataset,
    finite_positive,
    load_manifest,
    prepare_input,
    rms,
    run_decoder,
    sha256_file,
    summary,
)
from scripts.probing.depth_probe_common import (  # noqa: E402
    DEFAULT_DATA_YAML,
    DEFAULT_FAST_FEATURE_ROOT,
    DEFAULT_SPATIAL_FEATURES_SUBDIR,
    torch_dtype_from_name,
)
from scripts.probing.local_depth_probe_cache import install_forward_frame_loader  # noqa: E402


class ResidualObserver:
    def __init__(self, block: torch.nn.Module):
        self.block = block
        self.handles = []
        self.h_sq = 0.0
        self.h_count = 0
        self.delta_sq = 0.0
        self.delta_count = 0

    def __enter__(self):
        self.handles = [
            self.block.register_forward_pre_hook(self._on_input),
            self.block.register_forward_hook(self._on_output),
        ]
        return self

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _on_input(self, _module, inputs):
        value = inputs[0].detach().float()
        self.h_sq += float(value.square().sum().item())
        self.h_count += int(value.numel())

    def _on_output(self, _module, _inputs, output):
        value = output[0] if isinstance(output, (tuple, list)) else output
        value = value.detach().float()
        self.delta_sq += float(value.square().sum().item())
        self.delta_count += int(value.numel())

    def ratio(self) -> float:
        if self.h_count <= 0 or self.delta_count <= 0:
            raise RuntimeError("C2 residual observer did not see a cross-attention forward.")
        return finite_positive(
            math.sqrt(self.delta_sq / self.delta_count) / math.sqrt(self.h_sq / self.h_count),
            "C2 RMS(delta)/RMS(H)",
        )


def forward_samples(args, model, dataset, collator, samples, device, dtype, *, block=None, cca=False, logits=False, residual=False):
    ratios: list[float] = []
    pre_q_moments = []
    logit_observer = None
    if block is not None and (cca or logits):
        logit_observer = PairedQKObserver(block, collect_cca=cca, collect_logits=logits)
    context = logit_observer if logit_observer is not None else _NullContext()
    with context:
        for dataset_index, _entry in samples:
            prepared, _metadata, payload = prepare_input(
                args=args, model=model, collator=collator, dataset=dataset,
                dataset_index=dataset_index, device=device, model_dtype=dtype,
            )
            residual_observer = ResidualObserver(block) if block is not None and residual else _NullContext()
            with residual_observer:
                run_decoder(model, prepared, payload, "spatialstack_cross_attn_v1")
            if residual:
                ratios.append(residual_observer.ratio())
    if logit_observer is not None and logit_observer.pre_q_moment.count:
        pre_q_moments = {
            "count": logit_observer.pre_q_moment.count,
            "mean": logit_observer.pre_q_moment.total / logit_observer.pre_q_moment.count,
            "rms": math.sqrt(logit_observer.pre_q_moment.total_sq / logit_observer.pre_q_moment.count),
        }
    return logit_observer, ratios, pre_q_moments


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _state_dict_clone(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def _same_state(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> bool:
    return first.keys() == second.keys() and all(torch.equal(first[key], second[key]) for key in first)


def _correlation_summary(value: torch.Tensor) -> dict[str, Any]:
    flat = value.detach().double().flatten()
    return {
        "mean": float(flat.mean().item()), "median": float(flat.median().item()),
        "min": float(flat.min().item()), "max": float(flat.max().item()),
        "largest": [float(x) for x in flat.sort(descending=True).values[:8].tolist()],
    }


def calibrate(args, model, dataset, collator, samples, reference, device, dtype) -> dict[str, Any]:
    qk_contract, residual_contract = c1_stat_contract(reference)
    merger = model.get_model().get_cut3r_spatialstack_merger()
    if merger is None or str(merger.fusion_type).lower() != "cross_attn":
        raise RuntimeError("C2 requires the C1 SpatialStack V1 merger.")
    apply_spatialstack_c1(merger, qk_basis_mode=reference.get("qk_basis_mode", "shared_canonical"))
    layer_ids = [int(layer) for layer in merger.llm_layers]
    if args.max_layers is not None:
        layer_ids = layer_ids[: args.max_layers]
    for layer, block in merger.cross_attn_blocks.items():
        block.set_c1_state(enabled=True, qk_scale=1.0, residual_gain=0.0, collect_diagnostics=False)
        for parameter in block.parameters():
            parameter.requires_grad_(False)

    result: dict[str, Any] = {}
    target_logit_std = float(qk_contract["target_std"])
    r_target = finite_positive(reference["r0"], "C1 r0")
    for layer in layer_ids:
        block = merger.cross_attn_blocks[str(layer)]
        earlier = {str(site): float(merger.cross_attn_blocks[str(site)].c1_residual_gain.item()) for site in layer_ids if site < layer}
        # CCA sees the true layer input distribution: earlier layers have their
        # frozen C2 gains, this and later layers have zero gain.
        observer, _ratios, pre_q = forward_samples(
            args, model, dataset, collator, samples, device, dtype, block=block, cca=True,
        )
        if observer is None:
            raise RuntimeError(f"C2 L{layer} did not observe projected Q/K outputs.")
        state = observer.fit_cca(args.cca_regularization_relative)
        q_before = _state_dict_clone(block.q_proj)
        k_before = _state_dict_clone(block.k_proj)
        v_before, o_before = _state_dict_clone(block.v_proj), _state_dict_clone(block.out_proj)
        compose_c2_qk(block, state)
        if _same_state(q_before, _state_dict_clone(block.q_proj)):
            raise RuntimeError(f"C2 L{layer} Q weights did not change after CCA.")
        if _same_state(k_before, _state_dict_clone(block.k_proj)):
            raise RuntimeError(f"C2 L{layer} K weights did not change after CCA.")
        if not _same_state(v_before, _state_dict_clone(block.v_proj)) or not _same_state(o_before, _state_dict_clone(block.out_proj)):
            raise RuntimeError(f"C2 L{layer} modified V or O.")

        block.set_c1_state(qk_scale=1.0, residual_gain=0.0)
        raw_observer, _unused, _ = forward_samples(
            args, model, dataset, collator, samples, device, dtype, block=block, logits=True,
        )
        raw_std = raw_observer.logit_moment.std(f"C2 L{layer} raw logit std")
        qk_scale = math.sqrt(target_logit_std / raw_std)
        block.set_c1_state(qk_scale=qk_scale, residual_gain=1.0)
        scaled_observer, raw_ratios, _ = forward_samples(
            args, model, dataset, collator, samples, device, dtype, block=block, logits=True, residual=True,
        )
        post_std = scaled_observer.logit_moment.std(f"C2 L{layer} scaled logit std")
        if not math.isclose(post_std, target_logit_std, rel_tol=2e-3, abs_tol=2e-4):
            raise RuntimeError(
                f"C2 L{layer} Q/K scale did not reproduce C1 target logit std: "
                f"target={target_logit_std:.8g}, measured={post_std:.8g}."
            )
        raw_ratio = float(summary(raw_ratios)["median"])
        alpha = r_target / raw_ratio
        block.set_c1_state(qk_scale=qk_scale, residual_gain=alpha)
        _unused, final_ratios, _ = forward_samples(
            args, model, dataset, collator, samples, device, dtype, block=block, residual=True,
        )
        final_summary = summary(final_ratios)
        if not math.isclose(float(final_summary["median"]), r_target, rel_tol=2e-3, abs_tol=2e-5):
            raise RuntimeError(
                f"C2 L{layer} residual calibration did not reproduce C1 target: "
                f"target={r_target:.8g}, measured={float(final_summary['median']):.8g}."
            )
        result[str(layer)] = {
            **{key: value.cpu() for key, value in state.items()},
            "qk_scale": qk_scale,
            "residual_gain": alpha,
            "diagnostics": {
                "layer_id": layer,
                "paired_tokens": int(state["pair_count_per_head"][0].item()),
                "heads": int(block.num_heads), "head_dim": int(block.head_dim),
                "cca_regularization_relative": float(args.cca_regularization_relative),
                "canonical_correlations": _correlation_summary(state["canonical_correlations"]),
                "qk_logit_std_before_scale": raw_std,
                "qk_logit_std_after_scale": post_std,
                "qk_logit_target_std": target_logit_std,
                "raw_residual_over_hidden": summary(raw_ratios),
                "final_residual_over_hidden": final_summary,
                "residual_target": r_target, "alpha": alpha,
                "pre_q_input_moments": pre_q,
                "earlier_calibrated_residual_gains": earlier,
            },
        }
        print("[C2] " + json.dumps(result[str(layer)]["diagnostics"], sort_keys=True), flush=True)
    return {
        "schema_version": ARTIFACT_SCHEMA,
        "complete": len(layer_ids) == len(merger.llm_layers),
        "c1_reference": {
            "architecture": reference["architecture"], "canonicalization_scheme_version": reference["canonicalization_scheme_version"],
            "qk_basis_mode": reference.get("qk_basis_mode", "shared_canonical"), "r0": r_target,
            "qk_logit_calibration": qk_contract, "residual_calibration": residual_contract,
        },
        "layers": result,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c1-calibration-json", required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--siglip-path", required=True)
    parser.add_argument("--feature-root", default=str(DEFAULT_FAST_FEATURE_ROOT))
    parser.add_argument("--spatial-features-subdir", default=DEFAULT_SPATIAL_FEATURES_SUBDIR)
    parser.add_argument("--forward-frames-root", required=True)
    parser.add_argument("--train-data-json", default=str(DEFAULT_DATA_YAML))
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
    parser.add_argument("--cca-regularization-relative", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int, default=None, help="Smoke-only deterministic prefix.")
    parser.add_argument("--max-layers", type=int, default=None, help="Smoke-only prefix of native injection layers.")
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.max_layers is not None and args.max_layers <= 0:
        parser.error("--max-layers must be positive")
    args.model_loading_mode = "pre_sft_fusion"
    args.pre_sft_fusion_variant = "c1_ss_cross_attn_v1"
    # The shared C1 input/decoder helpers use this to select the native
    # SpatialStack payload construction and injection keyword.
    args.architecture = "spatialstack_cross_attn_v1"
    args.fusion_init_seed = 0
    args.common_model_init_seed = 0
    args.skip_spatial_tower_load = True
    args.seed = 42
    args.add_time_instruction = None
    args.spatial_feature_dir = args.feature_root
    return args


def main():
    args = parse_args()
    manifest_path, c1_path = Path(args.calibration_manifest).resolve(), Path(args.c1_calibration_json).resolve()
    with c1_path.open("r", encoding="utf-8") as handle:
        reference = json.load(handle)
    c1_stat_contract(reference)
    c1_manifest_sha256 = reference.get("calibration_manifest_sha256")
    manifest_sha256 = sha256_file(manifest_path)
    if not isinstance(c1_manifest_sha256, str) or c1_manifest_sha256 != manifest_sha256:
        raise RuntimeError(
            "C2 must use the exact calibration manifest used to establish the C1 residual target: "
            f"C1={c1_manifest_sha256!r}, supplied={manifest_sha256}."
        )
    manifest = load_manifest(manifest_path, args.max_samples)
    install_forward_frame_loader(Path(args.forward_frames_root).resolve())
    device, dtype = torch.device(args.device), torch_dtype_from_name(args.dtype)
    tokenizer, model, image_processor = load_model(args, device, dtype)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("C2 requires a fully frozen model.")
    dataset, collator, by_video = build_dataset(args, tokenizer, image_processor)
    samples = []
    for entry in manifest:
        video = str(entry["video_path"])
        if video not in by_video:
            raise RuntimeError(f"Calibration manifest video is absent from the SFT dataset: {video}")
        samples.append((by_video[video], entry))
    with torch.inference_mode():
        artifact = calibrate(args, model, dataset, collator, samples, reference, device, dtype)
    merger = model.get_model().get_cut3r_spatialstack_merger()
    if merger is None:
        raise RuntimeError("C2 merger disappeared after calibration.")
    artifact.update({
        "c1_calibration_path": str(c1_path), "c1_calibration_sha256": sha256_file(c1_path),
        "calibration_manifest": str(manifest_path), "calibration_manifest_sha256": manifest_sha256,
        "num_calibration_samples": len(samples), "calibration_input": "cached_32_frame_probe_preprocessing_with_real_sft_human_prompt_and_empty_assistant_turn",
        "feature_root": str(Path(args.feature_root).resolve()), "spatial_features_subdir": args.spatial_features_subdir,
        "model_identifier": {
            "fusion_type": str(merger.fusion_type),
            "hidden_size": int(merger.hidden_size),
            "feature_dim": int(merger.feature_dim),
            "cross_attn_heads": int(merger.cross_attn_heads),
            "llm_layers": [int(value) for value in merger.llm_layers],
            "cut3r_layers": [int(value) for value in merger.cut3r_layers],
            "model_path": str(Path(args.model_path).resolve()),
            "model_config_sha256": sha256_file(Path(args.model_path).resolve() / "config.json"),
        },
    })
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    print(f"[C2] wrote {'complete' if artifact['complete'] else 'smoke/incomplete'} artifact: {output}")


if __name__ == "__main__":
    main()
