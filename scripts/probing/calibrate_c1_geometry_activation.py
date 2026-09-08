#!/usr/bin/env python
"""Solve the unlabeled C1 activation scalar for GeoRoPE pre-SFT architectures.

This script consumes full-32-frame activation captures produced by
``extract_depth_probe_features.py --calibration-capture-pre-llm``.  It never
uses depth labels, a probe score, gradients, an optimizer, or post-SFT model
weights.  Dense architecture maps are regenerated from the frozen C1 VLM3R
artifact; only one outer activation scalar is solved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnose_layerwise_spatial_hidden_scan import load_model
from llava.model.c1_structured_isometry import apply_c1_calibration_artifact
from llava.model.multimodal_fusion_block.builder import CrossAttentionFusion
from llava.model.c1_structured_isometry import apply_vlm3r_c1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(value.detach().float().square().mean()).item())


def summary(values: list[float]) -> dict[str, float | int]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": int(tensor.numel()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "mean": float(tensor.mean().item()),
        "median": float(tensor.median().item()),
        "std": float(tensor.std(unbiased=False).item()),
    }


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    av, bv = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
    value = float(torch.nn.functional.cosine_similarity(av, bv, dim=0, eps=1e-12).item())
    return max(-1.0, min(1.0, value))


def module_device(module: torch.nn.Module, fallback: torch.device) -> torch.device:
    # Accelerate keeps CPU-offloaded parameters resident on CPU between calls,
    # but records the actual execution GPU on the module hook.  Calibration
    # invokes the frozen projector directly, so prefer that execution device
    # instead of accidentally running its fp16 MLP on CPU.
    # The projector container can have child Linear layers individually
    # hooked by Accelerate, so search the small module tree before inspecting
    # CPU-offloaded parameter storage.
    for candidate in module.modules():
        hook = getattr(candidate, "_hf_hook", None)
        execution_device = getattr(hook, "execution_device", None)
        if isinstance(execution_device, str):
            execution_device = torch.device(execution_device)
        if isinstance(execution_device, torch.device) and execution_device.type != "meta":
            return execution_device
    declared = getattr(module, "device", None)
    if isinstance(declared, torch.device) and declared.type != "meta":
        return declared
    for parameter in module.parameters():
        if not parameter.is_meta:
            return parameter.device
    return fallback


def module_dtype(module: torch.nn.Module, fallback: torch.dtype) -> torch.dtype:
    for parameter in module.parameters():
        if not parameter.is_meta and parameter.is_floating_point():
            return parameter.dtype
    return fallback


def move(value: Any, *, device: torch.device, dtype: torch.dtype) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype if value.is_floating_point() else None)
    if isinstance(value, tuple):
        return tuple(move(item, device=device, dtype=dtype) for item in value)
    if isinstance(value, list):
        return [move(item, device=device, dtype=dtype) for item in value]
    return value


def load_capture(path: Path, expected_architecture: str) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("architecture") != expected_architecture:
        raise RuntimeError(f"Invalid {expected_architecture} calibration capture: {path}")
    raw = payload.get("raw_pre_llm")
    if not isinstance(raw, dict) or not isinstance(raw.get("siglip_output"), torch.Tensor):
        raise RuntimeError(f"Calibration capture lacks raw SigLIP output: {path}")
    return payload


def load_tensor_payload(path: Path) -> dict[str, Any]:
    """Load one compact, calibration-owned branch materialization."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema_version") != "c1_geo_rope_branch_v1":
        raise RuntimeError(f"Invalid compact GeoRoPE branch materialization: {path}")
    return payload


def make_load_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_loading_mode="pre_sft_fusion",
        model_path=str(args.model_base),
        runtime_root=str(args.runtime_root),
        siglip_path=str(args.siglip_path),
        attn_implementation=None,
        device_map=args.device_map,
        pre_sft_defer_dispatch=False,
        pre_sft_gpu_weight_budget=args.pre_sft_gpu_weight_budget,
        pre_sft_cpu_offload_budget=args.pre_sft_cpu_offload_budget,
        common_model_init_seed=0,
        pre_sft_fusion_variant=(
            "c1_geo_rope_fusion" if args.architecture == "geo_rope_fusion" else "c1_visual_geo_rope"
        ),
        fusion_init_seed=0,
        spatialstack_cut3r_layers=None,
        spatialstack_llm_layers=None,
    )


def projector_pair(projector: torch.nn.Module, base: torch.Tensor, fused: torch.Tensor, fallback: torch.device, dtype: torch.dtype):
    device = module_device(projector, fallback)
    project_dtype = module_dtype(projector, dtype)
    with torch.no_grad():
        e_base = projector(base.to(device=device, dtype=project_dtype))
        e_fused = projector(fused.to(device=device, dtype=project_dtype))
    return e_base, e_fused


def projector_forward(projector: torch.nn.Module, value: torch.Tensor, fallback: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Run the frozen normal projector once at its dispatched placement."""
    target_device = module_device(projector, fallback)
    target_dtype = module_dtype(projector, dtype)
    with torch.no_grad():
        return projector(value.to(device=target_device, dtype=target_dtype))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", required=True, choices=("geo_rope_fusion", "visual_geo_rope"))
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--c1-reference-json", required=True, type=Path)
    parser.add_argument("--model-base", required=True, type=Path)
    parser.add_argument("--siglip-path", required=True, type=Path)
    parser.add_argument("--runtime-root", default=str(REPO_ROOT / ".offline_runtime"), type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", choices=("auto", "cuda:0", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--pre-sft-gpu-weight-budget", default="5GiB")
    parser.add_argument("--pre-sft-cpu-offload-budget", default="45GiB")
    parser.add_argument("--max-bisection-steps", type=int, default=24)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_architecture = "geo_rope_fusion" if args.architecture == "geo_rope_fusion" else "visual_3d_rope"
    capture_paths = sorted(args.capture_root.glob("*.pt"))
    if args.max_samples is not None:
        capture_paths = capture_paths[: int(args.max_samples)]
    if not capture_paths:
        raise RuntimeError(f"No calibration captures under {args.capture_root}")
    reference = json.loads(args.c1_reference_json.read_text(encoding="utf-8"))
    if reference.get("architecture") != "vlm3r":
        raise RuntimeError("The reference artifact must be the official C1 VLM3R calibration.")
    r0 = float(reference["r0"])
    reference_qk = float(reference["vlm3r"]["s_qk"])
    requested_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    device = torch.device(args.device)
    tokenizer, model, _processor = load_model(make_load_args(args), device, requested_dtype)
    del tokenizer
    apply_c1_calibration_artifact(model, reference)
    model.eval()
    projector = model.get_model().mm_projector
    # Calibration never invokes the decoder after the C1 fusion branch.  Make
    # the compact frozen projector resident on the otherwise unused second
    # visible GPU so repeated scalar evaluations do not execute its FP16 MLP
    # through CPU-offload hooks.  This is a placement-only operation: the
    # pretrained projector weights and its computation are unchanged.
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        projector_device = torch.device("cuda:1")
        try:
            from accelerate.hooks import remove_hook_from_module

            remove_hook_from_module(projector, recurse=True)
            projector.to(device=projector_device, dtype=requested_dtype).eval()
        except Exception as exc:
            raise RuntimeError(f"Could not materialize frozen projector on {projector_device} for calibration") from exc
    print(
        f"[INFO] calibration placement: projector_execution_device={module_device(projector, device)}",
        flush=True,
    )

    if args.architecture == "geo_rope_fusion":
        fusion = model.get_model().get_fusion_block()
        if fusion is None or not hasattr(fusion, "geo_rope_fusion_gate_q"):
            raise RuntimeError("Loaded C1 model is not GeoRoPE fusion.")
        # The newly constructed C1 fusion module starts on CPU even when the
        # base VLM was dispatched across GPUs.  Calibration is forward-only,
        # but must still use a CUDA-capable dtype for the canonical fp16 maps.
        compute_device = device if device.type == "cuda" else module_device(fusion, device)
        compute_dtype = module_dtype(fusion, requested_dtype)
        fusion.to(device=compute_device, dtype=compute_dtype)

        # This is the strict g=0 topology control, not a trainable model.
        # Keep its native outer dropout disabled exactly as the loaded C1
        # fusion is disabled by model.eval().
        baseline = CrossAttentionFusion(1152, 768, 1152, 18).to(
            device=compute_device, dtype=compute_dtype
        ).eval()
        apply_vlm3r_c1(baseline)
        baseline.set_c1_state(enabled=True, qk_scale=reference_qk, residual_gain=float(reference["vlm3r"]["lambda"]))

        # Lambda is an outer linear residual scale in this architecture:
        # fusion(clip, ..., lambda) == clip + lambda * branch.  Materialize
        # the two gate-dependent branches once, then sweep only the frozen
        # normal projector during the scalar solve.  This preserves the exact
        # executable forward map while avoiding hundreds of redundant Q/K
        # attention evaluations on the same 32 calibration videos.
        # Full calibration captures include 32 dense point maps and are about
        # 0.6 GB each.  Stream them into compact branch tensors instead of
        # retaining all 32 payloads in RAM: this is numerically identical but
        # avoids swapping the local host during the scalar solve.
        materialized_root = args.capture_root.parent / "geo_rope_branch_materializations"
        materialized_root.mkdir(parents=True, exist_ok=True)
        materialized_paths: list[Path] = []
        parity_errors, parity_ratios = [], []
        for capture_path in capture_paths:
            capture = load_capture(capture_path, expected_architecture)
            inputs = capture.get("fusion_inputs")
            if not isinstance(inputs, tuple) or len(inputs) != 4:
                raise RuntimeError(f"GeoRoPE capture lacks its four fusion inputs: {capture_path}")
            clip, patch, pos_clip, pos_spatial = move(inputs, device=compute_device, dtype=compute_dtype)
            with torch.no_grad():
                fusion.geo_rope_fusion_gate_q.fill_(1.0)
                fusion.geo_rope_fusion_gate_k.fill_(1.0)
                fusion.set_c1_state(enabled=True, residual_gain=1.0)
                full_fused, _ = fusion(clip, patch, pos_clip, pos_spatial)
                fusion.geo_rope_fusion_gate_q.zero_()
                fusion.geo_rope_fusion_gate_k.zero_()
                no_geo_fused, _ = fusion(clip, patch, pos_clip, pos_spatial)
                baseline_fused, _ = baseline(clip, patch)
                e_base = projector_forward(projector, clip, device, requested_dtype)
                e_no_geo = projector_forward(
                    projector,
                    clip + float(reference["vlm3r"]["lambda"]) * (no_geo_fused - clip),
                    device,
                    requested_dtype,
                )
                e_baseline = projector_forward(projector, baseline_fused, device, requested_dtype)
            no_geo_delta = (e_no_geo - e_base).detach().float().cpu()
            expected_delta = (e_baseline - e_base).detach().float().cpu()
            e_base_cpu = e_base.detach().float().cpu()
            parity_error = rms(no_geo_delta - expected_delta)
            parity_ratio = rms(no_geo_delta) / rms(e_base_cpu)
            parity_errors.append(parity_error)
            parity_ratios.append(parity_ratio)
            materialized_path = materialized_root / capture_path.name
            torch.save({
                "schema_version": "c1_geo_rope_branch_v1",
                "capture_name": capture_path.name,
                "clip": clip.detach().cpu(),
                "full_branch": (full_fused - clip).detach().cpu(),
                "no_geo_branch": (no_geo_fused - clip).detach().cpu(),
                "e_base": e_base_cpu,
                "parity_error": parity_error,
                "parity_ratio": parity_ratio,
            }, materialized_path)
            materialized_paths.append(materialized_path)
            del capture, inputs, clip, patch, pos_clip, pos_spatial
            del full_fused, no_geo_fused, baseline_fused, e_base, e_no_geo, e_baseline
            if compute_device.type == "cuda":
                torch.cuda.empty_cache()
        parity_error_summary = summary(parity_errors)
        if float(parity_error_summary["max"]) > 1e-3:
            raise RuntimeError(
                "GeoRoPE g=0 does not reproduce the mapped patch-only C1 baseline within numerical tolerance: "
                f"{parity_error_summary}"
            )

        def evaluate(lam: float, branch_key: str) -> tuple[list[float], list[torch.Tensor], list[torch.Tensor]]:
            ratios, deltas, bases = [], [], []
            for materialized_path in materialized_paths:
                entry = load_tensor_payload(materialized_path)
                clip = entry["clip"].to(device=compute_device, dtype=compute_dtype)
                branch = entry[branch_key].to(device=compute_device, dtype=compute_dtype)
                with torch.no_grad():
                    e_fused = projector_forward(projector, clip + float(lam) * branch, device, requested_dtype)
                delta = (e_fused.detach().float().cpu() - entry["e_base"])
                ratios.append(rms(delta) / rms(entry["e_base"]))
                deltas.append(delta)
                bases.append(entry["e_base"])
                del entry, clip, branch, e_fused
                if compute_device.type == "cuda":
                    torch.cuda.empty_cache()
            return ratios, deltas, bases
        lower, upper = 0.0, 1.0
        observed = []
        while True:
            ratios, _, _ = evaluate(upper, "full_branch")
            value = float(summary(ratios)["median"])
            observed.append({"scale": upper, "median_ratio": value})
            if value >= r0:
                break
            upper *= 2.0
            if upper > 2.0 ** 20:
                raise RuntimeError(f"Could not bracket GeoRoPE r0={r0}; observed={observed}")
        for _ in range(int(args.max_bisection_steps)):
            middle = (lower + upper) / 2.0
            ratios, _, _ = evaluate(middle, "full_branch")
            if float(summary(ratios)["median"]) < r0:
                lower = middle
            else:
                upper = middle
        activation = (lower + upper) / 2.0
        ratios, full_deltas, _ = evaluate(activation, "full_branch")
        _, no_geo_deltas, _ = evaluate(activation, "no_geo_branch")
        metadata = {
            "activation": {"gate_q": 1.0, "gate_k": 1.0, "lambda_geo": activation},
            "parity": {
                "gate_zero_reference_lambda": float(reference["vlm3r"]["lambda"]),
                "projected_delta_rms_error": parity_error_summary,
                "reference_ratio": summary(parity_ratios),
            },
            "full_geo_vs_no_geo_residual_cosine": summary([cosine(a, b) for a, b in zip(full_deltas, no_geo_deltas)]),
            "bracket": observed,
        }
    else:
        module = model.get_model().get_geometry_aware_projection()
        if module is None or len(module.layers) != 1:
            raise RuntimeError("Loaded C1 model is not a one-layer Visual GeoRoPE projection.")
        layer = module.layers[0]
        # As above, this C1-created projection layer is initially CPU-resident
        # rather than colocated with the dispatched pretrained visual tower.
        compute_device = device if device.type == "cuda" else module_device(layer, device)
        compute_dtype = module_dtype(layer, requested_dtype)
        layer.to(device=compute_device, dtype=compute_dtype)

        def evaluate(gamma: float) -> tuple[list[float], list[torch.Tensor], list[torch.Tensor]]:
            with torch.no_grad():
                layer.gamma_attn.fill_(float(gamma))
                layer.gamma_ffn.fill_(float(gamma))
            ratios, deltas, bases = [], [], []
            for capture_path in capture_paths:
                capture = load_capture(capture_path, expected_architecture)
                visual = capture["raw_pre_llm"]["siglip_output"].to(device=compute_device, dtype=compute_dtype)
                pos = capture["geometry_pos"].to(device=compute_device, dtype=compute_dtype)
                mask = capture["geometry_mask"].to(device=compute_device, dtype=torch.bool)
                with torch.no_grad():
                    fused = layer(visual, pos, attention_mask=mask)
                    e_base, e_fused = projector_pair(projector, visual, fused, device, requested_dtype)
                delta = (e_fused - e_base).detach().float().cpu()
                base = e_base.detach().float().cpu()
                ratios.append(rms(delta) / rms(base))
                deltas.append(delta)
                bases.append(base)
                del capture, visual, pos, mask, fused, e_base, e_fused
                if compute_device.type == "cuda":
                    torch.cuda.empty_cache()
            return ratios, deltas, bases

        zero_ratios, zero_deltas, _ = evaluate(0.0)
        zero_rms = [rms(value) for value in zero_deltas]
        scan = []
        upper = 1.0
        previous = 0.0
        while True:
            ratios, _, _ = evaluate(upper)
            value = float(summary(ratios)["median"])
            scan.append({"gamma": upper, "median_ratio": value})
            if value + 1e-6 < previous:
                raise RuntimeError(f"Visual GeoRoPE residual is non-monotonic during C1 bracketing: {scan}")
            previous = value
            if value >= r0:
                break
            upper *= 2.0
            if upper > 2.0 ** 20:
                raise RuntimeError(f"Could not bracket Visual GeoRoPE r0={r0}; scan={scan}")
        lower = 0.0
        for _ in range(int(args.max_bisection_steps)):
            middle = (lower + upper) / 2.0
            ratios, _, _ = evaluate(middle)
            if float(summary(ratios)["median"]) < r0:
                lower = middle
            else:
                upper = middle
        activation = (lower + upper) / 2.0
        ratios, full_deltas, full_bases = evaluate(activation)
        metadata = {
            "activation": {"gamma_c1": activation},
            "native_zero_identity_delta_rms": summary(zero_rms),
            "residual_cosine_vs_base": summary([cosine(delta, base) for delta, base in zip(full_deltas, full_bases)]),
            "bracket": scan,
        }

    achieved = summary(ratios)
    if abs(float(achieved["median"]) - r0) > 2e-4:
        raise RuntimeError(f"C1 calibration missed r0={r0}: achieved={achieved}")
    output = {
        "schema_version": "c1_geometry_activation_v1",
        "architecture": args.architecture,
        "no_training": True,
        "calibration_statistic": "median_s RMS(E_arch(s)-E_base(s))/RMS(E_base(s))",
        "r0": r0,
        "achieved_ratio": achieved,
        "activation": metadata.pop("activation"),
        "diagnostics": metadata,
        "reference_c1_artifact": str(args.c1_reference_json.resolve()),
        "reference_c1_artifact_sha256": sha256_file(args.c1_reference_json),
        "capture_root": str(args.capture_root.resolve()),
        "capture_count": len(capture_paths),
        "capture_architecture": expected_architecture,
        "geometry_source": "predicted CUT3R point_maps_ref",
        "qk_reference_scale": reference_qk,
        "projector_execution_device": str(module_device(projector, device)),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
