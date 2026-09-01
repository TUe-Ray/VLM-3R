#!/usr/bin/env python
"""Compare full and EoMT-selective CUT3R K/V under frozen C1-VLM3R scales.

This is intentionally a forward-only diagnostic.  It uses the fixed C1
calibration manifest, canonical C1-VLM3R matrices, and frozen lambda from the
given artifact.  It neither changes lambda nor constructs labels, targets,
optimizers, or probes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llava.model.c1_structured_isometry import apply_c1_calibration_artifact  # noqa: E402
from scripts.diagnose_layerwise_spatial_hidden_scan import load_model, make_data_args, move_to_device  # noqa: E402
from scripts.probing.c1_calibrate_fusion import (  # noqa: E402
    build_dataset,
    load_manifest,
    module_device,
    move_value,
    prompt_only_item,
    sha256_file,
)
from scripts.probing.depth_probe_common import (  # noqa: E402
    DEFAULT_SPATIAL_FEATURES_SUBDIR,
    torch_dtype_from_name,
)
from scripts.probing.extract_depth_probe_features import load_eomt_consumer_cache  # noqa: E402
from scripts.probing.local_depth_probe_cache import install_forward_frame_loader  # noqa: E402


DEFAULT_BASE = "/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2"
DEFAULT_SIGLIP = "/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384"
DEFAULT_FRAMES = "/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1"
DEFAULT_CUT3R = "/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features"
DEFAULT_C1 = "/home/shaoruei/probe_outputs/c1_vlm3r_v1/official/vlm3r.json"
DEFAULT_EOMT_ROOT = "/home/shaoruei/probe_cache/eomt_consumer_grid_v2"
DEFAULT_OUTPUT = "/home/shaoruei/probe_outputs/c1_eomt_selective_calibration_v1"
ACTIVE_EPSILON = 1e-6


def rms(value: torch.Tensor) -> float:
    value = value.detach().float()
    if value.numel() == 0:
        raise RuntimeError("RMS received an empty tensor")
    result = float(value.square().mean().sqrt().item())
    if not math.isfinite(result):
        raise RuntimeError(f"RMS is non-finite: {result}")
    return result


def finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError(f"{name} is non-finite: {value}")
    return value


def describe(values: Iterable[float | int]) -> dict[str, float | int]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    if tensor.numel() == 0 or not torch.isfinite(tensor).all():
        raise RuntimeError("Cannot summarize an empty or non-finite statistic")
    return {
        "count": int(tensor.numel()),
        "median": float(tensor.median().item()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--c1-calibration-json", default=DEFAULT_C1)
    parser.add_argument("--model-path", default=DEFAULT_BASE)
    parser.add_argument("--siglip-path", default=DEFAULT_SIGLIP)
    parser.add_argument("--forward-frames-root", default=DEFAULT_FRAMES)
    parser.add_argument("--feature-root", default=DEFAULT_CUT3R)
    parser.add_argument("--spatial-features-subdir", default=DEFAULT_SPATIAL_FEATURES_SUBDIR)
    parser.add_argument(
        "--train-data-json",
        default=str(REPO_ROOT / "scripts" / "probing" / "scannet_depth_probe_local_data.yaml"),
        help="Local prompt manifest used by the official C1 VLM3R calibration.",
    )
    parser.add_argument("--eomt-consumer-cache-root", default=DEFAULT_EOMT_ROOT)
    parser.add_argument("--eomt-cache-validation", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--pre-sft-gpu-weight-budget", default="5GiB")
    parser.add_argument("--pre-sft-cpu-offload-budget", default="45GiB")
    parser.add_argument("--frames-upbound", type=int, default=32)
    parser.add_argument("--image-folder", default=DEFAULT_CUT3R)
    parser.add_argument("--video-folder", default=DEFAULT_CUT3R)
    parser.add_argument("--runtime-root", default=None)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Smoke only; official run uses all 32 samples.")
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    args.model_loading_mode = "pre_sft_fusion"
    args.pre_sft_fusion_variant = "c1_vlm3r"
    # build_dataset() shares the C1 architecture switch to decide whether
    # CUT3R sidecars are required.
    args.architecture = "vlm3r"
    args.fusion_init_seed = 0
    args.common_model_init_seed = 0
    args.skip_spatial_tower_load = True
    args.seed = 42
    args.add_time_instruction = None
    args.spatial_feature_dir = args.feature_root
    # Reuse the hardened cache loader, which requires the validated v2 cache.
    args.post_sft_architecture = "eomt_selective"
    args.verify_eomt_file_checksum = False
    if args.eomt_cache_validation is None:
        args.eomt_cache_validation = str(Path(args.eomt_consumer_cache_root) / "validation.json")
    return args


def eomt_overlay(config: Any, enabled: bool) -> None:
    """Set the executable selective configuration; ``enabled`` is the sole switch."""
    values = {
        "mm_eomt_selective_3d_enable": enabled,
        "mm_eomt_selective_3d_gate_type": "soft",
        "mm_eomt_selective_3d_selector_mode": "confidence",
        "mm_eomt_selective_3d_score_threshold": 0.8,
        "mm_eomt_selective_3d_topk": -1,
        "mm_eomt_selective_3d_class_type": "things",
        "mm_eomt_selective_3d_merge_mode": "soft_max_union",
        "mm_eomt_selective_3d_word_match_enable": True,
        "mm_eomt_selective_3d_empty_fallback": "zero_3d",
        "mm_eomt_word_match_source": "visible_grounded_words",
        "mm_eomt_word_match_mode": "hybrid_safe",
        "mm_eomt_word_match_no_match": "keep_masks",
        "mm_eomt_word_match_similarity_threshold": 0.86,
    }
    for name, value in values.items():
        setattr(config, name, value)


def prepared_embeddings(
    *, model: torch.nn.Module, batch: dict[str, Any], eomt_payload: dict[str, Any], enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]] | None]:
    """Run normal multimodal preparation and return C1-captured E base/branch."""
    eomt_overlay(model.get_model().config, enabled)
    fusion = model.get_model().get_fusion_block()
    fusion._c1_last_clip_features = None
    fusion._c1_last_branch = None
    prepare_fn = model.prepare_inputs_labels_for_multimodal
    with torch.no_grad():
        prepare_fn(
            input_ids=batch["input_ids"],
            position_ids=None,
            attention_mask=batch["attention_mask"],
            past_key_values=None,
            labels=None,
            images=batch["images"],
            spatial_features=batch.get("spatial_features"),
            point_maps=None,
            modalities=batch.get("modalities"),
            image_sizes=batch.get("image_sizes"),
            return_visual_metadata=True,
            # The exact same cache payload is passed in both conditions; full
            # ignores it because its selective-enable flag is false.
            eomt_cached_outputs=[eomt_payload],
        )
    clip, branch = fusion._c1_last_clip_features, fusion._c1_last_branch
    if not isinstance(clip, torch.Tensor) or not isinstance(branch, torch.Tensor):
        raise RuntimeError("C1 fusion did not capture visual base/branch tensors")
    debug = getattr(model, "_last_eomt_selective_debug", None)
    if enabled and (not isinstance(debug, list) or len(debug) != 32):
        raise RuntimeError("Selective condition did not execute all 32 EoMT gate frames")
    if not enabled and debug is not None:
        raise RuntimeError("Full condition unexpectedly executed EoMT selection")
    return clip.detach(), branch.detach(), debug


def project_pair(model: torch.nn.Module, clip: torch.Tensor, branch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fusion = model.get_model().get_fusion_block()
    lam = float(fusion.c1_residual_gain.item())
    projector = model.get_model().mm_projector
    projector_dtype = next((parameter.dtype for parameter in projector.parameters() if not parameter.is_meta), clip.dtype)
    projector_device = module_device(projector, clip.device)
    with torch.no_grad():
        clip = clip.to(device=projector_device, dtype=projector_dtype)
        branch = branch.to(device=projector_device, dtype=projector_dtype)
        e_base = projector(clip)
        e_fused = projector(clip + lam * branch)
    return e_base.detach(), e_fused.detach()


def prepare_batch(args: argparse.Namespace, model: Any, collator: Any, dataset: Any, dataset_index: int, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    item = prompt_only_item(dataset, dataset_index)
    batch = move_to_device(collator([item]), device, dtype)
    vision_tower = model.get_vision_tower()
    if vision_tower is None:
        raise RuntimeError("C1 VLM3R model has no vision tower")
    batch["images"] = move_value(batch["images"], module_device(vision_tower, device), getattr(vision_tower, "dtype", dtype))
    spatial_tower = model.get_spatial_tower()
    if spatial_tower is None:
        raise RuntimeError("C1 VLM3R model has no CUT3R sidecar tower")
    if "spatial_features" not in batch:
        raise RuntimeError("C1 VLM3R diagnostic received no CUT3R sidecar features")
    batch["spatial_features"] = move_value(batch["spatial_features"], module_device(spatial_tower, device))
    return batch


def classification(rho_full: float, rho_sel: float, cosine: float) -> str:
    """Conservative, explicitly recorded interpretation heuristic."""
    attenuation = 1.0 - rho_sel / rho_full
    direction_change = 1.0 - cosine
    if attenuation >= 0.20 and direction_change >= 0.10:
        return "both substantially"
    if attenuation >= 0.20:
        return "mainly attenuates residual magnitude"
    if direction_change >= 0.10:
        return "mainly changes residual direction/content"
    return "neither substantially under the recorded 20% magnitude / 0.10 cosine-distance thresholds"


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.calibration_manifest).resolve()
    c1_path = Path(args.c1_calibration_json).resolve()
    samples_manifest = load_manifest(manifest_path, args.max_samples)
    c1_artifact = json.loads(c1_path.read_text(encoding="utf-8"))
    if c1_artifact.get("architecture") != "vlm3r":
        raise RuntimeError("This diagnostic requires a C1 VLM3R calibration artifact")
    lambda_artifact = float(c1_artifact.get("vlm3r", {}).get("lambda"))
    install_forward_frame_loader(Path(args.forward_frames_root).resolve())
    device, dtype = torch.device(args.device), torch_dtype_from_name(args.dtype)
    tokenizer, model, image_processor = load_model(args, device, dtype)
    apply_c1_calibration_artifact(model, c1_artifact)
    fusion = model.get_model().get_fusion_block()
    runtime_lambda = float(fusion.c1_residual_gain.item())
    lambda_tolerance = max(1e-7, float(torch.finfo(fusion.c1_residual_gain.dtype).eps))
    if not math.isclose(runtime_lambda, lambda_artifact, rel_tol=0.0, abs_tol=lambda_tolerance):
        raise RuntimeError("Loaded C1 lambda differs from the frozen artifact beyond runtime dtype rounding")
    fusion.set_c1_state(capture_branch=True, collect_diagnostics=False)
    model.eval()
    dataset, collator, by_video = build_dataset(args, tokenizer, image_processor)
    samples: list[tuple[int, dict[str, Any]]] = []
    for record in samples_manifest:
        video = str(record["video_path"])
        if video not in by_video:
            raise RuntimeError(f"Calibration manifest video is absent from prompt mapping: {video}")
        samples.append((by_video[video], record))

    rows: list[dict[str, Any]] = []
    all_frame_gate_means: list[float] = []
    all_frame_active: list[float] = []
    all_frame_selected: list[int] = []
    all_frame_fallback: list[int] = []
    for number, (dataset_index, record) in enumerate(samples, start=1):
        batch = prepare_batch(args, model, collator, dataset, dataset_index, device, dtype)
        payload = load_eomt_consumer_cache(args, record)
        if payload is None:
            raise RuntimeError("Validated EoMT selective cache was not loaded")
        # A TITAN V cannot safely retain two full 32-frame vision forwards.
        # Materialize one condition's diagnostic tensors on CPU before the
        # paired forward; this changes no model calculation or comparison.
        full_clip, full_branch, _ = prepared_embeddings(model=model, batch=batch, eomt_payload=payload, enabled=False)
        e_base, e_full = project_pair(model, full_clip, full_branch)
        full_clip_cpu, e_base_cpu, e_full_cpu = (full_clip.cpu(), e_base.cpu(), e_full.cpu())
        del full_clip, full_branch, e_base, e_full
        fusion._c1_last_clip_features = None
        fusion._c1_last_branch = None
        if device.type == "cuda":
            torch.cuda.empty_cache()

        sel_clip, sel_branch, debug = prepared_embeddings(model=model, batch=batch, eomt_payload=payload, enabled=True)
        e_base_sel, e_sel = project_pair(model, sel_clip, sel_branch)
        sel_clip_cpu, e_base_sel_cpu, e_sel_cpu = (sel_clip.cpu(), e_base_sel.cpu(), e_sel.cpu())
        del sel_clip, sel_branch, e_base_sel, e_sel
        fusion._c1_last_clip_features = None
        fusion._c1_last_branch = None
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if not torch.equal(full_clip_cpu, sel_clip_cpu):
            max_diff = float((full_clip_cpu.float() - sel_clip_cpu.float()).abs().max().item())
            raise RuntimeError(f"Base visual embeddings changed across conditions (max abs diff {max_diff})")
        if not torch.equal(e_base_cpu, e_base_sel_cpu):
            max_diff = float((e_base_cpu.float() - e_base_sel_cpu.float()).abs().max().item())
            raise RuntimeError(f"Projected E_base changed across conditions (max abs diff {max_diff})")
        delta_full, delta_sel = (e_full_cpu - e_base_cpu).float(), (e_sel_cpu - e_base_cpu).float()
        base_rms = rms(e_base_cpu)
        full_rms, sel_rms = rms(delta_full), rms(delta_sel)
        cosine = finite(float(torch.nn.functional.cosine_similarity(delta_sel.reshape(1, -1), delta_full.reshape(1, -1)).item()), "residual cosine")
        assert debug is not None
        gate_means = [float(frame["gate_mean"]) for frame in debug]
        active_fractions = [float(frame["active_patch_fraction"]) for frame in debug]
        selected_queries = [int(frame["selected_queries"]) for frame in debug]
        fallback = [int(frame.get("fallback") == "zero_3d") for frame in debug]
        all_frame_gate_means.extend(gate_means)
        all_frame_active.extend(active_fractions)
        all_frame_selected.extend(selected_queries)
        all_frame_fallback.extend(fallback)
        row = {
            "sample_index": number - 1,
            "video_path": str(record["video_path"]),
            "scene_id": str(payload.get("scene_id")),
            "e_base_rms": base_rms,
            "delta_full_rms": full_rms,
            "delta_sel_rms": sel_rms,
            "rho_full": finite(full_rms / base_rms, "rho_full"),
            "rho_sel": finite(sel_rms / base_rms, "rho_sel"),
            "attenuation_ratio": finite(sel_rms / full_rms, "attenuation ratio"),
            "delta_cosine_similarity": cosine,
            "gate_mean_over_32_frames": finite(sum(gate_means) / 32.0, "mean gate"),
            "active_patch_fraction_over_32_frames": finite(sum(active_fractions) / 32.0, "active fraction"),
            "zero3d_fallback_frame_fraction": finite(sum(fallback) / 32.0, "fallback fraction"),
            "selected_query_count_mean": finite(sum(selected_queries) / 32.0, "selected count mean"),
            "selected_query_count_median": float(torch.tensor(selected_queries, dtype=torch.float64).median().item()),
            "selected_query_count_max": max(selected_queries),
        }
        rows.append(row)
        print(
            f"[C1 EoMT {number:02d}/{len(samples)}] rho_full={row['rho_full']:.6f} "
            f"rho_sel={row['rho_sel']:.6f} attenuation={row['attenuation_ratio']:.6f} cosine={cosine:.6f}"
        )

    rho_full = describe(row["rho_full"] for row in rows)
    rho_sel = describe(row["rho_sel"] for row in rows)
    attenuation = describe(row["attenuation_ratio"] for row in rows)
    cosine = describe(row["delta_cosine_similarity"] for row in rows)
    output_dir = Path(args.output_dir).resolve()
    summary = {
        "schema_version": "c1_eomt_selective_calibration_diagnostic_v1",
        "purpose": "forward_only_full_vs_eomt_selective_cut3r_kv_under_frozen_c1_vlm3r",
        "no_training": True,
        "no_probe_fitting": True,
        "conditions": {
            "full": "same C1 VLM3R canonical K/V with EoMT gate disabled",
            "selective": "same C1 VLM3R canonical K/V with soft things-only EoMT max-union gate and zero_3d fallback enabled",
            "only_condition_difference": "mm_eomt_selective_3d_enable",
        },
        "inputs": {
            "calibration_manifest": str(manifest_path),
            "calibration_manifest_sha256": sha256_file(manifest_path),
            "num_samples": len(rows),
            "model_path": str(Path(args.model_path).resolve()),
            "siglip_path": str(Path(args.siglip_path).resolve()),
            "forward_frames_root": str(Path(args.forward_frames_root).resolve()),
            "cut3r_feature_root": str(Path(args.feature_root).resolve()),
            "cut3r_feature_subdir": args.spatial_features_subdir,
            "eomt_consumer_cache_root": str(Path(args.eomt_consumer_cache_root).resolve()),
            "eomt_validation_sha256": sha256_file(Path(args.eomt_cache_validation).resolve()),
        },
        "frozen_c1": {
            "artifact": str(c1_path),
            "artifact_sha256": sha256_file(c1_path),
            "lambda_artifact": lambda_artifact,
            "lambda_runtime": runtime_lambda,
            "lambda_runtime_buffer_dtype": str(fusion.c1_residual_gain.dtype),
            "lambda_runtime_rounding_tolerance": lambda_tolerance,
            "lambda_recalibrated": False,
            "qk_scale": float(fusion.c1_qk_scale.item()),
        },
        "definitions": {
            "effective_projected_residual": "DeltaE = mm_projector(clip + frozen_lambda * fusion_branch) - mm_projector(clip)",
            "rho": "RMS(DeltaE) / RMS(E_base), per sample; reported headline is median over samples",
            "attenuation_ratio": "rho_sel / rho_full, per sample; reported headline is median over samples",
            "active_patch_fraction": f"fraction of 27x27 gate values > {ACTIVE_EPSILON:g}; zero_3d fallback frames contribute zero",
        },
        "headline": {
            "rho_full": rho_full["median"],
            "rho_sel": rho_sel["median"],
            "attenuation_ratio": attenuation["median"],
            "delta_cosine_similarity": cosine["median"],
            "conclusion": classification(float(rho_full["median"]), float(rho_sel["median"]), float(cosine["median"])),
        },
        "per_sample_distributions": {
            "rho_full": rho_full,
            "rho_sel": rho_sel,
            "attenuation_ratio": attenuation,
            "delta_cosine_similarity": cosine,
            "gate_mean_over_32_frames": describe(row["gate_mean_over_32_frames"] for row in rows),
            "active_patch_fraction_over_32_frames": describe(row["active_patch_fraction_over_32_frames"] for row in rows),
            "zero3d_fallback_frame_fraction": describe(row["zero3d_fallback_frame_fraction"] for row in rows),
            "selected_query_count_mean": describe(row["selected_query_count_mean"] for row in rows),
        },
        "eomt_gate_statistics": {
            "frame_gate_mean": describe(all_frame_gate_means),
            "frame_active_patch_fraction": describe(all_frame_active),
            "zero3d_fallback_frame_rate": float(sum(all_frame_fallback) / len(all_frame_fallback)),
            "selected_query_count_per_frame": describe(all_frame_selected),
            "active_patch_epsilon": ACTIVE_EPSILON,
        },
    }
    json_dump(output_dir / "summary.json", summary)
    with (output_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(
        "[C1 EoMT summary] "
        f"rho_full={summary['headline']['rho_full']:.6f} rho_sel={summary['headline']['rho_sel']:.6f} "
        f"attenuation={summary['headline']['attenuation_ratio']:.6f} "
        f"cosine={summary['headline']['delta_cosine_similarity']:.6f} "
        f"=> {summary['headline']['conclusion']}"
    )
    print(f"[C1 EoMT summary] wrote {output_dir / 'summary.json'} and {output_dir / 'per_sample.csv'}")


if __name__ == "__main__":
    main()
