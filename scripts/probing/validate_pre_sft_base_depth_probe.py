#!/usr/bin/env python
"""Integrity checks and smoke attestation for the pre-SFT base-VLM depth probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PROBING_DIR = REPO_ROOT / "scripts" / "probing"
if str(PROBING_DIR) not in sys.path:
    sys.path.insert(0, str(PROBING_DIR))

from depth_probe_common import load_frame_records  # noqa: E402


MODEL_LABEL = "pre_sft_base_vlm"
LOADING_MODE = "pre_sft_base_vlm"
SMOKE_LEVELS = ("projected_features", "layer_6")
FULL_LEVELS = (
    "projected_features", "layer_0", "layer_1", "layer_2", "layer_3", "layer_6",
    "layer_9", "layer_12", "layer_15", "layer_18", "layer_21", "layer_24", "layer_27",
)
EXPECTED_VAL_TOKENS = 75656
RUNTIME_SOURCES = (
    "scripts/probing/extract_depth_probe_features.py",
    "scripts/probing/local_depth_probe_cache.py",
    "scripts/diagnose_layerwise_spatial_hidden_scan.py",
    "llava/model/language_model/llava_qwen.py",
    "scripts/probing/validate_pre_sft_base_depth_probe.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def git_value(*command: str) -> str:
    try:
        return subprocess.check_output(command, cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def runtime_source_fingerprint() -> dict[str, Any]:
    hashes = {name: sha256_file(REPO_ROOT / name) for name in RUNTIME_SOURCES}
    encoded = json.dumps(hashes, sort_keys=True).encode("utf-8")
    return {"sources": hashes, "sha256": hashlib.sha256(encoded).hexdigest()}


def run_identity(args: argparse.Namespace) -> dict[str, Any]:
    status = git_value("git", "status", "--short")
    return {
        "model_label": MODEL_LABEL,
        "model_loading_mode": LOADING_MODE,
        "base_model_path": str(args.base_model.resolve()),
        "base_model_config_sha256": sha256_file(args.base_model / "config.json"),
        "siglip_path": str(args.siglip.resolve()),
        "siglip_config_sha256": sha256_file(args.siglip / "config.json"),
        "sample_indices": str(args.sample_indices.resolve()),
        "sample_indices_sha256": sha256_file(args.sample_indices),
        "hidden_state_indexing": "requested_L -> hidden_states[L + 1]",
        "dtype": args.dtype,
        "device_map": args.device_map,
        "pre_sft_gpu_weight_budget": getattr(args, "pre_sft_gpu_weight_budget", "5GiB"),
        "pre_sft_cpu_offload_budget": getattr(args, "pre_sft_cpu_offload_budget", "45GiB"),
        "requested_attention_backend": args.attn_implementation or None,
        "git_commit": git_value("git", "rev-parse", "HEAD"),
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "runtime_source_fingerprint": runtime_source_fingerprint(),
    }


def expected_torch_dtype(dtype: str) -> str:
    names = {"float16": "torch.float16", "bfloat16": "torch.bfloat16", "float32": "torch.float32"}
    if dtype not in names:
        raise ValueError(f"Unsupported dtype identity value: {dtype!r}")
    return names[dtype]


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    errors: list[str] = []
    for path in (args.base_model / "config.json", args.siglip / "config.json", args.sample_indices):
        if not path.is_file():
            errors.append(f"missing {path}")
    for name in ("adapter_config.json", "adapter_model.bin", "non_lora_trainables.bin"):
        if (args.base_model / name).exists():
            errors.append(f"base model unexpectedly contains adapter file {name}")
    payload = json.loads(args.sample_indices.read_text(encoding="utf-8")) if args.sample_indices.is_file() else {}
    videos = payload.get("videos", [])
    train = sum(video.get("split") == "train" for video in videos)
    val = sum(video.get("split") == "val" for video in videos)
    if len(videos) != 1199 or train != 1006 or val != 193:
        errors.append(f"unexpected ScanNet split counts: videos={len(videos)} train={train} val={val}")
    if not all(len(video.get("frames", [])) == 2 for video in videos):
        errors.append("every video must have exactly two selected target frames")
    return {"assessment": "PASS" if not errors else "FAIL", "identity": run_identity(args), "errors": errors}


def feature_path(output_root: Path, level: str, frame_id: str) -> Path:
    return output_root / "features" / MODEL_LABEL / level / f"frame_{frame_id}.pt"


def attestation_path(output_root: Path) -> Path:
    return output_root / "provenance" / "pre_sft_base_vlm_smoke_attestation.json"


def create_smoke_attestation(args: argparse.Namespace) -> dict[str, Any]:
    preflight_report = preflight(args)
    if preflight_report["assessment"] != "PASS":
        raise RuntimeError(f"Base preflight failed: {preflight_report['errors']}")
    output_root = args.output_root
    provenance_path = output_root / "features" / MODEL_LABEL / "extraction_provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(provenance_path)
    extraction = json.loads(provenance_path.read_text(encoding="utf-8"))
    contract = extraction.get("base_forward_contract", {})
    evidence = contract.get("projector_loading_evidence", {})
    if not contract.get("no_vlm3r_sft_adapter_loaded") or not contract.get("no_cut3r_or_spatial_sidecar_usage"):
        raise RuntimeError("Smoke provenance does not prove the base-only forward contract.")
    if evidence.get("projector_missing_keys") or evidence.get("projector_mismatched_keys"):
        raise RuntimeError("Smoke provenance does not prove pretrained mm_projector restoration.")
    runtime_dtypes = extraction.get("runtime_dtype_summary", {})
    expected_dtype = expected_torch_dtype(args.dtype)
    required_runtime_dtype_keys = (
        "vision_tower_parameter_dtype",
        "vision_tower_forward_input_dtype",
        "vision_tower_forward_output_dtype",
        "mm_projector_forward_output_dtype",
        "projected_features_dtype",
        "llm_inputs_embeds_dtype",
        "layer_6_hidden_states_7_dtype",
    )
    missing_dtype_keys = [name for name in required_runtime_dtype_keys if not runtime_dtypes.get(name)]
    if missing_dtype_keys:
        raise RuntimeError(f"Smoke provenance is missing observed runtime dtypes: {missing_dtype_keys}")
    vision_dtype_keys = (
        "vision_tower_parameter_dtype",
        "vision_tower_forward_input_dtype",
        "vision_tower_forward_output_dtype",
    )
    wrong_vision_dtype = {
        name: runtime_dtypes.get(name)
        for name in vision_dtype_keys
        if runtime_dtypes.get(name) != [expected_dtype]
    }
    if wrong_vision_dtype:
        raise RuntimeError(
            "Smoke provenance does not prove explicit requested vision compute dtype: "
            f"expected={expected_dtype}, observed={wrong_vision_dtype}"
        )
    frame_records = load_frame_records(args.smoke_manifest)
    if len(frame_records) != 4:
        raise RuntimeError(f"Expected four smoke target frames, got {len(frame_records)}")
    shapes: dict[str, list[int]] = {}
    for record in frame_records:
        frame_id = str(record["frame_sample_id"])
        metadata = torch_load(output_root / "metadata" / f"frame_{frame_id}.pt")
        if int(metadata.get("source_video_num_frames", -1)) != 32:
            raise RuntimeError(f"{frame_id}: model input was not the 32-frame cache")
        if metadata.get("point_map_key") != "point_maps_cam" or metadata.get("depth_mode") != "camera_z":
            raise RuntimeError(f"{frame_id}: compact target semantics changed")
        for level in SMOKE_LEVELS:
            tensor = torch_load(feature_path(output_root, level, frame_id))
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape[:2]) != (14, 14):
                raise RuntimeError(f"{frame_id}/{level}: expected [14,14,D], got {getattr(tensor, 'shape', None)}")
            shapes.setdefault(level, list(tensor.shape))
    identity = run_identity(args)
    identity["projector_load_evidence_sha256"] = hashlib.sha256(
        json.dumps(evidence, sort_keys=True).encode("utf-8")
    ).hexdigest()
    identity["resolved_placement_sha256"] = hashlib.sha256(
        json.dumps(extraction.get("placement", {}), sort_keys=True).encode("utf-8")
    ).hexdigest()
    identity["runtime_dtypes_sha256"] = hashlib.sha256(
        json.dumps(runtime_dtypes, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "pre_sft_base_vlm_smoke_attestation_v1",
        "assessment": "PASS",
        "identity": identity,
        "projected_grid_shape": [14, 14],
        "visual_tokens_per_selected_frame": 196,
        "hidden_state_indexing": "requested_L -> hidden_states[L + 1]",
        "l6_hidden_state_index": 7,
        "feature_shapes": shapes,
        "projector_loading_evidence": evidence,
        "resolved_placement": extraction.get("placement", {}),
        "materialized_parameter_dtypes": extraction.get("materialized_parameter_dtypes", {}),
        "runtime_dtype_summary": runtime_dtypes,
        "extraction_provenance": str(provenance_path),
    }


def verify_smoke_attestation(args: argparse.Namespace) -> dict[str, Any]:
    path = attestation_path(args.smoke_root)
    if not path.is_file():
        return {"assessment": "FAIL", "errors": [f"missing smoke attestation {path}"]}
    attestation = json.loads(path.read_text(encoding="utf-8"))
    expected = run_identity(args)
    observed = attestation.get("identity", {})
    errors = []
    for key, value in expected.items():
        if observed.get(key) != value:
            errors.append(f"stale smoke identity {key}: expected={value!r} observed={observed.get(key)!r}")
    if attestation.get("assessment") != "PASS" or attestation.get("projected_grid_shape") != [14, 14]:
        errors.append("smoke attestation is not a passing 14x14 projected-grid attestation")
    if int(attestation.get("visual_tokens_per_selected_frame", -1)) != 196:
        errors.append("smoke attestation does not establish 196 selected visual tokens per frame")
    return {"assessment": "PASS" if not errors else "FAIL", "errors": errors, "attestation": attestation}


def verify_full(args: argparse.Namespace) -> dict[str, Any]:
    attestation = verify_smoke_attestation(args)
    errors = list(attestation.get("errors", []))
    records = load_frame_records(args.sample_indices)
    for level in FULL_LEVELS:
        for record in records:
            path = feature_path(args.output_root, level, str(record["frame_sample_id"]))
            if not path.is_file():
                errors.append(f"missing {level} feature {path}")
                break
        metrics_path = args.output_root / "probes" / MODEL_LABEL / level / "metrics.json"
        if not metrics_path.is_file():
            errors.append(f"missing {level} probe metrics")
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("model_label") != MODEL_LABEL or metrics.get("feature_level") != level:
            errors.append(f"wrong metrics identity for {level}")
        if int(metrics.get("num_tokens", -1)) != EXPECTED_VAL_TOKENS:
            errors.append(f"wrong validation token count for {level}: {metrics.get('num_tokens')}")
    return {"assessment": "PASS" if not errors else "FAIL", "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--write-smoke-attestation", action="store_true")
    mode.add_argument("--verify-smoke-attestation", action="store_true")
    mode.add_argument("--verify-full", action="store_true")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--siglip", type=Path, required=True)
    parser.add_argument("--sample-indices", type=Path, required=True)
    parser.add_argument("--smoke-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--pre-sft-gpu-weight-budget", default="5GiB")
    parser.add_argument("--pre-sft-cpu-offload-budget", default="45GiB")
    parser.add_argument("--attn-implementation", default=None)
    args = parser.parse_args()
    if args.smoke_root is None:
        args.smoke_root = args.output_root
    if args.preflight:
        report = preflight(args)
    elif args.write_smoke_attestation:
        if args.smoke_manifest is None:
            parser.error("--write-smoke-attestation requires --smoke-manifest")
        report = create_smoke_attestation(args)
        path = attestation_path(args.output_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report["attestation_path"] = str(path)
    elif args.verify_smoke_attestation:
        report = verify_smoke_attestation(args)
    else:
        report = verify_full(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["assessment"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
