#!/usr/bin/env python
"""Verify and lock official C1 artifacts for controlled fusion B/C/D/E/H."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llava.model.c1_structured_isometry import SCHEME_VERSION  # noqa: E402
from llava.model.controlled_fusion_pre_sft import (  # noqa: E402
    CONTROLLED_FUSION_PRE_SFT_SPECS,
    controlled_fusion_artifact_metadata,
)


SCHEMA_VERSION = "controlled_fusion_c1_manifest_v1"
RATIO_RTOL = 0.05


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def finite_positive(value: Any, label: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{label} must be finite and positive, got {value!r}")
    return numeric


def validate_ratio(actual: Any, target: float, label: str) -> float:
    value = finite_positive(actual, label)
    if not math.isclose(value, target, rel_tol=RATIO_RTOL, abs_tol=0.0):
        raise ValueError(
            f"{label}={value} differs from the shared C1 target r0={target} by more than {RATIO_RTOL:.0%}"
        )
    return value


def validate_artifact(
    identifier: str,
    path: Path,
    *,
    base_sha256: str,
    calibration_sha256: str,
    expected_r0: float,
) -> dict[str, Any]:
    spec = CONTROLLED_FUSION_PRE_SFT_SPECS[identifier]
    artifact = read_json(path)
    if artifact.get("schema_version") != "c1_calibration_v1":
        raise ValueError(f"{identifier} has an unsupported C1 schema")
    if artifact.get("canonicalization_scheme_version") != SCHEME_VERSION:
        raise ValueError(f"{identifier} has the wrong canonicalization scheme")
    if artifact.get("no_training") is not True:
        raise ValueError(f"{identifier} is not marked as no-training C1")
    if artifact.get("base_calibration_sha256") != base_sha256:
        raise ValueError(f"{identifier} does not use the locked base r0 artifact")
    if artifact.get("calibration_manifest_sha256") != calibration_sha256:
        raise ValueError(f"{identifier} does not use the locked calibration manifest")
    if int(artifact.get("num_calibration_samples", -1)) != 32:
        raise ValueError(f"{identifier} official C1 must use exactly 32 calibration samples")
    expected_metadata = controlled_fusion_artifact_metadata(spec)
    if artifact.get("controlled_fusion") != expected_metadata:
        raise ValueError(
            f"{identifier} topology metadata mismatch: "
            f"actual={artifact.get('controlled_fusion')}, expected={expected_metadata}"
        )
    if artifact.get("architecture") != spec.architecture:
        raise ValueError(f"{identifier} architecture does not match its exact topology")
    r0 = finite_positive(artifact.get("r0"), f"{identifier}.r0")
    if not math.isclose(r0, expected_r0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"{identifier}.r0={r0} differs from the locked base artifact r0={expected_r0}"
        )
    calibrated: dict[str, float] = {}
    if identifier == "B":
        values = artifact.get("pre_projector_add")
        if not isinstance(values, dict):
            raise ValueError("B lacks pre_projector_add calibration values")
        finite_positive(values.get("s_pre"), "B.s_pre")
        finite_positive(values.get("residual_gain"), "B.residual_gain")
        calibrated["pre_mm_projector"] = validate_ratio(
            values.get("calibrated_delta_over_siglip", {}).get("median"),
            r0,
            "B.calibrated_delta_over_siglip",
        )
    else:
        layers = artifact.get("layers")
        if not isinstance(layers, dict) or set(layers) != {
            str(value) for value in spec.llm_injection_layers
        }:
            raise ValueError(f"{identifier} C1 layer keys do not match its injection sites")
        for layer in spec.llm_injection_layers:
            values = layers[str(layer)]
            finite_positive(values.get("residual_gain"), f"{identifier}.L{layer}.residual_gain")
            scale_name = "s_pre" if spec.fusion_type == "add" else "s_qk"
            finite_positive(values.get(scale_name), f"{identifier}.L{layer}.{scale_name}")
            calibrated[str(layer)] = validate_ratio(
                values.get("calibrated_delta_over_h", {}).get("median"),
                r0,
                f"{identifier}.L{layer}.calibrated_delta_over_h",
            )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "architecture": spec.architecture,
        "topology": expected_metadata,
        "r0": r0,
        "calibrated_ratios": calibrated,
        "no_training": True,
        "post_sft_checkpoint_loaded": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--base-calibration", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--siglip-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = (
        args.base_calibration,
        args.calibration_manifest,
        args.base_model / "config.json",
        args.siglip_model / "config.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing controlled-fusion C1 provenance inputs: {missing}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite C1 manifest: {args.output}")
    base_sha256 = sha256_file(args.base_calibration)
    calibration_sha256 = sha256_file(args.calibration_manifest)
    base_artifact = read_json(args.base_calibration)
    expected_r0 = finite_positive(base_artifact.get("r0"), "base calibration r0")
    worktree_status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
    )
    if worktree_status:
        raise RuntimeError("Official controlled-fusion C1 artifacts require a clean Git worktree")
    artifacts = {
        identifier: validate_artifact(
            identifier,
            args.artifact_root / identifier / "c1.json",
            base_sha256=base_sha256,
            calibration_sha256=calibration_sha256,
            expected_r0=expected_r0,
        )
        for identifier in CONTROLLED_FUSION_PRE_SFT_SPECS
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "hostname": socket.gethostname(),
        "base_model": {
            "path": str(args.base_model.resolve()),
            "config_sha256": sha256_file(args.base_model / "config.json"),
            "forbidden_post_sft_artifacts_loaded": False,
        },
        "siglip_model": {
            "path": str(args.siglip_model.resolve()),
            "config_sha256": sha256_file(args.siglip_model / "config.json"),
        },
        "base_calibration": {
            "path": str(args.base_calibration.resolve()),
            "sha256": base_sha256,
        },
        "calibration_manifest": {
            "path": str(args.calibration_manifest.resolve()),
            "sha256": calibration_sha256,
            "samples": 32,
        },
        "artifacts": artifacts,
        "candidate_order": list(CONTROLLED_FUSION_PRE_SFT_SPECS),
        "formal_existing_five_candidate_roster_modified": False,
        "experiment_label": "controlled-fusion pre-SFT extension",
        "no_optimizer_constructed": True,
        "no_optimizer_step": True,
        "post_sft_state_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output), "artifacts": list(artifacts)}))


if __name__ == "__main__":
    main()
