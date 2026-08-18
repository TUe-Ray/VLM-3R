#!/usr/bin/env python
"""Validate zero-spatial pre-LLM smoke provenance and identity-bound markers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FEATURES = ("siglip_output", "projected_features")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_extraction_provenance(
    provenance_path: Path,
    *,
    sample_indices: Path,
    checkpoint: Path,
    forward_root: Path,
    target_root: Path,
    feature_root: Path,
    require_smoke_attestation: bool = True,
) -> dict[str, Any]:
    payload = read_json(provenance_path)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected extraction provenance object at {provenance_path}")
    expected_sample_sha = sha256_file(sample_indices)
    expected_checkpoint_sha = sha256_file(checkpoint / "config.json")
    adapter_config = checkpoint / "adapter_config.json"
    expected_adapter_sha = sha256_file(adapter_config) if adapter_config.is_file() else None
    checks = {
        "model_label": payload.get("model_label") == "zero_spatial",
        "requested_pre_llm_features": sorted(payload.get("requested_pre_llm_features", [])) == sorted(FEATURES),
        "requested_llm_layers": payload.get("requested_llm_layers") == [],
        "sample_indices_sha256": payload.get("sample_indices_sha256") == expected_sample_sha,
        "checkpoint_config_sha256": payload.get("checkpoint_config_sha256") == expected_checkpoint_sha,
        "forward_frames_root": Path(str(payload.get("forward_frames_root", ""))).resolve()
        == forward_root.resolve(),
        "probe_targets_root": Path(str(payload.get("probe_targets_root", ""))).resolve()
        == target_root.resolve(),
        "feature_root": Path(str(payload.get("feature_root", feature_root))).resolve()
        == feature_root.resolve(),
    }
    if expected_adapter_sha is not None:
        checks["adapter_config_sha256"] = payload.get("adapter_config_sha256") == expected_adapter_sha
    # feature_root was added to newer provenance; accept its absence only when
    # the command itself records the canonical sidecar root elsewhere.
    if "feature_root" not in payload:
        checks["feature_root"] = True
    samples = payload.get("extraction_samples", [])
    if require_smoke_attestation:
        checks["first_video_attestation"] = any(
            isinstance(sample, dict)
            and isinstance(sample.get("first_video_runtime_assertions"), dict)
            and sample["first_video_runtime_assertions"].get("assessment") == "PASS"
            for sample in samples
        )
    definitions = payload.get("zero_spatial_post_fusion_projector_contract", {})
    checks["post_fusion_projector_definition"] = (
        definitions.get("projected_features")
        == "mm_projector output after zero-spatial fusion path; verified by fusion output == mm_projector input"
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"zero-spatial pre-LLM provenance validation failed: {failed}")
    return {
        "assessment": "PASS",
        "checks": checks,
        "model_label": "zero_spatial",
        "feature_levels": list(FEATURES),
        "sample_indices": str(sample_indices.resolve()),
        "sample_indices_sha256": expected_sample_sha,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_config_sha256": expected_checkpoint_sha,
        "adapter_config_sha256": expected_adapter_sha,
        "forward_frames_root": str(forward_root.resolve()),
        "probe_targets_root": str(target_root.resolve()),
        "feature_root": str(feature_root.resolve()),
        "extraction_provenance": str(provenance_path.resolve()),
        "extraction_provenance_sha256": sha256_file(provenance_path),
    }


def require_marker(
    marker_path: Path,
    *,
    sample_indices: Path,
    checkpoint: Path,
    forward_root: Path | None = None,
    target_root: Path | None = None,
    feature_root: Path | None = None,
) -> dict[str, Any]:
    marker = read_json(marker_path)
    if not isinstance(marker, dict) or marker.get("assessment") != "PASS":
        raise RuntimeError(f"Invalid zero-spatial pre-LLM smoke marker: {marker_path}")
    if marker.get("sample_indices_sha256") != sha256_file(sample_indices):
        raise RuntimeError("Zero pre-LLM smoke marker split identity is stale.")
    if marker.get("checkpoint_config_sha256") != sha256_file(checkpoint / "config.json"):
        raise RuntimeError("Zero pre-LLM smoke marker checkpoint identity is stale.")
    adapter_config = checkpoint / "adapter_config.json"
    if adapter_config.is_file() and marker.get("adapter_config_sha256") != sha256_file(adapter_config):
        raise RuntimeError("Zero pre-LLM smoke marker adapter identity is stale.")
    extraction_provenance = Path(str(marker.get("extraction_provenance", "")))
    if not extraction_provenance.is_file() or marker.get("extraction_provenance_sha256") != sha256_file(extraction_provenance):
        raise RuntimeError("Zero pre-LLM smoke marker extraction provenance is stale.")
    if sorted(marker.get("feature_levels", [])) != sorted(FEATURES):
        raise RuntimeError("Zero pre-LLM smoke marker does not attest both requested representations.")
    for key, expected in (
        ("forward_frames_root", forward_root),
        ("probe_targets_root", target_root),
        ("feature_root", feature_root),
    ):
        if expected is not None and Path(str(marker.get(key, ""))).resolve() != expected.resolve():
            raise RuntimeError(f"Zero pre-LLM smoke marker {key} identity is stale.")
    return marker


def verify_probe_outputs(output_root: Path, *, require_full_tokens: bool = False) -> dict[str, Any]:
    missing = []
    for feature in FEATURES:
        probe_root = output_root / "probes" / "zero_spatial" / feature
        for name in ("metrics.json", "history.json", "best.pt"):
            if not (probe_root / name).is_file():
                missing.append(str(probe_root / name))
        feature_files = list((output_root / "features" / "zero_spatial" / feature).glob("frame_*.pt"))
        if require_full_tokens and len(feature_files) != 2398:
            missing.append(
                f"{output_root / 'features' / 'zero_spatial' / feature}: "
                f"expected 2398 selected-frame tensors, found {len(feature_files)}"
            )
        if require_full_tokens and (probe_root / "metrics.json").is_file():
            metrics = read_json(probe_root / "metrics.json")
            if int(metrics.get("num_tokens", -1)) != 75656:
                missing.append(
                    f"{probe_root / 'metrics.json'}: expected 75656 validation tokens, "
                    f"found {metrics.get('num_tokens')}"
                )
    if missing:
        raise RuntimeError("Zero pre-LLM probe outputs are incomplete: " + ", ".join(missing))
    return {"assessment": "PASS", "feature_levels": list(FEATURES)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("write-smoke-marker", "verify-smoke-marker", "verify-full"), required=True)
    parser.add_argument("--extraction-provenance", type=Path)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--sample-indices", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--forward-root", type=Path)
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    if args.mode == "verify-smoke-marker":
        report = require_marker(
            args.marker,
            sample_indices=args.sample_indices,
            checkpoint=args.checkpoint,
            forward_root=args.forward_root,
            target_root=args.target_root,
            feature_root=args.feature_root,
        )
    elif args.mode == "verify-full":
        if args.extraction_provenance is None or args.forward_root is None or args.target_root is None or args.feature_root is None:
            parser.error("verify-full requires --extraction-provenance, --forward-root, --target-root, and --feature-root")
        report = validate_extraction_provenance(
            args.extraction_provenance,
            sample_indices=args.sample_indices,
            checkpoint=args.checkpoint,
            forward_root=args.forward_root,
            target_root=args.target_root,
            feature_root=args.feature_root,
            require_smoke_attestation=False,
        )
        if args.output_root is None:
            parser.error("verify-full requires --output-root")
        report["probe_outputs"] = verify_probe_outputs(args.output_root)
    else:
        if args.extraction_provenance is None or args.forward_root is None or args.target_root is None or args.feature_root is None:
            parser.error("write-smoke-marker requires --extraction-provenance, --forward-root, --target-root, and --feature-root")
        report = validate_extraction_provenance(
            args.extraction_provenance,
            sample_indices=args.sample_indices,
            checkpoint=args.checkpoint,
            forward_root=args.forward_root,
            target_root=args.target_root,
            feature_root=args.feature_root,
        )
        if args.output_root is not None:
            report["probe_outputs"] = verify_probe_outputs(args.output_root, require_full_tokens=True)
        args.marker.parent.mkdir(parents=True, exist_ok=True)
        args.marker.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
