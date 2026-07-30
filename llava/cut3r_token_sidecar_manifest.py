"""Advisory provenance support for legacy CUT3R final-token sidecars."""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import torch

CUT3R_TOKEN_MANIFEST_POLICIES = {"off", "warn", "strict"}


def normalize_cut3r_token_manifest_policy(policy: str | None) -> str:
    value = str(policy or "warn").strip().lower()
    if value not in CUT3R_TOKEN_MANIFEST_POLICIES:
        raise ValueError(f"cut3r_token_manifest_policy must be one of {sorted(CUT3R_TOKEN_MANIFEST_POLICIES)}, got {policy!r}")
    return value


def _warn(callback, message: str) -> None:
    if callback:
        callback(message)
    else:
        warnings.warn(message, RuntimeWarning, stacklevel=3)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str | Path, *, include_hash: bool = False) -> dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    result: dict[str, Any] = {"size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
    if include_hash:
        result["sha256"] = sha256_file(resolved)
    return result


def _indices(value: Any) -> list[int]:
    return [int(item) for item in torch.as_tensor(value).flatten().tolist()]


def load_cut3r_token_sidecar_manifest(path: str | None, *, policy: str | None = "strict", warning_callback=None) -> dict[str, Any] | None:
    policy = normalize_cut3r_token_manifest_policy(policy)
    if policy == "off" or not path:
        return None
    manifest_path = Path(path).resolve()
    try:
        if not manifest_path.is_file():
            raise RuntimeError(f"CUT3R-token-only sidecar manifest is missing: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1 or not isinstance(payload.get("entries"), dict):
            raise RuntimeError(f"Invalid CUT3R-token-only sidecar manifest: {manifest_path}")
    except (OSError, json.JSONDecodeError, RuntimeError) as error:
        if policy == "strict":
            raise RuntimeError(str(error)) from error
        _warn(warning_callback, f"[CUT3R_TOKEN_ONLY][MANIFEST][WARN] {error}; using deterministic legacy fallback.")
        return None
    payload["_path"] = str(manifest_path)
    return payload


def _verify_identity(path: Path, expected: dict[str, Any], kind: str) -> None:
    actual = file_identity(path, include_hash="sha256" in expected)
    for key in ("size_bytes", "mtime_ns", "sha256"):
        if key in expected and actual.get(key) != expected[key]:
            raise RuntimeError(f"CUT3R-token-only manifest {kind} identity mismatch for {path}: {key}={actual.get(key)!r}, expected={expected[key]!r}")


def validate_cut3r_token_sidecar_manifest_entry(manifest: dict[str, Any] | None, *, video_path: str | Path, sidecar_path: str | Path, patch_tokens: torch.Tensor, selected_frame_indices: list[int], video_fps: int, frames_upbound: int, force_sample: bool, require_verified: bool = True, policy: str | None = "strict", warning_callback=None) -> list[int] | None:
    policy = normalize_cut3r_token_manifest_policy(policy)
    if policy == "off" or manifest is None:
        return None
    try:
        video = str(Path(video_path).resolve())
        entry = manifest["entries"].get(video)
        if not isinstance(entry, dict):
            raise RuntimeError(f"CUT3R-token-only manifest lacks an entry for video: {video}")
        if require_verified and entry.get("provenance_status") != "verified":
            raise RuntimeError(f"CUT3R-token-only manifest entry is not spot-parity verified: {video}")
        _verify_identity(Path(video), entry.get("video_identity", {}), "video")
        sidecar = Path(sidecar_path).resolve()
        if str(sidecar) != str(Path(entry.get("sidecar_path", "")).resolve()):
            raise RuntimeError(f"CUT3R-token-only manifest sidecar path mismatch for {video}")
        _verify_identity(sidecar, entry.get("sidecar_identity", {}), "sidecar")
        expected_shape = tuple(int(value) for value in entry.get("sidecar_shape", ()))
        if tuple(patch_tokens.shape) != expected_shape or tuple(patch_tokens.shape[1:]) != (729, 768):
            raise RuntimeError(f"CUT3R-token-only manifest sidecar shape mismatch for {sidecar}")
        active_sampling = {"video_fps": int(video_fps), "frames_upbound": int(frames_upbound), "force_sample": bool(force_sample)}
        if entry.get("sampling", {}) != active_sampling:
            raise RuntimeError(f"CUT3R-token-only sampler configuration differs from manifest for {video}")
        indices = _indices(entry.get("derived_frame_indices", []))
        if indices != [int(index) for index in selected_frame_indices] or len(indices) != int(patch_tokens.shape[0]):
            raise RuntimeError(f"CUT3R-token-only manifest frame order/count mismatch for {video}")
        return indices
    except (OSError, RuntimeError, KeyError, TypeError, ValueError) as error:
        if policy == "strict":
            raise RuntimeError(str(error)) from error
        _warn(warning_callback, f"[CUT3R_TOKEN_ONLY][MANIFEST][WARN] {error}; using deterministic legacy fallback.")
        return None
