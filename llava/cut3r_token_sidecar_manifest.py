"""Verified external provenance for legacy CUT3R final-token sidecars.

Legacy sidecars intentionally remain immutable.  This module supplies the
missing exact frame-order contract only for ``visual_token_source=cut3r_only``.
It is deliberately independent of the legacy SpatialStack/PI3X loaders.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str | Path, *, include_hash: bool = False) -> dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    identity: dict[str, Any] = {"size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
    if include_hash:
        identity["sha256"] = sha256_file(resolved)
    return identity


def _indices(value: Any) -> list[int]:
    return [int(item) for item in torch.as_tensor(value).flatten().tolist()]


def load_cut3r_token_sidecar_manifest(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise RuntimeError(f"CUT3R-token-only sidecar manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError(f"Invalid CUT3R-token-only sidecar manifest: {manifest_path}")
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise RuntimeError(f"CUT3R-token-only sidecar manifest has no entries map: {manifest_path}")
    payload["_path"] = str(manifest_path)
    return payload


def _verify_identity(path: Path, expected: dict[str, Any], kind: str) -> None:
    actual = file_identity(path, include_hash="sha256" in expected)
    for key in ("size_bytes", "mtime_ns", "sha256"):
        if key in expected and actual.get(key) != expected[key]:
            raise RuntimeError(
                f"CUT3R-token-only manifest {kind} identity mismatch for {path}: "
                f"{key}={actual.get(key)!r}, expected={expected[key]!r}"
            )


def validate_cut3r_token_sidecar_manifest_entry(
    manifest: dict[str, Any] | None,
    *,
    video_path: str | Path,
    sidecar_path: str | Path,
    patch_tokens: torch.Tensor,
    selected_frame_indices: list[int],
    video_fps: int,
    frames_upbound: int,
    force_sample: bool,
    require_verified: bool = True,
) -> list[int] | None:
    """Return verified external indices, or ``None`` when no manifest exists."""
    if manifest is None:
        return None
    video = str(Path(video_path).resolve())
    entry = manifest["entries"].get(video)
    if not isinstance(entry, dict):
        raise RuntimeError(f"CUT3R-token-only manifest lacks an entry for video: {video}")
    if require_verified and entry.get("provenance_status") != "verified":
        raise RuntimeError(f"CUT3R-token-only manifest entry is not spot-parity verified: {video}")
    _verify_identity(Path(video), entry.get("video_identity", {}), "video")
    resolved_sidecar = Path(sidecar_path).resolve()
    if str(resolved_sidecar) != str(Path(entry.get("sidecar_path", "")).resolve()):
        raise RuntimeError(
            f"CUT3R-token-only manifest sidecar path mismatch for {video}: "
            f"active={resolved_sidecar}, manifest={entry.get('sidecar_path')}"
        )
    _verify_identity(resolved_sidecar, entry.get("sidecar_identity", {}), "sidecar")
    expected_shape = tuple(int(value) for value in entry.get("sidecar_shape", ()))
    if tuple(patch_tokens.shape) != expected_shape or tuple(patch_tokens.shape[1:]) != (729, 768):
        raise RuntimeError(
            f"CUT3R-token-only manifest sidecar shape mismatch for {resolved_sidecar}: "
            f"active={tuple(patch_tokens.shape)}, manifest={expected_shape}"
        )
    sampling = entry.get("sampling", {})
    active_sampling = {"video_fps": int(video_fps), "frames_upbound": int(frames_upbound), "force_sample": bool(force_sample)}
    if sampling != active_sampling:
        raise RuntimeError(f"CUT3R-token-only sampler configuration differs from manifest for {video}: {active_sampling} != {sampling}")
    manifest_indices = _indices(entry.get("derived_frame_indices", []))
    if manifest_indices != [int(index) for index in selected_frame_indices]:
        raise RuntimeError(f"CUT3R-token-only manifest frame order mismatch for {video}: manifest={manifest_indices}, sampler={selected_frame_indices}")
    if len(manifest_indices) != int(patch_tokens.shape[0]):
        raise RuntimeError(f"CUT3R-token-only manifest frame count mismatch for {video}")
    return manifest_indices
