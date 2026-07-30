#!/usr/bin/env python3
"""Bounded checkpoint-delta evidence for CUT3R-token-only ZeRO runs."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch


def checkpoint_step(path: Path) -> int | None:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else None


def latest_checkpoint(root: Path) -> Path | None:
    candidates = [(step, path) for path in root.rglob("checkpoint-*")
                  if path.is_dir() and (step := checkpoint_step(path)) is not None]
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _adapter_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    binary = checkpoint / "adapter_model.bin"
    if binary.is_file():
        return _load_torch(binary)
    safetensors_path = checkpoint / "adapter_model.safetensors"
    if safetensors_path.is_file():
        from safetensors.torch import load_file
        return load_file(str(safetensors_path), device="cpu")
    return {}


def _canonical_key(name: str) -> str:
    """Normalize PEFT/default and wrapper prefixes without changing suffixes."""
    name = name.replace(".default.", ".")
    while True:
        for prefix in ("module.", "base_model.model.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        else:
            return name


def _canonical_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state.items():
        canonical = _canonical_key(key)
        if canonical in normalized:
            raise RuntimeError(f"Ambiguous checkpoint key after normalization: {canonical}")
        normalized[canonical] = value
    return normalized


def checkpoint_delta_evidence(run_dir: Path, checkpoint: Path | None = None) -> dict[str, Any]:
    """Compare only fixed initial slices against established saved artifacts."""
    checkpoint = checkpoint or latest_checkpoint(run_dir)
    initial_path = run_dir / "cut3r_token_only_initial_weight_samples.pt"
    if checkpoint is None or not initial_path.is_file():
        return {
            "complete": False,
            "checkpoint_path": str(checkpoint) if checkpoint else None,
            "initial_samples_path": str(initial_path),
            "reason": "missing checkpoint or initial bounded sample artifact",
        }
    initial = _load_torch(initial_path)
    groups = initial.get("groups") if isinstance(initial, dict) else None
    non_lora_path = checkpoint / "non_lora_trainables.bin"
    if not isinstance(groups, dict) or not non_lora_path.is_file():
        return {"complete": False, "checkpoint_path": str(checkpoint), "reason": "missing groups or non-LoRA state"}
    projector_state = _canonical_state(_load_torch(non_lora_path))
    lora_state = _canonical_state(_adapter_state(checkpoint))
    report: dict[str, Any] = {
        "complete": False,
        "checkpoint_path": str(checkpoint),
        "initial_samples_path": str(initial_path),
        "groups": {},
    }
    for group_name, state in (("projector", projector_state), ("lora", lora_state)):
        samples = groups.get(group_name)
        missing = []
        delta_sq = 0.0
        finite = True
        matched = []
        if not isinstance(samples, dict) or not samples:
            missing.append("<no initial samples>")
        for raw_name, before in (samples or {}).items():
            canonical = _canonical_key(str(raw_name))
            after = state.get(canonical)
            if after is None:
                missing.append(str(raw_name))
                continue
            before = before.detach().reshape(-1).float()
            after = after.detach().reshape(-1).float()
            if after.numel() < before.numel():
                missing.append(f"{raw_name} (checkpoint tensor too small)")
                continue
            delta = after[:before.numel()] - before
            finite = finite and bool(torch.isfinite(before).all() and torch.isfinite(after[:before.numel()]).all() and torch.isfinite(delta).all())
            delta_sq += float(delta.square().sum().item())
            matched.append(str(raw_name))
        delta_norm = delta_sq ** 0.5
        report["groups"][group_name] = {
            "sampled_parameter_names": sorted((samples or {}).keys()),
            "matched_parameter_names": sorted(matched),
            "missing_parameter_names": missing,
            "delta_norm": delta_norm,
            "finite": finite,
            "nonzero": bool(finite and delta_norm > 0.0),
        }
    report["complete"] = all(
        not report["groups"][group]["missing_parameter_names"]
        and bool(report["groups"][group]["matched_parameter_names"])
        for group in ("projector", "lora")
    )
    return report
