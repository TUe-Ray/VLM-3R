#!/usr/bin/env python3
"""Gate raw predicted evaluation on cached/online SigLIP feature parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionModel, SigLipVisionTower
from llava.utils import process_video_with_decord


def cached_tensor(path: Path):
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, torch.Tensor):
        return value
    for name in ("features", "siglip_features", "patch_tokens", "tensor"):
        if isinstance(value.get(name), torch.Tensor):
            return value[name]
    raise RuntimeError(f"No SigLIP tensor in {path}")


def metrics(left, right):
    left, right = left.float(), right.float()
    return {
        "cosine": float(F.cosine_similarity(left.reshape(-1, left.shape[-1]), right.reshape(-1, right.shape[-1]), dim=-1).mean()),
        "relative_l2": float((left - right).norm() / right.norm().clamp_min(1e-8)),
        "max_abs_difference": float((left - right).abs().max()),
        "mean_abs_difference": float((left - right).abs().mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached_feature", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--siglip_model", required=True)
    parser.add_argument("--frame_indices", required=True, help="JSON list recorded for the cached feature")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    frame_indices = [int(x) for x in json.loads(args.frame_indices)]
    cached = cached_tensor(Path(args.cached_feature))
    sampler = SimpleNamespace(video_fps=1, frames_upbound=len(frame_indices), force_sample=True)
    frames, _, _, _, online_indices = process_video_with_decord(args.video, sampler, return_indices=True)
    if [int(x) for x in online_indices] != frame_indices:
        raise RuntimeError(f"Frame order mismatch: online={online_indices}, cached={frame_indices}")
    device = torch.device(args.device)
    tower = SigLipVisionTower(args.siglip_model, SimpleNamespace())
    tower.load_model(); tower.to(device).eval()
    pixels = tower.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        online_short, raw_tap = tower(pixels, return_raw_features=True)
        full = SigLipVisionModel.from_pretrained(args.siglip_model).to(device).eval()
        full_minus2 = full(pixels.to(dtype=next(full.parameters()).dtype), output_hidden_states=True).hidden_states[-2].to(dtype=raw_tap.dtype)
    online_short, raw_tap, full_minus2 = (item.detach().cpu() for item in (online_short, raw_tap, full_minus2))
    if tuple(cached.shape) != tuple(raw_tap.shape) or tuple(raw_tap.shape) != (len(frame_indices), 729, 1152):
        raise RuntimeError(f"Raw feature shape mismatch: cached={tuple(cached.shape)}, online={tuple(raw_tap.shape)}")
    cached_to_online = metrics(raw_tap.to(dtype=cached.dtype), cached)
    short_to_full = metrics(raw_tap, full_minus2)
    passes = cached_to_online["cosine"] >= 0.99999 and cached_to_online["relative_l2"] <= 1e-3 and cached_to_online["max_abs_difference"] <= 1e-2
    report = {
        "cached_shape": list(cached.shape), "cached_dtype": str(cached.dtype), "online_shape": list(raw_tap.shape),
        "online_dtype": str(raw_tap.dtype), "frame_indices": frame_indices, "patch_order": "row_major (p=row*27+column)",
        "cached_vs_online": cached_to_online, "shortened_vs_full_hidden_states_minus2": short_to_full,
        "thresholds": {"cosine_min": 0.99999, "relative_l2_max": 1e-3, "max_abs_difference_max": 1e-2}, "passes": passes,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not passes:
        raise SystemExit("Online/offline SigLIP parity gate failed.")


if __name__ == "__main__":
    main()
