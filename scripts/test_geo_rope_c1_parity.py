#!/usr/bin/env python
"""Small deterministic g=0 parity check for C1 GeoRoPE fusion."""

from __future__ import annotations

import sys
from pathlib import Path
import argparse

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llava.model.c1_structured_isometry import apply_geo_rope_fusion_c1, apply_vlm3r_c1
from llava.model.multimodal_fusion_block.builder import CrossAttentionFusion, GeoRoPEFusionCrossAttention


def rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    args = parser.parse_args()
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    torch.manual_seed(0)
    qk_scale = 1.1072058170834018
    residual_gain = 0.5849834381369874
    baseline = CrossAttentionFusion(1152, 768, 1152, 18).to(device=device, dtype=dtype).eval()
    geo = GeoRoPEFusionCrossAttention(1152, 768, 1152, 18, dropout_rate=0.1).to(device=device, dtype=dtype).eval()
    apply_vlm3r_c1(baseline)
    baseline.set_c1_state(enabled=True, qk_scale=qk_scale, residual_gain=residual_gain)
    apply_geo_rope_fusion_c1(
        geo, qk_scale=qk_scale, residual_gain=residual_gain, gate_q=0.0, gate_k=0.0
    )
    clip = torch.randn(1, 196, 1152, device=device, dtype=dtype)
    patch = torch.randn(1, 729, 768, device=device, dtype=dtype)
    pos_clip = torch.randn(1, 196, 3, device=device, dtype=dtype)
    pos_patch = torch.randn(1, 729, 3, device=device, dtype=dtype)
    with torch.no_grad():
        baseline_output, _ = baseline(clip, patch)
        geo_output, _ = geo(clip, patch, pos_clip, pos_patch)
    error = rms(geo_output - baseline_output)
    report = {
        "baseline_rms": rms(baseline_output),
        "geo_rms": rms(geo_output),
        "difference_rms": error,
        "difference_max_abs": float((geo_output - baseline_output).abs().max().item()),
    }
    print(report)
    if error > 5e-4:
        raise RuntimeError(f"C1 GeoRoPE g=0 parity failed: {report}")


if __name__ == "__main__":
    main()
