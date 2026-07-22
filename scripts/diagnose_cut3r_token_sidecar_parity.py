#!/usr/bin/env python3
"""Fail a CUT3R-token-only launch when a recomputed sample disagrees with its sidecar."""

import argparse
import json
import sys

import torch


def frame_order(payload):
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    value = payload.get("frame_indices", payload.get("frame_order", metadata.get("frame_indices", metadata.get("frame_order"))))
    if value is None:
        return None
    return [int(x) for x in torch.as_tensor(value).flatten().tolist()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--recomputed", required=True, help="CUT3R output recomputed on the exact sampled frames.")
    parser.add_argument("--min-cosine", type=float, default=0.99999)
    parser.add_argument("--max-abs-diff", type=float, default=1e-4)
    args = parser.parse_args()
    saved = torch.load(args.sidecar, map_location="cpu")
    recomputed = torch.load(args.recomputed, map_location="cpu")
    saved_tokens = saved.get("patch_tokens") if isinstance(saved, dict) else None
    recomputed_tokens = recomputed.get("patch_tokens") if isinstance(recomputed, dict) else None
    if not isinstance(saved_tokens, torch.Tensor) or not isinstance(recomputed_tokens, torch.Tensor):
        raise RuntimeError("Both payloads must be CUT3R sidecar dicts containing patch_tokens.")
    if tuple(saved_tokens.shape) != tuple(recomputed_tokens.shape) or saved_tokens.ndim != 3 or saved_tokens.shape[1:] != (729, 768):
        raise RuntimeError(f"CUT3R sidecar shape mismatch: saved={tuple(saved_tokens.shape)} recomputed={tuple(recomputed_tokens.shape)}")
    saved_order, recomputed_order = frame_order(saved), frame_order(recomputed)
    if saved_order != recomputed_order:
        raise RuntimeError(f"CUT3R frame-order mismatch: saved={saved_order} recomputed={recomputed_order}")
    a, b = saved_tokens.float().flatten(1), recomputed_tokens.float().flatten(1)
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=1)
    diff = (a - b).abs()
    report = {"frame_order": saved_order, "saved_shape": list(saved_tokens.shape), "recomputed_shape": list(recomputed_tokens.shape), "cosine_min": float(cosine.min()), "cosine_mean": float(cosine.mean()), "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean())}
    print("[CUT3R_TOKEN_PARITY] " + json.dumps(report, sort_keys=True))
    if report["cosine_min"] < args.min_cosine or report["max_abs_diff"] > args.max_abs_diff:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
