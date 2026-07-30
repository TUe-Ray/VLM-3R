#!/usr/bin/env python3
"""Compute training-split SpatialStack mean residual templates without Qwen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping
import sys

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train_siglip_to_spatialstack_residual import (
    DEFAULT_SOURCE_LAYERS,
    FrozenSpatialStackTeacher,
    PairedResidualCache,
    collate,
    current_git_commit,
    dtype_from_name,
    keys_from_dataset_json,
    read_key_list,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--siglip_feature_cache", required=True)
    result.add_argument("--cut3r_feature_cache", required=True)
    result.add_argument("--cut3r_layer6_subdir", default="spatial_features_dec_6")
    result.add_argument("--cut3r_layer9_subdir", default="spatial_features_dec_9")
    result.add_argument("--cut3r_layer12_subdir", default="spatial_features")
    result.add_argument("--cut3r_layer6_cache", required=True)
    result.add_argument("--cut3r_layer9_cache", required=True)
    result.add_argument("--cut3r_layer12_cache", required=True)
    result.add_argument("--teacher_checkpoint", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--train_key_list")
    result.add_argument("--validation_key_list")
    result.add_argument("--dataset_json", action="append", default=[])
    result.add_argument("--train_dataset_json", action="append", default=[])
    result.add_argument("--validation_dataset_json", action="append", default=[])
    result.add_argument("--validation_fraction", type=float, default=0.1)
    result.add_argument("--split_seed", type=int, default=42)
    result.add_argument("--batch_size", type=int, default=1)
    result.add_argument("--max_train_samples", type=int, default=0)
    result.add_argument("--dtype", default="bfloat16")
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return result


def build_cache(args) -> PairedResidualCache:
    candidate_keys = keys_from_dataset_json(args.dataset_json) if args.dataset_json else None
    train_keys = read_key_list(args.train_key_list)
    validation_keys = read_key_list(args.validation_key_list)
    if args.train_dataset_json:
        train_keys = (train_keys or set()) | keys_from_dataset_json(args.train_dataset_json)
    if args.validation_dataset_json:
        validation_keys = (validation_keys or set()) | keys_from_dataset_json(args.validation_dataset_json)
    return PairedResidualCache(
        args.siglip_feature_cache,
        args.cut3r_feature_cache,
        {6: args.cut3r_layer6_subdir, 9: args.cut3r_layer9_subdir, 12: args.cut3r_layer12_subdir},
        layer_roots={6: args.cut3r_layer6_cache, 9: args.cut3r_layer9_cache, 12: args.cut3r_layer12_cache},
        train_keys=train_keys,
        validation_keys=validation_keys,
        candidate_keys=candidate_keys,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
    )


def main() -> None:
    args = parser().parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one.")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive.")
    cache = build_cache(args)
    train_keys = cache.train_keys[: args.max_train_samples] if args.max_train_samples > 0 else cache.train_keys
    teacher = FrozenSpatialStackTeacher(
        args.teacher_checkpoint, torch.device(args.device), dtype_from_name(args.dtype)
    )
    hidden_size = int(getattr(teacher.config, "hidden_size"))
    sums = {layer: torch.zeros(196, hidden_size, dtype=torch.float32, device=teacher.device) for layer in DEFAULT_SOURCE_LAYERS}
    valid_frame_counts = {layer: 0 for layer in DEFAULT_SOURCE_LAYERS}
    for start in range(0, len(train_keys), args.batch_size):
        selected = train_keys[start:start + args.batch_size]
        batch = collate([cache.load(key, strict=False) for key in selected])
        targets = teacher.targets(batch["cut3r"])
        valid_frames = batch["valid_mask"].to(device=teacher.device, dtype=torch.bool)
        frame_count = int(valid_frames.sum().item())
        if frame_count == 0:
            continue
        for layer in DEFAULT_SOURCE_LAYERS:
            target = targets[layer].float()
            sums[layer] += (target * valid_frames[:, :, None, None]).sum(dim=(0, 1))
            valid_frame_counts[layer] += frame_count
        if start == 0 or (start // args.batch_size + 1) % 100 == 0:
            print(json.dumps({"processed_train_samples": min(start + len(selected), len(train_keys)), "train_samples": len(train_keys)}), flush=True)
    means = {}
    valid_token_counts = {}
    for layer in DEFAULT_SOURCE_LAYERS:
        if valid_frame_counts[layer] == 0:
            raise RuntimeError(f"No valid training frames while computing layer-{layer} mean residual.")
        means[layer] = (sums[layer] / float(valid_frame_counts[layer])).cpu()
        valid_token_counts[str(layer)] = int(valid_frame_counts[layer] * 196)
    artifact = {
        "format": "siglip_spatialstack_mean_residual_v1",
        "mean_residuals": means,
        "source_to_llm_mapping": {str(source): int(llm) for source, llm in zip(teacher.source_layers, teacher.llm_layers)},
        "teacher_checkpoint": str(teacher.checkpoint),
        "teacher_config_hash": teacher.config_hash,
        "cache_roots": {
            "siglip": args.siglip_feature_cache,
            "cut3r": args.cut3r_feature_cache,
            "cut3r_layer6": args.cut3r_layer6_cache,
            "cut3r_layer9": args.cut3r_layer9_cache,
            "cut3r_layer12": args.cut3r_layer12_cache,
        },
        "split_seed": int(args.split_seed),
        "validation_fraction": float(args.validation_fraction),
        "train_key_count": len(train_keys),
        "valid_token_counts": valid_token_counts,
        "git_revision": current_git_commit(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    print(json.dumps({"artifact": str(output), "train_key_count": len(train_keys), "validation_key_count": len(cache.validation_keys), "valid_token_counts": valid_token_counts, "shapes": {str(layer): list(means[layer].shape) for layer in DEFAULT_SOURCE_LAYERS}}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
