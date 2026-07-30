#!/usr/bin/env python3
"""Fit train-split FP64 residual calibration artifacts for a frozen predictor."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

from llava.model.siglip_spatialstack_residual import (
    DEFAULT_SOURCE_LAYERS,
    load_residual_predictor_checkpoint,
    predictor_state_sha256,
)
from train_siglip_to_spatialstack_residual import (
    FrozenSpatialStackTeacher,
    PairedResidualCache,
    collate,
    current_git_commit,
    dtype_from_name,
    finalise,
    regression_metrics,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_sha256(checkpoint: Path) -> str:
    config = checkpoint / "config.json"
    return file_sha256(config) if config.is_file() else "missing"


def statistics(value: torch.Tensor) -> dict[str, float | int]:
    flat = value.detach().double().flatten().cpu()
    quantiles = torch.quantile(flat, torch.tensor([0, .01, .05, .25, .5, .75, .95, .99, 1], dtype=torch.float64))
    return {
        "count": int(flat.numel()), "min": float(flat.min()), "max": float(flat.max()),
        "mean": float(flat.mean()), "std": float(flat.std(unbiased=False)),
        "negative_fraction": float((flat < 0).double().mean()),
        "quantiles": {str(q): float(v) for q, v in zip((0, .01, .05, .25, .5, .75, .95, .99, 1), quantiles)},
    }


def metric_totals():
    return defaultdict(float)


def add_metrics(totals, prediction, target, valid_mask, args):
    for layer in DEFAULT_SOURCE_LAYERS:
        values = regression_metrics(prediction[layer], target[layer], valid_mask, 1.0, .1, 0.0, 0.0, args.teacher_norm_eps)
        valid = float(values["valid_weight"].cpu())
        direction = float(values["direction_weight"].cpu())
        totals[f"layer_{layer}_valid_weight"] += valid
        totals[f"layer_{layer}_direction_weight"] += direction
        totals[f"layer_{layer}_low_norm_excluded"] += float(values["low_norm_excluded"].cpu())
        for name in ("cosine_loss", "cosine", "smooth_l1", "relative_l2", "log_norm", "pred_norm", "teacher_norm", "norm_ratio"):
            totals[f"layer_{layer}_{name}_sum"] += float(values[f"{name}_sum"].cpu())


def evaluate(cache, keys, teacher, predictor, alphas, args):
    totals = metric_totals()
    with torch.no_grad():
        for key in keys:
            batch = collate([cache.load(key)])
            valid = batch["valid_mask"].to(teacher.device)
            inputs = teacher.inputs(batch["siglip"])
            target = teacher.targets(batch["cut3r"])
            prediction = predictor(inputs.float(), valid)
            calibrated = {layer: prediction[layer] * alphas[layer].to(prediction[layer].device, prediction[layer].dtype) for layer in DEFAULT_SOURCE_LAYERS}
            add_metrics(totals, calibrated, target, valid, args)
    return finalise(totals, SimpleNamespace(cosine_loss_weight=1.0, smooth_l1_weight=.1, relative_l2_loss_weight=0.0, log_norm_loss_weight=0.0))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--siglip_feature_cache", required=True)
    result.add_argument("--cut3r_feature_cache", required=True)
    result.add_argument("--cut3r_layer6_cache", required=True)
    result.add_argument("--cut3r_layer9_cache", required=True)
    result.add_argument("--cut3r_layer12_cache", required=True)
    result.add_argument("--teacher_checkpoint", required=True)
    result.add_argument("--predictor_checkpoint", required=True)
    result.add_argument("--output_dir", required=True)
    result.add_argument("--validation_fraction", type=float, default=.1)
    result.add_argument("--split_seed", type=int, default=42)
    result.add_argument("--dtype", default="bfloat16")
    result.add_argument("--device", default="cuda")
    result.add_argument("--teacher_norm_eps", type=float, default=1e-6)
    result.add_argument("--denominator_eps", type=float, default=1e-30)
    result.add_argument("--max_train_samples", type=int, default=0)
    result.add_argument("--max_validation_samples", type=int, default=0)
    return result


def main():
    args = parser().parse_args()
    predictor_path = Path(args.predictor_checkpoint).resolve()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache = PairedResidualCache(
        args.siglip_feature_cache, args.cut3r_feature_cache,
        {6: "spatial_features_dec_6", 9: "spatial_features_dec_9", 12: "spatial_features"},
        layer_roots={6: args.cut3r_layer6_cache, 9: args.cut3r_layer9_cache, 12: args.cut3r_layer12_cache},
        validation_fraction=args.validation_fraction, split_seed=args.split_seed,
    )
    if args.max_train_samples:
        cache.train_keys = cache.train_keys[:args.max_train_samples]
    if args.max_validation_samples:
        cache.validation_keys = cache.validation_keys[:args.max_validation_samples]
    device = torch.device(args.device)
    predictor, checkpoint = load_residual_predictor_checkpoint(predictor_path, map_location="cpu")
    predictor.to(device).eval()
    for parameter in predictor.parameters():
        parameter.requires_grad = False
    teacher = FrozenSpatialStackTeacher(args.teacher_checkpoint, device, dtype_from_name(args.dtype))
    numerator = {layer: torch.zeros(196, dtype=torch.float64, device=device) for layer in DEFAULT_SOURCE_LAYERS}
    denominator = {layer: torch.zeros(196, dtype=torch.float64, device=device) for layer in DEFAULT_SOURCE_LAYERS}
    token_counts = {layer: 0 for layer in DEFAULT_SOURCE_LAYERS}
    with torch.no_grad():
        for index, key in enumerate(cache.train_keys):
            batch = collate([cache.load(key)])
            valid = batch["valid_mask"].to(device).double().unsqueeze(-1)
            inputs = teacher.inputs(batch["siglip"])
            target = teacher.targets(batch["cut3r"])
            prediction = predictor(inputs.float(), valid.squeeze(-1).bool())
            for layer in DEFAULT_SOURCE_LAYERS:
                pred64, target64 = prediction[layer].double(), target[layer].double()
                weight = valid.unsqueeze(-1)
                numerator[layer] += (pred64 * target64 * weight).sum(dim=(0, 1, 3))
                denominator[layer] += (pred64.square() * weight).sum(dim=(0, 1, 3))
                token_counts[layer] += int(valid.sum().item()) * 196
            if (index + 1) % 100 == 0:
                print(json.dumps({"calibration_train_samples": index + 1}), flush=True)
    global_alphas, position_alphas, fallback = {}, {}, {}
    for layer in DEFAULT_SOURCE_LAYERS:
        global_denominator = denominator[layer].sum()
        global_fallback = bool(global_denominator.abs() <= args.denominator_eps)
        global_alphas[layer] = (torch.ones((), dtype=torch.float64, device=device) if global_fallback else numerator[layer].sum() / global_denominator).cpu()
        position_fallback = denominator[layer].abs() <= args.denominator_eps
        position = numerator[layer] / denominator[layer]
        position[position_fallback] = 1.0
        position_alphas[layer] = position[:, None].cpu()
        fallback[layer] = {"global": int(global_fallback), "position": int(position_fallback.sum().item())}
    base_alphas = {layer: torch.ones((), dtype=torch.float32) for layer in DEFAULT_SOURCE_LAYERS}
    before = evaluate(cache, cache.validation_keys, teacher, predictor, base_alphas, args)
    after_global = evaluate(cache, cache.validation_keys, teacher, predictor, global_alphas, args)
    after_position = evaluate(cache, cache.validation_keys, teacher, predictor, position_alphas, args)
    provenance = {
        "format_version": 1, "source_to_llm_mapping": {"6": 0, "9": 1, "12": 2},
        "teacher_checkpoint": str(Path(args.teacher_checkpoint).resolve()), "teacher_config_hash": config_sha256(Path(args.teacher_checkpoint)),
        "predictor_checkpoint": str(predictor_path), "predictor_file_sha256": file_sha256(predictor_path),
        "predictor_state_sha256": predictor_state_sha256(checkpoint["predictor"]),
        "cache_roots": {"siglip": args.siglip_feature_cache, "cut3r": args.cut3r_feature_cache, "layer6": args.cut3r_layer6_cache, "layer9": args.cut3r_layer9_cache, "layer12": args.cut3r_layer12_cache},
        "split_seed": args.split_seed, "train_key_count": len(cache.train_keys), "validation_key_count": len(cache.validation_keys),
        "valid_token_counts": token_counts, "git_revision": current_git_commit(), "validation_before": before,
    }
    for variant, alphas, after in (("global", global_alphas, after_global), ("per_position", position_alphas, after_position)):
        artifact = {**provenance, "calibration_variant": variant, "alphas": alphas, "numerator": {k: v.cpu() for k, v in numerator.items()}, "denominator": {k: v.cpu() for k, v in denominator.items()}, "fallback_counts": fallback, "alpha_statistics": {str(k): statistics(v) for k, v in alphas.items()}, "validation_after": after}
        path = output / f"residual_calibration_{variant}.pt"
        torch.save(artifact, path)
        print(json.dumps({"artifact": str(path), "variant": variant, "validation_before": before["loss"], "validation_after": after["loss"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
