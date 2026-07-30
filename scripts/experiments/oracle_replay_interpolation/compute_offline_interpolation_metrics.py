#!/usr/bin/env python3
"""Evaluate all fixed interpolation points on the established cache split."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch


def load_trainer(repo: Path):
    path = repo / "scripts/train/train_siglip_to_spatialstack_residual.py"
    spec = importlib.util.spec_from_file_location("residual_trainer_for_interpolation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--predictor-checkpoint", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--siglip-cache", required=True)
    parser.add_argument("--cut3r-cache", required=True)
    parser.add_argument("--layer6-cache", required=True)
    parser.add_argument("--layer9-cache", required=True)
    parser.add_argument("--layer12-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    trainer = load_trainer(Path(args.repo).resolve())
    predictor, _ = trainer.load_residual_predictor_checkpoint(args.predictor_checkpoint, expected_type=None)
    device = torch.device(args.device)
    teacher = trainer.FrozenSpatialStackTeacher(args.teacher_checkpoint, device, trainer.dtype_from_name(args.dtype))
    predictor.to(device=device, dtype=torch.float32).eval()
    cache = trainer.PairedResidualCache(args.siglip_cache, args.cut3r_cache, {6: "spatial_features_dec_6", 9: "spatial_features_dec_9", 12: "spatial_features"}, layer_roots={6: args.layer6_cache, 9: args.layer9_cache, 12: args.layer12_cache}, validation_fraction=args.validation_fraction, split_seed=args.split_seed)
    metric_args = SimpleNamespace(cosine_loss_weight=1.0, smooth_l1_weight=0.1, relative_l2_loss_weight=0.0, log_norm_loss_weight=0.0, teacher_norm_eps=1e-6)
    results = {}
    with torch.no_grad():
        for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
            totals = defaultdict(float)
            for key in cache.validation_keys:
                batch = trainer.collate([cache.load(key, strict=False)])
                x = teacher.inputs(batch["siglip"])
                target = teacher.targets(batch["cut3r"])
                prediction = predictor(x.float(), batch["valid_mask"].to(device=x.device))
                metrics = {}
                for layer in trainer.DEFAULT_SOURCE_LAYERS:
                    interpolated = target[layer] if beta == 0.0 else prediction[layer] if beta == 1.0 else ((1 - beta) * target[layer].float() + beta * prediction[layer].float()).to(target[layer].dtype)
                    metrics[layer] = trainer.regression_metrics(interpolated, target[layer], batch["valid_mask"].to(device=x.device), metric_args.cosine_loss_weight, metric_args.smooth_l1_weight, metric_args.relative_l2_loss_weight, metric_args.log_norm_loss_weight, metric_args.teacher_norm_eps)
                trainer.accumulate(totals, metrics)
            results[str(beta)] = trainer.finalise(totals, metric_args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"split_seed": args.split_seed, "validation_fraction": args.validation_fraction, "betas": results}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
