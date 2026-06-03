#!/usr/bin/env python
"""Diagnostics for VLM-3R H1 spatial ranking representations.

Input is a .pt file with at least:
  - cut3r_patch_tokens: Tensor[F, 729, C]
  - baseline_h1: Tensor[F, N, D]
  - ours_h1: Tensor[F, N, D]

Optional:
  - ours_pgeo_h1: Tensor[F, N, 256]
  - rgb: Tensor[F, 3, H, W] or Tensor[F, H, W, 3]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def pool_cut3r(tokens: torch.Tensor, target_tokens: int, pool_mode: str) -> torch.Tensor:
    if target_tokens == 729:
        return tokens
    if target_tokens != 196:
        raise ValueError(f"Unsupported target token count: {target_tokens}")
    grid = tokens.view(27, 27, -1)
    pool_mode = pool_mode.lower()
    if pool_mode == "bilinear":
        pooled = F.interpolate(
            grid.permute(2, 0, 1).unsqueeze(0).float(),
            size=(14, 14),
            mode="bilinear",
            align_corners=False,
        )[0].permute(1, 2, 0)
        return pooled.to(dtype=tokens.dtype).reshape(196, -1)
    padded = F.pad(grid.permute(2, 0, 1), (0, 1, 0, 1), value=0.0)
    valid = F.pad(torch.ones(1, 27, 27, device=tokens.device), (0, 1, 0, 1), value=0.0)
    if pool_mode == "average":
        counts = F.avg_pool2d(valid, 2, 2) * 4.0
        summed = F.avg_pool2d(padded.float(), 2, 2) * 4.0
        pooled = summed / counts.clamp_min(1.0)
    elif pool_mode == "max":
        masked = padded.float().masked_fill(valid.bool().expand_as(padded) == 0, -torch.finfo(torch.float32).max)
        pooled = F.max_pool2d(masked, 2, 2)
    else:
        raise ValueError(f"Unsupported pool mode: {pool_mode}")
    return pooled.permute(1, 2, 0).to(dtype=tokens.dtype).reshape(196, -1)


def ranking_accuracy(student: torch.Tensor, teacher: torch.Tensor, anchors: int, pos_pct: float, neg_pct: float) -> float:
    student = F.normalize(student.float(), dim=-1)
    teacher = F.normalize(teacher.float(), dim=-1)
    s_teacher = teacher @ teacher.T
    s_student = student @ student.T
    n = teacher.shape[0]
    anchor_ids = torch.randperm(n, device=teacher.device)[: min(anchors, n)]
    pos_k = max(1, math.ceil(n * pos_pct / 100.0))
    neg_k = max(1, math.ceil(n * neg_pct / 100.0))
    correct = []
    for anchor in anchor_ids:
        row = s_teacher[anchor].clone()
        row[anchor] = -float("inf")
        pos_pool = torch.topk(row, min(pos_k, n - 1), largest=True).indices
        row = s_teacher[anchor].clone()
        row[anchor] = float("inf")
        neg_pool = torch.topk(row, min(neg_k, n - 1), largest=False).indices
        pos = pos_pool[torch.randint(pos_pool.numel(), (1,), device=teacher.device)]
        neg = neg_pool[torch.randint(neg_pool.numel(), (1,), device=teacher.device)]
        correct.append((s_student[anchor, pos] > s_student[anchor, neg]).float())
    return float(torch.stack(correct).mean().item())


def similarity_map(features: torch.Tensor, roi_index: int) -> torch.Tensor:
    features = F.normalize(features.float(), dim=-1)
    sim = features @ features[roi_index].unsqueeze(-1)
    sim = sim.squeeze(-1)
    side = int(math.isqrt(sim.numel()))
    if side * side != sim.numel():
        raise ValueError(f"Similarity map needs square token count, got {sim.numel()}")
    return sim.view(side, side).detach().cpu()


def train_probe(student: torch.Tensor, teacher: torch.Tensor, dim: int, steps: int, lr: float) -> torch.Tensor:
    probe = nn.Sequential(nn.LayerNorm(student.shape[-1]), nn.Linear(student.shape[-1], dim)).to(student.device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    teacher = F.normalize(teacher.float(), dim=-1)
    teacher_sim = teacher @ teacher.T
    n = student.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (min(4096, n * n), 2), device=student.device)
        z = F.normalize(probe(student.float()), dim=-1)
        pred = (z[idx[:, 0]] * z[idx[:, 1]]).sum(dim=-1)
        target = teacher_sim[idx[:, 0], idx[:, 1]]
        loss = F.mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return F.normalize(probe(student.float()), dim=-1)


def save_maps(output_dir: Path, name: str, maps: dict[str, torch.Tensor]) -> None:
    if plt is None:
        print("[WARN] matplotlib is not installed; skipping similarity-map PNG output.")
        return
    fig, axes = plt.subplots(1, len(maps), figsize=(4 * len(maps), 4), squeeze=False)
    for ax, (title, value) in zip(axes[0], maps.items()):
        ax.imshow(value, cmap="viridis")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to tensor dump .pt file.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pool-mode", default="bilinear", choices=["bilinear", "average", "max"])
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--roi-index", type=int, default=0)
    parser.add_argument("--anchors-per-frame", type=int, default=128)
    parser.add_argument("--positive-top-percent", type=float, default=10.0)
    parser.add_argument("--negative-bottom-percent", type=float, default=30.0)
    parser.add_argument("--probe-steps", type=int, default=300)
    parser.add_argument("--probe-dim", type=int, default=256)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.input, map_location="cpu")

    teacher_raw = payload["cut3r_patch_tokens"][args.frame]
    baseline = payload["baseline_h1"][args.frame]
    ours = payload["ours_h1"][args.frame]
    teacher = pool_cut3r(teacher_raw, baseline.shape[0], args.pool_mode)
    if teacher.shape[0] != baseline.shape[0] or teacher.shape[0] != ours.shape[0]:
        raise ValueError(
            f"Token count mismatch: teacher={teacher.shape[0]}, baseline={baseline.shape[0]}, ours={ours.shape[0]}"
        )

    metrics = {
        "baseline_raw_h1_rank_acc": ranking_accuracy(
            baseline, teacher, args.anchors_per_frame, args.positive_top_percent, args.negative_bottom_percent
        ),
        "ours_raw_h1_rank_acc": ranking_accuracy(
            ours, teacher, args.anchors_per_frame, args.positive_top_percent, args.negative_bottom_percent
        ),
        "baseline_raw_corr": float(torch.corrcoef(torch.stack([
            (F.normalize(baseline.float(), dim=-1) @ F.normalize(baseline.float(), dim=-1).T).flatten(),
            (F.normalize(teacher.float(), dim=-1) @ F.normalize(teacher.float(), dim=-1).T).flatten(),
        ]))[0, 1].item()),
        "ours_raw_corr": float(torch.corrcoef(torch.stack([
            (F.normalize(ours.float(), dim=-1) @ F.normalize(ours.float(), dim=-1).T).flatten(),
            (F.normalize(teacher.float(), dim=-1) @ F.normalize(teacher.float(), dim=-1).T).flatten(),
        ]))[0, 1].item()),
    }

    maps = {
        "CUT3R teacher": similarity_map(teacher, args.roi_index),
        "CE raw H1": similarity_map(baseline, args.roi_index),
        "CE+rank raw H1": similarity_map(ours, args.roi_index),
    }
    if "ours_pgeo_h1" in payload:
        ours_pgeo = payload["ours_pgeo_h1"][args.frame]
        maps["CE+rank P_geo(H1)"] = similarity_map(ours_pgeo, args.roi_index)
        metrics["ours_pgeo_rank_acc"] = ranking_accuracy(
            ours_pgeo, teacher, args.anchors_per_frame, args.positive_top_percent, args.negative_bottom_percent
        )

    baseline_probe = train_probe(baseline, teacher, args.probe_dim, args.probe_steps, args.probe_lr)
    ours_probe = train_probe(ours, teacher, args.probe_dim, args.probe_steps, args.probe_lr)
    maps["CE posthoc probe"] = similarity_map(baseline_probe, args.roi_index)
    maps["CE+rank posthoc probe"] = similarity_map(ours_probe, args.roi_index)
    metrics["baseline_probe_rank_acc"] = ranking_accuracy(
        baseline_probe, teacher, args.anchors_per_frame, args.positive_top_percent, args.negative_bottom_percent
    )
    metrics["ours_probe_rank_acc"] = ranking_accuracy(
        ours_probe, teacher, args.anchors_per_frame, args.positive_top_percent, args.negative_bottom_percent
    )

    save_maps(output_dir, f"frame_{args.frame}_roi_{args.roi_index}", maps)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
