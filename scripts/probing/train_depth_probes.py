#!/usr/bin/env python
"""Train weak per-token MLP depth probes on cached VLM-3R frame features."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from depth_probe_common import (
    DEFAULT_OUTPUT_ROOT,
    LLM_LAYERS,
    MODEL_PRESETS,
    PRE_LLM_FEATURES,
    load_frame_records,
    metric_values,
    write_csv,
    write_json,
)


def split_layer_feature_path(output_root: Path, model_label: str, feature_level: str, frame_sample_id: str) -> Path:
    return output_root / "features" / model_label / feature_level / f"frame_{frame_sample_id}.pt"


def legacy_llm_layers_path(output_root: Path, model_label: str, frame_sample_id: str) -> Path:
    return output_root / "features" / model_label / "llm_layers" / f"frame_{frame_sample_id}.pt"


def feature_tensor_path(output_root: Path, model_label: str, feature_level: str, frame_sample_id: str) -> Path:
    if feature_level.startswith("layer_"):
        split_path = split_layer_feature_path(output_root, model_label, feature_level, frame_sample_id)
        if split_path.exists():
            return split_path
        return legacy_llm_layers_path(output_root, model_label, frame_sample_id)
    return output_root / "features" / model_label / feature_level / f"frame_{frame_sample_id}.pt"


def load_feature_tensor(output_root: Path, model_label: str, feature_level: str, frame_sample_id: str) -> torch.Tensor:
    path = feature_tensor_path(output_root, model_label, feature_level, frame_sample_id)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    if feature_level.startswith("layer_") and isinstance(payload, dict):
        return payload[feature_level]
    if not isinstance(payload, torch.Tensor):
        raise TypeError(f"Expected tensor feature at {path}, got {type(payload)}")
    return payload


class DepthProbeMLP(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CachedFrameDepthDataset(Dataset):
    def __init__(self, output_root: Path, model_label: str, feature_level: str, frame_records: list[dict[str, Any]]):
        self.output_root = Path(output_root)
        self.model_label = model_label
        self.feature_level = feature_level
        self.frame_records = list(frame_records)

    def __len__(self) -> int:
        return len(self.frame_records)

    def _feature_path(self, frame_sample_id: str) -> Path:
        return feature_tensor_path(self.output_root, self.model_label, self.feature_level, frame_sample_id)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.frame_records[index]
        fsid = str(record["frame_sample_id"])
        feature = load_feature_tensor(self.output_root, self.model_label, self.feature_level, fsid)
        gt = torch.load(self.output_root / "gt_depth" / f"frame_{fsid}.pt", map_location="cpu")
        meta = torch.load(self.output_root / "metadata" / f"frame_{fsid}.pt", map_location="cpu")
        valid = meta.get("gt_valid_mask", torch.isfinite(gt) & (gt > 0))
        x = feature.reshape(-1, feature.shape[-1]).float()
        y = gt.reshape(-1).float()
        valid = valid.reshape(-1).bool()
        return {"x": x, "y": y, "valid": valid}


def collate_frame_tokens(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    xs = []
    ys = []
    valids = []
    for item in batch:
        xs.append(item["x"])
        ys.append(item["y"])
        valids.append(item["valid"])
    return {
        "x": torch.cat(xs, dim=0),
        "y": torch.cat(ys, dim=0),
        "valid": torch.cat(valids, dim=0),
    }


def available_feature_levels(model_label: str) -> list[str]:
    levels = [f"layer_{layer}" for layer in LLM_LAYERS]
    if model_label != "zero_spatial":
        levels = PRE_LLM_FEATURES + levels
    return levels


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_tokens = 0
    preds = []
    gts = []
    valids = []
    loss_fn = nn.L1Loss(reduction="sum")

    for batch in loader:
        x = batch["x"].to(device=device, non_blocking=True)
        y = batch["y"].to(device=device, non_blocking=True)
        valid = batch["valid"].to(device=device, non_blocking=True)
        if valid.sum().item() == 0:
            continue
        if training:
            optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred[valid], y[valid])
        if training:
            loss.div(valid.sum().clamp_min(1)).backward()
            optimizer.step()
        total_loss += float(loss.detach().item())
        total_tokens += int(valid.sum().item())
        if not training:
            preds.append(pred.detach().cpu())
            gts.append(y.detach().cpu())
            valids.append(valid.detach().cpu())

    mae = total_loss / max(total_tokens, 1)
    if training or not preds:
        return {"mae": mae, "absrel": float("nan"), "delta125": float("nan"), "num_tokens": total_tokens}
    metrics = metric_values(torch.cat(preds), torch.cat(gts), torch.cat(valids))
    metrics["mae"] = mae
    return metrics


def infer_input_dim(dataset: CachedFrameDepthDataset) -> int:
    for idx in range(len(dataset)):
        item = dataset[idx]
        if item["x"].numel() > 0:
            return int(item["x"].shape[-1])
    raise RuntimeError("Could not infer feature dimension from empty dataset")


def train_one_probe(
    *,
    output_root: Path,
    model_label: str,
    feature_level: str,
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    args: argparse.Namespace,
    wandb_run: Any | None = None,
) -> dict[str, Any]:
    train_dataset = CachedFrameDepthDataset(output_root, model_label, feature_level, train_records)
    val_dataset = CachedFrameDepthDataset(output_root, model_label, feature_level, val_records)
    d_in = infer_input_dim(train_dataset)
    device = torch.device(args.device)
    model = DepthProbeMLP(d_in).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_frame_tokens,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_frame_tokens,
    )

    probe_dir = output_root / "probes" / model_label / feature_level
    probe_dir.mkdir(parents=True, exist_ok=True)
    best_mae = float("inf")
    best_epoch = -1
    history = []
    stale_epochs = 0
    probe_started_at = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started_at = datetime.now(timezone.utc).isoformat()
        epoch_start_time = time.perf_counter()
        train_start_time = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, device=device, optimizer=optimizer)
        train_seconds = time.perf_counter() - train_start_time
        val_start_time = time.perf_counter()
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, device=device)
        val_seconds = time.perf_counter() - val_start_time
        epoch_seconds = time.perf_counter() - epoch_start_time
        elapsed_seconds = time.perf_counter() - probe_started_at
        epoch_finished_at = datetime.now(timezone.utc).isoformat()
        row = {
            "epoch": epoch,
            "epoch_started_at": epoch_started_at,
            "epoch_finished_at": epoch_finished_at,
            "train_seconds": train_seconds,
            "val_seconds": val_seconds,
            "epoch_seconds": epoch_seconds,
            "elapsed_seconds": elapsed_seconds,
            "train_mae": train_metrics["mae"],
            "val_mae": val_metrics["mae"],
            "val_absrel": val_metrics["absrel"],
            "val_delta125": val_metrics["delta125"],
        }
        history.append(row)
        print(
            f"[{model_label}/{feature_level}] epoch={epoch} "
            f"train_mae={row['train_mae']:.6f} val_mae={row['val_mae']:.6f} "
            f"absrel={row['val_absrel']:.6f} delta125={row['val_delta125']:.6f} "
            f"train_s={train_seconds:.1f} val_s={val_seconds:.1f} epoch_s={epoch_seconds:.1f}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "model_label": model_label,
                    "feature_level": feature_level,
                    "epoch": epoch,
                    "train/mae": row["train_mae"],
                    "val/mae": row["val_mae"],
                    "val/absrel": row["val_absrel"],
                    "val/delta125": row["val_delta125"],
                    "time/train_seconds": row["train_seconds"],
                    "time/val_seconds": row["val_seconds"],
                    "time/epoch_seconds": row["epoch_seconds"],
                    "time/elapsed_seconds": row["elapsed_seconds"],
                    f"{model_label}/{feature_level}/train_mae": row["train_mae"],
                    f"{model_label}/{feature_level}/val_mae": row["val_mae"],
                    f"{model_label}/{feature_level}/val_absrel": row["val_absrel"],
                    f"{model_label}/{feature_level}/val_delta125": row["val_delta125"],
                    f"{model_label}/{feature_level}/train_seconds": row["train_seconds"],
                    f"{model_label}/{feature_level}/val_seconds": row["val_seconds"],
                    f"{model_label}/{feature_level}/epoch_seconds": row["epoch_seconds"],
                }
            )
        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "d_in": d_in,
                    "model_label": model_label,
                    "feature_level": feature_level,
                    "epoch": epoch,
                    "metrics": val_metrics,
                },
                probe_dir / "best.pt",
            )
        else:
            stale_epochs += 1
        if args.early_stop_patience > 0 and stale_epochs >= args.early_stop_patience:
            break

    write_json(probe_dir / "history.json", history)
    best = torch.load(probe_dir / "best.pt", map_location="cpu")
    result = {
        "model_label": model_label,
        "feature_level": feature_level,
        "d_in": d_in,
        "best_epoch": best_epoch,
        "mae": float(best["metrics"]["mae"]),
        "absrel": float(best["metrics"]["absrel"]),
        "delta125": float(best["metrics"]["delta125"]),
        "num_tokens": int(best["metrics"]["num_tokens"]),
    }
    write_json(probe_dir / "metrics.json", result)
    if wandb_run is not None:
        prefix = f"best/{model_label}/{feature_level}"
        wandb_run.log(
            {
                f"{prefix}/epoch": best_epoch,
                f"{prefix}/mae": result["mae"],
                f"{prefix}/absrel": result["absrel"],
                f"{prefix}/delta125": result["delta125"],
                f"{prefix}/num_tokens": result["num_tokens"],
            }
        )
        wandb_run.summary[f"{prefix}/mae"] = result["mae"]
        wandb_run.summary[f"{prefix}/absrel"] = result["absrel"]
        wandb_run.summary[f"{prefix}/delta125"] = result["delta125"]
        wandb_run.summary[f"{prefix}/epoch"] = best_epoch
    return result


def filter_existing_records(output_root: Path, model_label: str, feature_level: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept = []
    for record in records:
        fsid = str(record["frame_sample_id"])
        path = feature_tensor_path(output_root, model_label, feature_level, fsid)
        if path.exists() and (output_root / "gt_depth" / f"frame_{fsid}.pt").exists():
            kept.append(record)
    return kept


def load_existing_result(output_root: Path, model_label: str, feature_level: str) -> dict[str, Any] | None:
    metrics_path = output_root / "probes" / model_label / feature_level / "metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected metrics dict at {metrics_path}, got {type(payload)}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--sample-indices", default=str(DEFAULT_OUTPUT_ROOT / "sample_indices.json"))
    parser.add_argument("--model-labels", default=",".join(MODEL_PRESETS.keys()))
    parser.add_argument("--feature-levels", default=None, help="Comma-separated override, e.g. fusion_output,layer_0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse probes that already have metrics.json.")
    parser.add_argument("--wandb", action="store_true", help="Log probe training to Weights & Biases.")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "vlm3r-depth-probes"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-name", default=os.environ.get("WANDB_NAME"))
    parser.add_argument("--wandb-dir", default=os.environ.get("WANDB_DIR", str(Path(os.environ.get("WORK", "/leonardo_work/EUHPC_D32_006")) / "wandb")))
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "offline"))
    parser.add_argument("--wandb-tags", default=os.environ.get("WANDB_TAGS", "depth-probe"))
    args = parser.parse_args()

    output_root = Path(args.output_root)
    train_records = load_frame_records(Path(args.sample_indices), split="train")
    val_records = load_frame_records(Path(args.sample_indices), split="val")
    model_labels = [part.strip() for part in args.model_labels.split(",") if part.strip()]
    wandb_run = None
    if args.wandb:
        try:
            import wandb  # type: ignore
        except ImportError as exc:
            raise RuntimeError("wandb logging requested, but wandb is not installed in this environment.") from exc
        Path(args.wandb_dir).mkdir(parents=True, exist_ok=True)
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or f"depth-probes-{Path(args.output_root).name}",
            dir=args.wandb_dir,
            mode=args.wandb_mode,
            tags=[tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()],
            config={
                "output_root": str(output_root),
                "sample_indices": str(args.sample_indices),
                "model_labels": model_labels,
                "feature_levels": args.feature_levels,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "early_stop_patience": args.early_stop_patience,
                "train_frames": len(train_records),
                "val_frames": len(val_records),
            },
        )
    all_results = []
    try:
        for model_label in model_labels:
            levels = [part.strip() for part in args.feature_levels.split(",") if part.strip()] if args.feature_levels else available_feature_levels(model_label)
            for feature_level in levels:
                if args.skip_existing:
                    existing = load_existing_result(output_root, model_label, feature_level)
                    if existing is not None:
                        print(f"[INFO] Skipping existing probe {model_label}/{feature_level}", flush=True)
                        all_results.append(existing)
                        if wandb_run is not None:
                            prefix = f"best/{model_label}/{feature_level}"
                            wandb_run.summary[f"{prefix}/mae"] = existing.get("mae")
                            wandb_run.summary[f"{prefix}/absrel"] = existing.get("absrel")
                            wandb_run.summary[f"{prefix}/delta125"] = existing.get("delta125")
                            wandb_run.summary[f"{prefix}/epoch"] = existing.get("best_epoch")
                        continue
                train_kept = filter_existing_records(output_root, model_label, feature_level, train_records)
                val_kept = filter_existing_records(output_root, model_label, feature_level, val_records)
                if not args.allow_partial and (len(train_kept) != len(train_records) or len(val_kept) != len(val_records)):
                    raise FileNotFoundError(
                        f"Missing cached features for {model_label}/{feature_level}: "
                        f"train {len(train_kept)}/{len(train_records)}, val {len(val_kept)}/{len(val_records)}. "
                        "Use --allow-partial for smoke tests."
                    )
                if not train_kept or not val_kept:
                    print(f"[WARN] Skipping {model_label}/{feature_level}: no cached train or val records", file=sys.stderr)
                    continue
                all_results.append(
                    train_one_probe(
                        output_root=output_root,
                        model_label=model_label,
                        feature_level=feature_level,
                        train_records=train_kept,
                        val_records=val_kept,
                        args=args,
                        wandb_run=wandb_run,
                    )
                )
        write_json(output_root / "probes" / "results.json", all_results)
        write_csv(output_root / "probes" / "results.csv", all_results)
        print(f"[INFO] Wrote {output_root / 'probes' / 'results.csv'}", flush=True)
        if wandb_run is not None:
            wandb_run.summary["completed_probes"] = len(all_results)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
