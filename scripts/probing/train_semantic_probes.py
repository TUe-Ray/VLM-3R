#!/usr/bin/env python
"""Train per-token ScanNet20 semantic probes on cached VLM-3R features."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from depth_probe_common import DEFAULT_OUTPUT_ROOT, MODEL_PRESETS, load_frame_records, write_json
from semantic_probe_common import IGNORE_INDEX, metrics_from_confusion, squeeze_singleton_feature_batch, update_confusion, write_csv
from train_depth_probes import available_feature_levels, feature_tensor_path, load_feature_tensor


class SemanticProbeMLP(nn.Module):
    def __init__(self, d_in: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CachedFrameSemanticDataset(Dataset):
    def __init__(self, output_root: Path, model_label: str, feature_level: str, frame_records: list[dict[str, Any]]):
        self.output_root = Path(output_root)
        self.model_label = model_label
        self.feature_level = feature_level
        self.frame_records = list(frame_records)

    def __len__(self) -> int:
        return len(self.frame_records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.frame_records[index]
        fsid = str(record["frame_sample_id"])
        feature = squeeze_singleton_feature_batch(load_feature_tensor(self.output_root, self.model_label, self.feature_level, fsid))
        labels = torch.load(self.output_root / "semantic_gt_scannet" / f"frame_{fsid}.pt", map_location="cpu").long()
        if feature.ndim == 3:
            if tuple(feature.shape[:2]) != tuple(labels.shape):
                raise ValueError(f"Feature grid {tuple(feature.shape[:2])} does not match labels {tuple(labels.shape)} for {fsid}")
            x = feature.reshape(-1, feature.shape[-1]).float()
        elif feature.ndim == 2:
            if int(feature.shape[0]) != int(labels.numel()):
                raise ValueError(f"Feature tokens {feature.shape[0]} do not match labels {labels.numel()} for {fsid}")
            x = feature.float()
        else:
            raise ValueError(f"Unsupported feature shape {tuple(feature.shape)} for {fsid}")
        return {"x": x, "y": labels.reshape(-1)}


def collate_frame_tokens(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "x": torch.cat([item["x"] for item in batch], dim=0),
        "y": torch.cat([item["y"] for item in batch], dim=0),
    }


def infer_input_dim(dataset: CachedFrameSemanticDataset) -> int:
    for idx in range(len(dataset)):
        item = dataset[idx]
        if item["x"].numel() > 0:
            return int(item["x"].shape[-1])
    raise RuntimeError("Could not infer feature dimension from empty dataset")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    total_all_tokens = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for batch in loader:
        x = batch["x"].to(device=device, non_blocking=True)
        y = batch["y"].to(device=device, non_blocking=True)
        valid = y != ignore_index
        total_all_tokens += int(y.numel())
        if valid.sum().item() == 0:
            continue
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        if training:
            loss.div(valid.sum().clamp_min(1)).backward()
            optimizer.step()
        total_loss += float(loss.detach().item())
        total_tokens += int(valid.sum().item())
        if not training:
            pred = logits.detach().argmax(dim=-1).cpu()
            update_confusion(confusion, pred, y.detach().cpu(), num_classes=num_classes)
    metrics = metrics_from_confusion(confusion)
    metrics["loss"] = total_loss / max(total_tokens, 1)
    metrics["valid_token_count"] = total_tokens
    metrics["ignored_token_ratio"] = 1.0 - (total_tokens / max(total_all_tokens, 1))
    return metrics


def filter_existing_records(output_root: Path, model_label: str, feature_level: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept = []
    for record in records:
        fsid = str(record["frame_sample_id"])
        if (
            feature_tensor_path(output_root, model_label, feature_level, fsid).exists()
            and (output_root / "semantic_gt_scannet" / f"frame_{fsid}.pt").exists()
        ):
            kept.append(record)
    return kept


def train_one_probe(
    *,
    output_root: Path,
    probe_subdir: str,
    model_label: str,
    feature_level: str,
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    train_dataset = CachedFrameSemanticDataset(output_root, model_label, feature_level, train_records)
    val_dataset = CachedFrameSemanticDataset(output_root, model_label, feature_level, val_records)
    d_in = infer_input_dim(train_dataset)
    device = torch.device(args.device)
    model = SemanticProbeMLP(d_in, args.num_classes).to(device)
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

    probe_dir = output_root / probe_subdir / model_label / feature_level
    probe_dir.mkdir(parents=True, exist_ok=True)
    best_miou = -1.0
    best_epoch = -1
    stale_epochs = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started_at = datetime.now(timezone.utc).isoformat()
        train_metrics = run_epoch(
            model,
            train_loader,
            device=device,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            optimizer=optimizer,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                device=device,
                num_classes=args.num_classes,
                ignore_index=args.ignore_index,
            )
        row = {
            "epoch": epoch,
            "epoch_started_at": epoch_started_at,
            "epoch_finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_top1_accuracy": val_metrics["top1_accuracy"],
            "val_mIoU_gt_present": val_metrics["mIoU_gt_present"],
            "val_num_present_classes": val_metrics["num_present_classes"],
            "val_valid_token_count": val_metrics["valid_token_count"],
            "val_ignored_token_ratio": val_metrics["ignored_token_ratio"],
            "val_dominant_class_fraction": val_metrics["dominant_class_fraction"],
            "val_low_confidence": val_metrics["low_confidence"],
        }
        history.append(row)
        print(
            f"[{model_label}/{feature_level}] epoch={epoch} train_loss={row['train_loss']:.6f} "
            f"val_loss={row['val_loss']:.6f} top1={row['val_top1_accuracy']:.4f} "
            f"mIoU_gt_present={row['val_mIoU_gt_present']:.4f} present={row['val_num_present_classes']} "
            f"ignored={row['val_ignored_token_ratio']:.3f}",
            flush=True,
        )
        cur_miou = float(val_metrics["mIoU_gt_present"])
        score = cur_miou if cur_miou == cur_miou else -1.0
        if best_epoch < 0 or score > best_miou:
            best_miou = cur_miou
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "d_in": d_in,
                    "num_classes": args.num_classes,
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
    metrics = best["metrics"]
    result = {
        "model_label": model_label,
        "feature_level": feature_level,
        "d_in": d_in,
        "num_classes": int(args.num_classes),
        "best_epoch": int(best_epoch),
        "top1_accuracy": float(metrics["top1_accuracy"]),
        "mIoU": float(metrics["mIoU"]),
        "mIoU_gt_present": float(metrics["mIoU_gt_present"]),
        "per_class_IoU": metrics["per_class_IoU"],
        "present_classes": metrics["present_classes"],
        "num_present_classes": int(metrics["num_present_classes"]),
        "num_gt_present_classes": int(metrics["num_gt_present_classes"]),
        "valid_token_count": int(metrics["valid_token_count"]),
        "ignored_token_ratio": float(metrics["ignored_token_ratio"]),
        "dominant_class_fraction": float(metrics["dominant_class_fraction"]),
        "low_confidence": bool(metrics["low_confidence"]),
    }
    write_json(probe_dir / "metrics.json", result)
    return result


def write_summary(output_root: Path, rows: list[dict[str, Any]], probe_subdir: str) -> None:
    lines = [
        "# ScanNet-Only Semantic Probe Summary",
        "",
        "| Model | Feature | Top1 | mIoU GT-Present | GT-Present Classes | Dominant GT Fraction | Low Confidence |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_label']} | {row['feature_level']} | "
            f"{row['top1_accuracy']:.4f} | {row['mIoU_gt_present']:.4f} | "
            f"{row['num_gt_present_classes']} | {row['dominant_class_fraction']:.4f} | "
            f"{row['low_confidence']} |"
        )
    (output_root / probe_subdir / "semantic_probe_scannet_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_existing_result(output_root: Path, probe_subdir: str, model_label: str, feature_level: str) -> dict[str, Any] | None:
    path = output_root / probe_subdir / model_label / feature_level / "metrics.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--sample-indices", default=str(DEFAULT_OUTPUT_ROOT / "semantic_probe_scannet_final_usable_sample_indices.json"))
    parser.add_argument("--probe-subdir", default="semantic_probes_scannet")
    parser.add_argument("--model-labels", default="zero_spatial,vlm3r_baseline")
    parser.add_argument("--feature-levels", default=None)
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--ignore-index", type=int, default=IGNORE_INDEX)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--early-stop-patience", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--no-write-aggregate",
        action="store_true",
        help="Only write per-probe metrics/history. Useful for Slurm arrays where aggregate files would race.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    train_records = load_frame_records(Path(args.sample_indices), split="train")
    val_records = load_frame_records(Path(args.sample_indices), split="val")
    if not train_records or not val_records:
        raise RuntimeError("Semantic probe training requires non-empty train and val records")
    model_labels = [part.strip() for part in args.model_labels.split(",") if part.strip()]
    feature_override = [part.strip() for part in args.feature_levels.split(",") if part.strip()] if args.feature_levels else None
    results = []
    for model_label in model_labels:
        levels = feature_override if feature_override is not None else available_feature_levels(model_label)
        for feature_level in levels:
            if args.skip_existing:
                existing = load_existing_result(output_root, args.probe_subdir, model_label, feature_level)
                if existing is not None:
                    print(f"[INFO] Skipping existing semantic probe {model_label}/{feature_level}", flush=True)
                    results.append(existing)
                    continue
            train_kept = filter_existing_records(output_root, model_label, feature_level, train_records)
            val_kept = filter_existing_records(output_root, model_label, feature_level, val_records)
            if not args.allow_partial and (len(train_kept) != len(train_records) or len(val_kept) != len(val_records)):
                raise FileNotFoundError(
                    f"Missing semantic GT or features for {model_label}/{feature_level}: "
                    f"train {len(train_kept)}/{len(train_records)}, val {len(val_kept)}/{len(val_records)}. "
                    "Use --allow-partial for smoke tests."
                )
            if not train_kept or not val_kept:
                print(f"[WARN] Skipping {model_label}/{feature_level}: no cached train or val records", file=sys.stderr)
                continue
            results.append(
                train_one_probe(
                    output_root=output_root,
                    probe_subdir=args.probe_subdir,
                    model_label=model_label,
                    feature_level=feature_level,
                    train_records=train_kept,
                    val_records=val_kept,
                    args=args,
                )
            )
    if not args.no_write_aggregate:
        write_json(output_root / args.probe_subdir / "semantic_probe_scannet_results.json", results)
        write_csv(output_root / args.probe_subdir / "semantic_probe_scannet_results.csv", results)
        write_summary(output_root, results, args.probe_subdir)
        write_json(output_root / "semantic_probe_scannet_results.json", results)
        write_csv(output_root / "semantic_probe_scannet_results.csv", results)
        (output_root / "semantic_probe_scannet_summary.md").write_text(
            (output_root / args.probe_subdir / "semantic_probe_scannet_summary.md").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        print(f"[INFO] Wrote semantic probe results under {output_root / args.probe_subdir}", flush=True)
    else:
        print("[INFO] Skipped aggregate semantic result writing (--no-write-aggregate)", flush=True)


if __name__ == "__main__":
    main()
