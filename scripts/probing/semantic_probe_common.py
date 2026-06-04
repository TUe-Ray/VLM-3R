#!/usr/bin/env python
"""Shared helpers for ScanNet-only semantic probing."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch


IGNORE_INDEX = -100
SCANNET20_LABEL_SPACE = "scannet20_nyu40"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalize_tsv_header(name: str) -> str:
    return name.strip().lstrip("\ufeff").lower()


def read_scannet_raw_to_nyu40(label_tsv: Path) -> dict[int, int]:
    """Read official ScanNet `scannet-labels.combined.tsv` id -> nyu40id."""
    with label_tsv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Missing TSV header in {label_tsv}")
        name_map = {normalize_tsv_header(name): name for name in reader.fieldnames}
        if "id" not in name_map or "nyu40id" not in name_map:
            raise ValueError(
                f"{label_tsv} must contain 'id' and 'nyu40id' columns; got {reader.fieldnames}"
            )
        raw_to_nyu40: dict[int, int] = {}
        for row in reader:
            raw_text = str(row[name_map["id"]]).strip()
            nyu_text = str(row[name_map["nyu40id"]]).strip()
            if not raw_text or not nyu_text:
                continue
            raw_to_nyu40[int(raw_text)] = int(nyu_text)
    if not raw_to_nyu40:
        raise ValueError(f"No id -> nyu40id rows loaded from {label_tsv}")
    return raw_to_nyu40


def read_scannet20_class_file(class_file: Path, nyu40id_to_name: dict[int, str] | None = None) -> list[tuple[int, str]]:
    """Read ScanNet20 class order from official benchmark class file."""
    nyu40id_to_name = nyu40id_to_name or {}
    classes: list[tuple[int, str]] = []
    with class_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            nyu40id = int(parts[0])
            name = " ".join(parts[1:]) if len(parts) > 1 else nyu40id_to_name.get(nyu40id, f"class_{nyu40id}")
            classes.append((nyu40id, name))
    if len(classes) != 20:
        raise ValueError(f"Expected 20 ScanNet benchmark classes in {class_file}, got {len(classes)}")
    return classes


def infer_label_value_space(
    unique_values: Iterable[int],
    *,
    raw_to_nyu40: dict[int, int],
) -> str:
    """Infer whether observed labels are raw ScanNet ids or already NYU40 ids.

    Conservative by design: if observed non-ignore values are valid as both raw
    ids and NYU40 ids, the value space is ambiguous and the caller must stop.
    """
    observed = {int(v) for v in unique_values if int(v) not in (IGNORE_INDEX, 0, 255)}
    if not observed:
        raise ValueError("Cannot infer semantic label value space from only ignore/background values")
    raw_ids = set(raw_to_nyu40)
    nyu40_ids = {int(v) for v in raw_to_nyu40.values()}
    raw_supported = observed.issubset(raw_ids)
    nyu_supported = observed.issubset(nyu40_ids)
    if raw_supported and not nyu_supported:
        return "raw_id"
    if nyu_supported and not raw_supported:
        return "nyu40id"
    if raw_supported and nyu_supported:
        raise ValueError(
            "Ambiguous semantic label value space: observed values are valid as both raw ScanNet ids and NYU40 ids"
        )
    raise ValueError("Ambiguous semantic label value space: observed values are not fully supported by either id space")


def load_label_tensor(path: Path) -> torch.Tensor:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required to load semantic label images") from exc
    image = Image.open(path)
    return torch.as_tensor(list(image.getdata()), dtype=torch.long).reshape(image.height, image.width)


def tensor_label_stats(path: Path, tensor: torch.Tensor) -> dict[str, Any]:
    unique = torch.unique(tensor.cpu()).tolist()
    return {
        "semantic_label_path": str(path),
        "dtype": str(tensor.dtype),
        "min_label_value": int(tensor.min().item()) if tensor.numel() else None,
        "max_label_value": int(tensor.max().item()) if tensor.numel() else None,
        "unique_label_values": [int(v) for v in unique],
    }


def build_label_mapping(
    *,
    label_tsv: Path,
    scannet20_class_file: Path,
    label_value_space: str,
    ignore_index: int = IGNORE_INDEX,
) -> dict[str, Any]:
    raw_to_nyu40 = read_scannet_raw_to_nyu40(label_tsv)
    raw_id_to_name = read_scannet_raw_id_to_name(label_tsv)
    nyu40id_to_name_from_tsv = {}
    for raw_id, nyu40id in raw_to_nyu40.items():
        if nyu40id > 0 and nyu40id not in nyu40id_to_name_from_tsv and raw_id in raw_id_to_name:
            nyu40id_to_name_from_tsv[nyu40id] = raw_id_to_name[raw_id]
    classes = read_scannet20_class_file(scannet20_class_file, nyu40id_to_name=nyu40id_to_name_from_tsv)
    nyu40_to_train = {nyu40id: idx for idx, (nyu40id, _name) in enumerate(classes)}
    nyu40_to_name = {nyu40id: name for nyu40id, name in classes}
    if label_value_space == "raw_id":
        label_to_train = {
            raw_id: nyu40_to_train.get(nyu40id, ignore_index)
            for raw_id, nyu40id in raw_to_nyu40.items()
        }
    elif label_value_space == "nyu40id":
        label_to_train = {
            nyu40id: nyu40_to_train.get(nyu40id, ignore_index)
            for nyu40id in sorted(set(raw_to_nyu40.values()) | set(nyu40_to_train))
        }
    else:
        raise ValueError(f"Unsupported label value space: {label_value_space}")
    return {
        "label_space_name": SCANNET20_LABEL_SPACE,
        "label_value_space": label_value_space,
        "num_classes": 20,
        "ignore_index": ignore_index,
        "source_scannet_label_tsv": str(label_tsv),
        "source_scannet20_class_file": str(scannet20_class_file),
        "class_index_to_name": {idx: name for idx, (_nyu40id, name) in enumerate(classes)},
        "class_index_to_nyu40id": {idx: nyu40id for idx, (nyu40id, _name) in enumerate(classes)},
        "nyu40id_to_class_name": nyu40_to_name,
        "nyu40id_to_train_label": nyu40_to_train,
        "raw_id_to_nyu40id": raw_to_nyu40,
        "raw_label_to_train_label": label_to_train,
    }


def read_scannet_raw_id_to_name(label_tsv: Path) -> dict[int, str]:
    """Best-effort raw id -> class name lookup from the official TSV."""
    with label_tsv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            return {}
        name_map = {normalize_tsv_header(name): name for name in reader.fieldnames}
        if "id" not in name_map:
            return {}
        name_column = None
        for candidate in ("raw_category", "category", "nyu40class", "name"):
            if candidate in name_map:
                name_column = name_map[candidate]
                break
        if name_column is None:
            return {}
        out: dict[int, str] = {}
        for row in reader:
            raw_text = str(row[name_map["id"]]).strip()
            name = str(row[name_column]).strip()
            if raw_text and name:
                out[int(raw_text)] = name
        return out


def map_label_tensor_to_train_labels(label_tensor: torch.Tensor, mapping: dict[str, Any]) -> torch.Tensor:
    ignore_index = int(mapping["ignore_index"])
    out = torch.full_like(label_tensor.long(), fill_value=ignore_index)
    for label_value, train_label in mapping["raw_label_to_train_label"].items():
        train_label = int(train_label)
        if train_label == ignore_index:
            continue
        out[label_tensor.long() == int(label_value)] = train_label
    return out


def squeeze_singleton_feature_batch(feature: torch.Tensor) -> torch.Tensor:
    """Remove an unambiguous singleton batch dimension from cached features."""
    if feature.ndim == 4 and int(feature.shape[0]) == 1:
        return feature.squeeze(0)
    if feature.ndim == 3 and int(feature.shape[0]) == 1:
        return feature.squeeze(0)
    return feature


def downsample_train_labels_majority(
    train_labels: torch.Tensor,
    grid_shape: tuple[int, int],
    *,
    num_classes: int = 20,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    if train_labels.ndim != 2:
        raise ValueError(f"Expected [H,W] semantic labels, got {tuple(train_labels.shape)}")
    h, w = int(train_labels.shape[0]), int(train_labels.shape[1])
    h_tok, w_tok = int(grid_shape[0]), int(grid_shape[1])
    out = torch.full((h_tok, w_tok), fill_value=ignore_index, dtype=torch.long)
    for iy in range(h_tok):
        y0 = math.floor(iy * h / h_tok)
        y1 = math.floor((iy + 1) * h / h_tok)
        for ix in range(w_tok):
            x0 = math.floor(ix * w / w_tok)
            x1 = math.floor((ix + 1) * w / w_tok)
            patch = train_labels[y0:y1, x0:x1].reshape(-1)
            valid = patch[(patch >= 0) & (patch < num_classes)]
            if valid.numel() == 0:
                continue
            counts = torch.bincount(valid.long(), minlength=num_classes)
            out[iy, ix] = int(torch.argmax(counts).item())
    return out


def update_confusion(confusion: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, *, num_classes: int = 20) -> None:
    pred = pred.reshape(-1).long().cpu()
    target = target.reshape(-1).long().cpu()
    valid = (target >= 0) & (target < num_classes)
    if valid.sum().item() == 0:
        return
    encoded = target[valid] * num_classes + pred[valid].clamp(0, num_classes - 1)
    confusion += torch.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def metrics_from_confusion(confusion: torch.Tensor) -> dict[str, Any]:
    confusion = confusion.long()
    tp = torch.diag(confusion).float()
    gt_count = confusion.sum(dim=1).float()
    pred_count = confusion.sum(dim=0).float()
    union = gt_count + pred_count - tp
    iou = torch.where(union > 0, tp / union.clamp_min(1), torch.full_like(tp, float("nan")))
    present = gt_count > 0
    total_valid = int(gt_count.sum().item())
    top1 = float(tp.sum().item() / max(total_valid, 1))
    present_ious = iou[present]
    miou = float(present_ious.mean().item()) if present_ious.numel() else float("nan")
    dominant_fraction = float(gt_count.max().item() / max(total_valid, 1)) if total_valid else float("nan")
    return {
        "top1_accuracy": top1,
        "mIoU": miou,
        "mIoU_gt_present": miou,
        "per_class_IoU": [None if torch.isnan(v) else float(v.item()) for v in iou],
        "present_classes": [int(i) for i in torch.where(present)[0].tolist()],
        "num_present_classes": int(present.sum().item()),
        "num_gt_present_classes": int(present.sum().item()),
        "valid_token_count": total_valid,
        "dominant_class_fraction": dominant_fraction,
        "low_confidence": bool(int(present.sum().item()) < 5 or dominant_fraction > 0.8),
    }


def candidate_semantic_label_paths(label_root: Path, scene_id: str, frame_index: int) -> list[Path]:
    names = [
        f"{frame_index}.png",
        f"{frame_index:06d}.png",
        f"{frame_index}.jpg",
        f"{frame_index:06d}.jpg",
    ]
    dirs = [
        label_root / scene_id / "label-filt",
        label_root / scene_id / "label",
        label_root / "label-filt" / scene_id,
        label_root / "label" / scene_id,
        label_root / scene_id,
    ]
    return [directory / name for directory in dirs for name in names]


def find_semantic_label_path(label_root: Path, scene_id: str, frame_index: int) -> Path | None:
    for candidate in candidate_semantic_label_paths(label_root, scene_id, frame_index):
        if candidate.exists():
            return candidate
    return None
