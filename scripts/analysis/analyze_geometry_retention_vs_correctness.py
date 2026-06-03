#!/usr/bin/env python
"""Analyze whether hidden-state CUT3R geometry retention predicts correctness.

This is a diagnostic script: it does not train, does not update model weights,
and uses model visual-token metadata rather than guessing token positions.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - plotting is optional at import time
    plt = None

try:
    from scipy import stats
except ImportError:  # pragma: no cover - handled at runtime
    stats = None


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MAJOR_CATEGORIES = {
    "Abs Dist",
    "Rel Dist",
    "Obj Size",
    "Room Size",
    "Rel Dir",
    "Route Plan",
    "Appr Order",
    "Obj Count",
}

PER_SAMPLE_COLUMNS = [
    "sample_id",
    "category",
    "question",
    "correctness",
    "model_answer",
    "ground_truth",
    "layer",
    "representation",
    "geometry_gap_mean",
    "geometry_gap_median",
    "geometry_rank_acc",
    "geometry_margin_loss",
    "teacher_gap_mean",
    "teacher_sim_pos_mean",
    "teacher_sim_neg_mean",
    "student_sim_pos_mean",
    "student_sim_neg_mean",
    "num_triplets",
    "num_frames",
    "tokens_per_frame",
    "model_path",
]

CATEGORY_COLUMNS = [
    "category",
    "layer",
    "representation",
    "n_correct",
    "n_wrong",
    "correct_gap_mean",
    "wrong_gap_mean",
    "gap_difference",
    "gap_difference_ci_low",
    "gap_difference_ci_high",
    "correct_rank_acc_mean",
    "wrong_rank_acc_mean",
    "rank_acc_difference",
    "rank_acc_difference_ci_low",
    "rank_acc_difference_ci_high",
    "correct_margin_loss_mean",
    "wrong_margin_loss_mean",
    "margin_loss_difference",
    "margin_loss_difference_ci_low",
    "margin_loss_difference_ci_high",
]


@dataclass
class PredictionRecord:
    sample_id: str
    question: str
    category: str
    model_answer: str
    ground_truth: str
    correctness: int


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_int_list(value: str) -> list[int]:
    layers = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not layers:
        raise argparse.ArgumentTypeError("--layers must contain at least one layer index")
    if any(layer < 1 for layer in layers):
        raise argparse.ArgumentTypeError("Layer ids are 1-based: H1 is layer 1")
    return layers


def parse_categories(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def load_json_or_jsonl(path: Path) -> Any:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_prediction_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("logs", "results", "predictions", "samples", "data", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        if all(isinstance(v, dict) for v in payload.values()):
            return [v for v in payload.values()]
    raise ValueError("Prediction JSON must be a list, a dict of records, or contain a results/predictions list.")


def first_present(record: dict[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def normalize_question_type(question_type: str) -> str:
    mapping = {
        "object_abs_distance": "Abs Dist",
        "object_rel_distance": "Rel Dist",
        "object_size_estimation": "Obj Size",
        "room_size_estimation": "Room Size",
        "object_rel_direction_easy": "Rel Dir",
        "object_rel_direction_medium": "Rel Dir",
        "object_rel_direction_hard": "Rel Dir",
        "route_planning": "Route Plan",
        "obj_appearance_order": "Appr Order",
        "object_counting": "Obj Count",
        "camera_obj_abs_dist": "Abs Dist",
        "camera_obj_rel_dist_v1": "Rel Dist",
        "camera_obj_rel_dist_v2": "Rel Dist",
        "camera_obj_rel_dist_v3": "Rel Dist",
        "camera_displacement": "Abs Dist",
        "camera_movement_direction_v1": "Rel Dir",
        "camera_movement_direction_v2": "Rel Dir",
        "camera_movement_direction_v3": "Rel Dir",
        "obj_obj_relative_pos_lr": "Rel Dir",
        "obj_obj_relative_pos_ud": "Rel Dir",
        "obj_obj_relative_pos_nf": "Rel Dir",
    }
    return mapping.get(str(question_type), str(question_type))


def normalize_correctness(value: Any, sample_id: str) -> int:
    if value is None:
        raise ValueError(
            f"Prediction for sample_id={sample_id!r} is missing correctness. "
            "Please provide an evaluated prediction file with correctness: 1 or 0."
        )
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return int(value)
    if isinstance(value, float):
        if value in (0.0, 1.0):
            return int(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "correct", "yes"}:
            return 1
        if lowered in {"0", "false", "wrong", "incorrect", "no"}:
            return 0
    raise ValueError(f"Invalid correctness value for sample_id={sample_id!r}: {value!r}")


def normalize_answer_text(value: str) -> str:
    text = str(value).strip().lower()
    text = text.replace("</s>", "").replace("<|im_end|>", "").strip()
    text = re.sub(r"^[\s\"'`]+|[\s\"'`]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_option_letter(value: str) -> str | None:
    text = normalize_answer_text(value)
    match = re.match(r"^(?:option\s*)?([a-d])(?:[\.\):,\s]|$)", text)
    if match:
        return match.group(1).upper()
    return None


def extract_first_number(value: str) -> float | None:
    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def match_generated_answer(model_answer: str, ground_truth: str, numeric_tolerance: float) -> int:
    pred_letter = extract_option_letter(model_answer)
    gt_letter = extract_option_letter(ground_truth)
    if gt_letter is not None:
        return int(pred_letter == gt_letter)

    gt_num = extract_first_number(ground_truth)
    pred_num = extract_first_number(model_answer)
    if gt_num is not None:
        return int(pred_num is not None and abs(pred_num - gt_num) <= numeric_tolerance)

    pred = normalize_answer_text(model_answer)
    gt = normalize_answer_text(ground_truth)
    return int(pred == gt or pred.startswith(gt) or gt.startswith(pred))


def extract_ground_truth(raw_item: dict[str, Any]) -> str:
    for key in ("ground_truth", "gt", "target", "label", "answer"):
        if raw_item.get(key) is not None:
            return str(raw_item[key])
    conversations = raw_item.get("conversations")
    if isinstance(conversations, list):
        for turn in reversed(conversations):
            if isinstance(turn, dict) and str(turn.get("from", "")).lower() in {"gpt", "assistant"}:
                return str(turn.get("value", ""))
    return ""


def load_predictions(path: Path) -> dict[str, PredictionRecord]:
    records = flatten_prediction_payload(load_json_or_jsonl(path))
    predictions: dict[str, PredictionRecord] = {}
    for idx, record in enumerate(records):
        doc = record.get("doc") if isinstance(record.get("doc"), dict) else record
        score_doc = record.get("vsibench_score") if isinstance(record.get("vsibench_score"), dict) else doc
        sample_id = str(first_present(record, ("sample_id", "doc_id", "id", "question_id", "uid"), first_present(doc, ("sample_id", "id", "question_id", "uid"), idx)))
        correctness_value = first_present(record, ("correctness", "correct", "is_correct"), None)
        if correctness_value is None and isinstance(score_doc, dict):
            if "accuracy" in score_doc:
                correctness_value = score_doc["accuracy"]
            elif "MRA:.5:.95:.05" in score_doc:
                correctness_value = 1 if float(score_doc["MRA:.5:.95:.05"]) >= 1.0 else 0
        correctness = normalize_correctness(correctness_value, sample_id)
        predictions[sample_id] = PredictionRecord(
            sample_id=sample_id,
            question=str(first_present(score_doc, ("question", "prompt", "query"), first_present(record, ("question", "prompt", "query"), ""))),
            category=normalize_question_type(str(first_present(score_doc, ("category", "question_type", "task", "task_type", "type"), first_present(record, ("category", "question_type", "task", "task_type", "type"), "")))),
            model_answer=str(first_present(score_doc, ("model_answer", "prediction", "pred", "answer", "response"), first_present(record, ("model_answer", "prediction", "pred", "answer", "response", "filtered_resps"), ""))),
            ground_truth=str(first_present(score_doc, ("ground_truth", "gt", "target", "label", "answer_gt"), first_present(record, ("ground_truth", "gt", "target", "label", "answer_gt"), ""))),
            correctness=correctness,
        )
    if not predictions:
        raise ValueError(f"No prediction records found in {path}")
    return predictions


def extract_question(raw_item: dict[str, Any], fallback: str = "") -> str:
    question = first_present(raw_item, ("question", "prompt", "query"), "")
    if question:
        return str(question)
    conversations = raw_item.get("conversations")
    if isinstance(conversations, list) and conversations:
        value = str(conversations[0].get("value", ""))
        return value.replace("<image>", "").replace("<video>", "").strip()
    return fallback


def raw_sample_ids(raw_item: dict[str, Any], index: int) -> list[str]:
    ids = []
    for key in ("sample_id", "id", "question_id", "uid"):
        if raw_item.get(key) is not None:
            ids.append(str(raw_item[key]))
    ids.append(str(index))
    return list(dict.fromkeys(ids))


def move_to_device(value: Any, device: torch.device, dtype: torch.dtype) -> Any:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            return value.to(device=device, dtype=dtype, non_blocking=True)
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device, dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device, dtype) for item in value)
    return value


def get_transformer_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return get_transformer_layers(model.base_model.model)
    raise RuntimeError("Could not locate LLM transformer blocks at model.model.layers.")


def pool_cut3r_teacher_to_student_grid(
    teacher_tokens: torch.Tensor,
    target_tokens: int,
    pool_mode: str,
) -> torch.Tensor:
    token_count = int(teacher_tokens.shape[0])
    if token_count == target_tokens:
        return teacher_tokens
    if token_count != 729:
        raise ValueError(f"CUT3R teacher must have 729 raw tokens before pooling, got {token_count}")
    if target_tokens != 196:
        raise ValueError(f"Unsupported visual token count per frame: {target_tokens}")

    teacher_grid = teacher_tokens.view(27, 27, -1)
    pool_mode = str(pool_mode or "bilinear").lower()
    if pool_mode == "bilinear":
        pooled = F.interpolate(
            teacher_grid.permute(2, 0, 1).unsqueeze(0).float(),
            size=(14, 14),
            mode="bilinear",
            align_corners=False,
        )[0].permute(1, 2, 0)
        return pooled.to(dtype=teacher_tokens.dtype).reshape(196, -1)

    padded = F.pad(teacher_grid.permute(2, 0, 1), (0, 1, 0, 1), value=0.0)
    valid = F.pad(torch.ones(1, 27, 27, device=teacher_tokens.device), (0, 1, 0, 1), value=0.0)
    if pool_mode == "average":
        counts = F.avg_pool2d(valid, kernel_size=2, stride=2) * 4.0
        summed = F.avg_pool2d(padded.float(), kernel_size=2, stride=2) * 4.0
        pooled = summed / counts.clamp_min(1.0)
    elif pool_mode == "max":
        masked = padded.float().masked_fill(valid.bool().expand_as(padded) == 0, -torch.finfo(torch.float32).max)
        pooled = F.max_pool2d(masked, kernel_size=2, stride=2)
    else:
        raise ValueError(f"Unsupported CUT3R pool mode: {pool_mode}")
    return pooled.permute(1, 2, 0).to(dtype=teacher_tokens.dtype).reshape(196, -1)


def percentile_pool(row: torch.Tensor, low_pct: float, high_pct: float, exclude_index: int) -> torch.Tensor:
    valid = torch.ones(row.numel(), dtype=torch.bool, device=row.device)
    valid[exclude_index] = False
    candidates = torch.where(valid)[0]
    scores = row[candidates]
    order = torch.argsort(scores, descending=False)
    n = int(order.numel())
    lo = int(math.floor(n * low_pct / 100.0))
    hi = int(math.ceil(n * high_pct / 100.0))
    lo = max(0, min(lo, n - 1))
    hi = max(lo + 1, min(hi, n))
    return candidates[order[lo:hi]]


def sample_triplets_for_frame(
    teacher: torch.Tensor,
    student: torch.Tensor,
    anchors_per_frame: int,
    positive_top_percent: float,
    negative_bottom_percent: float,
    negative_mode: str,
    semihard_neg_low_percent: float,
    semihard_neg_high_percent: float,
    margin: float,
    generator: torch.Generator,
    triplet_indices: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    teacher = F.normalize(teacher.float(), dim=-1)
    student = F.normalize(student.float(), dim=-1)
    teacher_sim = teacher @ teacher.T
    student_sim = student @ student.T
    num_tokens = int(teacher.shape[0])
    if num_tokens < 3:
        raise ValueError(f"Need at least 3 visual tokens per frame, got {num_tokens}")

    selected_triplets: list[tuple[int, int, int]] = []
    if triplet_indices is not None:
        if triplet_indices.dim() != 2 or triplet_indices.shape[1] != 3:
            raise ValueError(f"Expected triplet_indices [n,3], got {tuple(triplet_indices.shape)}")
        for anchor, pos, neg in triplet_indices.cpu().long().tolist():
            if not (0 <= anchor < num_tokens and 0 <= pos < num_tokens and 0 <= neg < num_tokens):
                raise ValueError("Triplet index out of range for frame token count")
            selected_triplets.append((int(anchor), int(pos), int(neg)))
    else:
        anchor_count = min(int(anchors_per_frame), num_tokens)
        anchors = torch.randperm(num_tokens, generator=generator)[:anchor_count]
        pos_k = max(1, int(math.ceil(num_tokens * positive_top_percent / 100.0)))
        neg_k = max(1, int(math.ceil(num_tokens * negative_bottom_percent / 100.0)))

        for anchor_tensor in anchors:
            anchor = int(anchor_tensor.item())
            row = teacher_sim[anchor].clone()
            row[anchor] = -float("inf")
            pos_pool = torch.topk(row, k=min(pos_k, num_tokens - 1), largest=True).indices

            if negative_mode == "bottom":
                neg_row = teacher_sim[anchor].clone()
                neg_row[anchor] = float("inf")
                neg_pool = torch.topk(neg_row, k=min(neg_k, num_tokens - 1), largest=False).indices
            elif negative_mode == "semihard":
                neg_pool = percentile_pool(teacher_sim[anchor], semihard_neg_low_percent, semihard_neg_high_percent, anchor)
            else:
                raise ValueError(f"Unsupported negative_mode: {negative_mode}")
            if pos_pool.numel() == 0 or neg_pool.numel() == 0:
                continue

            pos = int(pos_pool[torch.randint(pos_pool.numel(), (1,), generator=generator)].item())
            neg = int(neg_pool[torch.randint(neg_pool.numel(), (1,), generator=generator)].item())
            selected_triplets.append((anchor, pos, neg))

    values = defaultdict(list)

    for anchor, pos, neg in selected_triplets:
        teacher_pos = teacher_sim[anchor, pos]
        teacher_neg = teacher_sim[anchor, neg]
        student_pos = student_sim[anchor, pos]
        student_neg = student_sim[anchor, neg]

        values["teacher_pos"].append(teacher_pos)
        values["teacher_neg"].append(teacher_neg)
        values["student_pos"].append(student_pos)
        values["student_neg"].append(student_neg)
        values["gap"].append(student_pos - student_neg)
        values["rank_correct"].append((student_pos > student_neg).float())
        values["margin_loss"].append(F.relu(torch.tensor(margin, device=student.device) - student_pos + student_neg))
        values["anchor_index"].append(torch.tensor(anchor, dtype=torch.long, device=student.device))
        values["positive_index"].append(torch.tensor(pos, dtype=torch.long, device=student.device))
        values["negative_index"].append(torch.tensor(neg, dtype=torch.long, device=student.device))

    if not values["gap"]:
        raise ValueError("Triplet sampling produced no triplets.")
    return {key: torch.stack(items) for key, items in values.items()}


def compute_geometry_retention(
    frame_students: list[torch.Tensor],
    patch_tokens: torch.Tensor,
    args: argparse.Namespace,
    generator: torch.Generator,
    triplets_by_frame: list[torch.Tensor] | None = None,
) -> dict[str, Any]:
    if patch_tokens.dim() == 4 and patch_tokens.shape[0] == 1:
        patch_tokens = patch_tokens[0]
    if patch_tokens.dim() != 3:
        raise ValueError(f"Expected CUT3R patch_tokens [frames,tokens,dim], got {tuple(patch_tokens.shape)}")
    if len(frame_students) != int(patch_tokens.shape[0]):
        raise ValueError(f"Frame count mismatch: hidden={len(frame_students)} CUT3R={patch_tokens.shape[0]}")
    if triplets_by_frame is not None and len(triplets_by_frame) != len(frame_students):
        raise ValueError(f"Triplet frame count mismatch: triplets={len(triplets_by_frame)} hidden={len(frame_students)}")

    all_values = defaultdict(list)
    triplet_rows = []
    sampled_triplets_by_frame = []
    tokens_per_frame = []
    for frame_idx, student_frame in enumerate(frame_students):
        student_frame = student_frame.detach().cpu()
        teacher_frame = pool_cut3r_teacher_to_student_grid(
            patch_tokens[frame_idx].detach().cpu(),
            int(student_frame.shape[0]),
            args.pool_mode,
        )
        if int(student_frame.shape[0]) != int(teacher_frame.shape[0]):
            raise ValueError(
                f"Token count mismatch in frame {frame_idx}: hidden={student_frame.shape[0]} CUT3R={teacher_frame.shape[0]}"
            )
        tokens_per_frame.append(int(student_frame.shape[0]))
        frame_values = sample_triplets_for_frame(
            teacher=teacher_frame,
            student=student_frame,
            anchors_per_frame=args.anchors_per_frame,
            positive_top_percent=args.positive_top_percent,
            negative_bottom_percent=args.negative_bottom_percent,
            negative_mode=args.negative_mode,
            semihard_neg_low_percent=args.semihard_neg_low_percent,
            semihard_neg_high_percent=args.semihard_neg_high_percent,
            margin=args.margin,
            generator=generator,
            triplet_indices=triplets_by_frame[frame_idx] if triplets_by_frame is not None else None,
        )
        sampled_triplets_by_frame.append(
            torch.stack(
                [
                    frame_values["anchor_index"].long(),
                    frame_values["positive_index"].long(),
                    frame_values["negative_index"].long(),
                ],
                dim=1,
            )
        )
        for key, value in frame_values.items():
            all_values[key].append(value)
        if args.save_per_triplet:
            num_triplets = int(frame_values["gap"].numel())
            for triplet_idx in range(num_triplets):
                triplet_rows.append({
                    "frame_id": frame_idx,
                    "anchor_index": int(frame_values["anchor_index"][triplet_idx].item()),
                    "positive_index": int(frame_values["positive_index"][triplet_idx].item()),
                    "negative_index": int(frame_values["negative_index"][triplet_idx].item()),
                    "teacher_sim_pos": float(frame_values["teacher_pos"][triplet_idx].item()),
                    "teacher_sim_neg": float(frame_values["teacher_neg"][triplet_idx].item()),
                    "student_sim_pos": float(frame_values["student_pos"][triplet_idx].item()),
                    "student_sim_neg": float(frame_values["student_neg"][triplet_idx].item()),
                    "gap": float(frame_values["gap"][triplet_idx].item()),
                    "rank_correct": int(frame_values["rank_correct"][triplet_idx].item()),
                    "margin_loss": float(frame_values["margin_loss"][triplet_idx].item()),
                })

    merged = {key: torch.cat(value, dim=0) for key, value in all_values.items()}
    teacher_gap = merged["teacher_pos"] - merged["teacher_neg"]
    if float(teacher_gap.mean().item()) <= 0.0:
        raise ValueError(f"Teacher sanity check failed: teacher_gap_mean={teacher_gap.mean().item():.6f}")

    token_count_set = sorted(set(tokens_per_frame))
    return {
        "geometry_gap_mean": float(merged["gap"].mean().item()),
        "geometry_gap_median": float(merged["gap"].median().item()),
        "geometry_rank_acc": float(merged["rank_correct"].mean().item()),
        "geometry_margin_loss": float(merged["margin_loss"].mean().item()),
        "teacher_gap_mean": float(teacher_gap.mean().item()),
        "teacher_sim_pos_mean": float(merged["teacher_pos"].mean().item()),
        "teacher_sim_neg_mean": float(merged["teacher_neg"].mean().item()),
        "student_sim_pos_mean": float(merged["student_pos"].mean().item()),
        "student_sim_neg_mean": float(merged["student_neg"].mean().item()),
        "num_triplets": int(merged["gap"].numel()),
        "num_frames": len(frame_students),
        "tokens_per_frame": token_count_set[0] if len(token_count_set) == 1 else json.dumps(token_count_set),
        "_triplets": triplet_rows,
        "_triplets_by_frame": sampled_triplets_by_frame,
    }


def make_data_args(args: argparse.Namespace, image_processor: Any) -> SimpleNamespace:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    return SimpleNamespace(
        data_path=args.data_json,
        lazy_preprocess=True,
        is_multimodal=True,
        early_mix_text=False,
        image_folder=args.image_folder,
        image_aspect_ratio=getattr(cfg, "image_aspect_ratio", "anyres_max_9"),
        image_grid_pinpoints=getattr(cfg, "image_grid_pinpoints", None),
        image_crop_resolution=getattr(cfg, "image_crop_resolution", None),
        image_split_resolution=getattr(cfg, "image_split_resolution", None),
        video_folder=args.video_folder,
        video_fps=1,
        frames_upbound=args.frames_upbound,
        add_time_instruction=args.add_time_instruction if args.add_time_instruction is not None else str2bool(getattr(cfg, "add_time_instruction", True)),
        force_sample=str2bool(getattr(cfg, "force_sample", True)),
        train_data_percentage=100.0,
        train_data_percentage_seed=args.seed,
        train_data_shuffle=False,
        deterministic_data_order=False,
        zero_spatial_features=False,
        spatial_tower_type="cut3r",
        spatial_features_root=args.spatial_feature_dir,
        spatial_features_subdir=args.spatial_features_subdir,
        image_processor=image_processor,
        mm_use_im_start_end=False,
    )


def load_model(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    model_name = args.model_name or get_model_name_from_path(args.model_path)
    load_zero_spatial = bool(args.skip_spatial_tower_load)

    original_build_spatial_tower = None
    if load_zero_spatial:
        import llava.model.llava_arch as llava_arch

        class Cut3rSidecarOnlySpatialTower(nn.Module):
            def __init__(self, spatial_tower: str = "cut3r"):
                super().__init__()
                self.spatial_tower_name = spatial_tower
                self.is_loaded = False
                self.config = SimpleNamespace()

            def load_model(self, device_map=None):
                raise RuntimeError(
                    "Runtime CUT3R tower loading was intentionally skipped. "
                    "This analysis expects precomputed CUT3R sidecar features."
                )

        original_build_spatial_tower = llava_arch.build_spatial_tower

        def build_sidecar_only_spatial_tower(spatial_tower_cfg, **kwargs):
            spatial_tower = getattr(spatial_tower_cfg, "mm_spatial_tower", getattr(spatial_tower_cfg, "spatial_tower", "cut3r"))
            if "cut3r" in str(spatial_tower).lower():
                return Cut3rSidecarOnlySpatialTower(str(spatial_tower))
            return original_build_spatial_tower(spatial_tower_cfg, **kwargs)

        llava_arch.build_spatial_tower = build_sidecar_only_spatial_tower

    try:
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path,
            args.model_base,
            model_name,
            device_map=str(device),
            torch_dtype="bfloat16" if dtype == torch.bfloat16 else "float16",
            attn_implementation=args.attn_implementation,
            overwrite_config={
                "delay_load": False,
                "mm_spatial_pool_stride": args.mm_spatial_pool_stride,
                "mm_spatial_pool_mode": args.pool_mode,
                # Loading with zero_spatial=True prevents LlavaQwen.from_pretrained()
                # from calling load_model() on the sidecar-only tower. We flip this
                # back below before inference so precomputed CUT3R tokens still fuse.
                "zero_spatial_features": load_zero_spatial,
            },
        )
    finally:
        if original_build_spatial_tower is not None:
            import llava.model.llava_arch as llava_arch
            llava_arch.build_spatial_tower = original_build_spatial_tower

    model.to(device=device, dtype=dtype)
    model.eval()
    model.config.use_cache = False
    model.config.spatial_rank_loss_enable = False
    model.config.zero_spatial_features = False
    return tokenizer, model, image_processor


def candidate_feature_paths(raw_item: dict[str, Any], spatial_feature_dir: Path) -> list[Path]:
    names = []
    for key in ("video", "image", "sample_id", "id", "question_id", "uid"):
        value = raw_item.get(key)
        if isinstance(value, list):
            value = value[0] if value else None
        if value is None:
            continue
        value = str(value)
        names.append(Path(value).with_suffix(".pt"))
        names.append(Path(value).stem + ".pt")
    candidates = []
    dataset = raw_item.get("dataset")
    scene_name = raw_item.get("scene_name")
    if dataset and scene_name:
        candidates.append(spatial_feature_dir / str(dataset) / "spatial_features" / f"{scene_name}.pt")
        candidates.append(spatial_feature_dir / str(dataset) / f"{scene_name}.pt")
        candidates.append(spatial_feature_dir / "spatial_features" / str(dataset) / f"{scene_name}.pt")
    for name in names:
        path = Path(name)
        candidates.append(spatial_feature_dir / path)
        candidates.append(spatial_feature_dir / path.name)
    return list(dict.fromkeys(candidates))


def ensure_spatial_features(item: dict[str, Any], raw_item: dict[str, Any], spatial_feature_dir: Path) -> None:
    if "spatial_features" in item and isinstance(item["spatial_features"], dict) and "patch_tokens" in item["spatial_features"]:
        return
    for path in candidate_feature_paths(raw_item, spatial_feature_dir):
        if path.exists():
            item["spatial_features"] = torch.load(path, map_location="cpu")
            return
    raise FileNotFoundError(f"No CUT3R feature file found for sample; tried roots under {spatial_feature_dir}")


def extract_hidden_states(
    model: torch.nn.Module,
    batch: dict[str, Any],
    layers: list[int],
    device: torch.device,
    dtype: torch.dtype,
    use_projection_head: bool,
) -> tuple[dict[int, list[torch.Tensor]], dict[int, list[torch.Tensor]], dict[str, Any]]:
    batch = move_to_device(batch, device, dtype)
    prepare_fn = getattr(model, "prepare_inputs_labels_for_multimodal", None)
    if prepare_fn is None:
        raise RuntimeError("Model does not expose prepare_inputs_labels_for_multimodal().")
    if "return_visual_metadata" not in inspect.signature(prepare_fn).parameters:
        raise RuntimeError(
            "prepare_inputs_labels_for_multimodal() does not support return_visual_metadata. "
            "Please use code/checkpoints with visual metadata support."
        )

    prepare_kwargs = {
        "input_ids": batch["input_ids"],
        "position_ids": None,
        "attention_mask": batch["attention_mask"],
        "past_key_values": None,
        "labels": None,
        "images": batch["images"],
        "spatial_features": batch.get("spatial_features"),
        "point_maps": batch.get("point_maps"),
        "modalities": batch.get("modalities"),
        "image_sizes": batch.get("image_sizes"),
        "return_visual_metadata": True,
    }
    with torch.no_grad():
        prepared = prepare_fn(**prepare_kwargs)
    input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, _labels, visual_metadata = prepared
    if not visual_metadata:
        raise RuntimeError("Model did not return visual metadata.")
    metadata = visual_metadata[0]

    blocks = get_transformer_layers(model)
    max_layer = max(layers)
    if max_layer > len(blocks):
        raise ValueError(f"Requested H{max_layer}, but model only has {len(blocks)} transformer blocks.")

    holders: dict[int, torch.Tensor] = {}
    handles = []
    for layer in layers:
        block_idx = layer - 1

        def capture(_module, _inputs, output, layer_id=layer):
            holders[layer_id] = output[0] if isinstance(output, (tuple, list)) else output

        handles.append(blocks[block_idx].register_forward_hook(capture))

    try:
        with torch.no_grad():
            model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    visual_indices = metadata["visual_token_indices"].to(device=device)
    frame_ids = metadata["visual_frame_ids"].to(device=device)
    if visual_indices.numel() == 0:
        raise RuntimeError("No valid visual tokens were returned by metadata.")
    excluded = torch.cat([
        metadata.get("newline_token_indices", torch.empty(0, dtype=torch.long, device=device)).to(device=device),
        metadata.get("padding_token_indices", torch.empty(0, dtype=torch.long, device=device)).to(device=device),
        metadata.get("answer_token_indices", torch.empty(0, dtype=torch.long, device=device)).to(device=device),
        metadata.get("text_token_indices", torch.empty(0, dtype=torch.long, device=device)).to(device=device),
        metadata.get("special_token_indices", torch.empty(0, dtype=torch.long, device=device)).to(device=device),
        metadata.get("camera_prefix_token_indices", torch.empty(0, dtype=torch.long, device=device)).to(device=device),
    ])
    if excluded.numel() > 0 and torch.isin(visual_indices, excluded).any():
        raise RuntimeError("Visual metadata overlaps excluded text/newline/padding/special/prefix tokens.")

    frame_order = [int(x) for x in metadata.get("frame_order", [])]
    if not frame_order:
        frame_order = sorted({int(x) for x in frame_ids.detach().cpu().tolist()})

    raw_by_layer: dict[int, list[torch.Tensor]] = {}
    projected_by_layer: dict[int, list[torch.Tensor]] = {}
    rank_head = getattr(model, "spatial_rank_head", None)
    for layer in layers:
        if layer not in holders:
            raise RuntimeError(f"H{layer} hook did not fire.")
        hidden = holders[layer]
        raw_frames = []
        projected_frames = []
        for frame_id in frame_order:
            indices = visual_indices[frame_ids == frame_id]
            frame_hidden = hidden[0, indices].detach()
            raw_frames.append(frame_hidden.float().cpu())
            if use_projection_head and rank_head is not None:
                projected_frames.append(rank_head(frame_hidden.to(dtype=dtype)).detach().float().cpu())
        raw_by_layer[layer] = raw_frames
        if projected_frames:
            projected_by_layer[layer] = projected_frames
    return raw_by_layer, projected_by_layer, metadata


def prompt_only_batch(batch: dict[str, Any], tokenizer: Any) -> tuple[torch.Tensor, torch.Tensor]:
    from llava.constants import IGNORE_INDEX

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    if input_ids.shape[0] != 1:
        raise ValueError("Generated-answer scoring currently expects batch size 1.")
    answer_positions = torch.where(labels[0] != IGNORE_INDEX)[0]
    if answer_positions.numel() == 0:
        prompt_len = int(batch["attention_mask"][0].sum().item())
    else:
        prompt_len = int(answer_positions[0].item())
    prompt_len = max(1, prompt_len)
    prompt_ids = input_ids[:, :prompt_len]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    attention_mask = prompt_ids.ne(pad_id).long()
    return prompt_ids, attention_mask


def generate_answer(
    model: torch.nn.Module,
    tokenizer: Any,
    batch: dict[str, Any],
    raw_item: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
) -> str:
    from llava import conversation as conversation_lib
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token

    question = extract_question(raw_item)
    if DEFAULT_IMAGE_TOKEN not in question and ("video" in raw_item or "image" in raw_item):
        question = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv = conversation_lib.default_conversation.copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    prompt_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    attention_mask = prompt_ids.ne(pad_id).long()
    gen_batch = move_to_device(
        {
            "input_ids": prompt_ids,
            "attention_mask": attention_mask,
            "images": batch.get("images"),
            "spatial_features": batch.get("spatial_features"),
            "modalities": batch.get("modalities"),
            "image_sizes": batch.get("image_sizes"),
        },
        device,
        dtype,
    )
    with torch.no_grad():
        output_ids = model.generate(
            inputs=gen_batch["input_ids"],
            images=gen_batch.get("images"),
            spatial_features=gen_batch.get("spatial_features"),
            attention_mask=gen_batch["attention_mask"],
            modalities=gen_batch.get("modalities"),
            image_sizes=gen_batch.get("image_sizes"),
            use_cache=True,
            do_sample=False,
            temperature=0.0,
            top_p=None,
            num_beams=1,
            max_new_tokens=max_new_tokens,
        )
    prompt_len = int(gen_batch["input_ids"].shape[1])
    generated = output_ids[:, prompt_len:] if output_ids.shape[1] > prompt_len else output_ids
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def bootstrap_difference_ci(
    correct_values: list[float],
    wrong_values: list[float],
    seed: int,
    resamples: int = 1000,
) -> tuple[float, float]:
    if not correct_values or not wrong_values:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    correct = np.asarray(correct_values, dtype=np.float64)
    wrong = np.asarray(wrong_values, dtype=np.float64)
    diffs = []
    for _ in range(resamples):
        c = rng.choice(correct, size=correct.size, replace=True)
        w = rng.choice(wrong, size=wrong.size, replace=True)
        diffs.append(float(np.nanmean(c) - np.nanmean(w)))
    return float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def build_category_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    output = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row["category"], row["layer"], row["representation"])].append(row)
    for (category, layer, representation), items in sorted(groups.items(), key=lambda x: (str(x[0][0]), int(x[0][1]), str(x[0][2]))):
        correct = [r for r in items if int(r["correctness"]) == 1]
        wrong = [r for r in items if int(r["correctness"]) == 0]
        cg = [float(r["geometry_gap_mean"]) for r in correct]
        wg = [float(r["geometry_gap_mean"]) for r in wrong]
        cr = [float(r["geometry_rank_acc"]) for r in correct]
        wr = [float(r["geometry_rank_acc"]) for r in wrong]
        cm = [float(r["geometry_margin_loss"]) for r in correct]
        wm = [float(r["geometry_margin_loss"]) for r in wrong]
        gap_ci = bootstrap_difference_ci(cg, wg, seed)
        rank_ci = bootstrap_difference_ci(cr, wr, seed)
        margin_ci = bootstrap_difference_ci(cm, wm, seed)
        output.append({
            "category": category,
            "layer": layer,
            "representation": representation,
            "n_correct": len(correct),
            "n_wrong": len(wrong),
            "correct_gap_mean": finite_mean(cg),
            "wrong_gap_mean": finite_mean(wg),
            "gap_difference": finite_mean(cg) - finite_mean(wg),
            "gap_difference_ci_low": gap_ci[0],
            "gap_difference_ci_high": gap_ci[1],
            "correct_rank_acc_mean": finite_mean(cr),
            "wrong_rank_acc_mean": finite_mean(wr),
            "rank_acc_difference": finite_mean(cr) - finite_mean(wr),
            "rank_acc_difference_ci_low": rank_ci[0],
            "rank_acc_difference_ci_high": rank_ci[1],
            "correct_margin_loss_mean": finite_mean(cm),
            "wrong_margin_loss_mean": finite_mean(wm),
            "margin_loss_difference": finite_mean(cm) - finite_mean(wm),
            "margin_loss_difference_ci_low": margin_ci[0],
            "margin_loss_difference_ci_high": margin_ci[1],
        })
    return output


def safe_corr(x: list[float], y: list[float], corr_type: str) -> tuple[float, float]:
    if stats is None:
        return float("nan"), float("nan")
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    if x_arr.size < 3 or len(set(y_arr.tolist())) < 2:
        return float("nan"), float("nan")
    if corr_type == "point_biserial":
        value, p_value = stats.pointbiserialr(y_arr, x_arr)
    elif corr_type == "spearman":
        value, p_value = stats.spearmanr(y_arr, x_arr)
    else:
        raise ValueError(corr_type)
    return float(value), float(p_value)


def build_correlation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row["layer"], row["representation"])].append(row)
    metrics = ("geometry_gap_mean", "geometry_rank_acc", "geometry_margin_loss")
    for (layer, representation), items in sorted(groups.items(), key=lambda x: (int(x[0][0]), str(x[0][1]))):
        correctness = [float(r["correctness"]) for r in items]
        for metric in metrics:
            values = [float(r[metric]) for r in items]
            for corr_type in ("point_biserial", "spearman"):
                corr, p_value = safe_corr(values, correctness, corr_type)
                output.append({
                    "layer": layer,
                    "representation": representation,
                    "metric": metric,
                    "correlation_type": corr_type,
                    "correlation_value": corr,
                    "p_value": p_value,
                    "n_samples": len(items),
                })
    return output


def build_bin_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row["layer"], row["representation"])].append(row)
    for (layer, representation), items in sorted(groups.items(), key=lambda x: (int(x[0][0]), str(x[0][1]))):
        ordered = sorted(items, key=lambda r: float(r["geometry_gap_mean"]))
        bins = np.array_split(np.asarray(ordered, dtype=object), 5)
        for bin_idx, bin_items in enumerate(bins):
            bin_list = list(bin_items)
            if not bin_list:
                continue
            gaps = [float(r["geometry_gap_mean"]) for r in bin_list]
            output.append({
                "layer": layer,
                "representation": representation,
                "bin_id": bin_idx + 1,
                "bin_range": ["lowest 20%", "20-40%", "40-60%", "60-80%", "highest 20%"][bin_idx],
                "n_samples": len(bin_list),
                "mean_geometry_gap": finite_mean(gaps),
                "mean_geometry_rank_acc": finite_mean([float(r["geometry_rank_acc"]) for r in bin_list]),
                "accuracy": finite_mean([float(r["correctness"]) for r in bin_list]),
            })
    return output


def save_plots(rows: list[dict[str, Any]], category_rows: list[dict[str, Any]], bin_rows: list[dict[str, Any]], output_dir: Path, min_category_samples: int) -> None:
    if plt is None:
        print("[WARN] matplotlib is not installed; skipping plots.")
        return
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    group_keys = sorted({(int(r["layer"]), str(r["representation"])) for r in rows})
    for layer, representation in group_keys:
        suffix = f"layer{layer}_{representation}"
        cat_items = [
            r for r in category_rows
            if int(r["layer"]) == layer and r["representation"] == representation and (int(r["n_correct"]) + int(r["n_wrong"])) > 0
        ]
        cats = [r["category"] for r in cat_items]
        x = np.arange(len(cats))
        if cats:
            width = 0.38
            fig, ax = plt.subplots(figsize=(max(8, len(cats) * 0.7), 4.5))
            ax.bar(x - width / 2, [float(r["correct_gap_mean"]) for r in cat_items], width, label="correct")
            ax.bar(x + width / 2, [float(r["wrong_gap_mean"]) for r in cat_items], width, label="wrong")
            ax.set_xticks(x)
            ax.set_xticklabels(cats, rotation=35, ha="right")
            ax.set_ylabel("geometry_gap_mean")
            ax.set_title(f"H{layer} {representation}: correct vs wrong gap")
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_dir / f"correct_vs_wrong_gap_by_category_{suffix}.png", dpi=180)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(max(8, len(cats) * 0.7), 4.5))
            ax.bar(x - width / 2, [float(r["correct_rank_acc_mean"]) for r in cat_items], width, label="correct")
            ax.bar(x + width / 2, [float(r["wrong_rank_acc_mean"]) for r in cat_items], width, label="wrong")
            ax.set_xticks(x)
            ax.set_xticklabels(cats, rotation=35, ha="right")
            ax.set_ylabel("geometry_rank_acc")
            ax.set_title(f"H{layer} {representation}: correct vs wrong rank accuracy")
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_dir / f"correct_vs_wrong_rankacc_by_category_{suffix}.png", dpi=180)
            plt.close(fig)

        bins = [r for r in bin_rows if int(r["layer"]) == layer and r["representation"] == representation]
        if bins:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot([r["bin_range"] for r in bins], [float(r["accuracy"]) for r in bins], marker="o")
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("accuracy")
            ax.set_xlabel("geometry_gap_mean bin")
            ax.set_title(f"H{layer} {representation}: binned accuracy")
            ax.tick_params(axis="x", rotation=25)
            fig.tight_layout()
            fig.savefig(plot_dir / f"geometry_gap_bins_accuracy_{suffix}.png", dpi=180)
            if representation == "raw":
                fig.savefig(plot_dir / f"geometry_gap_bins_accuracy_layer{layer}.png", dpi=180)
            plt.close(fig)

        layer_rows = [r for r in rows if int(r["layer"]) == layer and r["representation"] == representation]
        if layer_rows:
            categories = sorted({r["category"] for r in layer_rows})
            color_map = {category: idx for idx, category in enumerate(categories)}
            fig, ax = plt.subplots(figsize=(7, 4.5))
            xs = [float(r["geometry_gap_mean"]) for r in layer_rows]
            ys = [float(r["correctness"]) + np.random.default_rng(0).normal(0.0, 0.025) for r in layer_rows]
            colors = [color_map[r["category"]] for r in layer_rows]
            scatter = ax.scatter(xs, ys, c=colors, s=16, alpha=0.75, cmap="tab20")
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["wrong", "correct"])
            ax.set_xlabel("geometry_gap_mean")
            ax.set_ylabel("correctness")
            ax.set_title(f"H{layer} {representation}: geometry gap vs correctness")
            if categories and len(categories) <= 12:
                handles, _ = scatter.legend_elements(num=len(categories))
                ax.legend(handles, categories, fontsize=7, loc="best")
            fig.tight_layout()
            fig.savefig(plot_dir / f"geometry_gap_vs_correctness_{suffix}.png", dpi=180)
            if representation == "raw":
                fig.savefig(plot_dir / f"geometry_gap_vs_correctness_layer{layer}.png", dpi=180)
            plt.close(fig)

        for category in MAJOR_CATEGORIES:
            cat_layer_rows = [r for r in layer_rows if r["category"] == category]
            if len(cat_layer_rows) < min_category_samples:
                continue
            ordered = sorted(cat_layer_rows, key=lambda r: float(r["geometry_gap_mean"]))
            cat_bins = np.array_split(np.asarray(ordered, dtype=object), 5)
            fig, ax = plt.subplots(figsize=(6, 4))
            acc = [finite_mean([float(r["correctness"]) for r in list(bin_items)]) for bin_items in cat_bins]
            ax.plot(["lowest", "20-40", "40-60", "60-80", "highest"], acc, marker="o")
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("accuracy")
            ax.set_xlabel("geometry_gap_mean bin")
            ax.set_title(f"{category} H{layer} {representation}: binned accuracy")
            fig.tight_layout()
            safe_cat = category.lower().replace(" ", "_")
            fig.savefig(plot_dir / f"geometry_gap_bins_accuracy_{safe_cat}_{suffix}.png", dpi=180)
            plt.close(fig)


def print_skip_summary(total_requested: int, processed_sample_ids: set[str], skip_reasons: Counter[str]) -> None:
    skipped = sum(skip_reasons.values())
    print("\n=== Geometry retention analysis summary ===")
    print(f"total samples requested: {total_requested}")
    print(f"total samples processed: {len(processed_sample_ids)}")
    print(f"total skipped: {skipped}")
    if skip_reasons:
        print("skip reasons:")
        for reason, count in skip_reasons.most_common():
            print(f"  {reason}: {count}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_json", required=True)
    parser.add_argument("--prediction_json", default=None)
    parser.add_argument("--spatial_feature_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layers", required=True, type=parse_int_list)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--sample_start", type=int, default=0, help="Dataset index to start from, useful for smoke tests.")
    parser.add_argument("--categories", default=None)
    parser.add_argument("--anchors_per_frame", type=int, default=128)
    parser.add_argument("--positive_top_percent", type=float, default=10.0)
    parser.add_argument("--negative_bottom_percent", type=float, default=30.0)
    parser.add_argument("--negative_mode", choices=["bottom", "semihard"], default="bottom")
    parser.add_argument("--semihard_neg_low_percent", type=float, default=30.0)
    parser.add_argument("--semihard_neg_high_percent", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_projection_head", type=str2bool, default=False)
    parser.add_argument("--save_per_triplet", type=str2bool, default=False)
    parser.add_argument("--generate_predictions", type=str2bool, default=False)
    parser.add_argument("--generation_max_new_tokens", type=int, default=16)
    parser.add_argument("--numeric_tolerance", type=float, default=0.1)
    parser.add_argument("--min_category_samples_for_plot", type=int, default=30)
    parser.add_argument("--margin", type=float, default=0.2)

    parser.add_argument("--model_base", default=None, help="Optional base model path for LoRA checkpoints.")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--image_folder", default=".")
    parser.add_argument("--video_folder", default=".")
    parser.add_argument("--spatial_features_subdir", default="")
    parser.add_argument("--frames_upbound", type=int, default=32)
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=2)
    parser.add_argument("--pool_mode", default="bilinear", choices=["bilinear", "average", "max"])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--add_time_instruction", type=str2bool, default=None)
    parser.add_argument("--skip_spatial_tower_load", type=str2bool, default=False)
    args = parser.parse_args()

    if args.negative_mode == "semihard" and args.semihard_neg_low_percent >= args.semihard_neg_high_percent:
        raise ValueError("--semihard_neg_low_percent must be < --semihard_neg_high_percent")

    try:
        torch._dynamo.disable()
    except Exception:
        pass

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions(Path(args.prediction_json)) if args.prediction_json else {}
    if not predictions and not args.generate_predictions:
        raise ValueError("--prediction_json is required unless --generate_predictions true is set.")
    category_filter = parse_categories(args.categories)
    device = torch.device(args.device)
    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float32":
        dtype = torch.float32
    else:
        dtype = torch.float16

    from llava import conversation as conversation_lib
    from llava.train.train import DataCollatorForSupervisedDataset, LazySupervisedDataset

    if "qwen_1_5" in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates["qwen_1_5"]

    print("[INFO] Loading model and tokenizer...")
    tokenizer, model, image_processor = load_model(args, device, dtype)
    print("[INFO] Model loaded; building dataset...")
    data_args = make_data_args(args, image_processor)
    dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=args.data_json, data_args=data_args)
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    print(f"[INFO] Dataset ready: {len(dataset)} samples available.")

    rows: list[dict[str, Any]] = []
    triplet_rows: list[dict[str, Any]] = []
    skip_reasons: Counter[str] = Counter()
    processed_sample_ids: set[str] = set()
    total_requested = len(dataset) if args.num_samples is None else min(args.num_samples, len(dataset))

    if args.sample_start < 0 or args.sample_start >= len(dataset):
        raise ValueError(f"--sample_start must be in [0, {len(dataset) - 1}], got {args.sample_start}")

    for index in range(args.sample_start, len(dataset)):
        if args.num_samples is not None and len(processed_sample_ids) >= args.num_samples:
            break
        raw_item = dataset.list_data_dict[index]
        sample_id = next((sid for sid in raw_sample_ids(raw_item, index) if sid in predictions), None)
        if sample_id is None and args.generate_predictions:
            sample_id = raw_sample_ids(raw_item, index)[0]
        if sample_id is None:
            skip_reasons["prediction missing for sample"] += 1
            continue
        pred = predictions.get(sample_id)
        category = (
            pred.category if pred is not None else ""
        ) or normalize_question_type(str(first_present(raw_item, ("category", "question_type", "task", "task_type", "type"), "")))
        if not category:
            print(f"[WARN] category label missing for sample_id={sample_id}")
            category = "UNKNOWN"
        if category_filter is not None and category not in category_filter:
            skip_reasons["filtered by category"] += 1
            continue

        try:
            print(f"[INFO] Processing sample_id={sample_id} index={index}")
            item = dataset[index]
            ensure_spatial_features(item, raw_item, Path(args.spatial_feature_dir))
            if not isinstance(item.get("spatial_features"), dict) or "patch_tokens" not in item["spatial_features"]:
                raise ValueError("sample missing CUT3R spatial_features['patch_tokens']")
            batch = collator([item])
            if pred is None and args.generate_predictions:
                ground_truth = extract_ground_truth(raw_item)
                if not ground_truth:
                    raise ValueError("sample missing ground-truth answer for generated scoring")
                print(f"[INFO] Generating answer for sample_id={sample_id}")
                model_answer = generate_answer(
                    model=model,
                    tokenizer=tokenizer,
                    batch=batch,
                    raw_item=raw_item,
                    device=device,
                    dtype=dtype,
                    max_new_tokens=args.generation_max_new_tokens,
                )
                pred = PredictionRecord(
                    sample_id=sample_id,
                    question=extract_question(raw_item),
                    category=category,
                    model_answer=model_answer,
                    ground_truth=ground_truth,
                    correctness=match_generated_answer(model_answer, ground_truth, args.numeric_tolerance),
                )
                predictions[sample_id] = pred
                print(
                    f"[PRED] sample_id={sample_id} correctness={pred.correctness} "
                    f"answer={model_answer!r} gt={ground_truth!r}"
                )
            print(f"[INFO] Extracting hidden states for sample_id={sample_id}")
            raw_by_layer, projected_by_layer, _metadata = extract_hidden_states(
                model=model,
                batch=batch,
                layers=args.layers,
                device=device,
                dtype=dtype,
                use_projection_head=args.use_projection_head,
            )
            patch_tokens = item["spatial_features"]["patch_tokens"]
            if patch_tokens.dim() == 4 and patch_tokens.shape[0] == 1:
                patch_tokens = patch_tokens[0]
            expected_visual_count = sum(frame.shape[0] for frame in next(iter(raw_by_layer.values())))
            expected_teacher_count = 0
            for frame_idx, frame in enumerate(next(iter(raw_by_layer.values()))):
                expected_teacher_count += int(pool_cut3r_teacher_to_student_grid(patch_tokens[frame_idx], int(frame.shape[0]), args.pool_mode).shape[0])
            if expected_visual_count != expected_teacher_count:
                raise ValueError(f"visual token count matches failed: hidden={expected_visual_count} CUT3R={expected_teacher_count}")

            question = pred.question or extract_question(raw_item)
            shared_triplets_by_frame = None
            for layer in args.layers:
                print(f"[INFO] Scoring sample_id={sample_id} H{layer}")
                for representation, frame_states in (("raw", raw_by_layer[layer]),):
                    metrics = compute_geometry_retention(
                        frame_states,
                        patch_tokens,
                        args,
                        generator,
                        triplets_by_frame=shared_triplets_by_frame,
                    )
                    if shared_triplets_by_frame is None:
                        shared_triplets_by_frame = metrics.pop("_triplets_by_frame")
                    else:
                        metrics.pop("_triplets_by_frame", None)
                    sampled_triplets = metrics.pop("_triplets", [])
                    if metrics["teacher_gap_mean"] < 1e-4:
                        print(f"[WARN] teacher gap is very small for sample_id={sample_id} H{layer}: {metrics['teacher_gap_mean']:.6f}")
                    if not np.isfinite(metrics["geometry_rank_acc"]):
                        print(f"[WARN] rank metric is NaN for sample_id={sample_id} H{layer}")
                    for triplet in sampled_triplets:
                        triplet_rows.append({
                            "sample_id": sample_id,
                            "category": category,
                            "correctness": pred.correctness,
                            "layer": layer,
                            "representation": representation,
                            **triplet,
                        })
                    rows.append({
                        "sample_id": sample_id,
                        "category": category,
                        "question": question,
                        "correctness": pred.correctness,
                        "model_answer": pred.model_answer,
                        "ground_truth": pred.ground_truth,
                        "layer": layer,
                        "representation": representation,
                        **metrics,
                        "model_path": args.model_path,
                    })
                if args.use_projection_head and layer in projected_by_layer:
                    metrics = compute_geometry_retention(
                        projected_by_layer[layer],
                        patch_tokens,
                        args,
                        generator,
                        triplets_by_frame=shared_triplets_by_frame,
                    )
                    metrics.pop("_triplets_by_frame", None)
                    sampled_triplets = metrics.pop("_triplets", [])
                    for triplet in sampled_triplets:
                        triplet_rows.append({
                            "sample_id": sample_id,
                            "category": category,
                            "correctness": pred.correctness,
                            "layer": layer,
                            "representation": "p_geo",
                            **triplet,
                        })
                    rows.append({
                        "sample_id": sample_id,
                        "category": category,
                        "question": question,
                        "correctness": pred.correctness,
                        "model_answer": pred.model_answer,
                        "ground_truth": pred.ground_truth,
                        "layer": layer,
                        "representation": "p_geo",
                        **metrics,
                        "model_path": args.model_path,
                    })
            processed_sample_ids.add(sample_id)
            print(f"[OK] processed {len(processed_sample_ids)}/{total_requested}: sample_id={sample_id}")
        except Exception as exc:
            reason = str(exc).split("\n", 1)[0]
            if "patch_tokens" in reason or "CUT3R" in reason:
                key = "sample missing CUT3R features"
            elif "correctness" in reason:
                key = "prediction missing correctness"
            elif "Token count mismatch" in reason or "Frame count mismatch" in reason:
                key = "token/frame count mismatch"
            else:
                key = reason[:160]
            skip_reasons[key] += 1
            print(f"[WARN] skipped sample_id={sample_id} index={index}: {reason}")

    if not rows:
        print_skip_summary(total_requested, processed_sample_ids, skip_reasons)
        raise RuntimeError("No samples were processed; see warnings and skip summary above.")

    per_sample_path = output_dir / "geometry_retention_per_sample.csv"
    write_csv(per_sample_path, rows, PER_SAMPLE_COLUMNS)
    category_rows = build_category_rows(rows, args.seed)
    write_csv(output_dir / "category_correct_vs_wrong.csv", category_rows, CATEGORY_COLUMNS)
    corr_rows = build_correlation_rows(rows)
    write_csv(
        output_dir / "overall_correlations.csv",
        corr_rows,
        ["layer", "representation", "metric", "correlation_type", "correlation_value", "p_value", "n_samples"],
    )
    bin_rows = build_bin_rows(rows)
    write_csv(
        output_dir / "geometry_bins_accuracy.csv",
        bin_rows,
        ["layer", "representation", "bin_id", "bin_range", "n_samples", "mean_geometry_gap", "mean_geometry_rank_acc", "accuracy"],
    )
    if args.save_per_triplet and triplet_rows:
        write_csv(
            output_dir / "geometry_retention_triplets.csv",
            triplet_rows,
            [
                "sample_id",
                "category",
                "correctness",
                "layer",
                "representation",
                "frame_id",
                "anchor_index",
                "positive_index",
                "negative_index",
                "teacher_sim_pos",
                "teacher_sim_neg",
                "student_sim_pos",
                "student_sim_neg",
                "gap",
                "rank_correct",
                "margin_loss",
            ],
        )
    if args.generate_predictions:
        generated_rows = [
            {
                "sample_id": pred.sample_id,
                "question": pred.question,
                "category": pred.category,
                "model_answer": pred.model_answer,
                "ground_truth": pred.ground_truth,
                "correctness": pred.correctness,
            }
            for sid, pred in sorted(predictions.items(), key=lambda item: str(item[0]))
            if sid in processed_sample_ids
        ]
        write_csv(
            output_dir / "generated_predictions_with_correctness.csv",
            generated_rows,
            ["sample_id", "question", "category", "model_answer", "ground_truth", "correctness"],
        )
    save_plots(rows, category_rows, bin_rows, output_dir, args.min_category_samples_for_plot)
    print_skip_summary(total_requested, processed_sample_ids, skip_reasons)
    print(f"\nWrote {per_sample_path}")
    print(f"Wrote {output_dir / 'category_correct_vs_wrong.csv'}")
    print(f"Wrote {output_dir / 'overall_correlations.csv'}")
    print(f"Wrote {output_dir / 'geometry_bins_accuracy.csv'}")


if __name__ == "__main__":
    main()
