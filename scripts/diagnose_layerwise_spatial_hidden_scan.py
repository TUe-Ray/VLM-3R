#!/usr/bin/env python
"""Layer-wise VLM-3R hidden-state spatial scan against CUT3R token topology.

This is an inference-only diagnostic. It never trains, never updates model
weights, and uses VLM-3R visual-token metadata rather than guessing visual
token locations.
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
except ImportError:  # pragma: no cover
    plt = None

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CATEGORIES = (
    "room_size",
    "relative_direction",
    "relative_distance",
    "absolute_distance",
    "route_planning",
    "appearance_order",
    "object_count",
)

PERTURBATIONS = (
    "zero_cut3r",
    "shuffle_cut3r_within_frame",
    "replace_cut3r",
    "zero_camera",
    "shuffle_camera",
)

PER_SAMPLE_COLUMNS = [
    "sample_id",
    "category",
    "layer",
    "model_path",
    "num_frames",
    "tokens_per_frame",
    "num_triplets",
    "geometry_gap_mean",
    "geometry_gap_median",
    "geometry_rank_acc",
    "geometry_margin_loss",
    "teacher_gap_mean",
    "student_sim_pos_mean",
    "student_sim_neg_mean",
    "roi_spearman_mean",
    "roi_spearman_median",
    "roi_pearson_mean",
    "num_roi_anchors",
    "correct_option",
    "predicted_option",
    "is_correct",
    "correct_margin",
]

SUMMARY_COLUMNS = [
    "category",
    "layer",
    "n_samples",
    "geometry_gap_mean",
    "geometry_rank_acc_mean",
    "geometry_margin_loss_mean",
    "roi_spearman_mean",
    "correct_margin_mean",
    "accuracy",
    "corr_gap_margin_pearson",
    "corr_gap_margin_spearman",
    "corr_rankacc_margin_pearson",
    "corr_rankacc_margin_spearman",
    "corr_roi_spearman_margin_pearson",
    "corr_roi_spearman_margin_spearman",
    "pointbiserial_gap_correct",
    "pointbiserial_rankacc_correct",
    "pointbiserial_roi_spearman_correct",
    "correct_gap_mean",
    "wrong_gap_mean",
    "correct_minus_wrong_gap",
    "correct_rankacc_mean",
    "wrong_rankacc_mean",
    "correct_minus_wrong_rankacc",
]


@dataclass
class BehaviorRecord:
    sample_id: str
    correct_option: str
    predicted_option: str
    is_correct: float
    correct_margin: float


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def parse_layers(value: str) -> list[str]:
    layers: list[str] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if part.lower() == "final":
            layers.append("final")
            continue
        layer = int(part)
        if layer < 1:
            raise argparse.ArgumentTypeError("Layer ids are 1-based: H1 means output after block 0")
        layers.append(str(layer))
    if not layers:
        raise argparse.ArgumentTypeError("--layers must contain at least one layer or final")
    return layers


def parse_categories(value: str | None) -> set[str] | None:
    if not value:
        return None
    cats = {canonical_category(part.strip()) for part in value.split(",") if part.strip()}
    return cats or None


def layer_sort_key(layer: str) -> tuple[int, int]:
    return (1, 10**9) if str(layer) == "final" else (0, int(layer))


def load_json_or_jsonl(path: Path) -> Any:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("selected_samples", "samples", "data", "records", "results", "predictions", "logs"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        if all(isinstance(v, dict) for v in payload.values()):
            return [v for v in payload.values()]
    raise ValueError("Expected a list of records or a dict containing samples/data/records.")


def first_present(record: dict[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def canonical_category(value: Any) -> str:
    text = str(value or "").strip()
    lowered = text.lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "room_size_estimation": "room_size",
        "room_size": "room_size",
        "room": "room_size",
        "room_size_est": "room_size",
        "object_rel_direction_easy": "relative_direction",
        "object_rel_direction_medium": "relative_direction",
        "object_rel_direction_hard": "relative_direction",
        "object_rel_direction": "relative_direction",
        "relative_direction": "relative_direction",
        "rel_dir": "relative_direction",
        "camera_movement_direction_v1": "relative_direction",
        "camera_movement_direction_v2": "relative_direction",
        "camera_movement_direction_v3": "relative_direction",
        "obj_obj_relative_pos_lr": "relative_direction",
        "obj_obj_relative_pos_ud": "relative_direction",
        "obj_obj_relative_pos_nf": "relative_direction",
        "object_rel_distance": "relative_distance",
        "relative_distance": "relative_distance",
        "rel_dist": "relative_distance",
        "camera_obj_rel_dist_v1": "relative_distance",
        "camera_obj_rel_dist_v2": "relative_distance",
        "camera_obj_rel_dist_v3": "relative_distance",
        "object_abs_distance": "absolute_distance",
        "absolute_distance": "absolute_distance",
        "abs_dist": "absolute_distance",
        "camera_obj_abs_dist": "absolute_distance",
        "camera_displacement": "absolute_distance",
        "route_planning": "route_planning",
        "route_plan": "route_planning",
        "obj_appearance_order": "appearance_order",
        "appearance_order": "appearance_order",
        "appr_order": "appearance_order",
        "object_counting": "object_count",
        "object_count": "object_count",
        "obj_count": "object_count",
    }
    return mapping.get(lowered, lowered or "unknown")


def raw_sample_ids(raw_item: dict[str, Any], index: int) -> list[str]:
    ids = []
    for key in ("sample_id", "doc_id", "id", "question_id", "uid"):
        if raw_item.get(key) is not None:
            ids.append(str(raw_item[key]))
    ids.append(str(index))
    return list(dict.fromkeys(ids))


def sample_category(raw_item: dict[str, Any]) -> str:
    return canonical_category(first_present(raw_item, ("category", "question_type", "task", "task_type", "type"), ""))


def normalize_answer_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("</s>", "").replace("<|im_end|>", "").strip()
    text = re.sub(r"^[\s\"'`]+|[\s\"'`]+$", "", text)
    return re.sub(r"\s+", " ", text)


def extract_option_letter(value: Any) -> str:
    text = normalize_answer_text(value)
    match = re.match(r"^(?:option\s*)?([a-h])(?:[\.\):,\s]|$)", text)
    return match.group(1).upper() if match else ""


def extract_question(raw_item: dict[str, Any]) -> str:
    question = first_present(raw_item, ("question", "prompt", "query"), "")
    if question:
        return str(question)
    conversations = raw_item.get("conversations")
    if isinstance(conversations, list) and conversations:
        for turn in conversations:
            if isinstance(turn, dict) and str(turn.get("from", "")).lower() in {"human", "user"}:
                return str(turn.get("value", "")).replace("<image>", "").replace("<video>", "").strip()
    return ""


def extract_correct_option(raw_item: dict[str, Any]) -> str:
    for key in ("correct_option", "answer_option", "gt_option", "label", "answer", "ground_truth", "gt"):
        if raw_item.get(key) is not None:
            letter = extract_option_letter(raw_item[key])
            if letter:
                return letter
            if key in {"correct_option", "answer_option", "gt_option", "label"}:
                text = str(raw_item[key]).strip().upper()
                if re.fullmatch(r"[A-H]", text):
                    return text
    conversations = raw_item.get("conversations")
    if isinstance(conversations, list):
        for turn in reversed(conversations):
            if isinstance(turn, dict) and str(turn.get("from", "")).lower() in {"gpt", "assistant"}:
                letter = extract_option_letter(turn.get("value", ""))
                if letter:
                    return letter
    return ""


def extract_options(raw_item: dict[str, Any]) -> dict[str, str]:
    for key in ("options", "choices", "candidates"):
        value = raw_item.get(key)
        if isinstance(value, dict):
            out = {}
            for k, v in value.items():
                letter = extract_option_letter(k) or str(k).strip().upper()
                if re.fullmatch(r"[A-H]", letter):
                    out[letter] = str(v)
            if out:
                return dict(sorted(out.items()))
        if isinstance(value, list):
            out = {}
            for idx, item in enumerate(value):
                letter = chr(ord("A") + idx)
                out[letter] = str(item)
            if out:
                return out

    question = extract_question(raw_item)
    matches = re.findall(
        r"(?:^|\n|\s)([A-H])[\.\):]\s*(.*?)(?=(?:\n|\s)[A-H][\.\):]\s*|$)",
        question,
        flags=re.DOTALL,
    )
    options = {}
    for letter, text in matches:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if cleaned:
            options[letter.upper()] = cleaned
    if options:
        return dict(sorted(options.items()))

    return {}


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


def patch_runtime_checkpoint(
    checkpoint: str,
    runtime_root: Path | None,
    siglip_path: str | None,
    cut3r_weights: str | None,
) -> str:
    if runtime_root is None and siglip_path is None and cut3r_weights is None:
        return checkpoint
    src = Path(checkpoint)
    runtime_root = runtime_root or (REPO_ROOT / ".offline_runtime")
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in src.name)
    dst = runtime_root / f"{safe_name}_layerwise_scan_runtime"
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.name == "config.json":
            continue
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(child)
    with open(src / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if siglip_path is not None:
        cfg["mm_vision_tower"] = siglip_path
        cfg["vision_tower"] = siglip_path
    if cut3r_weights is not None:
        cfg["weights_path"] = cut3r_weights
    with open(dst / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    return str(dst)


def load_model(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    model_name = args.model_name or get_model_name_from_path(args.model_path)
    model_path = patch_runtime_checkpoint(
        args.model_path,
        Path(args.runtime_root) if args.runtime_root else None,
        args.siglip_path,
        args.cut3r_weights,
    )
    skip_spatial_tower_load = bool(args.skip_spatial_tower_load)
    load_zero_spatial = bool(getattr(args, "zero_spatial_features", skip_spatial_tower_load))

    original_build_spatial_tower = None
    if skip_spatial_tower_load:
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
                    "This diagnostic expects precomputed CUT3R token sidecars."
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
            model_path,
            args.model_base,
            model_name,
            device_map=str(device),
            torch_dtype="bfloat16" if dtype == torch.bfloat16 else "float16" if dtype == torch.float16 else "float32",
            attn_implementation=args.attn_implementation,
            overwrite_config={
                "delay_load": False,
                "mm_spatial_pool_stride": args.mm_spatial_pool_stride,
                "mm_spatial_pool_mode": args.pool_mode,
                "zero_spatial_features": load_zero_spatial,
                "spatial_tower_preextracted_only": skip_spatial_tower_load,
            },
        )
    finally:
        if original_build_spatial_tower is not None:
            import llava.model.llava_arch as llava_arch
            llava_arch.build_spatial_tower = original_build_spatial_tower

    model.to(device=device, dtype=dtype)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.config.use_cache = False
    model.config.spatial_rank_loss_enable = False
    model.config.zero_spatial_features = False
    model.config.spatial_tower_preextracted_only = skip_spatial_tower_load
    return tokenizer, model, image_processor


def make_data_args(args: argparse.Namespace, image_processor: Any) -> SimpleNamespace:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    return SimpleNamespace(
        data_path=args.train_data_json,
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


def candidate_feature_paths(raw_item: dict[str, Any], spatial_feature_dir: Path) -> list[Path]:
    names = []
    for key in ("video", "image", "sample_id", "id", "question_id", "uid", "scene_name"):
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
        candidates.extend([
            spatial_feature_dir / str(dataset) / "spatial_features" / f"{scene_name}.pt",
            spatial_feature_dir / str(dataset) / f"{scene_name}.pt",
            spatial_feature_dir / "spatial_features" / str(dataset) / f"{scene_name}.pt",
        ])
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
    raise FileNotFoundError(f"No CUT3R feature file found for sample; searched under {spatial_feature_dir}")


def pool_cut3r_teacher_to_student_grid(
    teacher_tokens: torch.Tensor,
    target_tokens: int,
    pool_mode: str,
) -> torch.Tensor:
    token_count = int(teacher_tokens.shape[0])
    if token_count == target_tokens:
        return teacher_tokens
    if token_count != 729:
        raise ValueError(f"CUT3R teacher must have 729 raw latent tokens before pooling, got {token_count}")
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
    lo = max(0, min(int(math.floor(n * low_pct / 100.0)), n - 1))
    hi = max(lo + 1, min(int(math.ceil(n * high_pct / 100.0)), n))
    return candidates[order[lo:hi]]


def sample_triplet_indices_for_frame(
    teacher: torch.Tensor,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> torch.Tensor:
    teacher = F.normalize(teacher.float(), dim=-1)
    teacher_sim = teacher @ teacher.T
    num_tokens = int(teacher.shape[0])
    if num_tokens < 3:
        raise ValueError(f"Need at least 3 visual tokens per frame, got {num_tokens}")
    anchor_count = min(int(args.anchors_per_frame), num_tokens)
    anchors = torch.randperm(num_tokens, generator=generator)[:anchor_count]
    pos_k = max(1, int(math.ceil(num_tokens * args.positive_top_percent / 100.0)))
    neg_k = max(1, int(math.ceil(num_tokens * args.negative_bottom_percent / 100.0)))
    triplets = []
    for anchor_tensor in anchors:
        anchor = int(anchor_tensor.item())
        row = teacher_sim[anchor].clone()
        row[anchor] = -float("inf")
        pos_pool = torch.topk(row, k=min(pos_k, num_tokens - 1), largest=True).indices
        if args.negative_mode == "bottom":
            neg_row = teacher_sim[anchor].clone()
            neg_row[anchor] = float("inf")
            neg_pool = torch.topk(neg_row, k=min(neg_k, num_tokens - 1), largest=False).indices
        else:
            neg_pool = percentile_pool(
                teacher_sim[anchor],
                args.semihard_neg_low_percent,
                args.semihard_neg_high_percent,
                anchor,
            )
        if pos_pool.numel() == 0 or neg_pool.numel() == 0:
            continue
        pos = int(pos_pool[torch.randint(pos_pool.numel(), (1,), generator=generator)].item())
        neg = int(neg_pool[torch.randint(neg_pool.numel(), (1,), generator=generator)].item())
        triplets.append((anchor, pos, neg))
    if not triplets:
        raise ValueError("Triplet sampling produced no triplets.")
    return torch.tensor(triplets, dtype=torch.long)


def prepare_teachers_and_triplets(
    patch_tokens: torch.Tensor,
    frame_students: list[torch.Tensor],
    args: argparse.Namespace,
    generator: torch.Generator,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if patch_tokens.dim() == 4 and patch_tokens.shape[0] == 1:
        patch_tokens = patch_tokens[0]
    if patch_tokens.dim() != 3:
        raise ValueError(f"Expected CUT3R patch_tokens [frames,tokens,dim], got {tuple(patch_tokens.shape)}")
    if len(frame_students) != int(patch_tokens.shape[0]):
        raise ValueError(f"Frame count mismatch: hidden={len(frame_students)} CUT3R={patch_tokens.shape[0]}")
    teachers = []
    triplets = []
    for frame_idx, student_frame in enumerate(frame_students):
        teacher_frame = pool_cut3r_teacher_to_student_grid(
            patch_tokens[frame_idx].detach().cpu(),
            int(student_frame.shape[0]),
            args.pool_mode,
        )
        if int(student_frame.shape[0]) != int(teacher_frame.shape[0]):
            raise ValueError(
                f"Token count mismatch in frame {frame_idx}: hidden={student_frame.shape[0]} CUT3R={teacher_frame.shape[0]}"
            )
        teachers.append(teacher_frame)
        triplets.append(sample_triplet_indices_for_frame(teacher_frame, args, generator))
    return teachers, triplets


def compute_triplet_metrics(
    teachers: list[torch.Tensor],
    frame_students: list[torch.Tensor],
    triplets_by_frame: list[torch.Tensor],
    margin: float,
) -> dict[str, Any]:
    all_values = defaultdict(list)
    tokens_per_frame = []
    for frame_idx, (teacher, student, triplets) in enumerate(zip(teachers, frame_students, triplets_by_frame)):
        teacher = F.normalize(teacher.float(), dim=-1)
        student = F.normalize(student.detach().cpu().float(), dim=-1)
        if teacher.shape[0] != student.shape[0]:
            raise ValueError(f"Token count mismatch in frame {frame_idx}: hidden={student.shape[0]} CUT3R={teacher.shape[0]}")
        if torch.isnan(student).any() or torch.isnan(teacher).any():
            raise ValueError(f"NaN detected in frame {frame_idx} hidden/teacher tokens")
        teacher_sim = teacher @ teacher.T
        student_sim = student @ student.T
        tokens_per_frame.append(int(student.shape[0]))
        for anchor, pos, neg in triplets.long().tolist():
            teacher_pos = teacher_sim[anchor, pos]
            teacher_neg = teacher_sim[anchor, neg]
            student_pos = student_sim[anchor, pos]
            student_neg = student_sim[anchor, neg]
            all_values["teacher_pos"].append(teacher_pos)
            all_values["teacher_neg"].append(teacher_neg)
            all_values["student_pos"].append(student_pos)
            all_values["student_neg"].append(student_neg)
            all_values["gap"].append(student_pos - student_neg)
            all_values["rank_correct"].append((student_pos > student_neg).float())
            all_values["margin_loss"].append(F.relu(torch.tensor(margin) - student_pos + student_neg))
    merged = {key: torch.stack(value) for key, value in all_values.items()}
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
        "student_sim_pos_mean": float(merged["student_pos"].mean().item()),
        "student_sim_neg_mean": float(merged["student_neg"].mean().item()),
        "num_triplets": int(merged["gap"].numel()),
        "num_frames": len(frame_students),
        "tokens_per_frame": token_count_set[0] if len(token_count_set) == 1 else json.dumps(token_count_set),
    }


def finite_corr(x: Iterable[float], y: Iterable[float], method: str) -> float:
    x_arr = np.asarray(list(x), dtype=np.float64)
    y_arr = np.asarray(list(y), dtype=np.float64)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    if x_arr.size < 3 or np.unique(x_arr).size < 2 or np.unique(y_arr).size < 2:
        return float("nan")
    if stats is not None:
        if method == "spearman":
            return float(stats.spearmanr(x_arr, y_arr).correlation)
        return float(stats.pearsonr(x_arr, y_arr).statistic)
    if method == "spearman":
        x_arr = rankdata_fallback(x_arr)
        y_arr = rankdata_fallback(y_arr)
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def rankdata_fallback(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    sums = np.zeros_like(counts, dtype=np.float64)
    np.add.at(sums, inverse, ranks)
    return sums[inverse] / counts[inverse]


def roi_anchor_indices(tokens_per_frame: int, mode: str, grid_size: int, generator: torch.Generator) -> list[int]:
    side = int(math.isqrt(tokens_per_frame))
    if side * side != tokens_per_frame:
        raise ValueError(f"ROI Spearman requires square per-frame token grid, got {tokens_per_frame}")
    if mode == "center":
        return [(side // 2) * side + (side // 2)]
    if mode == "grid":
        if grid_size <= 1:
            coords = [(side // 2, side // 2)]
        else:
            points = np.linspace(0, side - 1, int(grid_size)).round().astype(int).tolist()
            coords = [(r, c) for r in points for c in points]
        return [r * side + c for r, c in coords]
    if mode == "random":
        count = min(tokens_per_frame, max(1, int(grid_size) * int(grid_size)))
        return torch.randperm(tokens_per_frame, generator=generator)[:count].tolist()
    raise ValueError(f"Unsupported roi_anchor_mode: {mode}")


def selected_roi_frames(num_frames: int) -> list[int]:
    preferred = [8, 16, 24]
    frames = [idx for idx in preferred if idx < num_frames]
    if frames:
        return frames
    if num_frames <= 0:
        return []
    return sorted(set([num_frames // 4, num_frames // 2, (3 * num_frames) // 4]))[:3]


def compute_roi_metrics(
    teachers: list[torch.Tensor],
    frame_students: list[torch.Tensor],
    roi_anchors_by_frame: dict[int, list[int]],
) -> dict[str, Any]:
    spearman_values = []
    pearson_values = []
    for frame_idx, anchors in roi_anchors_by_frame.items():
        teacher = F.normalize(teachers[frame_idx].float(), dim=-1)
        student = F.normalize(frame_students[frame_idx].detach().cpu().float(), dim=-1)
        if teacher.shape[0] != student.shape[0]:
            raise ValueError(f"ROI token count mismatch in frame {frame_idx}: hidden={student.shape[0]} CUT3R={teacher.shape[0]}")
        teacher_sim = teacher @ teacher.T
        student_sim = student @ student.T
        for anchor in anchors:
            t_map = teacher_sim[anchor].numpy()
            s_map = student_sim[anchor].numpy()
            spearman_values.append(finite_corr(t_map, s_map, "spearman"))
            pearson_values.append(finite_corr(t_map, s_map, "pearson"))
    spearman_arr = np.asarray(spearman_values, dtype=np.float64)
    pearson_arr = np.asarray(pearson_values, dtype=np.float64)
    return {
        "roi_spearman_mean": float(np.nanmean(spearman_arr)) if spearman_arr.size else float("nan"),
        "roi_spearman_median": float(np.nanmedian(spearman_arr)) if spearman_arr.size else float("nan"),
        "roi_pearson_mean": float(np.nanmean(pearson_arr)) if pearson_arr.size else float("nan"),
        "num_roi_anchors": int(np.isfinite(spearman_arr).sum()),
    }


def validate_metric_row(metrics: dict[str, Any]) -> None:
    roi_disabled = int(metrics.get("num_roi_anchors", 0) or 0) == 0
    for key, value in metrics.items():
        if key in {"tokens_per_frame", "num_frames", "num_triplets", "num_roi_anchors"}:
            continue
        if roi_disabled and key in {"roi_spearman_mean", "roi_spearman_median", "roi_pearson_mean"}:
            continue
        if value == "":
            continue
        if not np.isfinite(float(value)):
            raise ValueError(f"Metric {key} is not finite: {value}")


def extract_hidden_states(
    model: torch.nn.Module,
    batch: dict[str, Any],
    layers: list[str],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, list[torch.Tensor]], dict[str, Any]]:
    batch = move_to_device(batch, device, dtype)
    prepare_fn = getattr(model, "prepare_inputs_labels_for_multimodal", None)
    if prepare_fn is None:
        raise RuntimeError("Model does not expose prepare_inputs_labels_for_multimodal().")
    if "return_visual_metadata" not in inspect.signature(prepare_fn).parameters:
        raise RuntimeError(
            "prepare_inputs_labels_for_multimodal() does not support return_visual_metadata. "
            "Use a VLM-3R code/checkpoint version with visual metadata support."
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
    requested_int_layers = [int(layer) for layer in layers if layer != "final"]
    if requested_int_layers and max(requested_int_layers) > len(blocks):
        raise ValueError(f"Requested H{max(requested_int_layers)}, but model only has {len(blocks)} transformer blocks.")

    holders: dict[str, torch.Tensor] = {}
    handles = []
    for layer in requested_int_layers:
        block_idx = layer - 1

        def capture(_module, _inputs, output, layer_id=str(layer)):
            holders[layer_id] = output[0] if isinstance(output, (tuple, list)) else output

        handles.append(blocks[block_idx].register_forward_hook(capture))

    try:
        with torch.no_grad():
            outputs = model.model(
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

    if "final" in layers:
        if hasattr(outputs, "last_hidden_state"):
            holders["final"] = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)) and outputs:
            holders["final"] = outputs[0]
        else:
            raise RuntimeError("Could not access final hidden states from model.model output.")

    visual_indices = metadata["visual_token_indices"].to(device=device)
    frame_ids = metadata["visual_frame_ids"].to(device=device)
    if visual_indices.numel() == 0:
        raise RuntimeError("No valid visual tokens were returned by metadata.")
    excluded_parts = []
    for key in (
        "newline_token_indices",
        "padding_token_indices",
        "answer_token_indices",
        "text_token_indices",
        "special_token_indices",
        "camera_prefix_token_indices",
    ):
        value = metadata.get(key, torch.empty(0, dtype=torch.long, device=device))
        if isinstance(value, torch.Tensor):
            excluded_parts.append(value.to(device=device))
    excluded = torch.cat(excluded_parts) if excluded_parts else torch.empty(0, dtype=torch.long, device=device)
    if excluded.numel() > 0 and torch.isin(visual_indices, excluded).any():
        raise RuntimeError("Visual metadata overlaps excluded text/newline/padding/special/camera-prefix tokens.")

    frame_order = [int(x) for x in metadata.get("frame_order", [])]
    if not frame_order:
        frame_order = sorted({int(x) for x in frame_ids.detach().cpu().tolist()})
    by_layer: dict[str, list[torch.Tensor]] = {}
    for layer in layers:
        if layer not in holders:
            raise RuntimeError(f"H{layer} hook did not fire.")
        hidden = holders[layer]
        frames = []
        for frame_id in frame_order:
            indices = visual_indices[frame_ids == frame_id]
            frames.append(hidden[0, indices].detach().float().cpu())
        token_counts = sorted({int(frame.shape[0]) for frame in frames})
        if len(token_counts) != 1:
            raise RuntimeError(f"Variable visual token counts per frame are not supported: {token_counts}")
        by_layer[layer] = frames
    return by_layer, metadata


def load_selected_samples(path: Path | None) -> tuple[set[str], set[int]]:
    if path is None:
        return set(), set()
    rows = flatten_payload(load_json_or_jsonl(path))
    sample_ids: set[str] = set()
    indices: set[int] = set()
    for idx, row in enumerate(rows):
        sid = first_present(row, ("sample_id", "doc_id", "id", "question_id", "uid"), None)
        if sid is not None:
            sample_ids.add(str(sid))
        sample_index = first_present(row, ("sample_index", "index", "dataset_index"), None)
        if sample_index is not None:
            try:
                indices.add(int(sample_index))
            except (TypeError, ValueError):
                pass
        if sid is None and sample_index is None:
            sample_ids.add(str(idx))
    if not sample_ids and not indices:
        raise ValueError(f"No sample ids or indices found in {path}")
    return sample_ids, indices


def sample_is_selected(raw_item: dict[str, Any], index: int, selected_ids: set[str], selected_indices: set[int]) -> bool:
    if not selected_ids and not selected_indices:
        return False
    if index in selected_indices:
        return True
    return any(sid in selected_ids for sid in raw_sample_ids(raw_item, index))


def load_behavior_csv(path: Path | None) -> dict[str, BehaviorRecord]:
    if path is None:
        return {}
    records = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            sid = str(first_present(row, ("sample_id", "doc_id", "id", "question_id", "uid"), row_idx))
            correct_margin = first_present(
                row,
                ("correct_margin_normal", "normal_correct_margin", "correct_margin", "margin_normal", "margin"),
                float("nan"),
            )
            predicted_option = first_present(
                row,
                ("predicted_option_normal", "normal_predicted_option", "predicted_option", "prediction", "pred"),
                "",
            )
            is_correct = first_present(row, ("is_correct_normal", "normal_is_correct", "is_correct", "correctness", "correct"), float("nan"))
            records[sid] = BehaviorRecord(
                sample_id=sid,
                correct_option=str(first_present(row, ("correct_option", "gt_option", "answer_option"), "")),
                predicted_option=str(predicted_option),
                is_correct=float(is_correct) if str(is_correct) != "" else float("nan"),
                correct_margin=float(correct_margin) if str(correct_margin) != "" else float("nan"),
            )
    return records


def continuation_logprob(
    model: torch.nn.Module,
    tokenizer: Any,
    batch: dict[str, Any],
    prompt: str,
    continuation: str,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token

    prompt_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    cont_ids = tokenizer((" " + continuation).strip(), add_special_tokens=False, return_tensors="pt").input_ids
    if cont_ids.numel() == 0:
        return -float("inf")
    input_ids = torch.cat([prompt_ids, cont_ids], dim=1)
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = IGNORE_INDEX
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    attention_mask = input_ids.ne(pad_id).long()
    score_batch = move_to_device(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": batch.get("images"),
            "spatial_features": batch.get("spatial_features"),
            "point_maps": batch.get("point_maps"),
            "modalities": batch.get("modalities"),
            "image_sizes": batch.get("image_sizes"),
        },
        device,
        dtype,
    )
    with torch.no_grad():
        outputs = model(
            input_ids=score_batch["input_ids"],
            attention_mask=score_batch["attention_mask"],
            labels=score_batch["labels"],
            images=score_batch.get("images"),
            spatial_features=score_batch.get("spatial_features"),
            point_maps=score_batch.get("point_maps"),
            modalities=score_batch.get("modalities"),
            image_sizes=score_batch.get("image_sizes"),
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    if outputs.loss is None:
        return -float("inf")
    return float(-outputs.loss.detach().float().item())


def compute_behavior_fallback(
    model: torch.nn.Module,
    tokenizer: Any,
    batch: dict[str, Any],
    raw_item: dict[str, Any],
    sample_id: str,
    device: torch.device,
    dtype: torch.dtype,
) -> BehaviorRecord:
    from llava import conversation as conversation_lib
    from llava.constants import DEFAULT_IMAGE_TOKEN

    question = extract_question(raw_item)
    if DEFAULT_IMAGE_TOKEN not in question and ("video" in raw_item or "image" in raw_item):
        question = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv = conversation_lib.default_conversation.copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    options = extract_options(raw_item)
    scores = {
        letter: continuation_logprob(model, tokenizer, batch, prompt, letter, device, dtype)
        for letter in options.keys()
    }
    correct_option = extract_correct_option(raw_item)
    predicted_option = max(scores.items(), key=lambda item: item[1])[0] if scores else ""
    correct_score = scores.get(correct_option, float("nan"))
    distractor_scores = [score for letter, score in scores.items() if letter != correct_option]
    best_distractor = max(distractor_scores) if distractor_scores else float("nan")
    correct_margin = float(correct_score - best_distractor) if np.isfinite(correct_score) and np.isfinite(best_distractor) else float("nan")
    return BehaviorRecord(
        sample_id=sample_id,
        correct_option=correct_option,
        predicted_option=predicted_option,
        is_correct=float(predicted_option == correct_option) if correct_option else float("nan"),
        correct_margin=correct_margin,
    )


def load_perturbation_deltas(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    output: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            sid = str(first_present(row, ("sample_id", "doc_id", "id", "question_id", "uid"), row_idx))
            perturbation = str(first_present(row, ("perturbation", "mode", "ablation"), ""))
            if perturbation:
                delta = first_present(row, ("delta_margin", "correct_margin_delta", "margin_delta", "delta_correct_margin"), "")
                if str(delta) != "":
                    output[sid][perturbation] = float(delta)
                continue
            for pert in PERTURBATIONS:
                for col in (f"{pert}_delta_margin", f"delta_margin_{pert}", f"{pert}_correct_margin_delta"):
                    if col in row and str(row[col]) != "":
                        output[sid][pert] = float(row[col])
                        break
    return dict(output)


def mean_finite(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def build_summary_rows(rows: list[dict[str, Any]], group_by_category: bool) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        key = (row["category"] if group_by_category else "ALL", row["layer"])
        groups[key].append(row)
    out = []
    for (category, layer), items in sorted(groups.items(), key=lambda item: (str(item[0][0]), layer_sort_key(str(item[0][1])))):
        correct = [r for r in items if np.isfinite(float(r.get("is_correct", float("nan")))) and int(float(r["is_correct"])) == 1]
        wrong = [r for r in items if np.isfinite(float(r.get("is_correct", float("nan")))) and int(float(r["is_correct"])) == 0]
        gap = [float(r["geometry_gap_mean"]) for r in items]
        rank = [float(r["geometry_rank_acc"]) for r in items]
        roi = [float(r.get("roi_spearman_mean", float("nan"))) for r in items]
        margins = [float(r.get("correct_margin", float("nan"))) for r in items]
        correctness = [float(r.get("is_correct", float("nan"))) for r in items]
        row = {
            "category": category,
            "layer": layer,
            "n_samples": len(items),
            "geometry_gap_mean": mean_finite(gap),
            "geometry_rank_acc_mean": mean_finite(rank),
            "geometry_margin_loss_mean": mean_finite([float(r["geometry_margin_loss"]) for r in items]),
            "roi_spearman_mean": mean_finite(roi),
            "correct_margin_mean": mean_finite(margins),
            "accuracy": mean_finite(correctness),
            "corr_gap_margin_pearson": finite_corr(gap, margins, "pearson"),
            "corr_gap_margin_spearman": finite_corr(gap, margins, "spearman"),
            "corr_rankacc_margin_pearson": finite_corr(rank, margins, "pearson"),
            "corr_rankacc_margin_spearman": finite_corr(rank, margins, "spearman"),
            "corr_roi_spearman_margin_pearson": finite_corr(roi, margins, "pearson"),
            "corr_roi_spearman_margin_spearman": finite_corr(roi, margins, "spearman"),
            "pointbiserial_gap_correct": finite_corr(gap, correctness, "pearson"),
            "pointbiserial_rankacc_correct": finite_corr(rank, correctness, "pearson"),
            "pointbiserial_roi_spearman_correct": finite_corr(roi, correctness, "pearson"),
            "correct_gap_mean": mean_finite([float(r["geometry_gap_mean"]) for r in correct]),
            "wrong_gap_mean": mean_finite([float(r["geometry_gap_mean"]) for r in wrong]),
            "correct_rankacc_mean": mean_finite([float(r["geometry_rank_acc"]) for r in correct]),
            "wrong_rankacc_mean": mean_finite([float(r["geometry_rank_acc"]) for r in wrong]),
        }
        row["correct_minus_wrong_gap"] = float(row["correct_gap_mean"] - row["wrong_gap_mean"])
        row["correct_minus_wrong_rankacc"] = float(row["correct_rankacc_mean"] - row["wrong_rankacc_mean"])
        out.append(row)
    return out


def build_perturbation_rows(rows: list[dict[str, Any]], deltas: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["category"], row["layer"])].append(row)
    metrics = ("geometry_gap_mean", "geometry_rank_acc", "geometry_margin_loss", "roi_spearman_mean")
    out = []
    for (category, layer), items in sorted(groups.items(), key=lambda item: (str(item[0][0]), layer_sort_key(str(item[0][1])))):
        for perturbation in PERTURBATIONS:
            delta_items = [r for r in items if r["sample_id"] in deltas and perturbation in deltas[r["sample_id"]]]
            if not delta_items:
                continue
            delta_values = [float(deltas[r["sample_id"]][perturbation]) for r in delta_items]
            for metric in metrics:
                metric_values = [float(r.get(metric, float("nan"))) for r in delta_items]
                out.append({
                    "category": category,
                    "layer": layer,
                    "perturbation": perturbation,
                    "metric": metric,
                    "corr_with_delta_margin_pearson": finite_corr(metric_values, delta_values, "pearson"),
                    "corr_with_delta_margin_spearman": finite_corr(metric_values, delta_values, "spearman"),
                    "n_samples": len(delta_items),
                })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def save_line_plot(
    rows: list[dict[str, Any]],
    output_path: Path,
    y_column: str,
    title: str,
    ylabel: str,
    group_column: str = "category",
) -> None:
    if plt is None or not rows:
        return
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[group_column]].append(row)
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = sorted({str(r["layer"]) for r in rows}, key=layer_sort_key)
    x = np.arange(len(labels))
    for group, items in sorted(grouped.items()):
        by_layer = {str(r["layer"]): r for r in items}
        y = [float(by_layer[label].get(y_column, float("nan"))) if label in by_layer else float("nan") for label in labels]
        ax.plot(x, y, marker="o", label=str(group))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("layer")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_plots(category_rows: list[dict[str, Any]], global_rows: list[dict[str, Any]], output_dir: Path) -> None:
    if plt is None:
        print("[WARN] matplotlib is not installed; skipping plots.")
        return
    plot_dir = output_dir / "plots"
    save_line_plot(
        category_rows,
        plot_dir / "layer_curve_geometry_rank_acc_by_category.png",
        "geometry_rank_acc_mean",
        "Geometry rank accuracy by category",
        "geometry_rank_acc",
    )
    save_line_plot(
        category_rows,
        plot_dir / "layer_curve_geometry_gap_by_category.png",
        "geometry_gap_mean",
        "Geometry gap by category",
        "geometry_gap_mean",
    )
    save_line_plot(
        category_rows,
        plot_dir / "layer_curve_roi_spearman_by_category.png",
        "roi_spearman_mean",
        "ROI Spearman by category",
        "roi_spearman_mean",
    )
    save_line_plot(
        category_rows,
        plot_dir / "layer_curve_corr_gap_with_margin_by_category.png",
        "corr_gap_margin_spearman",
        "Spearman corr: geometry gap vs correct margin",
        "corr_gap_margin_spearman",
    )
    save_line_plot(
        category_rows,
        plot_dir / "layer_curve_corr_rankacc_with_margin_by_category.png",
        "corr_rankacc_margin_spearman",
        "Spearman corr: rank accuracy vs correct margin",
        "corr_rankacc_margin_spearman",
    )

    labels = [str(r["layer"]) for r in sorted(global_rows, key=lambda r: layer_sort_key(str(r["layer"])))]
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(8, 5))
    sorted_rows = sorted(global_rows, key=lambda r: layer_sort_key(str(r["layer"])))
    ax1.plot(x, [float(r["geometry_rank_acc_mean"]) for r in sorted_rows], marker="o", label="rank_acc")
    ax1.plot(x, [float(r["geometry_gap_mean"]) for r in sorted_rows], marker="o", label="gap")
    ax1.plot(x, [float(r["roi_spearman_mean"]) for r in sorted_rows], marker="o", label="roi_spearman")
    ax1.plot(x, [float(r["corr_gap_margin_spearman"]) for r in sorted_rows], marker="o", label="corr_gap_margin")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("layer")
    ax1.set_title("Global layer curves")
    ax1.legend(fontsize=8, loc="best")
    fig.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / "global_layer_curves.png", dpi=180)
    plt.close(fig)


def print_diagnostic_summary(global_rows: list[dict[str, Any]], category_rows: list[dict[str, Any]]) -> None:
    print("\n=== Layer-wise diagnostic summary ===")
    if not global_rows:
        print("No global summary rows were produced.")
        return
    best_gap_corr = max(global_rows, key=lambda r: np.nan_to_num(float(r["corr_gap_margin_spearman"]), nan=-999.0))
    best_rank_corr = max(global_rows, key=lambda r: np.nan_to_num(float(r["corr_rankacc_margin_spearman"]), nan=-999.0))
    best_roi = max(global_rows, key=lambda r: np.nan_to_num(float(r["roi_spearman_mean"]), nan=-999.0))
    print(
        f"Best global Spearman corr(gap, correct_margin): layer {best_gap_corr['layer']} "
        f"= {float(best_gap_corr['corr_gap_margin_spearman']):.4f}"
    )
    print(
        f"Best global Spearman corr(rank_acc, correct_margin): layer {best_rank_corr['layer']} "
        f"= {float(best_rank_corr['corr_rankacc_margin_spearman']):.4f}"
    )
    print(f"Highest global CUT3R ROI Spearman: layer {best_roi['layer']} = {float(best_roi['roi_spearman_mean']):.4f}")
    if str(best_roi["layer"]) != str(best_gap_corr["layer"]):
        print(
            "ROI topology agreement and answer-margin correlation peak at different layers; "
            "CUT3R-like topology may not by itself identify task-relevant spatial behavior."
        )
    late_layers = {"16", "24", "final"}
    if str(best_gap_corr["layer"]) in late_layers or str(best_rank_corr["layer"]) in late_layers:
        print("The strongest answer-behavior correlation is mid/late, which is consistent with transformed task-useful spatial signals.")
    early = [r for r in global_rows if str(r["layer"]) == "1"]
    if early and float(early[0]["roi_spearman_mean"]) >= float(best_roi["roi_spearman_mean"]) - 1e-6:
        print("H1 has the strongest ROI topology agreement; check whether its answer-margin correlations are weaker before treating it as task-useful.")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_data_json", required=True)
    parser.add_argument("--spatial_feature_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--selected_samples_json", default=None)
    parser.add_argument("--num_per_category", type=int, default=20)
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("1,8,16,24,final"))
    parser.add_argument("--anchors_per_frame", type=int, default=64)
    parser.add_argument("--positive_top_percent", type=float, default=10.0)
    parser.add_argument("--negative_bottom_percent", type=float, default=30.0)
    parser.add_argument("--negative_mode", choices=["bottom", "semihard"], default="bottom")
    parser.add_argument("--semihard_neg_low_percent", type=float, default=30.0)
    parser.add_argument("--semihard_neg_high_percent", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compute_roi_spearman", type=str2bool, default=True)
    parser.add_argument("--roi_anchor_mode", choices=["grid", "random", "center"], default="grid")
    parser.add_argument("--roi_grid_size", type=int, default=3)
    parser.add_argument("--option_margin_csv", default=None)
    parser.add_argument("--perturbation_delta_csv", default=None)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--model_base", default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--image_folder", default=".")
    parser.add_argument("--video_folder", default=".")
    parser.add_argument("--spatial_features_subdir", default="")
    parser.add_argument("--frames_upbound", type=int, default=32)
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=2)
    parser.add_argument("--pool_mode", choices=["bilinear", "average", "max"], default="bilinear")
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--add_time_instruction", type=str2bool, default=None)
    parser.add_argument("--skip_spatial_tower_load", type=str2bool, default=False)
    parser.add_argument("--runtime_root", default=str(REPO_ROOT / ".offline_runtime"))
    parser.add_argument("--siglip_path", default=None)
    parser.add_argument("--cut3r_weights", default=None)
    args = parser.parse_args()

    if args.batch_size != 1:
        print("[WARN] --batch_size > 1 was requested, but this diagnostic uses batch_size=1 for metadata-safe hooks.")
    if args.negative_mode == "semihard" and args.semihard_neg_low_percent >= args.semihard_neg_high_percent:
        raise ValueError("--semihard_neg_low_percent must be < --semihard_neg_high_percent")

    os.environ["TORCH_COMPILE_DISABLE"] = "1"
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
    category_filter = parse_categories(args.categories)
    selected_ids, selected_indices = load_selected_samples(Path(args.selected_samples_json) if args.selected_samples_json else None)
    behavior_records = load_behavior_csv(Path(args.option_margin_csv) if args.option_margin_csv else None)
    perturbation_deltas = load_perturbation_deltas(Path(args.perturbation_delta_csv) if args.perturbation_delta_csv else None)

    device = torch.device(args.device)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    from llava import conversation as conversation_lib
    from llava.train.train import DataCollatorForSupervisedDataset, LazySupervisedDataset

    if "qwen_1_5" in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates["qwen_1_5"]

    print("[INFO] Loading model...")
    tokenizer, model, image_processor = load_model(args, device, dtype)
    print("[INFO] Building dataset...")
    data_args = make_data_args(args, image_processor)
    dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=args.train_data_json, data_args=data_args)
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    print(f"[INFO] Dataset ready: {len(dataset)} samples.")

    rows: list[dict[str, Any]] = []
    skip_reasons: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()
    processed_ids: set[str] = set()

    for index in range(len(dataset)):
        raw_item = dataset.list_data_dict[index]
        sample_ids = raw_sample_ids(raw_item, index)
        sample_id = next((sid for sid in sample_ids if sid in behavior_records), sample_ids[0])
        category = sample_category(raw_item)
        if category_filter is not None and category not in category_filter:
            skip_reasons["filtered by category"] += 1
            continue
        if selected_ids or selected_indices:
            if not sample_is_selected(raw_item, index, selected_ids, selected_indices):
                skip_reasons["not in selected_samples_json"] += 1
                continue
        else:
            if selected_counts[category] >= args.num_per_category:
                skip_reasons["category quota reached"] += 1
                continue
        if sample_id in processed_ids:
            skip_reasons["duplicate sample_id"] += 1
            continue

        try:
            print(f"[INFO] Processing sample_id={sample_id} category={category} index={index}")
            item = dataset[index]
            ensure_spatial_features(item, raw_item, Path(args.spatial_feature_dir))
            if not isinstance(item.get("spatial_features"), dict) or "patch_tokens" not in item["spatial_features"]:
                raise ValueError("sample missing CUT3R spatial_features['patch_tokens']")
            batch = collator([item])
            behavior = behavior_records.get(sample_id)
            if behavior is None:
                behavior = compute_behavior_fallback(model, tokenizer, batch, raw_item, sample_id, device, dtype)
                behavior_records[sample_id] = behavior

            by_layer, metadata = extract_hidden_states(model, batch, args.layers, device, dtype)
            first_layer_frames = by_layer[args.layers[0]]
            patch_tokens = item["spatial_features"]["patch_tokens"]
            teachers, triplets_by_frame = prepare_teachers_and_triplets(patch_tokens, first_layer_frames, args, generator)

            frame_count = len(first_layer_frames)
            tokens_per_frame_set = sorted({int(frame.shape[0]) for frame in first_layer_frames})
            if len(tokens_per_frame_set) != 1:
                raise ValueError(f"tokens per frame are not consistent: {tokens_per_frame_set}")
            total_visual = sum(int(frame.shape[0]) for frame in first_layer_frames)
            total_teacher = sum(int(frame.shape[0]) for frame in teachers)
            if total_visual != total_teacher:
                raise ValueError(f"hidden/CUT3R token mismatch: hidden={total_visual} CUT3R={total_teacher}")
            if "visual_grid_shapes" in metadata:
                print(
                    f"[CHECK] sample_id={sample_id} frames={frame_count} "
                    f"tokens_per_frame={tokens_per_frame_set[0]} visual_tokens={total_visual} "
                    f"grid={metadata.get('visual_grid_shapes')[:1]}"
                )
            roi_anchors_by_frame: dict[int, list[int]] = {}
            if args.compute_roi_spearman:
                for frame_idx in selected_roi_frames(frame_count):
                    roi_anchors_by_frame[frame_idx] = roi_anchor_indices(
                        tokens_per_frame_set[0],
                        args.roi_anchor_mode,
                        args.roi_grid_size,
                        generator,
                    )

            for layer in args.layers:
                triplet_metrics = compute_triplet_metrics(teachers, by_layer[layer], triplets_by_frame, args.margin)
                roi_metrics = (
                    compute_roi_metrics(teachers, by_layer[layer], roi_anchors_by_frame)
                    if args.compute_roi_spearman
                    else {
                        "roi_spearman_mean": float("nan"),
                        "roi_spearman_median": float("nan"),
                        "roi_pearson_mean": float("nan"),
                        "num_roi_anchors": 0,
                    }
                )
                combined_metrics = {**triplet_metrics, **roi_metrics}
                validate_metric_row(combined_metrics)
                rows.append({
                    "sample_id": sample_id,
                    "category": category,
                    "layer": layer,
                    "model_path": args.model_path,
                    **combined_metrics,
                    "correct_option": behavior.correct_option,
                    "predicted_option": behavior.predicted_option,
                    "is_correct": behavior.is_correct,
                    "correct_margin": behavior.correct_margin,
                })
                print(
                    f"[OK] sample_id={sample_id} layer={layer} "
                    f"rank_acc={combined_metrics['geometry_rank_acc']:.4f} "
                    f"gap={combined_metrics['geometry_gap_mean']:.4f} "
                    f"roi={combined_metrics['roi_spearman_mean']:.4f}"
                )
            processed_ids.add(sample_id)
            selected_counts[category] += 1
        except Exception as exc:
            reason = str(exc).split("\n", 1)[0]
            if "CUT3R" in reason or "patch_tokens" in reason:
                key = "sample missing CUT3R features"
            elif "metadata" in reason or "visual token" in reason:
                key = "visual metadata/token issue"
            elif "Token count mismatch" in reason or "Frame count mismatch" in reason or "mismatch" in reason:
                key = "token/frame count mismatch"
            else:
                key = reason[:180]
            skip_reasons[key] += 1
            print(f"[WARN] skipped sample_id={sample_id} index={index}: {reason}")

        if not (selected_ids or selected_indices) and category_filter is not None:
            if all(selected_counts[category] >= args.num_per_category for category in category_filter):
                break

    if not rows:
        print("[ERROR] No samples processed.")
        for reason, count in skip_reasons.most_common():
            print(f"  {reason}: {count}")
        raise RuntimeError("No samples were processed; see skip reasons above.")

    per_sample_path = output_dir / "per_sample_layerwise_spatial_metrics.csv"
    write_csv(per_sample_path, rows, PER_SAMPLE_COLUMNS)
    category_rows = build_summary_rows(rows, group_by_category=True)
    global_rows = build_summary_rows(rows, group_by_category=False)
    write_csv(output_dir / "category_layer_summary.csv", category_rows, SUMMARY_COLUMNS)
    write_csv(output_dir / "global_layer_summary.csv", global_rows, SUMMARY_COLUMNS)
    if perturbation_deltas:
        pert_rows = build_perturbation_rows(rows, perturbation_deltas)
        write_csv(
            output_dir / "layer_perturbation_correlation.csv",
            pert_rows,
            [
                "category",
                "layer",
                "perturbation",
                "metric",
                "corr_with_delta_margin_pearson",
                "corr_with_delta_margin_spearman",
                "n_samples",
            ],
        )
    save_plots(category_rows, global_rows, output_dir)

    print("\n=== Skip summary ===")
    print(f"processed samples: {len(processed_ids)}")
    for reason, count in skip_reasons.most_common():
        print(f"  {reason}: {count}")
    print(f"\nWrote {per_sample_path}")
    print(f"Wrote {output_dir / 'category_layer_summary.csv'}")
    print(f"Wrote {output_dir / 'global_layer_summary.csv'}")
    if perturbation_deltas:
        print(f"Wrote {output_dir / 'layer_perturbation_correlation.csv'}")
    print_diagnostic_summary(global_rows, category_rows)


if __name__ == "__main__":
    main()
