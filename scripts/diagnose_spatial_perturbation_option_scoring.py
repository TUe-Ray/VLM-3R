#!/usr/bin/env python
"""Inference-only spatial perturbation diagnostics for VLM-3R training data.

This script intentionally does not train or update model weights.  It scores
multiple-choice options under paired perturbations of pre-extracted CUT3R
sidecar features.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional runtime dependency
    pd = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional runtime dependency
    plt = None

from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model

try:
    from llava.model.language_model.llava_qwen import LlavaQwenConfig

    AutoConfig.register("llava_qwen", LlavaQwenConfig)
except Exception:
    pass

try:
    from llava.model.language_model.llava_llama import LlavaConfig

    AutoConfig.register("llava_llama", LlavaConfig)
except Exception:
    pass

try:
    from decord import VideoReader, cpu
except Exception:  # pragma: no cover - optional runtime dependency
    VideoReader = None
    cpu = None


LOGGER = logging.getLogger("spatial_perturb_diag")

DEFAULT_CATEGORIES = (
    "room_size",
    "relative_direction",
    "relative_distance",
    "absolute_distance",
    "route_planning",
    "appearance_order",
    "object_count",
)

DEFAULT_PERTURBATIONS = (
    "normal",
    "zero_cut3r",
    "shuffle_cut3r_within_frame",
    "replace_cut3r",
    "zero_camera",
    "shuffle_camera",
)

PATCH_KEYS = {
    "patch_tokens",
    "patch_token",
    "patch_features",
    "spatial_tokens",
    "spatial_features",
    "last_hidden_state",
    "features",
    # Coordinate consistency rule:
    # point_maps_ref/pts3d_in_other_view is CUT3R reference/anchor-frame;
    # point_maps_cam/pts3d_in_self_view is per-frame camera coordinates.
    # Diagnostics should match the coordinate source used by train/eval.
    "point_maps",
    "point_map",
    "points",
    "pts3d",
    "point_maps_ref",
    "pts3d_in_other_view",
    "point_maps_cam",
    "pts3d_in_self_view",
}
CAMERA_KEYS = {
    "camera_tokens",
    "camera_token",
    "pose_tokens",
    "pose_token",
    "view_tokens",
    "view_token",
}

QUESTION_TYPE_TO_CATEGORY = {
    "room_size_estimation": "room_size",
    "object_rel_direction_easy": "relative_direction",
    "object_rel_direction_medium": "relative_direction",
    "object_rel_direction_hard": "relative_direction",
    "object_rel_distance": "relative_distance",
    "object_abs_distance": "absolute_distance",
    "route_planning": "route_planning",
    "obj_appearance_order": "appearance_order",
    "object_counting": "object_count",
    "object_size_estimation": "object_size",
}


@dataclass
class SampleRecord:
    raw: Dict[str, Any]
    index: int
    sample_id: str
    category: str
    question: str
    prompt_text: str
    options: Dict[str, str]
    correct_option: str
    score_option_text: bool
    video_path: Optional[str]
    feature_path: Optional[Path]


@dataclass
class FeatureStats:
    num_frames: Optional[int] = None
    cut3r_tokens_per_frame: Optional[int] = None
    camera_token_count: int = 0
    patch_shapes: str = ""
    camera_shapes: str = ""
    camera_found: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_data_json", required=True)
    parser.add_argument("--spatial_feature_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_per_category", type=int, default=20)
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perturbations", default=",".join(DEFAULT_PERTURBATIONS))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--option_scoring_mode", default="teacher_forced")
    parser.add_argument("--save_logits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_debug_tensors", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--model_base", default=None, help="Optional base model for LoRA checkpoints.")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--conv_template", default="auto")
    parser.add_argument("--video_root", default=None, help="Optional root for relative video/image paths.")
    parser.add_argument("--max_frames_num", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--torch_dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--mm_resampler_type", default="spatial_pool")
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=2)
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--mm_spatial_pool_mode", default="bilinear")
    parser.add_argument("--mm_newline_position", default="grid")
    parser.add_argument("--mm_pooling_position", default="after")
    parser.add_argument("--spatial_features_subdir", default="spatial_features")
    parser.add_argument("--score_full_option_text", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def setup_logging(output_dir: Path, level: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "diagnostic.log", mode="w"),
        ],
    )


def split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("data", "samples", "annotations"):
            if isinstance(data.get(key), list):
                return data[key]
        raise ValueError(f"Could not find a sample list in {path}")
    if not isinstance(data, list):
        raise ValueError(f"Expected list or dict JSON in {path}, got {type(data)}")
    return data


def normalize_category(sample: Dict[str, Any]) -> str:
    for key in ("category", "category_label", "task_category"):
        if sample.get(key):
            return str(sample[key])
    question_type = str(sample.get("question_type") or sample.get("type") or "")
    if question_type and question_type != "<missing>":
        return QUESTION_TYPE_TO_CATEGORY.get(question_type, question_type)

    text = first_human_message(sample).lower()
    if "standing by" in text and "facing" in text and ("front-left" in text or "front-right" in text):
        return "relative_direction"
    if (
        ("which object" in text or "which of these objects" in text)
        and ("closest" in text or "farthest" in text or "nearer" in text)
    ):
        return "relative_distance"
    if "appeared first" in text or "appeared last" in text or "appearance order" in text:
        return "appearance_order"
    if "how many" in text or "number of" in text:
        return "object_count"
    if "room" in text and ("size" in text or "area" in text or "square" in text):
        return "room_size"
    if "direct distance" in text or "in meters" in text:
        return "absolute_distance"
    return question_type


def first_human_message(sample: Dict[str, Any]) -> str:
    conversations = sample.get("conversations") or sample.get("messages") or []
    for turn in conversations:
        role = turn.get("from", turn.get("role", ""))
        if role in {"human", "user"}:
            return str(turn.get("value", turn.get("content", "")))
    return ""


def assistant_answer(sample: Dict[str, Any]) -> str:
    conversations = sample.get("conversations") or sample.get("messages") or []
    for turn in conversations:
        role = turn.get("from", turn.get("role", ""))
        if role in {"gpt", "assistant"}:
            return str(turn.get("value", turn.get("content", ""))).strip()
    return ""


def parse_options(value: Any) -> Dict[str, str]:
    options: Dict[str, str] = {}
    if isinstance(value, dict):
        for key, text in value.items():
            letter = str(key).strip().upper()[:1]
            if letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                options[letter] = str(text).strip()
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            text = str(item).strip()
            match = re.match(r"^\s*([A-Z])\s*[\.\):：-]\s*(.*)$", text, flags=re.I)
            if match:
                options[match.group(1).upper()] = match.group(2).strip() or text
            else:
                options[chr(ord("A") + idx)] = text
    elif isinstance(value, str):
        for line in value.splitlines():
            match = re.match(r"^\s*([A-Z])\s*[\.\):：-]\s*(.*)$", line, flags=re.I)
            if match:
                options[match.group(1).upper()] = match.group(2).strip()
    return options


def extract_options(sample: Dict[str, Any], prompt_text: str) -> Dict[str, str]:
    for key in ("options", "choices", "answer_options"):
        options = parse_options(sample.get(key))
        if options:
            return options
    return parse_options(prompt_text)


def normalize_correct_option(sample: Dict[str, Any], options: Dict[str, str]) -> str:
    for key in ("correct_option", "answer_letter", "label", "target", "ground_truth", "answer"):
        if sample.get(key) is None:
            continue
        value = str(sample[key]).strip()
        match = re.match(r"^\s*([A-Z])(?:[\.\):：-]|\s|$)", value, flags=re.I)
        if match:
            return match.group(1).upper()
        for letter, text in options.items():
            if value.lower() == str(text).strip().lower():
                return letter
    answer = assistant_answer(sample)
    match = re.match(r"^\s*([A-Z])(?:[\.\):：-]|\s|$)", answer, flags=re.I)
    if match:
        return match.group(1).upper()
    for letter, text in options.items():
        if answer.lower() == str(text).strip().lower():
            return letter
    return ""


def synthesize_numeric_options(sample: Dict[str, Any]) -> Tuple[Dict[str, str], str]:
    answer = assistant_answer(sample)
    try:
        correct = float(re.findall(r"[-+]?\d*\.?\d+", answer)[0])
    except Exception:
        return {}, ""
    decimals = 1 if re.search(r"\d+\.\d", answer) else 0
    if correct == 0:
        values = [0.0, 0.5, 1.0, 1.5]
    else:
        offsets = [-0.25, -0.10, 0.0, 0.20]
        values = [max(0.0, correct * (1.0 + off)) for off in offsets]
    rounded = []
    for value in values:
        text = f"{value:.{decimals}f}" if decimals > 0 else str(int(round(value)))
        if text not in rounded:
            rounded.append(text)
    step = 0.5 if decimals > 0 else 1.0
    next_value = correct + step
    while len(rounded) < 4:
        text = f"{next_value:.{decimals}f}" if decimals > 0 else str(int(round(next_value)))
        if text not in rounded:
            rounded.append(text)
        next_value += step
    correct_text = f"{correct:.{decimals}f}" if decimals > 0 else str(int(round(correct)))
    if correct_text not in rounded[:4]:
        rounded[2] = correct_text
    options = {chr(ord("A") + idx): text for idx, text in enumerate(rounded[:4])}
    correct_letter = next(letter for letter, text in options.items() if text == correct_text)
    return options, correct_letter


def resolve_video_path(sample: Dict[str, Any], video_root: Optional[Path]) -> Optional[str]:
    for key in ("video", "video_path", "image", "image_path", "visual_path"):
        value = sample.get(key)
        if not value:
            continue
        if isinstance(value, list):
            value = value[0] if value else None
        if not value:
            continue
        path = Path(str(value))
        if path.is_absolute() and path.exists():
            return str(path)
        candidates = []
        if video_root is not None:
            candidates.append(video_root / path)
        candidates.append(REPO_ROOT / path)
        candidates.append(path)
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(candidates[0])
    return None


def resolve_feature_path(
    sample: Dict[str, Any],
    sample_id: str,
    video_path: Optional[str],
    spatial_feature_dir: Path,
    spatial_features_subdir: str,
) -> Optional[Path]:
    explicit_keys = (
        "spatial_feature_path",
        "spatial_features_path",
        "cut3r_feature_path",
        "feature_path",
        "spatial_key",
    )
    candidates: List[Path] = []
    for key in explicit_keys:
        value = sample.get(key)
        if not value:
            continue
        path = Path(str(value))
        if path.suffix != ".pt":
            path = path.with_suffix(".pt")
        candidates.append(path if path.is_absolute() else spatial_feature_dir / path)
        candidates.append(path if path.is_absolute() else REPO_ROOT / path)

    candidates.append(spatial_feature_dir / f"{sample_id}.pt")
    if video_path:
        video = Path(video_path)
        rel_candidates = [video.with_suffix(".pt")]
        raw_video = sample.get("video") or sample.get("video_path") or sample.get("visual_path") or ""
        if isinstance(raw_video, list):
            raw_video = raw_video[0] if raw_video else ""
        if raw_video:
            rel_candidates.append(Path(str(raw_video)).with_suffix(".pt"))
        parts = list(Path(str(raw_video)).with_suffix(".pt").parts if raw_video else video.with_suffix(".pt").parts)
        if "videos" in parts:
            replaced = list(parts)
            replaced[replaced.index("videos")] = spatial_features_subdir
            rel_candidates.append(Path(*replaced).with_suffix(".pt"))
        for rel in rel_candidates:
            candidates.append(rel if rel.is_absolute() else spatial_feature_dir / rel)
            candidates.append(spatial_feature_dir / rel.name)
            try:
                candidates.append(spatial_feature_dir / rel.relative_to(rel.anchor))
            except Exception:
                pass

    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def make_sample_records(
    samples: Sequence[Dict[str, Any]],
    categories: Sequence[str],
    spatial_feature_dir: Path,
    video_root: Optional[Path],
    spatial_features_subdir: str,
) -> List[SampleRecord]:
    allowed = set(categories)
    records: List[SampleRecord] = []
    for idx, sample in enumerate(samples):
        category = normalize_category(sample)
        if category not in allowed:
            continue
        sample_id = str(sample.get("sample_id") or sample.get("id") or sample.get("question_id") or idx)
        prompt_text = str(sample.get("prompt") or sample.get("question") or first_human_message(sample)).strip()
        question = str(sample.get("question") or prompt_text).replace(DEFAULT_IMAGE_TOKEN, "").strip()
        options = extract_options(sample, prompt_text)
        correct = normalize_correct_option(sample, options)
        score_option_text = False
        if not options:
            options, correct = synthesize_numeric_options(sample)
            score_option_text = bool(options)
        video_path = resolve_video_path(sample, video_root)
        feature_path = resolve_feature_path(sample, sample_id, video_path, spatial_feature_dir, spatial_features_subdir)
        if not options:
            LOGGER.warning("Sample %s has no parseable answer options; skipping.", sample_id)
            continue
        if not correct:
            LOGGER.warning("Sample %s has no parseable correct option; skipping.", sample_id)
            continue
        records.append(
            SampleRecord(
                raw=sample,
                index=idx,
                sample_id=sample_id,
                category=category,
                question=question,
                prompt_text=prompt_text,
                options=options,
                correct_option=correct,
                score_option_text=score_option_text,
                video_path=video_path,
                feature_path=feature_path,
            )
        )
    return records


def stratified_subset(
    records: Sequence[SampleRecord],
    categories: Sequence[str],
    num_per_category: int,
    max_samples: Optional[int],
    seed: int,
) -> List[SampleRecord]:
    rng = random.Random(seed)
    by_cat: Dict[str, List[SampleRecord]] = defaultdict(list)
    for record in records:
        by_cat[record.category].append(record)

    selected: List[SampleRecord] = []
    for category in categories:
        group = list(by_cat.get(category, []))
        rng.shuffle(group)
        if len(group) < num_per_category:
            LOGGER.warning(
                "Category %s has %d examples, fewer than requested %d; using all.",
                category,
                len(group),
                num_per_category,
            )
        selected.extend(group[:num_per_category])

    if max_samples is not None and len(selected) > max_samples:
        rng.shuffle(selected)
        selected = selected[:max_samples]
    selected.sort(key=lambda item: (item.category, item.sample_id))
    return selected


def clone_nested(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: clone_nested(val) for key, val in value.items()}
    if isinstance(value, list):
        return [clone_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_nested(item) for item in value)
    return copy.deepcopy(value)


def move_nested(value: Any, device: torch.device, dtype: Optional[torch.dtype] = None) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device)
        if dtype is not None and torch.is_floating_point(tensor):
            tensor = tensor.to(dtype=dtype)
        return tensor
    if isinstance(value, dict):
        return {key: move_nested(val, device, dtype) for key, val in value.items()}
    if isinstance(value, list):
        return [move_nested(item, device, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(move_nested(item, device, dtype) for item in value)
    return value


def iter_key_tensors(value: Any, path: Tuple[str, ...] = ()) -> Iterable[Tuple[Tuple[str, ...], torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, val in value.items():
            yield from iter_key_tensors(val, path + (str(key),))
    elif isinstance(value, (list, tuple)):
        for idx, val in enumerate(value):
            yield from iter_key_tensors(val, path + (str(idx),))


def path_key(path: Tuple[str, ...]) -> str:
    for part in reversed(path):
        if not part.isdigit():
            return part
    return path[-1] if path else ""


def is_patch_path(path: Tuple[str, ...]) -> bool:
    return path_key(path) in PATCH_KEYS


def is_camera_path(path: Tuple[str, ...]) -> bool:
    return path_key(path) in CAMERA_KEYS


def set_by_path(container: Any, path: Tuple[str, ...], new_value: Any) -> None:
    cur = container
    for part in path[:-1]:
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    last = path[-1]
    if isinstance(cur, list):
        cur[int(last)] = new_value
    else:
        cur[last] = new_value


def tensor_frame_token_shape(tensor: torch.Tensor) -> Tuple[Optional[int], Optional[int]]:
    if tensor.ndim == 0:
        return None, None
    frames = int(tensor.shape[0])
    if tensor.ndim == 1:
        return frames, 1
    if tensor.ndim == 2:
        return 1, int(tensor.shape[0])
    if tensor.ndim == 3:
        return frames, int(tensor.shape[1])
    if tensor.ndim >= 4:
        return frames, int(np.prod(tensor.shape[1:-1]))
    return None, None


def flatten_frame_tokens(tensor: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    original_shape = tuple(tensor.shape)
    if tensor.ndim < 2:
        return tensor.reshape(1, -1, 1), original_shape
    if tensor.ndim == 2:
        return tensor.unsqueeze(0), original_shape
    if tensor.ndim == 3:
        return tensor, original_shape
    return tensor.reshape(tensor.shape[0], int(np.prod(tensor.shape[1:-1])), tensor.shape[-1]), original_shape


def unflatten_frame_tokens(flat: torch.Tensor, original_shape: Tuple[int, ...]) -> torch.Tensor:
    if len(original_shape) == 2:
        return flat.squeeze(0)
    return flat.reshape(original_shape)


def zero_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(tensor)


def shuffle_within_frame(tensor: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    flat, shape = flatten_frame_tokens(tensor)
    shuffled = flat.clone()
    for frame_idx in range(flat.shape[0]):
        perm = torch.randperm(flat.shape[1], generator=generator, device=flat.device)
        shuffled[frame_idx] = flat[frame_idx, perm]
    return unflatten_frame_tokens(shuffled, shape)


def shuffle_camera_tensor(tensor: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    if tensor.numel() <= 1:
        return tensor.clone()
    shuffled = tensor.clone()
    if tensor.ndim >= 3 and tensor.shape[0] > 1:
        perm = torch.randperm(tensor.shape[0], generator=generator, device=tensor.device)
        return shuffled[perm]
    if tensor.ndim >= 2 and tensor.shape[-2] > 1:
        perm = torch.randperm(tensor.shape[-2], generator=generator, device=tensor.device)
        return shuffled.index_select(tensor.ndim - 2, perm)
    return shuffled


def collect_feature_stats(features: Any) -> FeatureStats:
    patch_shapes = []
    camera_shapes = []
    num_frames = None
    tokens_per_frame = None
    camera_token_count = 0
    for path, tensor in iter_key_tensors(features):
        frames, tokens = tensor_frame_token_shape(tensor)
        if is_patch_path(path):
            patch_shapes.append(f"{'.'.join(path)}:{tuple(tensor.shape)}")
            if num_frames is None and frames is not None:
                num_frames = frames
            if tokens_per_frame is None and tokens is not None:
                tokens_per_frame = tokens
        if is_camera_path(path):
            camera_shapes.append(f"{'.'.join(path)}:{tuple(tensor.shape)}")
            if tokens is not None:
                camera_token_count += int(tokens) * (int(frames) if frames and tensor.ndim != 2 else 1)
    return FeatureStats(
        num_frames=num_frames,
        cut3r_tokens_per_frame=tokens_per_frame,
        camera_token_count=camera_token_count,
        patch_shapes=";".join(patch_shapes),
        camera_shapes=";".join(camera_shapes),
        camera_found=bool(camera_shapes),
    )


def feature_signature(features: Any) -> Tuple[Tuple[str, Tuple[int, ...]], ...]:
    items = []
    for path, tensor in iter_key_tensors(features):
        if is_patch_path(path):
            items.append((".".join(path), tuple(tensor.shape)))
    return tuple(sorted(items))


def replacement_matches(a: Any, b: Any) -> bool:
    return feature_signature(a) == feature_signature(b)


def apply_patch_replacement(base: Any, replacement: Any) -> Tuple[Any, bool]:
    result = clone_nested(base)
    replaced = False
    rep_by_path = {".".join(path): tensor for path, tensor in iter_key_tensors(replacement) if is_patch_path(path)}
    for path, tensor in iter_key_tensors(result):
        joined = ".".join(path)
        if is_patch_path(path) and joined in rep_by_path and tuple(tensor.shape) == tuple(rep_by_path[joined].shape):
            set_by_path(result, path, rep_by_path[joined].clone().to(device=tensor.device, dtype=tensor.dtype))
            replaced = True
    return result, replaced


def apply_perturbation(
    features: Any,
    mode: str,
    generator: torch.Generator,
    replacement: Optional[Any] = None,
) -> Tuple[Any, List[str]]:
    debug: List[str] = []
    perturbed = clone_nested(features)
    if mode == "normal":
        return perturbed, ["normal: spatial features unchanged"]
    if mode == "replace_cut3r":
        if replacement is None:
            return perturbed, ["replace_cut3r: no shape-compatible replacement found"]
        replaced, ok = apply_patch_replacement(perturbed, replacement)
        return replaced, [f"replace_cut3r: patch/point tensors replaced={ok}; camera tokens unchanged"]

    touched_patch = 0
    touched_camera = 0
    for path, tensor in list(iter_key_tensors(perturbed)):
        if mode == "zero_cut3r" and is_patch_path(path):
            new_tensor = zero_tensor(tensor)
            set_by_path(perturbed, path, new_tensor)
            touched_patch += 1
            debug.append(f"{'.'.join(path)} norm {tensor.float().norm().item():.6g}->0")
        elif mode == "shuffle_cut3r_within_frame" and is_patch_path(path):
            before_norms = flatten_frame_tokens(tensor.float())[0].norm(dim=-1).sort(dim=-1).values
            new_tensor = shuffle_within_frame(tensor, generator)
            after_norms = flatten_frame_tokens(new_tensor.float())[0].norm(dim=-1).sort(dim=-1).values
            ok = torch.allclose(before_norms.cpu(), after_norms.cpu(), rtol=1e-4, atol=1e-5)
            set_by_path(perturbed, path, new_tensor)
            touched_patch += 1
            debug.append(f"{'.'.join(path)} shuffled within frame; norm_multiset_preserved={ok}")
        elif mode == "zero_camera" and is_camera_path(path):
            new_tensor = zero_tensor(tensor)
            set_by_path(perturbed, path, new_tensor)
            touched_camera += 1
            debug.append(f"{'.'.join(path)} camera norm {tensor.float().norm().item():.6g}->0")
        elif mode == "shuffle_camera" and is_camera_path(path):
            new_tensor = shuffle_camera_tensor(tensor, generator)
            set_by_path(perturbed, path, new_tensor)
            touched_camera += 1
            debug.append(f"{'.'.join(path)} camera tokens shuffled")

    if mode in {"zero_cut3r", "shuffle_cut3r_within_frame"}:
        debug.append(f"{mode}: touched_patch_tensors={touched_patch}; SigLIP/2D visual tensor unchanged by construction")
    if mode in {"zero_camera", "shuffle_camera"}:
        if touched_camera == 0:
            debug.append(f"{mode}: no separately exposed camera/view tokens found; no approximation applied")
        else:
            debug.append(f"{mode}: touched_camera_tensors={touched_camera}; patch tensors unchanged")
    return perturbed, debug


def load_sidecar(path: Path) -> Any:
    sidecar = torch.load(path, map_location="cpu")
    return sidecar


def read_video(video_path: str, max_frames_num: int) -> np.ndarray:
    path = Path(video_path)
    if path.is_dir():
        frames = []
        for image_file in sorted(path.iterdir()):
            if image_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                frames.append(np.array(Image.open(image_file).convert("RGB")))
        if not frames:
            raise FileNotFoundError(f"No image frames found in {video_path}")
        idx = np.linspace(0, len(frames) - 1, min(max_frames_num, len(frames)), dtype=int)
        return np.stack([frames[i] for i in idx], axis=0)

    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return np.array(Image.open(path).convert("RGB"))[None, ...]

    if VideoReader is None:
        raise RuntimeError("decord is not available; cannot decode video.")
    vr = VideoReader(str(path), ctx=cpu(0))
    total = len(vr)
    if total <= 0:
        raise RuntimeError(f"Video has no frames: {video_path}")
    idx = np.linspace(0, total - 1, min(max_frames_num, total), dtype=int)
    return vr.get_batch(idx.tolist()).asnumpy()


def build_prompt(prompt_text: str, conv_template: str, model: Any) -> str:
    qs = prompt_text.strip()
    if DEFAULT_IMAGE_TOKEN not in qs:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    elif getattr(model.config, "mm_use_im_start_end", False):
        image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        qs = qs.replace(DEFAULT_IMAGE_TOKEN, image_token)
    conv = conv_templates[conv_template].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def option_strings_for_sample(sample: SampleRecord, score_full_option_text: bool) -> Dict[str, str]:
    if not score_full_option_text and not sample.score_option_text:
        return {letter: letter for letter in sorted(sample.options.keys())}
    if sample.score_option_text:
        return {letter: sample.options[letter] for letter in sorted(sample.options.keys())}
    return {letter: f"{letter}. {sample.options[letter]}" for letter in sorted(sample.options.keys())}


def score_option_logprob(
    model: Any,
    tokenizer: Any,
    prompt_prefix: str,
    option_string: str,
    image_tensor: torch.Tensor,
    spatial_features: Any,
    device: torch.device,
) -> Tuple[float, int]:
    prefix_ids = tokenizer_image_token(
        prompt_prefix,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )
    full_ids = tokenizer_image_token(
        prompt_prefix + option_string,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )
    if full_ids.numel() <= prefix_ids.numel():
        raise ValueError(f"Option string produced no tokens: {option_string!r}")

    input_ids = full_ids.unsqueeze(0).to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    labels = input_ids.clone()
    labels[:, : prefix_ids.numel()] = -100

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        images=[image_tensor],
        spatial_features=[spatial_features] if spatial_features is not None else None,
        modalities=["video"],
        use_cache=False,
    )
    logits = outputs.logits.float()
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = labels[:, 1:].ne(-100)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    safe_shift_labels = shift_labels.masked_fill(~shift_mask, 0)
    gathered = log_probs.gather(-1, safe_shift_labels.unsqueeze(-1)).squeeze(-1)
    total = gathered.masked_select(shift_mask).sum().item()
    return float(total), int(shift_mask.sum().item())


def score_sample(
    model: Any,
    tokenizer: Any,
    conv_template: str,
    sample: SampleRecord,
    image_tensor: torch.Tensor,
    features: Any,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    prompt_prefix = build_prompt(sample.prompt_text, conv_template, model)
    option_strings = option_strings_for_sample(sample, args.score_full_option_text)
    option_logprobs: Dict[str, float] = {}
    option_token_counts: Dict[str, int] = {}
    with torch.no_grad():
        for letter, option_string in option_strings.items():
            logprob, token_count = score_option_logprob(
                model,
                tokenizer,
                prompt_prefix,
                option_string,
                image_tensor,
                features,
                device,
            )
            option_logprobs[letter] = logprob
            option_token_counts[letter] = token_count

    predicted = max(option_logprobs.items(), key=lambda kv: kv[1])[0]
    correct_logprob = option_logprobs.get(sample.correct_option, float("nan"))
    wrong_logprobs = [v for k, v in option_logprobs.items() if k != sample.correct_option]
    best_wrong = max(wrong_logprobs) if wrong_logprobs else float("nan")
    margin = correct_logprob - best_wrong
    row = {
        "sample_id": sample.sample_id,
        "category": sample.category,
        "question": sample.question,
        "correct_option": sample.correct_option,
        "predicted_option": predicted,
        "is_correct": predicted == sample.correct_option,
        "correct_logprob": correct_logprob,
        "best_wrong_logprob": best_wrong,
        "correct_margin": margin,
        "scored_option_strings": json.dumps(option_strings, ensure_ascii=False),
        "option_token_counts": json.dumps(option_token_counts),
    }
    for letter in ("A", "B", "C", "D"):
        row[f"option_logprob_{letter}"] = option_logprobs.get(letter, "")
    return row


def choose_replacement(
    sample: SampleRecord,
    selected: Sequence[SampleRecord],
    feature_cache: Dict[str, Any],
    rng: random.Random,
) -> Tuple[Optional[SampleRecord], Optional[Any]]:
    candidates = [record for record in selected if record.sample_id != sample.sample_id]
    rng.shuffle(candidates)
    base = feature_cache.get(sample.sample_id)
    if base is None:
        return None, None
    for candidate in candidates:
        repl = feature_cache.get(candidate.sample_id)
        if repl is None:
            continue
        if replacement_matches(base, repl):
            return candidate, repl
    return None, None


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_summaries(delta_rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    def summarize(rows: Sequence[Dict[str, Any]], category: Optional[str] = None) -> List[Dict[str, Any]]:
        by_key: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if category is not None and row["category"] != category:
                continue
            by_key[row["perturbation"]].append(row)
        out = []
        for perturbation, group in sorted(by_key.items()):
            deltas = np.array([float(r["delta_margin"]) for r in group], dtype=np.float64)
            margin_n = np.array([float(r["margin_normal"]) for r in group], dtype=np.float64)
            margin_p = np.array([float(r["margin_perturbed"]) for r in group], dtype=np.float64)
            acc_n = np.array([bool(r["is_correct_normal"]) for r in group], dtype=np.float64)
            acc_p = np.array([bool(r["is_correct_perturbed"]) for r in group], dtype=np.float64)
            flips = np.array([bool(r["answer_flipped"]) for r in group], dtype=np.float64)
            c2w = np.array([bool(r["correct_to_wrong_flip"]) for r in group], dtype=np.float64)
            w2c = np.array([bool(r["wrong_to_correct_flip"]) for r in group], dtype=np.float64)
            out.append(
                {
                    "category": category if category is not None else "ALL",
                    "perturbation": perturbation,
                    "n_samples": len(group),
                    "mean_delta_margin": float(np.mean(deltas)),
                    "median_delta_margin": float(np.median(deltas)),
                    "std_delta_margin": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                    "mean_margin_normal": float(np.mean(margin_n)),
                    "mean_margin_perturbed": float(np.mean(margin_p)),
                    "accuracy_normal": float(np.mean(acc_n)),
                    "accuracy_perturbed": float(np.mean(acc_p)),
                    "accuracy_delta": float(np.mean(acc_n) - np.mean(acc_p)),
                    "flip_rate": float(np.mean(flips)),
                    "correct_to_wrong_flip_rate": float(np.mean(c2w)),
                    "wrong_to_correct_flip_rate": float(np.mean(w2c)),
                }
            )
        return out

    categories = sorted({row["category"] for row in delta_rows})
    category_rows: List[Dict[str, Any]] = []
    for category in categories:
        category_rows.extend(summarize(delta_rows, category))
    global_rows = summarize(delta_rows, None)
    return category_rows, global_rows


def make_plots(output_dir: Path, category_summary: Sequence[Dict[str, Any]], delta_rows: Sequence[Dict[str, Any]]) -> None:
    if plt is None or pd is None:
        LOGGER.warning("matplotlib or pandas unavailable; skipping plots.")
        return
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(category_summary)
    deltas = pd.DataFrame(delta_rows)
    if summary.empty or deltas.empty:
        return

    def grouped_bar(metric: str, filename: str, ylabel: str) -> None:
        pivot = summary.pivot(index="category", columns="perturbation", values=metric).fillna(0.0)
        ax = pivot.plot(kind="bar", figsize=(max(10, len(pivot) * 1.3), 5))
        ax.set_ylabel(ylabel)
        ax.set_xlabel("category")
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(plots_dir / filename, dpi=160)
        plt.close()

    grouped_bar("mean_delta_margin", "category_delta_margin_bar.png", "mean delta margin")
    grouped_bar("flip_rate", "category_flip_rate_bar.png", "flip rate")
    grouped_bar("accuracy_delta", "category_accuracy_delta_bar.png", "accuracy normal - perturbed")

    plt.figure(figsize=(10, 5))
    for perturbation, group in deltas.groupby("perturbation"):
        plt.hist(group["delta_margin"].astype(float), bins=30, alpha=0.45, label=perturbation)
    plt.xlabel("delta margin")
    plt.ylabel("count")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "delta_margin_distribution_by_perturbation.png", dpi=160)
    plt.close()


def interpret(category_summary: Sequence[Dict[str, Any]]) -> List[str]:
    lines = ["Automatic interpretation:"]
    by_cat = defaultdict(dict)
    for row in category_summary:
        by_cat[row["category"]][row["perturbation"]] = float(row["mean_delta_margin"])
    for category, vals in sorted(by_cat.items()):
        if not vals:
            continue
        strongest = max(vals.items(), key=lambda kv: kv[1])
        notes = [f"{category}: strongest={strongest[0]} ({strongest[1]:.4f})"]
        zero = vals.get("zero_cut3r", 0.0)
        shuffle = vals.get("shuffle_cut3r_within_frame", 0.0)
        zero_cam = vals.get("zero_camera", 0.0)
        shuf_cam = vals.get("shuffle_camera", 0.0)
        if zero > 0.5:
            notes.append("zero_cut3r positive: model likely relies on CUT3R spatial patch features")
        if zero > 0.5 and shuffle < max(0.2, zero * 0.35):
            notes.append("zero_cut3r >> shuffle: effect may be global feature distribution more than topology")
        if shuffle > 0.5:
            notes.append("shuffle_cut3r positive: patch-level topology/arrangement likely matters")
        if category in {"relative_direction", "route_planning"} and max(zero_cam, shuf_cam) > 0.5:
            notes.append("camera perturbation positive: viewpoint/order tokens likely important")
        if category == "object_count" and max(zero, shuffle) > 0.5:
            notes.append("object_count spatial sensitivity: check for over-destructive or non-spatial side effects")
        lines.append(" | ".join(notes))
    return lines


def load_model_and_processor(args: argparse.Namespace):
    model_name = args.model_name or get_model_name_from_path(args.model_path)
    overwrite_config = {
        "mm_resampler_type": args.mm_resampler_type,
        "mm_spatial_pool_stride": args.mm_spatial_pool_stride,
        "mm_spatial_pool_out_channels": args.mm_spatial_pool_out_channels,
        "mm_spatial_pool_mode": args.mm_spatial_pool_mode,
        "mm_pooling_position": args.mm_pooling_position,
        "mm_newline_position": args.mm_newline_position,
        "add_faster_video": False,
        "delay_load": False,
        "zero_spatial_features": False,
    }
    cfg = AutoConfig.from_pretrained(args.model_path)
    for key in (
        "spatial_tower",
        "spatial_feature_dim",
        "spatial_tower_select_feature",
        "fusion_block",
        "geo_rope_fusion_mode",
        "geo_rope_fusion_max_depth",
        "geo_rope_fusion_group_split",
        "geo_rope_fusion_log_stats",
    ):
        if hasattr(cfg, key):
            overwrite_config[key] = getattr(cfg, key)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        device_map=args.device,
        attn_implementation=args.attn_implementation,
        torch_dtype=args.torch_dtype,
        overwrite_config=overwrite_config,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return tokenizer, model, image_processor, context_len


def auto_conv_template(args: argparse.Namespace, model: Any) -> str:
    if args.conv_template != "auto":
        return args.conv_template
    model_type = str(getattr(model.config, "model_type", "")).lower()
    if "qwen" in model_type:
        return "qwen_1_5"
    return "vicuna_v1"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    setup_logging(output_dir, args.log_level)

    if args.option_scoring_mode != "teacher_forced":
        raise ValueError("Only --option_scoring_mode teacher_forced is implemented.")
    if args.batch_size != 1:
        LOGGER.warning("This model path handles pre-extracted spatial sidecars per sample; processing sequentially despite --batch_size=%s.", args.batch_size)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)

    categories = split_csv(args.categories)
    perturbations = split_csv(args.perturbations)
    unknown = [p for p in perturbations if p not in DEFAULT_PERTURBATIONS]
    if unknown:
        raise ValueError(f"Unknown perturbations: {unknown}")

    spatial_feature_dir = Path(args.spatial_feature_dir).expanduser()
    video_root = Path(args.video_root).expanduser() if args.video_root else None

    raw_samples = load_json_or_jsonl(Path(args.train_data_json))
    records = make_sample_records(raw_samples, categories, spatial_feature_dir, video_root, args.spatial_features_subdir)
    selected = stratified_subset(records, categories, args.num_per_category, args.max_samples, args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "selected_samples.json").open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "sample_id": r.sample_id,
                    "category": r.category,
                    "index": r.index,
                    "video_path": r.video_path,
                    "feature_path": str(r.feature_path) if r.feature_path else None,
                    "correct_option": r.correct_option,
                    "score_option_text": r.score_option_text,
                }
                for r in selected
            ],
            f,
            indent=2,
        )

    LOGGER.info("Selected %d samples across categories: %s", len(selected), Counter(r.category for r in selected))

    tokenizer, model, image_processor, _ = load_model_and_processor(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    conv_template = auto_conv_template(args, model)
    if conv_template not in conv_templates:
        raise ValueError(f"Unknown conversation template {conv_template}. Available: {sorted(conv_templates)}")
    LOGGER.info("Using conv_template=%s", conv_template)

    dtype = torch.float16 if args.torch_dtype == "float16" else torch.bfloat16
    feature_cache: Dict[str, Any] = {}
    skipped_reasons: Counter[str] = Counter()
    for sample in selected:
        if sample.feature_path is None:
            skipped_reasons["missing_spatial_feature"] += 1
            LOGGER.warning("Missing spatial feature path for %s", sample.sample_id)
            continue
        try:
            feature_cache[sample.sample_id] = load_sidecar(sample.feature_path)
        except Exception as exc:
            skipped_reasons[f"feature_load_error:{type(exc).__name__}"] += 1
            LOGGER.exception("Could not load sidecar for %s from %s", sample.sample_id, sample.feature_path)

    per_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []
    saved_logits: Dict[str, Any] = {}
    processed_samples = 0
    rng = random.Random(args.seed)

    for sample_idx, sample in enumerate(selected):
        if sample.sample_id not in feature_cache:
            continue
        if not sample.video_path:
            skipped_reasons["missing_video_path"] += 1
            LOGGER.warning("Missing video path for %s", sample.sample_id)
            continue
        try:
            video = read_video(sample.video_path, args.max_frames_num)
            image_tensor = image_processor.preprocess(video, return_tensors="pt")["pixel_values"]
            image_tensor = image_tensor.to(device=device, dtype=dtype)
        except Exception as exc:
            skipped_reasons[f"video_load_error:{type(exc).__name__}"] += 1
            LOGGER.exception("Could not load visual input for %s from %s", sample.sample_id, sample.video_path)
            continue

        normal_row: Optional[Dict[str, Any]] = None
        sample_rows: Dict[str, Dict[str, Any]] = {}
        base_features_cpu = feature_cache[sample.sample_id]
        base_stats = collect_feature_stats(base_features_cpu)
        LOGGER.info(
            "[%d/%d] %s cat=%s frames=%s cut3r_tokens_per_frame=%s camera_found=%s camera_count=%s patch_shapes=%s camera_shapes=%s",
            sample_idx + 1,
            len(selected),
            sample.sample_id,
            sample.category,
            base_stats.num_frames,
            base_stats.cut3r_tokens_per_frame,
            base_stats.camera_found,
            base_stats.camera_token_count,
            base_stats.patch_shapes,
            base_stats.camera_shapes,
        )

        for pert_idx, perturbation in enumerate(perturbations):
            replacement_sample_id = ""
            replacement_cpu = None
            if perturbation == "replace_cut3r":
                repl_sample, replacement_cpu = choose_replacement(sample, selected, feature_cache, rng)
                if repl_sample is not None:
                    replacement_sample_id = repl_sample.sample_id
                else:
                    skipped_reasons["replace_no_shape_match"] += 1

            generator = torch.Generator(device="cpu")
            generator.manual_seed(args.seed + sample_idx * 1009 + pert_idx * 9176)
            perturbed_cpu, debug = apply_perturbation(base_features_cpu, perturbation, generator, replacement_cpu)
            stats = collect_feature_stats(perturbed_cpu)
            debug_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "category": sample.category,
                    "perturbation": perturbation,
                    "feature_shape_before": base_stats.patch_shapes,
                    "feature_shape_after": stats.patch_shapes,
                    "camera_found": stats.camera_found,
                    "camera_token_count": stats.camera_token_count,
                    "replacement_sample_id": replacement_sample_id,
                    "debug": " | ".join(debug),
                }
            )
            LOGGER.debug("%s %s: %s", sample.sample_id, perturbation, " | ".join(debug))

            try:
                perturbed_gpu = move_nested(perturbed_cpu, device=device, dtype=dtype)
                row = score_sample(
                    model,
                    tokenizer,
                    conv_template,
                    sample,
                    image_tensor,
                    perturbed_gpu,
                    device,
                    args,
                )
            except Exception as exc:
                skipped_reasons[f"score_error:{type(exc).__name__}"] += 1
                LOGGER.exception("Scoring failed for %s perturbation=%s", sample.sample_id, perturbation)
                continue

            row.update(
                {
                    "perturbation": perturbation,
                    "replacement_sample_id": replacement_sample_id,
                    "num_frames": stats.num_frames if stats.num_frames is not None else "",
                    "cut3r_tokens_per_frame": stats.cut3r_tokens_per_frame if stats.cut3r_tokens_per_frame is not None else "",
                    "camera_token_count": stats.camera_token_count,
                    "model_path": args.model_path,
                }
            )
            per_rows.append(row)
            sample_rows[perturbation] = row
            if perturbation == "normal":
                normal_row = row
            if args.save_logits:
                saved_logits[f"{sample.sample_id}:{perturbation}"] = {
                    key: value for key, value in row.items() if key.startswith("option_logprob_")
                }
            if args.save_debug_tensors:
                debug_dir = output_dir / "debug_tensors" / sample.sample_id
                debug_dir.mkdir(parents=True, exist_ok=True)
                torch.save(perturbed_cpu, debug_dir / f"{perturbation}.pt")

        if normal_row is None:
            skipped_reasons["missing_normal_score"] += 1
            continue
        processed_samples += 1
        for perturbation, perturbed_row in sample_rows.items():
            if perturbation == "normal":
                continue
            delta_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "category": sample.category,
                    "perturbation": perturbation,
                    "margin_normal": normal_row["correct_margin"],
                    "margin_perturbed": perturbed_row["correct_margin"],
                    "delta_margin": float(normal_row["correct_margin"]) - float(perturbed_row["correct_margin"]),
                    "predicted_normal": normal_row["predicted_option"],
                    "predicted_perturbed": perturbed_row["predicted_option"],
                    "is_correct_normal": bool(normal_row["is_correct"]),
                    "is_correct_perturbed": bool(perturbed_row["is_correct"]),
                    "answer_flipped": perturbed_row["predicted_option"] != normal_row["predicted_option"],
                    "correct_to_wrong_flip": bool(normal_row["is_correct"]) and not bool(perturbed_row["is_correct"]),
                    "wrong_to_correct_flip": (not bool(normal_row["is_correct"])) and bool(perturbed_row["is_correct"]),
                }
            )

    per_fields = [
        "sample_id",
        "category",
        "question",
        "correct_option",
        "perturbation",
        "predicted_option",
        "is_correct",
        "correct_logprob",
        "best_wrong_logprob",
        "correct_margin",
        "option_logprob_A",
        "option_logprob_B",
        "option_logprob_C",
        "option_logprob_D",
        "replacement_sample_id",
        "num_frames",
        "cut3r_tokens_per_frame",
        "camera_token_count",
        "model_path",
        "scored_option_strings",
        "option_token_counts",
    ]
    delta_fields = [
        "sample_id",
        "category",
        "perturbation",
        "margin_normal",
        "margin_perturbed",
        "delta_margin",
        "predicted_normal",
        "predicted_perturbed",
        "is_correct_normal",
        "is_correct_perturbed",
        "answer_flipped",
        "correct_to_wrong_flip",
        "wrong_to_correct_flip",
    ]
    summary_fields = [
        "category",
        "perturbation",
        "n_samples",
        "mean_delta_margin",
        "median_delta_margin",
        "std_delta_margin",
        "mean_margin_normal",
        "mean_margin_perturbed",
        "accuracy_normal",
        "accuracy_perturbed",
        "accuracy_delta",
        "flip_rate",
        "correct_to_wrong_flip_rate",
        "wrong_to_correct_flip_rate",
    ]

    category_summary, global_summary = make_summaries(delta_rows)
    write_csv(output_dir / "per_sample_perturbation_scores.csv", per_rows, per_fields)
    write_csv(output_dir / "per_sample_perturbation_deltas.csv", delta_rows, delta_fields)
    write_csv(output_dir / "category_perturbation_summary.csv", category_summary, summary_fields)
    write_csv(output_dir / "global_perturbation_summary.csv", global_summary, summary_fields)
    write_csv(
        output_dir / "perturbation_debug.csv",
        debug_rows,
        [
            "sample_id",
            "category",
            "perturbation",
            "feature_shape_before",
            "feature_shape_after",
            "camera_found",
            "camera_token_count",
            "replacement_sample_id",
            "debug",
        ],
    )
    if args.save_logits:
        torch.save(saved_logits, output_dir / "option_logprobs.pt")
    make_plots(output_dir, category_summary, delta_rows)

    print("\n=== Spatial Perturbation Diagnostic Complete ===")
    print(f"selected_samples: {len(selected)}")
    print(f"processed_successfully: {processed_samples}")
    print(f"skipped_total: {sum(skipped_reasons.values())}")
    print(f"skipped_reasons: {dict(skipped_reasons)}")
    print("\nCategory-wise mean delta_margin:")
    if category_summary:
        for row in category_summary:
            print(f"{row['category']:24s} {row['perturbation']:30s} n={row['n_samples']:4d} mean_delta={row['mean_delta_margin']:.4f}")
    print("\nStrongest perturbation effect per category:")
    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in category_summary:
        by_cat[row["category"]].append(row)
    for category, rows in sorted(by_cat.items()):
        strongest = max(rows, key=lambda r: float(r["mean_delta_margin"]))
        print(f"{category}: {strongest['perturbation']} ({float(strongest['mean_delta_margin']):.4f})")
    print()
    for line in interpret(category_summary):
        print(line)
    print(f"\noutput_dir: {output_dir}")


if __name__ == "__main__":
    main()
