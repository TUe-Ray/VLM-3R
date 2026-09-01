"""Small cached-EoMT consumers used only by post-SFT probing.

The historical checkpoints consumed sigmoid mask probabilities resized to the
visual-token (14x14) or CUT3R patch-token (27x27) grid plus class logits.  The
frozen EoMT network is deliberately not loaded in the VLM process: the exact
consumer tensors live in ``eomt_consumer_grid_v1``.
"""

from __future__ import annotations

from typing import Any

import torch


NUM_FRAMES = 32
NUM_QUERIES = 200
NUM_CLASSES_WITH_NO_OBJECT = 134
THING_CLASS_COUNT = 80
SELECTIVE_GATE_ACTIVE_EPSILON = 1e-6


def configure_selective_kv_gate(config: Any, *, enabled: bool) -> dict[str, Any]:
    """Apply the checkpoint-equivalent executable selective-K/V configuration.

    The current runtime has no grounded-word input socket.  Word-match options
    remain enabled to faithfully represent the checkpoint configuration, while
    the gate records that they are a no-op for this execution path.
    """
    settings = {
        "mm_eomt_selective_3d_enable": bool(enabled),
        "mm_eomt_selective_3d_gate_type": "soft",
        "mm_eomt_selective_3d_selector_mode": "confidence",
        "mm_eomt_selective_3d_score_threshold": 0.8,
        "mm_eomt_selective_3d_topk": -1,
        "mm_eomt_selective_3d_class_type": "things",
        "mm_eomt_selective_3d_merge_mode": "soft_max_union",
        "mm_eomt_selective_3d_word_match_enable": True,
        "mm_eomt_selective_3d_empty_fallback": "zero_3d",
        "mm_eomt_word_match_source": "visible_grounded_words",
        "mm_eomt_word_match_mode": "hybrid_safe",
        "mm_eomt_word_match_no_match": "keep_masks",
        "mm_eomt_word_match_similarity_threshold": 0.86,
    }
    for name, value in settings.items():
        setattr(config, name, value)
    return settings


def _flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def validate_cached_outputs(payload: Any, *, consumer: str) -> dict[str, torch.Tensor]:
    """Fail closed on the compact cache schema needed by one VLM forward."""
    if not isinstance(payload, dict):
        raise TypeError("EoMT cached outputs must be a dict")
    classes = payload.get("class_logits")
    masks = payload.get("soft_masks")
    expected_hw = (14, 14) if consumer == "object" else (27, 27)
    if not isinstance(classes, torch.Tensor) or tuple(classes.shape) != (
        NUM_FRAMES,
        NUM_QUERIES,
        NUM_CLASSES_WITH_NO_OBJECT,
    ):
        raise ValueError(f"Invalid EoMT class_logits shape: {getattr(classes, 'shape', None)}")
    if not isinstance(masks, torch.Tensor) or tuple(masks.shape) != (
        NUM_FRAMES,
        NUM_QUERIES,
        *expected_hw,
    ):
        raise ValueError(f"Invalid EoMT {consumer} mask shape: {getattr(masks, 'shape', None)}")
    if classes.dtype != torch.float32 or masks.dtype != torch.float32:
        raise TypeError("EoMT consumer-grid cache must retain FP32 tensors")
    if not torch.isfinite(classes).all() or not torch.isfinite(masks).all():
        raise RuntimeError("EoMT cached tensors contain non-finite values")
    return {"class_logits": classes, "soft_masks": masks}


def _foreground_scores(class_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(class_logits.float(), dim=-1)[..., :-1]
    return probabilities.max(dim=-1)


def _selected_queries(class_logits: torch.Tensor, *, threshold: float, things_only: bool) -> list[list[int]]:
    scores, class_ids = _foreground_scores(class_logits)
    selected: list[list[int]] = []
    for frame in range(int(class_logits.shape[0])):
        valid = scores[frame] >= float(threshold)
        if things_only:
            valid &= class_ids[frame] < THING_CLASS_COUNT
        indices = torch.where(valid)[0]
        if indices.numel():
            order = torch.argsort(scores[frame, indices], descending=True)
            indices = indices[order]
        selected.append([int(index) for index in indices.detach().cpu().tolist()])
    return selected


def gate_cut3r_patch_tokens(
    patch_tokens: torch.Tensor,
    payload: Any,
    config: Any,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Apply the historical soft, things-only, zero-3D selective gate."""
    cached = validate_cached_outputs(payload, consumer="selective")
    if patch_tokens.ndim != 3 or tuple(patch_tokens.shape[:2]) != (NUM_FRAMES, 729):
        raise ValueError(f"Selective EoMT expects CUT3R [32,729,D], got {tuple(patch_tokens.shape)}")
    if not _flag(getattr(config, "mm_eomt_selective_3d_enable", False)):
        return patch_tokens, []
    threshold = float(getattr(config, "mm_eomt_selective_3d_score_threshold", 0.8))
    masks = cached["soft_masks"].to(device=patch_tokens.device, dtype=patch_tokens.dtype)
    classes = cached["class_logits"].to(device=patch_tokens.device)
    selected = _selected_queries(classes, threshold=threshold, things_only=True)
    result, debug = [], []
    word_match_enabled = _flag(getattr(config, "mm_eomt_selective_3d_word_match_enable", False))
    for frame, query_ids in enumerate(selected):
        current = patch_tokens[frame : frame + 1]
        if not query_ids:
            result.append(torch.zeros_like(current))
            debug.append(
                {
                    "frame_index": frame,
                    "selected_queries": 0,
                    "fallback": "zero_3d",
                    # Keep the effective zero-3D gate observable for
                    # forward-only diagnostics without changing the gate.
                    "gate_mean": 0.0,
                    "active_patch_fraction": 0.0,
                    "word_match_enabled": word_match_enabled,
                    "no_words_available": True,
                    # This is metadata from the sole executable selector: it
                    # receives no word payload, so it never enters a word
                    # filtering path or alters the effective mask for words.
                    "word_match_applied": False,
                    "word_match_effective_noop": word_match_enabled,
                }
            )
            continue
        gate = masks[frame, query_ids].amax(dim=0).reshape(1, 729, 1)
        result.append(current * gate)
        debug.append(
            {
                "frame_index": frame,
                "selected_queries": len(query_ids),
                "fallback": None,
                "gate_mean": float(gate.float().mean().item()),
                "active_patch_fraction": float((gate > SELECTIVE_GATE_ACTIVE_EPSILON).float().mean().item()),
                "word_match_enabled": word_match_enabled,
                "no_words_available": True,
                "word_match_applied": False,
                "word_match_effective_noop": word_match_enabled,
            }
        )
    return torch.cat(result, dim=0), debug


def _text_phrase_proxy(hidden_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    text = "Object information from the image:"
    score = sum((index + 1) * ord(char) for index, char in enumerate(text)) or 1
    indices = torch.arange(hidden_size, device=device, dtype=torch.float32)
    value = torch.sin(indices / 17.0 + float(score) / 997.0) + torch.cos(
        indices / 29.0 + float(score) / 577.0
    )
    return (value / value.norm(p=2).clamp_min(1e-6)).to(dtype=dtype).unsqueeze(0)


def object_tokens_from_cached(
    visual_tokens: torch.Tensor,
    visual_metadata: dict[str, Any],
    payload: Any,
    config: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Pool selected masks over ordinary 14x14 projected visual tokens."""
    cached = validate_cached_outputs(payload, consumer="object")
    indices = visual_metadata.get("visual_token_indices")
    frame_ids = visual_metadata.get("visual_frame_ids")
    if not isinstance(indices, torch.Tensor) or not isinstance(frame_ids, torch.Tensor):
        raise RuntimeError("EoMT object consumer requires visual-token metadata")
    if indices.numel() != NUM_FRAMES * 196 or frame_ids.numel() != indices.numel():
        raise RuntimeError("EoMT object consumer requires exactly 32x196 ordinary visual tokens")
    # The historical object branch uses MaskGuidedPooler's class-confidence
    # top-k selection.  Its configured ``-1`` is clamped to one query per
    # frame by that pooler; it is not a confidence threshold.  The selective
    # branch's .8 threshold must therefore not be reused here.
    top_k = max(1, min(int(getattr(config, "eomt_pool_top_k", 5)), NUM_QUERIES))
    per_frame_budget = int(getattr(config, "mm_eomt_object_block_max_per_frame", 2))
    global_budget = int(getattr(config, "mm_eomt_object_block_max_objects", 8))
    masks = cached["soft_masks"].to(device=visual_tokens.device, dtype=visual_tokens.dtype)
    classes = cached["class_logits"].to(device=visual_tokens.device)
    scores, class_ids = _foreground_scores(classes)
    stuff_class_ids = {int(value) for value in payload.get("stuff_class_ids", ())}
    keep_stuff = _flag(getattr(config, "mm_eomt_selector_keep_stuff", True))
    keep_things = _flag(getattr(config, "mm_eomt_selector_keep_things", True))
    candidates: list[tuple[int, float, int]] = []
    for frame in range(NUM_FRAMES):
        query_ids = torch.argsort(scores[frame], descending=True)[:top_k]
        for query in query_ids[:per_frame_budget].detach().cpu().tolist():
            class_id = int(class_ids[frame, query].item())
            is_stuff = class_id in stuff_class_ids
            if (is_stuff and not keep_stuff) or ((not is_stuff) and not keep_things):
                continue
            candidates.append((frame, float(scores[frame, query].item()), int(query)))
    # With no external word socket, historical word_match_then_frame_score
    # reduces to deterministic frame-then-score ordering.
    candidates.sort(key=lambda value: (value[0], -value[1], value[2]))
    candidates = candidates[:global_budget]
    pooled: list[torch.Tensor] = []
    for frame, _score, query in candidates:
        frame_indices = indices[frame_ids == frame].to(device=visual_tokens.device)
        frame_tokens = visual_tokens[frame_indices]
        if frame_tokens.shape[0] != 196:
            raise RuntimeError(f"EoMT frame {frame} has {frame_tokens.shape[0]} ordinary tokens, expected 196")
        weights = masks[frame, query].reshape(-1).clamp_min(0)
        weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
        # CPU Half GEMM is unsupported in the local torch build; preserving
        # FP16 output after a small FP32 pooling operation also keeps CPU
        # parity tests valid without changing CUDA inference semantics.
        if frame_tokens.device.type == "cpu" and frame_tokens.dtype == torch.float16:
            pooled_token = (weights.float().unsqueeze(0) @ frame_tokens.float()).squeeze(0).to(frame_tokens.dtype)
        else:
            pooled_token = (weights.unsqueeze(0) @ frame_tokens).squeeze(0)
        pooled.append(pooled_token)
    if not pooled:
        return visual_tokens.new_empty((0, visual_tokens.shape[-1])), {
            "enabled": True,
            "selected_count": 0,
            "fallback": "no_object_queries",
            "pool_top_k_effective": top_k,
            "stuff_taxonomy_available": bool(stuff_class_ids),
        }
    objects = torch.stack(pooled, dim=0)
    if str(getattr(config, "mm_eomt_obj_info_mode", "none")) == "text_phrase":
        objects = torch.cat((_text_phrase_proxy(objects.shape[-1], device=objects.device, dtype=objects.dtype), objects), dim=0)
    return objects, {
        "enabled": True,
        "selected_count": len(candidates),
        "object_block_token_count": int(objects.shape[0]),
        "selected_pairs": [(int(frame), int(query)) for frame, _score, query in candidates],
        "pool_top_k_effective": top_k,
        "stuff_taxonomy_available": bool(stuff_class_ids),
        "fallback": None,
    }
