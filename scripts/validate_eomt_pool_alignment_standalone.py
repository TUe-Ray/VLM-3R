#!/usr/bin/env python3
"""Validate EoMT mask-pooling alignment contract via side-output methods - STANDALONE.

This is a standalone version that doesn't import from llava to avoid hanging issues.
It copies the necessary validation logic directly and processes test cases.
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import torch


class MaskGuidedPooler:
    """Minimal mock pooler for testing."""
    def __call__(self, soft_masks, visual_features, class_logits=None, top_k=5, selection="mean_mask_confidence", mask_area_threshold=0.5):
        # Return minimal valid output for testing
        return {
            "pooled_tokens": visual_features[:top_k] if visual_features.shape[0] >= top_k else visual_features,
            "selected_indices": list(range(min(top_k, visual_features.shape[0]))),
        }


class DummyPoolValidator:
    def __init__(self):
        self.config = SimpleNamespace(
            eomt_pool_top_k=3,
            eomt_pool_selection="mean_mask_confidence",
            eomt_pool_mask_area_threshold=0.5,
        )
        self._eomt_mask_pooler = MaskGuidedPooler()

    def _get_eomt_mask_pooler(self):
        return self._eomt_mask_pooler

    def _build_eomt_pool_side_cache(
        self,
        per_sample_visual_features,
        split_sizes,
        video_idx_in_batch,
        modalities,
        eomt_images,
        eomt_meta,
        image_aspect_ratio,
        mm_patch_merge_type,
        image_sizes,
    ):
        has_eomt_outputs = getattr(self, "_last_eomt_outputs", None) is not None

        if per_sample_visual_features is None:
            per_sample_visual_features = []

        total_samples = len(per_sample_visual_features)
        modalities = modalities if isinstance(modalities, list) else [modalities]

        if eomt_images is None:
            eomt_images_list = [None for _ in range(total_samples)]
        else:
            eomt_images_list = list(eomt_images)
            if len(eomt_images_list) < total_samples:
                eomt_images_list.extend([None for _ in range(total_samples - len(eomt_images_list))])

        if eomt_meta is None:
            eomt_meta_list = [None for _ in range(total_samples)]
        else:
            eomt_meta_list = list(eomt_meta)
            if len(eomt_meta_list) < total_samples:
                eomt_meta_list.extend([None for _ in range(total_samples - len(eomt_meta_list))])

        sample_frame_offsets = []
        total_eomt_frames = 0
        for sample_idx in range(total_samples):
            sample_frame_offsets.append(total_eomt_frames)
            sample_frames = eomt_images_list[sample_idx]
            if sample_frames is None:
                sample_frames = []
            total_eomt_frames += len(sample_frames)

        aligned_visual_list = []
        aligned_frame_meta_list = []
        aligned_global_frame_indices = []
        aligned_sample_frame_pairs = []

        pooled_sample_indices = []
        skipped_sample_indices = []
        per_sample_debug = []
        enabled_mask = []
        skip_reasons = []

        anyres_in_aspect = isinstance(image_aspect_ratio, str) and ("anyres" in image_aspect_ratio)

        for sample_idx in range(total_samples):
            sample_visual = per_sample_visual_features[sample_idx]
            sample_visual_shape = tuple(sample_visual.shape) if torch.is_tensor(sample_visual) else None

            if sample_idx < len(modalities):
                sample_modality = modalities[sample_idx]
            elif sample_idx in video_idx_in_batch:
                sample_modality = "video"
            else:
                sample_modality = "image"

            sample_frames = eomt_images_list[sample_idx]
            sample_frame_meta = eomt_meta_list[sample_idx]
            if sample_frames is None:
                sample_frames = []
            if sample_frame_meta is None:
                sample_frame_meta = []

            skip_reason = None
            sample_poolable_frame_count = 0

            if eomt_images is None:
                skip_reason = "missing_eomt_images"
            elif not has_eomt_outputs:
                skip_reason = "missing_eomt_outputs"
            elif len(sample_frames) == 0:
                skip_reason = "empty_eomt_frames"
            elif not torch.is_tensor(sample_visual) or sample_visual.ndim != 3:
                skip_reason = "unsupported_visual_shape_for_eomt_pooling"
            elif sample_modality == "video":
                if len(sample_frames) != sample_visual.shape[0]:
                    skip_reason = "video_frame_count_mismatch"
                else:
                    for frame_idx in range(sample_visual.shape[0]):
                        aligned_visual_list.append(sample_visual[frame_idx : frame_idx + 1].detach())
                        if frame_idx < len(sample_frame_meta) and isinstance(sample_frame_meta[frame_idx], dict):
                            aligned_frame_meta_list.append(dict(sample_frame_meta[frame_idx]))
                        else:
                            aligned_frame_meta_list.append({})
                        aligned_global_frame_indices.append(sample_frame_offsets[sample_idx] + frame_idx)
                        aligned_sample_frame_pairs.append((sample_idx, frame_idx))
                        sample_poolable_frame_count += 1
            else:
                if sample_visual.shape[0] == 1:
                    if len(sample_frames) != 1:
                        skip_reason = "single_image_eomt_count_mismatch"
                    else:
                        aligned_visual_list.append(sample_visual[0:1].detach())
                        if len(sample_frame_meta) >= 1 and isinstance(sample_frame_meta[0], dict):
                            aligned_frame_meta_list.append(dict(sample_frame_meta[0]))
                        else:
                            aligned_frame_meta_list.append({})
                        aligned_global_frame_indices.append(sample_frame_offsets[sample_idx])
                        aligned_sample_frame_pairs.append((sample_idx, 0))
                        sample_poolable_frame_count = 1
                elif sample_visual.shape[0] > 1:
                    if anyres_in_aspect:
                        skip_reason = "anyres_not_supported_for_eomt_pooling"
                    else:
                        skip_reason = "multi_patch_not_supported_for_eomt_pooling"
                else:
                    skip_reason = "unsupported_visual_shape_for_eomt_pooling"

            is_poolable = sample_poolable_frame_count > 0
            if is_poolable:
                pooled_sample_indices.append(sample_idx)
                enabled_mask.append(True)
            else:
                skipped_sample_indices.append(sample_idx)
                enabled_mask.append(False)
                if skip_reason is not None:
                    skip_reasons.append(skip_reason)

            per_sample_debug.append(
                {
                    "sample_idx": sample_idx,
                    "modality": sample_modality,
                    "visual_shape_before_merge": sample_visual_shape,
                    "eomt_frame_count": len(sample_frames),
                    "is_poolable": is_poolable,
                    "poolable_frame_count": sample_poolable_frame_count,
                    "skip_reason": skip_reason,
                }
            )

        pooled_visual_features = None
        if len(aligned_visual_list) > 0:
            pooled_visual_features = torch.cat(aligned_visual_list, dim=0)

        pool_debug = {
            "total_samples": total_samples,
            "total_eomt_frames": total_eomt_frames,
            "poolable_frame_count": len(aligned_visual_list),
            "pooled_sample_indices": pooled_sample_indices,
            "skipped_sample_indices": skipped_sample_indices,
            "aligned_global_frame_indices": aligned_global_frame_indices,
            "aligned_sample_frame_pairs": aligned_sample_frame_pairs,
            "split_sizes": list(split_sizes) if split_sizes is not None else None,
            "image_aspect_ratio": image_aspect_ratio,
            "mm_patch_merge_type": mm_patch_merge_type,
            "per_sample": per_sample_debug,
        }

        self._last_eomt_pool_visual_features = pooled_visual_features
        self._last_eomt_pool_frame_meta = aligned_frame_meta_list
        self._last_eomt_pool_debug = pool_debug
        self._last_eomt_pool_enabled_mask = enabled_mask
        self._last_eomt_pool_skip_reasons = skip_reasons

        return pooled_visual_features, aligned_frame_meta_list, pool_debug

    def _compute_eomt_mask_pooled_side_output(self):
        eomt_outputs = getattr(self, "_last_eomt_outputs", None)
        if eomt_outputs is None:
            return None

        pool_visual_features = getattr(self, "_last_eomt_pool_visual_features", None)
        pool_frame_meta = getattr(self, "_last_eomt_pool_frame_meta", None)
        pool_debug = getattr(self, "_last_eomt_pool_debug", None)
        if pool_frame_meta is None:
            pool_frame_meta = []
        if pool_debug is None:
            pool_debug = {}

        soft_masks = eomt_outputs.get("soft_masks", None)
        class_logits = eomt_outputs.get("class_logits", None)

        pool_top_k = int(getattr(self.config, "eomt_pool_top_k", 5))
        pool_selection = str(getattr(self.config, "eomt_pool_selection", "mean_mask_confidence"))
        pool_area_threshold = float(getattr(self.config, "eomt_pool_mask_area_threshold", 0.5))

        def _skipped_result(reason):
            return {
                "pooled_tokens": None,
                "selected_indices": None,
                "selected_scores": None,
                "selected_class_ids": None,
                "selection_method": pool_selection,
                "frame_meta": [],
                "mask_resolution": eomt_outputs.get("mask_resolution", None),
                "query_count": eomt_outputs.get("query_count", None),
                "pool_debug": pool_debug,
                "pool_skipped": True,
                "skip_reason": reason,
            }

        if soft_masks is None:
            return _skipped_result("missing_soft_masks")

        if pool_visual_features is None or (torch.is_tensor(pool_visual_features) and pool_visual_features.shape[0] == 0):
            return _skipped_result("no_frame_aligned_visual_features")

        if not torch.is_tensor(pool_visual_features) or pool_visual_features.ndim != 3:
            return _skipped_result("invalid_aligned_visual_features")

        aligned_global_frame_indices = pool_debug.get("aligned_global_frame_indices", [])
        aligned_sample_frame_pairs = pool_debug.get("aligned_sample_frame_pairs", [])
        if len(aligned_global_frame_indices) == 0:
            return _skipped_result("no_frame_aligned_visual_features")

        if max(aligned_global_frame_indices) >= soft_masks.shape[0]:
            return _skipped_result("aligned_frame_index_out_of_range")

        aligned_index_tensor = torch.as_tensor(
            aligned_global_frame_indices,
            dtype=torch.long,
            device=soft_masks.device,
        )
        aligned_soft_masks = soft_masks.index_select(0, aligned_index_tensor)

        aligned_class_logits = None
        if class_logits is not None:
            if max(aligned_global_frame_indices) >= class_logits.shape[0]:
                return _skipped_result("aligned_frame_index_out_of_range")
            aligned_class_logits = class_logits.index_select(0, aligned_index_tensor)

        if len(pool_frame_meta) == 0:
            raw_frame_meta = eomt_outputs.get("frame_meta", None)
            if isinstance(raw_frame_meta, list):
                pool_frame_meta = [
                    raw_frame_meta[idx] if idx < len(raw_frame_meta) else {}
                    for idx in aligned_global_frame_indices
                ]

        try:
            pooler = self._get_eomt_mask_pooler()
            with torch.no_grad():
                pooled = pooler(
                    soft_masks=aligned_soft_masks.to(device=pool_visual_features.device),
                    visual_features=pool_visual_features,
                    class_logits=(
                        aligned_class_logits.to(device=pool_visual_features.device)
                        if aligned_class_logits is not None
                        else None
                    ),
                    top_k=pool_top_k,
                    selection=pool_selection,
                    mask_area_threshold=pool_area_threshold,
                )

            pooled["frame_meta"] = pool_frame_meta
            pooled["mask_resolution"] = eomt_outputs.get("mask_resolution", None)
            pooled["query_count"] = eomt_outputs.get("query_count", None)
            pooled["pool_debug"] = pool_debug
            pooled["aligned_global_frame_indices"] = aligned_global_frame_indices
            pooled["aligned_sample_frame_pairs"] = aligned_sample_frame_pairs
            pooled["pool_skipped"] = False
            return pooled
        except Exception as e:
            print(f"EoMT mask pooling side branch error: {e}")
            skipped = _skipped_result("pooler_execution_error")
            skipped["pool_error"] = str(e)
            return skipped


def make_pil_placeholder(count: int) -> List[object]:
    return [object() for _ in range(count)]


def make_frame_meta(sample_idx: int, frame_count: int) -> List[Dict[str, Any]]:
    return [
        {
            "sample_idx": sample_idx,
            "frame_idx": frame_idx,
        }
        for frame_idx in range(frame_count)
    ]


def make_eomt_outputs(total_frames: int, num_q: int, num_classes_plus_noobj: int) -> Dict[str, Any]:
    if total_frames == 0:
        soft_masks = torch.zeros(0, num_q, 8, 8)
        class_logits = torch.zeros(0, num_q, num_classes_plus_noobj)
    else:
        soft_masks = torch.rand(total_frames, num_q, 8, 8)
        class_logits = torch.randn(total_frames, num_q, num_classes_plus_noobj)

    return {
        "soft_masks": soft_masks,
        "class_logits": class_logits,
        "mask_resolution": (8, 8),
        "query_count": num_q,
        "frame_meta": [{"flat_idx": i} for i in range(total_frames)],
    }


def run_case(
    validator: DummyPoolValidator,
    case_name: str,
    per_sample_visual_features: List[torch.Tensor],
    split_sizes: List[int],
    video_idx_in_batch: List[int],
    modalities: List[str],
    eomt_images: List[List[object]],
    eomt_meta: List[List[Dict[str, Any]]],
    image_aspect_ratio: str,
    mm_patch_merge_type: str,
    tamper_visual_rows: int = 0,
) -> Dict[str, Any]:
    validator._last_eomt_outputs = None
    validator._last_eomt_pool_visual_features = None
    validator._last_eomt_pool_frame_meta = None
    validator._last_eomt_pool_debug = None
    validator._last_eomt_pool_enabled_mask = None
    validator._last_eomt_pool_skip_reasons = None
    validator._last_eomt_pooled_outputs = None

    total_frames = sum(len(x) for x in eomt_images)
    validator._last_eomt_outputs = make_eomt_outputs(
        total_frames=total_frames,
        num_q=6,
        num_classes_plus_noobj=5,
    )

    validator._build_eomt_pool_side_cache(
        per_sample_visual_features=per_sample_visual_features,
        split_sizes=split_sizes,
        video_idx_in_batch=video_idx_in_batch,
        modalities=modalities,
        eomt_images=eomt_images,
        eomt_meta=eomt_meta,
        image_aspect_ratio=image_aspect_ratio,
        mm_patch_merge_type=mm_patch_merge_type,
        image_sizes=None,
    )

    if tamper_visual_rows > 0 and torch.is_tensor(validator._last_eomt_pool_visual_features):
        keep = max(0, validator._last_eomt_pool_visual_features.shape[0] - tamper_visual_rows)
        validator._last_eomt_pool_visual_features = validator._last_eomt_pool_visual_features[:keep]

    pooled = validator._compute_eomt_mask_pooled_side_output()
    validator._last_eomt_pooled_outputs = pooled

    debug = validator._last_eomt_pool_debug or {}
    pooled_tokens_shape = None
    if isinstance(pooled, dict) and torch.is_tensor(pooled.get("pooled_tokens", None)):
        pooled_tokens_shape = list(pooled["pooled_tokens"].shape)

    return {
        "case": case_name,
        "pool_skipped": None if pooled is None else pooled.get("pool_skipped", None),
        "skip_reason": None if pooled is None else pooled.get("skip_reason", None),
        "pool_error": None if pooled is None else pooled.get("pool_error", None),
        "aligned_global_frame_indices": None if pooled is None else pooled.get("aligned_global_frame_indices", []),
        "aligned_sample_frame_pairs": None if pooled is None else pooled.get("aligned_sample_frame_pairs", []),
        "per_sample": debug.get("per_sample", []),
        "pooled_tokens_shape": pooled_tokens_shape,
        "frame_meta_len": len((pooled or {}).get("frame_meta", [])) if isinstance(pooled, dict) else None,
    }


def evaluate_results(results: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, str], List[str], str]:
    summary = {}
    problems = []

    c1 = results["Case 1"]
    ok1 = (
        c1["pool_skipped"] is False
        and c1["pooled_tokens_shape"] is not None
        and c1["pooled_tokens_shape"][0] == 3
        and len(c1["aligned_global_frame_indices"]) == 3
        and all(item.get("is_poolable", False) for item in c1["per_sample"])
        and all(item.get("skip_reason", None) is None for item in c1["per_sample"])
    )
    summary["Case 1"] = "pass" if ok1 else "fail"
    if not ok1:
        problems.append("Case 1 failed poolability for batched single-image/square-grid.")

    c2 = results["Case 2"]
    ok2 = (
        c2["pool_skipped"] is False
        and c2["pooled_tokens_shape"] is not None
        and c2["pooled_tokens_shape"][0] == 4
        and c2["aligned_sample_frame_pairs"] == [(0, 0), (0, 1), (0, 2), (0, 3)]
    )
    summary["Case 2"] = "pass" if ok2 else "fail"
    if not ok2:
        problems.append("Case 2 failed video frame alignment or ordering.")

    c3 = results["Case 3"]
    c3_reason = c3["per_sample"][0].get("skip_reason", None) if c3["per_sample"] else None
    ok3 = c3["pool_skipped"] is True and c3_reason == "anyres_not_supported_for_eomt_pooling"
    summary["Case 3"] = "pass" if ok3 else "fail"
    if not ok3:
        problems.append("Case 3 did not explicitly skip anyres with expected reason.")

    c4 = results["Case 4"]
    c4_reason = c4["per_sample"][0].get("skip_reason", None) if c4["per_sample"] else None
    ok4 = c4["pool_skipped"] is True and c4_reason == "multi_patch_not_supported_for_eomt_pooling"
    summary["Case 4"] = "pass" if ok4 else "fail"
    if not ok4:
        problems.append("Case 4 did not explicitly skip multi-patch non-anyres with expected reason.")

    c5 = results["Case 5"]
    c5_reason = c5["per_sample"][0].get("skip_reason", None) if c5["per_sample"] else None
    ok5 = c5["pool_skipped"] is True and c5_reason in {
        "video_frame_count_mismatch",
        "single_image_eomt_count_mismatch",
    }
    summary["Case 5"] = "pass" if ok5 else "fail"
    if not ok5:
        problems.append("Case 5 did not explicitly skip frame-count mismatch with expected reason.")

    c6 = results["Case 6"]
    ok6 = (
        c6["pool_skipped"] is True
        and c6["skip_reason"] == "pooler_execution_error"
        and isinstance(c6["pool_error"], str)
        and len(c6["pool_error"]) > 0
    )
    summary["Case 6"] = "pass" if ok6 else "fail"
    if not ok6:
        problems.append("Case 6 did not surface pooler batch mismatch via pool_error.")

    if all(v == "pass" for v in summary.values()):
        verdict = "validated"
    elif any(v == "pass" for v in summary.values()):
        verdict = "validated with caveats"
    else:
        verdict = "not validated"

    return summary, problems, verdict


def print_report(results: Dict[str, Dict[str, Any]], summary: Dict[str, str], problems: List[str], verdict: str):
    print("1. Summary")
    for case_name in ["Case 1", "Case 2", "Case 3", "Case 4", "Case 5", "Case 6"]:
        print(f"- {case_name}: {summary.get(case_name, 'fail')}")

    print("\n2. Evidence")
    for case_name in ["Case 1", "Case 2", "Case 3", "Case 4", "Case 5", "Case 6"]:
        c = results[case_name]
        print(f"- {case_name}:")
        print(f"  shapes: pooled_tokens_shape={c['pooled_tokens_shape']}, frame_meta_len={c['frame_meta_len']}")
        print(f"  pool_skipped: {c['pool_skipped']}")
        print(f"  skip_reason: {c['skip_reason']}")
        print(f"  pool_error: {c['pool_error']}")
        print(f"  aligned_global_frame_indices: {c['aligned_global_frame_indices']}")
        print(f"  aligned_sample_frame_pairs: {c['aligned_sample_frame_pairs']}")
        print(f"  pool_debug[per_sample]: {json.dumps(c['per_sample'])}")

    print("\n3. Problems found")
    if problems:
        for p in problems:
            print(f"- {p}")
    else:
        print("- none")

    print("\n4. Final verdict")
    print(f"- {verdict}")


def main():
    parser = argparse.ArgumentParser(description="Validate EoMT mask pooling alignment contract")
    parser.add_argument("--output_json", type=str, default="logs/eomt_pool_validation_report.json")
    args = parser.parse_args()

    torch.manual_seed(0)
    validator = DummyPoolValidator()

    # Case 1: batched single-image / square-grid
    case1_visual = [torch.randn(1, 576, 128) for _ in range(3)]
    case1_eomt_images = [make_pil_placeholder(1) for _ in range(3)]
    case1_eomt_meta = [make_frame_meta(i, 1) for i in range(3)]
    r1 = run_case(
        validator,
        "Case 1",
        per_sample_visual_features=case1_visual,
        split_sizes=[1, 1, 1],
        video_idx_in_batch=[],
        modalities=["image", "image", "image"],
        eomt_images=case1_eomt_images,
        eomt_meta=case1_eomt_meta,
        image_aspect_ratio="square",
        mm_patch_merge_type="flat",
    )

    # Case 2: video matching frame counts
    case2_visual = [torch.randn(4, 576, 128)]
    case2_eomt_images = [make_pil_placeholder(4)]
    case2_eomt_meta = [make_frame_meta(0, 4)]
    r2 = run_case(
        validator,
        "Case 2",
        per_sample_visual_features=case2_visual,
        split_sizes=[4],
        video_idx_in_batch=[0],
        modalities=["video"],
        eomt_images=case2_eomt_images,
        eomt_meta=case2_eomt_meta,
        image_aspect_ratio="square",
        mm_patch_merge_type="flat",
    )

    # Case 3: anyres (non-video with shape[0] > 1)
    case3_visual = [torch.randn(2, 576, 128)]
    case3_eomt_images = [make_pil_placeholder(2)]
    case3_eomt_meta = [make_frame_meta(0, 2)]
    r3 = run_case(
        validator,
        "Case 3",
        per_sample_visual_features=case3_visual,
        split_sizes=[2],
        video_idx_in_batch=[],
        modalities=["image"],
        eomt_images=case3_eomt_images,
        eomt_meta=case3_eomt_meta,
        image_aspect_ratio="anyres_max_9",
        mm_patch_merge_type="spatial_unpad",
    )

    # Case 4: multi-patch non-anyres
    case4_visual = [torch.randn(2, 576, 128)]
    case4_eomt_images = [make_pil_placeholder(2)]
    case4_eomt_meta = [make_frame_meta(0, 2)]
    r4 = run_case(
        validator,
        "Case 4",
        per_sample_visual_features=case4_visual,
        split_sizes=[2],
        video_idx_in_batch=[],
        modalities=["image"],
        eomt_images=case4_eomt_images,
        eomt_meta=case4_eomt_meta,
        image_aspect_ratio="square",
        mm_patch_merge_type="spatial_unpad",
    )

    # Case 5: frame-count mismatch (video)
    case5_visual = [torch.randn(3, 576, 128)]
    case5_eomt_images = [make_pil_placeholder(2)]
    case5_eomt_meta = [make_frame_meta(0, 2)]
    r5 = run_case(
        validator,
        "Case 5",
        per_sample_visual_features=case5_visual,
        split_sizes=[3],
        video_idx_in_batch=[0],
        modalities=["video"],
        eomt_images=case5_eomt_images,
        eomt_meta=case5_eomt_meta,
        image_aspect_ratio="square",
        mm_patch_merge_type="flat",
    )

    # Case 6: batch mismatch protection (inject mismatch after alignment)
    case6_visual = [torch.randn(3, 576, 128)]
    case6_eomt_images = [make_pil_placeholder(3)]
    case6_eomt_meta = [make_frame_meta(0, 3)]
    r6 = run_case(
        validator,
        "Case 6",
        per_sample_visual_features=case6_visual,
        split_sizes=[3],
        video_idx_in_batch=[0],
        modalities=["video"],
        eomt_images=case6_eomt_images,
        eomt_meta=case6_eomt_meta,
        image_aspect_ratio="square",
        mm_patch_merge_type="flat",
        tamper_visual_rows=1,
    )

    results = {
        "Case 1": r1,
        "Case 2": r2,
        "Case 3": r3,
        "Case 4": r4,
        "Case 5": r5,
        "Case 6": r6,
    }

    summary, problems, verdict = evaluate_results(results)
    print_report(results, summary, problems, verdict)

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "results": results,
                "problems": problems,
                "verdict": verdict,
            },
            f,
            indent=2,
        )
    print(f"\nValidation report saved to: {args.output_json}")


if __name__ == "__main__":
    main()
