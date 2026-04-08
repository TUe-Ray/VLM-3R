#!/usr/bin/env python3
"""Validate EoMT mask-pooling alignment contract via side-output methods.

This script does not touch the main fusion path. It exercises:
- _build_eomt_pool_side_cache
- _compute_eomt_mask_pooled_side_output

and reports the required validation sections.
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Avoid hanging imports by importing directly
try:
    from llava.model.llava_arch import LlavaMetaForCausalLM
except ImportError as e:
    print(f"Warning: Failed to import LlavaMetaForCausalLM: {e}", file=sys.stderr)
    print("Creating mock class instead...", file=sys.stderr)
    # Create a mock class if import fails
    class LlavaMetaForCausalLM:
        def get_model(self):
            return self
        def _get_eomt_mask_pooler(self):
            from llava.model.multimodal_eomt import MaskGuidedPooler
            pooler = getattr(self, "_eomt_mask_pooler", None)
            if pooler is None:
                pooler = MaskGuidedPooler()
                self._eomt_mask_pooler = pooler
            return pooler


class DummyPoolValidator(LlavaMetaForCausalLM):
    def __init__(self):
        self.config = SimpleNamespace(
            eomt_pool_top_k=3,
            eomt_pool_selection="mean_mask_confidence",
            eomt_pool_mask_area_threshold=0.5,
        )

    def get_model(self):
        return self


def make_pil_placeholder(count: int) -> List[object]:
    # Pool side-cache only needs list lengths; placeholders are enough.
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


if __name__ == "__main__":
    main()
