"""Debug script to verify EoMT frame alignment with VLM-3R training pipeline.

Loads training samples from LazySupervisedDataset and uses the eomt_images /
eomt_meta fields that the dataset already populated.  EoMT is run on exactly
those same raw PIL frames — no independent reloading or resampling.

This script is intended for strict alignment checks, so key dataset settings
that affect frame sampling and preprocessing are exposed via CLI and passed into
LazySupervisedDataset.

Usage:
    python scripts/debug_eomt_alignment.py \
        --data_path scripts/VLM_3R/vsibench_data.yaml \
        --video_folder /path/to/data \
        --eomt_config_path third_party/EoMT/configs/dinov2/coco/panoptic/eomt_large_640.yaml \
        --eomt_ckpt_path /path/to/eomt_weights.bin \
        --num_samples 5 \
        --output_dir debug_eomt
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so llava imports work
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_sample_indices(raw: Optional[str]) -> List[int]:
    if raw is None or raw.strip() == "":
        return []
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part == "":
            continue
        values.append(int(part))
    return values


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------
def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    cmap_name: str = "jet",
) -> np.ndarray:
    cmap = plt.get_cmap(cmap_name)
    heatmap = (cmap(mask)[..., :3] * 255).astype(np.uint8)
    blended = alpha * image.astype(np.float32) + (1 - alpha) * heatmap.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Debug EoMT frame alignment with VLM-3R training pipeline."
    )
    # Core inputs
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--eomt_config_path", type=str, required=True)
    parser.add_argument("--eomt_ckpt_path", type=str, required=True)

    # Dataset/data-path arguments (Task A: strict alignment)
    parser.add_argument("--video_folder", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--video_fps", type=int, default=1)
    parser.add_argument("--frames_upbound", type=int, default=32)
    parser.add_argument("--force_sample", type=str2bool, default=False)
    parser.add_argument("--image_aspect_ratio", type=str, default="square")
    parser.add_argument("--add_time_instruction", type=str2bool, default=False)
    parser.add_argument("--train_data_percentage", type=float, default=100.0)
    parser.add_argument("--train_data_percentage_seed", type=int, default=42)
    parser.add_argument("--train_data_shuffle", type=str2bool, default=False)

    # Processor/tokenizer controls to better mirror training setup
    parser.add_argument(
        "--processor_name_or_path",
        type=str,
        default="google/siglip-so400m-patch14-384",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="Qwen/Qwen2-0.5B",
    )

    # Sampling controls
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument(
        "--sample_indices",
        type=str,
        default="",
        help="Comma-separated absolute dataset indices (overrides num_samples/start_index).",
    )

    # Outputs
    parser.add_argument("--output_dir", type=str, default="debug_eomt")
    parser.add_argument("--top_k_masks", type=int, default=5)

    # Runtime
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build EoMT extractor (reuse the project wrapper)
    # ------------------------------------------------------------------
    from llava.model.multimodal_eomt import EoMTExtractor

    eomt_cfg = {
        "config_path": args.eomt_config_path,
        "ckpt_path": args.eomt_ckpt_path,
        "device": args.device,
    }
    print(f"[EoMT] Loading from {args.eomt_ckpt_path} ...")
    eomt = EoMTExtractor(eomt_cfg)
    eomt.to(device)
    eomt.eval()

    # ------------------------------------------------------------------
    # 2. Build dataset
    # ------------------------------------------------------------------
    try:
        from transformers import SiglipImageProcessor
        processor = SiglipImageProcessor.from_pretrained(args.processor_name_or_path)
    except Exception:
        try:
            from transformers import CLIPImageProcessor
            processor = CLIPImageProcessor.from_pretrained(args.processor_name_or_path)
        except Exception:
            print("WARNING: Could not load image processor. Dataset loading may fail.")
            processor = None

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    except Exception:
        print("WARNING: Could not load tokenizer.")
        tokenizer = None

    from llava.train.train import DataArguments, LazySupervisedDataset

    data_args = DataArguments(
        data_path=args.data_path,
        lazy_preprocess=True,
        is_multimodal=True,
        early_mix_text=False,
        video_folder=args.video_folder,
        image_folder=args.image_folder,
        video_fps=args.video_fps,
        frames_upbound=args.frames_upbound,
        add_time_instruction=args.add_time_instruction,
        force_sample=args.force_sample,
        image_aspect_ratio=args.image_aspect_ratio,
        train_data_percentage=args.train_data_percentage,
        train_data_percentage_seed=args.train_data_percentage_seed,
        train_data_shuffle=args.train_data_shuffle,
    )
    # LazySupervisedDataset expects this runtime-attached field.
    data_args.image_processor = processor

    print(f"[Dataset] Loading from {args.data_path} ...")
    dataset = LazySupervisedDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        data_args=data_args,
    )
    print(f"[Dataset] Total samples: {len(dataset)}")

    explicit_indices = parse_sample_indices(args.sample_indices)
    if explicit_indices:
        sample_indices = [idx for idx in explicit_indices if 0 <= idx < len(dataset)]
    else:
        start = max(args.start_index, 0)
        end = min(start + args.num_samples, len(dataset))
        sample_indices = list(range(start, end))

    print(f"[Dataset] Processing indices: {sample_indices}")
    if not sample_indices:
        print("No valid sample indices to process. Exiting.")
        return

    # ------------------------------------------------------------------
    # 3. Iterate over samples
    # ------------------------------------------------------------------
    for iter_idx, sample_idx in enumerate(sample_indices):
        print(f"\n{'='*60}")
        print(f"Processing sample index {sample_idx} ({iter_idx + 1}/{len(sample_indices)})")

        # Get the sample — this populates eomt_images and eomt_meta
        try:
            sample = dataset[sample_idx]
        except Exception as e:
            print(f"  WARNING: dataset[{sample_idx}] failed: {e}")
            continue

        if "eomt_images" not in sample or not sample["eomt_images"]:
            print(f"  Sample {sample_idx} has no eomt_images, skipping.")
            continue

        pil_frames = sample["eomt_images"]   # list of PIL.Image
        frame_metas = sample.get("eomt_meta", [{} for _ in pil_frames])

        sample_dir = os.path.join(args.output_dir, f"sample_{sample_idx}")
        os.makedirs(sample_dir, exist_ok=True)

        print(f"  Frames: {len(pil_frames)}")
        print(f"  Modality: {frame_metas[0].get('modality', '?')}")
        print(f"  Source: {frame_metas[0].get('source_path', '?')}")

        # ---- Run EoMT on the exact frames the dataset produced ----
        try:
            with torch.no_grad():
                eomt_outputs = eomt(pil_frames, frame_metas)
        except Exception as e:
            print(f"  WARNING: EoMT inference failed: {e}")
            continue

        soft_masks = eomt_outputs["soft_masks"]    # (B, num_q, H, W)
        class_logits = eomt_outputs["class_logits"]  # (B, num_q, num_classes+1)
        num_queries = int(soft_masks.shape[1])

        frame_reports: List[Dict[str, Any]] = []

        # ---- Visualise per frame ----
        for fidx, (pil_img, fmeta) in enumerate(zip(pil_frames, frame_metas)):
            frame_index = fmeta.get("frame_index", fidx)

            # Save original frame
            orig_path = os.path.join(sample_dir, f"original_frame_{fidx}_vidx{frame_index}.png")
            pil_img.save(orig_path)
            print(f"  Saved {orig_path}")

            frame_report: Dict[str, Any] = {
                "frame_list_index": fidx,
                "video_frame_index": frame_index,
                "frame_path": fmeta.get("frame_path", ""),
                "original_frame_path": orig_path,
                "frame_meta": to_jsonable(fmeta),
                "topk": [],
            }

            if num_queries <= 0:
                frame_reports.append(frame_report)
                continue

            # Top-k mask overlays
            mask_batch = soft_masks[fidx]        # (num_q, H, W)
            cls_batch = class_logits[fidx]       # (num_q, num_classes+1)

            class_probs = torch.softmax(cls_batch.float(), dim=-1)
            max_scores, max_class_ids = class_probs[:, :-1].max(dim=-1)
            topk_count = min(args.top_k_masks, max_scores.shape[0])
            topk_indices = torch.argsort(max_scores, descending=True)[:topk_count]

            orig_np = np.array(pil_img)
            orig_h, orig_w = orig_np.shape[:2]

            for rank, q_idx in enumerate(topk_indices):
                q_idx_int = q_idx.item()
                score = max_scores[q_idx_int].item()
                cls_id = max_class_ids[q_idx_int].item()
                mask = mask_batch[q_idx_int]  # (H, W) already at EoMT img_size

                import torch.nn.functional as F
                mask_resized = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze().cpu().float().numpy()

                overlay = overlay_mask_on_image(orig_np, mask_resized, alpha=0.5)

                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                ax.imshow(overlay)
                ax.set_title(
                    f"Frame {fidx} (video_idx={frame_index}) | Query {q_idx_int} (rank {rank}) | "
                    f"class {cls_id} | score {score:.3f}",
                    fontsize=9,
                )
                ax.axis("off")

                save_path = os.path.join(
                    sample_dir,
                    f"soft_mask_frame{fidx}_vidx{frame_index}_query{q_idx_int}_rank{rank}.png",
                )
                fig.savefig(save_path, bbox_inches="tight", dpi=120)
                plt.close(fig)
                print(f"  Saved {save_path}")

                frame_report["topk"].append(
                    {
                        "rank": rank,
                        "query_index": q_idx_int,
                        "class_id": cls_id,
                        "score": float(score),
                        "overlay_path": save_path,
                    }
                )

            frame_reports.append(frame_report)

        # ---- Save metadata ----
        frame_indices = [int(fm.get("frame_index", idx)) for idx, fm in enumerate(frame_metas)]
        frame_order_non_decreasing = all(
            frame_indices[j] <= frame_indices[j + 1]
            for j in range(len(frame_indices) - 1)
        )

        meta_out = {
            "sample_idx": sample_idx,
            "sample_id": sample.get("id", sample_idx),
            "num_frames": len(pil_frames),
            "eomt_query_count": eomt_outputs["query_count"],
            "mask_resolution": list(eomt_outputs["mask_resolution"]),
            "dataset_config": {
                "data_path": args.data_path,
                "video_folder": args.video_folder,
                "image_folder": args.image_folder,
                "video_fps": args.video_fps,
                "frames_upbound": args.frames_upbound,
                "force_sample": args.force_sample,
                "image_aspect_ratio": args.image_aspect_ratio,
                "add_time_instruction": args.add_time_instruction,
                "train_data_percentage": args.train_data_percentage,
                "train_data_percentage_seed": args.train_data_percentage_seed,
                "train_data_shuffle": args.train_data_shuffle,
            },
            "frame_indices": frame_indices,
            "frame_order_non_decreasing": frame_order_non_decreasing,
            "frame_metas": to_jsonable(frame_metas),
            "frame_reports": frame_reports,
        }
        meta_path = os.path.join(sample_dir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta_out, f, indent=2)
        print(f"  Saved {meta_path}")

    print(f"\nDone. Debug outputs saved to {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
