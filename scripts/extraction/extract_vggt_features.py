import argparse
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from PIL import ImageFile

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
from llava.model.multimodal_spatial_encoder.vggt_spatial_encoder import (
    _load_vggt_model,
    prepare_input,
)
from llava.utils import process_video_with_decord, rank0_print

# GeoRoPE coordinate consistency rule:
# If VGGT-derived geometry is used for training, evaluation for that checkpoint
# must use the same VGGT geometry provider and coordinate convention. Do not
# compare against eval runs that silently switch to CUT3R ref/cam point maps.

try:
    from decord import VideoReader, cpu
except Exception:
    VideoReader = None
    cpu = None


ImageFile.LOAD_TRUNCATED_IMAGES = True
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]
DEFAULT_LAYER_INDICES = (4, 11, 17, 23)


@dataclass
class DataArguments:
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = None
    image_crop_resolution: Optional[int] = None
    image_split_resolution: Optional[int] = None
    video_fps: int = 1
    frames_upbound: int = 32
    force_sample: bool = True


def find_video_files(input_dir):
    input_path = Path(input_dir)
    video_files = []
    rank0_print(f"Scanning for video files in {input_dir}...")
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(input_path.rglob(f"*{ext}"))
        video_files.extend(input_path.rglob(f"*{ext.upper()}"))
    unique_video_files = sorted(set(video_files))
    rank0_print(f"Found {len(unique_video_files)} potential video files.")
    return [str(path) for path in unique_video_files]


def get_output_path(input_file_path: Path, input_base_dir: Path, output_base_dir: Path) -> Path:
    try:
        relative_path = input_file_path.relative_to(input_base_dir)
    except ValueError:
        rank0_print(
            f"Warning: {input_file_path} is not within {input_base_dir}; "
            f"saving directly under {output_base_dir}."
        )
        relative_path = Path(input_file_path.name)
    return output_base_dir / relative_path.with_suffix(".pt")


def compute_frame_idx_with_decord(video_path, data_args):
    if VideoReader is None or cpu is None:
        return None

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    avg_fps = round(vr.get_avg_fps() / data_args.video_fps)
    if avg_fps <= 0:
        avg_fps = 1
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]

    if data_args.frames_upbound > 0:
        if len(frame_idx) > data_args.frames_upbound or data_args.force_sample:
            frame_idx = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int).tolist()

    vr.seek(0)
    return torch.tensor(frame_idx, dtype=torch.long)


def load_and_preprocess_video_frames(video_path, data_args, processor, rank=0):
    try:
        sampled_frame_idx = compute_frame_idx_with_decord(video_path, data_args)
        video_frames, _, _, _ = process_video_with_decord(video_path, data_args)
        try:
            processed_output = processor.preprocess(images=video_frames, return_tensors="pt")
            frames_tensor = processed_output["pixel_values"]
        except Exception as exc:
            rank0_print(f"[GPU {rank}] Error processing video frames for {video_path}: {exc}")
            return None

        if sampled_frame_idx is not None and len(sampled_frame_idx) != frames_tensor.shape[0]:
            rank0_print(
                f"[GPU {rank}] Warning: frame_idx length ({len(sampled_frame_idx)}) "
                f"does not match frame tensor count ({frames_tensor.shape[0]}) for {video_path}."
            )
            sampled_frame_idx = None

        if sampled_frame_idx is None:
            sampled_frame_idx = torch.arange(frames_tensor.shape[0], dtype=torch.long)

        return {
            "frames_tensor": frames_tensor,
            "frame_idx": sampled_frame_idx,
        }
    except Exception as exc:
        rank0_print(f"[GPU {rank}] Error processing video {video_path}: {exc}")
        return None


@torch.no_grad()
def extract_vggt_tokens(vggt, pixel_values, image_size, layer_indices):
    views = prepare_input(pixel_values=pixel_values, image_size=image_size)
    aggregated_tokens_list, patch_start_idx = vggt.aggregator(views)

    missing = [idx for idx in layer_indices if idx < 0 or idx >= len(aggregated_tokens_list)]
    if missing:
        raise RuntimeError(
            f"Requested VGGT layers {missing}, but aggregator returned "
            f"{len(aggregated_tokens_list)} layers."
        )

    tokens_by_layer = {}
    for idx in layer_indices:
        tokens = aggregated_tokens_list[idx].detach()
        if tokens.shape[0] == 1:
            tokens = tokens.squeeze(0)
        tokens_by_layer[str(idx)] = tokens.to(dtype=torch.bfloat16, device="cpu")
    return tokens_by_layer, int(patch_start_idx), views.shape[-2:]


def process_videos_on_gpu(rank, gpu_id, args, video_files_chunk, input_base_dir, output_dir):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    processed_count = 0
    skipped_count = 0
    total_files = len(video_files_chunk)
    batch_size = args.batch_size
    layer_indices = tuple(int(value) for value in args.layer_indices.split(",") if value.strip())

    rank0_print(f"[GPU {gpu_id}] Worker started, assigned {total_files} files.")

    try:
        if not args.vggt_weights_path:
            rank0_print(f"[GPU {gpu_id}] ERROR: --vggt-weights-path is required.")
            return 0, total_files
        if os.path.exists(args.vggt_weights_path):
            rank0_print(f"[GPU {gpu_id}] Loading VGGT from local path: {args.vggt_weights_path}")
        else:
            rank0_print(f"[GPU {gpu_id}] Loading VGGT from model id: {args.vggt_weights_path}")

        vggt = _load_vggt_model(args.vggt_weights_path)
        vggt.eval()
        for param in vggt.parameters():
            param.requires_grad = False

        model_dtype = (
            torch.bfloat16
            if args.precision == "bf16"
            else torch.float16
            if args.precision == "fp16"
            else torch.float32
        )
        vggt.to(device=device, dtype=model_dtype).eval()

        with open(args.processor_config_path, "r") as handle:
            processor_config = json.load(handle)
        size_config = processor_config.get("size", {"height": 384, "width": 384})
        image_processor = SigLipImageProcessor(
            image_mean=processor_config.get("image_mean", (0.5, 0.5, 0.5)),
            image_std=processor_config.get("image_std", (0.5, 0.5, 0.5)),
            size=(size_config["height"], size_config["width"]),
            resample=processor_config.get("resample", 3),
            rescale_factor=processor_config.get("rescale_factor", 1 / 255.0),
        )
        rank0_print(f"[GPU {gpu_id}] VGGT feature extractor loaded on {device}.")
    except Exception as exc:
        rank0_print(f"[GPU {gpu_id}] Error during initialization: {exc}\n{traceback.format_exc()}")
        return 0, total_files

    data_args = DataArguments(video_fps=args.video_fps, frames_upbound=args.frames_upbound)

    for i in range(0, total_files, batch_size):
        batch_paths = [Path(path) for path in video_files_chunk[i : i + batch_size]]
        processed_in_batch = 0
        skipped_in_batch = 0

        for video_path in batch_paths:
            output_path = get_output_path(video_path, input_base_dir, output_dir)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists() and not args.overwrite:
                skipped_in_batch += 1
                continue

            preprocessed = load_and_preprocess_video_frames(
                str(video_path), data_args, image_processor, rank=gpu_id
            )
            if preprocessed is None or preprocessed["frames_tensor"].nelement() == 0:
                skipped_in_batch += 1
                rank0_print(f"[GPU {gpu_id}] Failed to load/preprocess {video_path}. Skipping.")
                continue

            try:
                pixel_values = preprocessed["frames_tensor"].to(device=device, dtype=model_dtype)
                with torch.cuda.amp.autocast(dtype=model_dtype):
                    tokens_by_layer, patch_start_idx, model_hw = extract_vggt_tokens(
                        vggt, pixel_values, args.vggt_input_size, layer_indices
                    )

                num_frames = int(preprocessed["frames_tensor"].shape[0])
                payload = {
                    "frames": {
                        "aggregated_tokens": tokens_by_layer,
                        "frame_idx": preprocessed["frame_idx"].clone(),
                    },
                    "meta": {
                        "source_video": str(video_path),
                        "vggt_weights_path": args.vggt_weights_path,
                        "num_frames": num_frames,
                        "input_size": int(args.vggt_input_size),
                        "model_image_hw": tuple(int(v) for v in model_hw),
                        "patch_size": int(vggt.aggregator.patch_size),
                        "patch_start_idx": int(patch_start_idx),
                        "feature_dim": int(next(iter(tokens_by_layer.values())).shape[-1]),
                        "intermediate_layer_idx": list(layer_indices),
                        "token_dtype": "bfloat16",
                        "schema": "vggt_aggregated_tokens_v1",
                    },
                }
                torch.save(payload, output_path)
                processed_in_batch += 1
            except Exception as exc:
                skipped_in_batch += 1
                rank0_print(
                    f"[GPU {gpu_id}] Error during VGGT inference/save for {video_path}: "
                    f"{exc}\n{traceback.format_exc()}"
                )
                if output_path.exists():
                    try:
                        output_path.unlink()
                    except OSError:
                        rank0_print(f"Warning: could not remove partial file {output_path}")

        processed_count += processed_in_batch
        skipped_count += skipped_in_batch
        if (i // batch_size) % 10 == 0:
            rank0_print(
                f"[GPU {gpu_id}] Progress: batch {i // batch_size + 1}/"
                f"{math.ceil(total_files / batch_size)}, processed={processed_count}, "
                f"skipped={skipped_count}"
            )

    rank0_print(f"[GPU {gpu_id}] Worker finished. Processed: {processed_count}, Skipped: {skipped_count}")
    return processed_count, skipped_count


def spawn_worker(rank, gpu_ids, args, chunks, input_base_dir, output_dir):
    return process_videos_on_gpu(
        rank, gpu_ids[rank], args, chunks[rank], input_base_dir, output_dir
    )


def parse_gpu_ids(gpu_ids_arg):
    if gpu_ids_arg.lower() == "all":
        if not torch.cuda.is_available():
            raise RuntimeError("'all' GPUs requested, but CUDA is not available.")
        return list(range(torch.cuda.device_count()))
    gpu_ids = [int(value.strip()) for value in gpu_ids_arg.split(",") if value.strip()]
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        invalid = [gpu_id for gpu_id in gpu_ids if gpu_id < 0 or gpu_id >= num_gpus]
        if invalid:
            raise RuntimeError(f"Invalid GPU IDs {invalid}. Available GPUs: {list(range(num_gpus))}")
    return gpu_ids


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Extract VGGT aggregated-token feature sidecar .pt files.")
    parser.add_argument("--vggt-weights-path", type=str, required=True)
    parser.add_argument("--vggt-input-size", type=int, default=518)
    parser.add_argument("--layer-indices", type=str, default=",".join(str(x) for x in DEFAULT_LAYER_INDICES))
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--input-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--processor-config-path", type=str, required=True)
    parser.add_argument("--gpu-ids", type=str, default="0")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--video-fps", type=int, default=1)
    parser.add_argument("--frames-upbound", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)

    args = parser.parse_args()

    if not args.input_dir and not args.input_file:
        parser.error("Either --input-dir or --input-file must be specified.")
    if args.input_dir and args.input_file:
        rank0_print("Warning: both --input-dir and --input-file provided; using --input-file.")
        args.input_dir = None

    if args.vggt_input_size % 14 != 0:
        parser.error("--vggt-input-size must be a multiple of 14 for VGGT patching")

    try:
        gpu_ids = parse_gpu_ids(args.gpu_ids)
    except Exception as exc:
        rank0_print(f"Error: {exc}")
        sys.exit(1)
    if not gpu_ids:
        rank0_print("Error: no valid GPUs specified.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_file:
        input_file_path = Path(args.input_file)
        if not input_file_path.is_file():
            rank0_print(f"Error: --input-file not found: {args.input_file}")
            sys.exit(1)
        video_files = [str(input_file_path)]
        input_base_dir = input_file_path.parent
    else:
        input_base_dir = Path(args.input_dir)
        if not input_base_dir.is_dir():
            rank0_print(f"Error: --input-dir not found: {args.input_dir}")
            sys.exit(1)
        video_files = find_video_files(input_base_dir)

    if not video_files:
        rank0_print("No video files found to process.")
        sys.exit(0)

    random_state = torch.Generator().manual_seed(0)
    permutation = torch.randperm(len(video_files), generator=random_state).tolist()
    video_files = [video_files[idx] for idx in permutation]

    chunks = [video_files[idx:: len(gpu_ids)] for idx in range(len(gpu_ids))]
    rank0_print(f"Processing {len(video_files)} videos with GPUs {gpu_ids}.")
    rank0_print(f"Saving VGGT aggregated-token sidecars under: {output_dir}")

    if len(gpu_ids) == 1:
        processed, skipped = process_videos_on_gpu(
            0, gpu_ids[0], args, chunks[0], input_base_dir, output_dir
        )
        rank0_print(f"Done. Processed: {processed}, skipped: {skipped}")
    else:
        mp.spawn(
            spawn_worker,
            args=(gpu_ids, args, chunks, input_base_dir, output_dir),
            nprocs=len(gpu_ids),
            join=True,
        )
