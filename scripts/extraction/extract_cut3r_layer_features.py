#!/usr/bin/env python
"""Extract selected CUT3R decoder-layer token sidecars.

The output schema intentionally matches the existing VLM-3R CUT3R sidecars:

    {
      "camera_tokens": Tensor[F, 1, 768],
      "patch_tokens": Tensor[F, 729, 768],
      "metadata": {...}
    }

Each requested layer is written into its own output subdirectory so training can
switch layers by changing only spatial_features_subdir.

Coordinate consistency rule: these token sidecars may be paired with CUT3R
point-map geometry sidecars for GeoRoPE. The point-map coordinate source
(`point_maps_ref` vs `point_maps_cam`) must stay identical between training and
evaluation for a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from PIL import ImageFile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
from llava.model.multimodal_spatial_encoder.cut3r_spatial_encoder import (
    Cut3rSpatialConfig,
    prepare_input,
)
from llava.utils import process_video_with_decord, rank0_print


def _resolve_cut3r_root() -> str:
    candidates = [
        REPO_ROOT / "third_party" / "CUT3R",
        REPO_ROOT / "CUT3R",
    ]
    for path in candidates:
        if path.is_dir():
            return str(path)
    return str(candidates[0])


_CUT3R_ROOT = _resolve_cut3r_root()
if _CUT3R_ROOT in sys.path:
    sys.path.remove(_CUT3R_ROOT)
sys.path.insert(0, _CUT3R_ROOT)

from src.dust3r.model import ARCroco3DStereo  # noqa: E402


ImageFile.LOAD_TRUNCATED_IMAGES = True
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]


@dataclass
class DataArguments:
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = None
    image_crop_resolution: Optional[int] = None
    image_split_resolution: Optional[int] = None
    video_fps: int = 1
    frames_upbound: int = 32
    force_sample: bool = True


def parse_layers(value: str) -> list[int]:
    layers = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not layers:
        raise argparse.ArgumentTypeError("At least one CUT3R layer is required.")
    return layers


def layer_subdir(layer: int, prefix: str) -> str:
    if layer < 0:
        suffix = f"m{abs(layer)}"
    else:
        suffix = str(layer)
    return f"{prefix}{suffix}"


def find_video_files(input_dir: str) -> list[str]:
    input_path = Path(input_dir)
    video_files: list[Path] = []
    rank0_print(f"Scanning for video files in {input_dir}...")
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(input_path.rglob(f"*{ext}"))
        video_files.extend(input_path.rglob(f"*{ext.upper()}"))
    unique = sorted(set(video_files))
    rank0_print(f"Found {len(unique)} potential video files.")
    return [str(path) for path in unique]


def get_output_path(input_file_path: Path, input_base_dir: Path, output_base_dir: Path) -> Path:
    try:
        relative_path = input_file_path.relative_to(input_base_dir)
    except ValueError:
        rank0_print(
            f"Warning: {input_file_path} is not under {input_base_dir}; "
            f"saving directly under {output_base_dir}."
        )
        relative_path = Path(input_file_path.name)
    return output_base_dir / relative_path.with_suffix(".pt")


def load_and_preprocess_video_frames(video_path: str, data_args: DataArguments, processor, target_size=(432, 432), rank=0):
    try:
        video_frames, _, _, _ = process_video_with_decord(video_path, data_args)
        try:
            processed = processor.preprocess(images=video_frames, return_tensors="pt")
            frames_tensor = processed["pixel_values"]
        except Exception as exc:
            rank0_print(f"[GPU {rank}] Error processing frames for {video_path}: {exc}")
            return None

        frames, _, height, width = frames_tensor.shape
        if (height, width) == target_size:
            return frames_tensor

        scaled = []
        for frame_idx in range(frames):
            scaled.append(
                nn.functional.interpolate(
                    frames_tensor[frame_idx].unsqueeze(0),
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            )
        return torch.stack(scaled)
    except Exception as exc:
        rank0_print(f"[GPU {rank}] Error processing video {video_path}: {exc}")
        return None


def resolve_layers(layers: list[int], dec_depth: int) -> dict[int, int]:
    resolved = {}
    for layer in layers:
        idx = layer if layer >= 0 else dec_depth + 1 + layer
        if idx < 0 or idx > dec_depth:
            raise RuntimeError(f"CUT3R layer {layer} resolves to {idx}, valid range is [0,{dec_depth}].")
        resolved[layer] = idx
    return resolved


@torch.no_grad()
def run_cut3r_layers(cut3r, pixel_values: torch.Tensor, layers: list[int]) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    views = prepare_input(pixel_values=pixel_values)
    _, feat_ls, pos = cut3r._encode_views(views)
    feat = feat_ls[-1]
    state_feat, state_pos = cut3r._init_state(feat[0], pos[0])
    mem = cut3r.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
    init_state_feat = state_feat.clone()
    init_mem = mem.clone()

    resolved = resolve_layers(layers, int(cut3r.dec_depth))
    camera_tokens: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    patch_tokens: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}

    for frame_idx in range(len(views)):
        feat_i = feat[frame_idx].to(pixel_values.dtype)
        pos_i = pos[frame_idx]
        if cut3r.pose_head_flag:
            global_img_feat_i = cut3r._get_img_level_feat(feat_i)
            if frame_idx == 0:
                pose_feat_i = cut3r.pose_token.expand(feat_i.shape[0], -1, -1)
            else:
                pose_feat_i = cut3r.pose_retriever.inquire(global_img_feat_i, mem)
            pose_pos_i = -torch.ones(feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype)
        else:
            global_img_feat_i = None
            pose_feat_i = None
            pose_pos_i = None

        new_state_feat, dec = cut3r._recurrent_rollout(
            state_feat,
            state_pos,
            feat_i,
            pos_i,
            pose_feat_i,
            pose_pos_i,
            init_state_feat,
            img_mask=views[frame_idx]["img_mask"],
            reset_mask=views[frame_idx]["reset"],
            update=views[frame_idx].get("update", None),
        )
        assert len(dec) == cut3r.dec_depth + 1

        for layer, resolved_idx in resolved.items():
            layer_tokens = dec[resolved_idx]
            camera_tokens[layer].append(layer_tokens[:, :1].clone())
            patch_tokens[layer].append(layer_tokens[:, 1:].clone())

        if global_img_feat_i is not None:
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = cut3r.pose_retriever.update_mem(mem, global_img_feat_i, out_pose_feat_i)
        else:
            new_mem = mem

        img_mask = views[frame_idx]["img_mask"]
        update = views[frame_idx].get("update", None)
        update_mask = img_mask & update if update is not None else img_mask
        update_mask = update_mask[:, None, None].to(pixel_values.dtype)
        state_feat = new_state_feat * update_mask + state_feat * (1 - update_mask)
        mem = new_mem * update_mask + mem * (1 - update_mask)

        reset_mask = views[frame_idx]["reset"]
        if reset_mask is not None:
            reset_mask = reset_mask[:, None, None].to(pixel_values.dtype)
            state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
            mem = init_mem * reset_mask + mem * (1 - reset_mask)

    outputs = {}
    for layer in layers:
        cams = torch.stack(camera_tokens[layer], dim=0)
        patches = torch.stack(patch_tokens[layer], dim=0)
        cams = cams.permute(1, 0, 2, 3).contiguous()
        patches = patches.permute(1, 0, 2, 3).contiguous()
        outputs[layer] = (cams, patches)
    return outputs


def process_videos_on_gpu(rank, gpu_id, args, video_files_chunk, input_base_dir: Path, output_dirs: dict[int, Path]):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    processed_count = 0
    skipped_count = 0
    total_files = len(video_files_chunk)
    batch_size = args.batch_size
    reference_token_dir = Path(args.reference_token_dir) if args.reference_token_dir else None

    rank0_print(f"[GPU {gpu_id}] Worker started, assigned {total_files} files.")
    try:
        if not os.path.exists(args.cut3r_weights_path):
            rank0_print(f"[GPU {gpu_id}] ERROR: CUT3R weights not found: {args.cut3r_weights_path}")
            return 0, total_files

        config = Cut3rSpatialConfig(weights_path=args.cut3r_weights_path)
        cut3r = ARCroco3DStereo.from_pretrained(config.weights_path)
        cut3r.eval()
        for param in cut3r.parameters():
            param.requires_grad_(False)

        model_dtype = (
            torch.bfloat16
            if args.precision == "bf16"
            else torch.float16
            if args.precision == "fp16"
            else torch.float32
        )
        cut3r.to(device=device, dtype=model_dtype).eval()

        with open(args.processor_config_path, "r", encoding="utf-8") as f:
            processor_config = json.load(f)
        size_config = processor_config.get("size", {"height": 384, "width": 384})
        image_processor = SigLipImageProcessor(
            image_mean=processor_config.get("image_mean", (0.5, 0.5, 0.5)),
            image_std=processor_config.get("image_std", (0.5, 0.5, 0.5)),
            size=(size_config["height"], size_config["width"]),
            resample=processor_config.get("resample", 3),
            rescale_factor=processor_config.get("rescale_factor", 1 / 255.0),
        )
        rank0_print(f"[GPU {gpu_id}] CUT3R layer extractor loaded on {device}.")
    except Exception as exc:
        rank0_print(f"[GPU {gpu_id}] Error during initialization: {exc}\n{traceback.format_exc()}")
        return 0, total_files

    data_args = DataArguments(video_fps=args.video_fps, frames_upbound=args.frames_upbound)

    for batch_start in range(0, total_files, batch_size):
        batch_paths = [Path(path) for path in video_files_chunk[batch_start : batch_start + batch_size]]
        batch_data = []
        skipped_in_batch = 0
        max_frames = 0

        for video_path in batch_paths:
            output_paths = {
                layer: get_output_path(video_path, input_base_dir, output_dir)
                for layer, output_dir in output_dirs.items()
            }
            if reference_token_dir is not None:
                reference_path = get_output_path(video_path, input_base_dir, reference_token_dir)
                if args.require_reference_token_file and not reference_path.exists():
                    skipped_in_batch += 1
                    rank0_print(f"[GPU {gpu_id}] Missing reference token sidecar, skipping: {reference_path}")
                    continue
            else:
                reference_path = None

            if not args.overwrite and all(path.exists() for path in output_paths.values()):
                skipped_in_batch += 1
                continue
            for path in output_paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)

            preprocessed = load_and_preprocess_video_frames(str(video_path), data_args, image_processor, rank=gpu_id)
            if preprocessed is None or preprocessed.nelement() == 0:
                skipped_in_batch += 1
                rank0_print(f"[GPU {gpu_id}] Failed to load/preprocess {video_path}. Skipping.")
                continue

            batch_data.append((preprocessed, output_paths, video_path, reference_path))
            max_frames = max(max_frames, int(preprocessed.shape[0]))

        processed_in_batch = 0
        if batch_data:
            padded_tensors = []
            frame_counts = []
            for preprocessed, _, _, _ in batch_data:
                frame_counts.append(int(preprocessed.shape[0]))
                padding_needed = max_frames - int(preprocessed.shape[0])
                padded_tensors.append(
                    torch.nn.functional.pad(
                        preprocessed,
                        (0, 0, 0, 0, 0, 0, 0, padding_needed),
                        mode="constant",
                        value=0,
                    )
                )

            try:
                batch_tensor = torch.stack(padded_tensors, dim=0).to(device=device, dtype=model_dtype)
                spatial_input = batch_tensor.permute(1, 0, 2, 3, 4)
                outputs = run_cut3r_layers(cut3r, spatial_input, args.layers)

                for idx, (_, output_paths, video_path, reference_path) in enumerate(batch_data):
                    try:
                        num_frames = frame_counts[idx]
                        for layer, output_path in output_paths.items():
                            if output_path.exists() and not args.overwrite:
                                continue
                            camera_tokens, patch_tokens = outputs[layer]
                            payload = {
                                "camera_tokens": camera_tokens[idx, :num_frames].detach().to(dtype=torch.float16, device="cpu"),
                                "patch_tokens": patch_tokens[idx, :num_frames].detach().to(dtype=torch.float16, device="cpu"),
                                "metadata": {
                                    "source_video": str(video_path),
                                    "reference_token_feature_path": str(reference_path) if reference_path else None,
                                    "cut3r_weights_path": args.cut3r_weights_path,
                                    "selected_cut3r_decoder_layer": int(layer),
                                    "selected_cut3r_decoder_layer_resolved": int(resolve_layers([layer], int(cut3r.dec_depth))[layer]),
                                    "num_frames": int(num_frames),
                                    "frames_upbound": int(args.frames_upbound),
                                    "video_fps": int(args.video_fps),
                                    "token_dtype": "float16",
                                    "schema": "cut3r_decoder_layer_token_sidecar_v1",
                                },
                            }
                            torch.save(payload, output_path)
                        processed_in_batch += 1
                    except Exception as exc:
                        skipped_in_batch += 1
                        rank0_print(f"[GPU {gpu_id}] Error saving features for {video_path}: {exc}\n{traceback.format_exc()}")
                        for path in output_paths.values():
                            if path.exists():
                                try:
                                    path.unlink()
                                except OSError:
                                    rank0_print(f"Warning: could not remove partial file {path}")
            except Exception as exc:
                skipped_in_batch += len(batch_data) - processed_in_batch
                rank0_print(f"[GPU {gpu_id}] Error during batch inference: {exc}\n{traceback.format_exc()}")

        processed_count += processed_in_batch
        skipped_count += skipped_in_batch
        if (batch_start // batch_size) % 10 == 0:
            rank0_print(
                f"[GPU {gpu_id}] Progress: batch {batch_start // batch_size + 1}/"
                f"{math.ceil(total_files / batch_size)}, processed={processed_count}, skipped={skipped_count}"
            )

    rank0_print(f"[GPU {gpu_id}] Worker finished. Processed: {processed_count}, Skipped: {skipped_count}")
    return processed_count, skipped_count


def spawn_worker(rank, gpu_ids, args, chunks, input_base_dir, output_dirs):
    return process_videos_on_gpu(rank, gpu_ids[rank], args, chunks[rank], input_base_dir, output_dirs)


def parse_gpu_ids(gpu_ids_arg: str) -> list[int]:
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


def main() -> None:
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cut3r-weights-path", required=True)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-subdir-prefix", default="spatial_features_dec_")
    parser.add_argument("--reference-token-dir", default=None)
    parser.add_argument("--require-reference-token-file", action="store_true")
    parser.add_argument("--processor-config-path", required=True)
    parser.add_argument("--layers", type=parse_layers, default="-2,-4")
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
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

    try:
        gpu_ids = parse_gpu_ids(args.gpu_ids)
    except Exception as exc:
        rank0_print(f"Error: {exc}")
        sys.exit(1)
    if not gpu_ids:
        rank0_print("Error: no valid GPUs specified.")
        sys.exit(1)

    output_root = Path(args.output_root)
    output_dirs = {layer: output_root / layer_subdir(layer, args.output_subdir_prefix) for layer in args.layers}
    for output_dir in output_dirs.values():
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
        video_files = find_video_files(str(input_base_dir))

    if not video_files:
        rank0_print("No video files found to process.")
        sys.exit(0)

    generator = torch.Generator().manual_seed(0)
    permutation = torch.randperm(len(video_files), generator=generator).tolist()
    video_files = [video_files[idx] for idx in permutation]
    chunks = [video_files[idx:: len(gpu_ids)] for idx in range(len(gpu_ids))]

    rank0_print(f"Processing {len(video_files)} videos with GPUs {gpu_ids}.")
    for layer, output_dir in output_dirs.items():
        rank0_print(f"Saving CUT3R layer {layer} features under: {output_dir}")

    if len(gpu_ids) == 1:
        processed, skipped = process_videos_on_gpu(0, gpu_ids[0], args, chunks[0], input_base_dir, output_dirs)
        rank0_print(f"Done. Processed: {processed}, skipped: {skipped}")
    else:
        mp.spawn(
            spawn_worker,
            args=(gpu_ids, args, chunks, input_base_dir, output_dirs),
            nprocs=len(gpu_ids),
            join=True,
        )


if __name__ == "__main__":
    main()
