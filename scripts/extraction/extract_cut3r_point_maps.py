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

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
from llava.model.multimodal_spatial_encoder.cut3r_spatial_encoder import (
    Cut3rSpatialConfig,
    prepare_input,
)
from llava.utils import process_video_with_decord, rank0_print


def _resolve_cut3r_root():
    repo_root = _REPO_ROOT
    candidates = [
        os.path.join(repo_root, "third_party", "CUT3R"),
        os.path.join(repo_root, "CUT3R"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


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


def load_and_preprocess_video_frames(video_path, data_args, processor, target_size=(432, 432), rank=0):
    try:
        video_frames, _, _, _ = process_video_with_decord(video_path, data_args)
        try:
            processed_output = processor.preprocess(images=video_frames, return_tensors="pt")
            frames_tensor = processed_output["pixel_values"]
        except Exception as exc:
            rank0_print(f"[GPU {rank}] Error processing video frames for {video_path}: {exc}")
            return None

        f, _, h_proc, w_proc = frames_tensor.shape
        if (h_proc, w_proc) == target_size:
            return frames_tensor

        frames_scaled = []
        for frame_idx in range(f):
            frame_scaled = nn.functional.interpolate(
                frames_tensor[frame_idx].unsqueeze(0),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            frames_scaled.append(frame_scaled)
        return torch.stack(frames_scaled)
    except Exception as exc:
        rank0_print(f"[GPU {rank}] Error processing video {video_path}: {exc}")
        return None


@torch.no_grad()
def run_cut3r_pointmaps(cut3r, pixel_values):
    views = prepare_input(pixel_values=pixel_values)
    shape, feat_ls, pos = cut3r._encode_views(views)
    feat = feat_ls[-1]
    state_feat, state_pos = cut3r._init_state(feat[0], pos[0])
    mem = cut3r.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
    init_state_feat = state_feat.clone()
    init_mem = mem.clone()

    point_maps_cam = []
    point_maps_ref = []
    camera_poses = []

    for i in range(len(views)):
        feat_i = feat[i].to(pixel_values.dtype)
        pos_i = pos[i]
        if cut3r.pose_head_flag:
            global_img_feat_i = cut3r._get_img_level_feat(feat_i)
            if i == 0:
                pose_feat_i = cut3r.pose_token.expand(feat_i.shape[0], -1, -1)
            else:
                pose_feat_i = cut3r.pose_retriever.inquire(global_img_feat_i, mem)
            pose_pos_i = -torch.ones(
                feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
            )
        else:
            raise RuntimeError("CUT3R point-map sidecar extraction requires pose_head outputs.")

        new_state_feat, dec = cut3r._recurrent_rollout(
            state_feat,
            state_pos,
            feat_i,
            pos_i,
            pose_feat_i,
            pose_pos_i,
            init_state_feat,
            img_mask=views[i]["img_mask"],
            reset_mask=views[i]["reset"],
            update=views[i].get("update", None),
        )
        out_pose_feat_i = dec[-1][:, 0:1]
        new_mem = cut3r.pose_retriever.update_mem(mem, global_img_feat_i, out_pose_feat_i)
        assert len(dec) == cut3r.dec_depth + 1

        head_input = [
            dec[0],
            dec[cut3r.dec_depth * 2 // 4][:, 1:],
            dec[cut3r.dec_depth * 3 // 4][:, 1:],
            dec[cut3r.dec_depth],
        ]
        res = cut3r._downstream_head(head_input, shape[i], pos=pos_i)
        missing = [
            key
            for key in ("pts3d_in_self_view", "pts3d_in_other_view", "camera_pose")
            if key not in res
        ]
        if missing:
            raise RuntimeError(f"CUT3R head output missing keys: {missing}")

        # Coordinate contract for GeoRoPE experiments:
        # point_maps_cam is per-frame camera coordinates; point_maps_ref is the
        # CUT3R reference/anchor-frame coordinates. Train/eval must use the
        # same key for a checkpoint.
        point_maps_cam.append(res["pts3d_in_self_view"])
        point_maps_ref.append(res["pts3d_in_other_view"])
        camera_poses.append(res["camera_pose"])

        img_mask = views[i]["img_mask"]
        update = views[i].get("update", None)
        update_mask = img_mask & update if update is not None else img_mask
        update_mask = update_mask[:, None, None].to(pixel_values.dtype)
        state_feat = new_state_feat * update_mask + state_feat * (1 - update_mask)
        mem = new_mem * update_mask + mem * (1 - update_mask)

        reset_mask = views[i]["reset"]
        if reset_mask is not None:
            reset_mask = reset_mask[:, None, None].to(pixel_values.dtype)
            state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
            mem = init_mem * reset_mask + mem * (1 - reset_mask)

    return {
        "point_maps_cam": torch.stack(point_maps_cam, dim=0).permute(1, 0, 2, 3, 4),
        "point_maps_ref": torch.stack(point_maps_ref, dim=0).permute(1, 0, 2, 3, 4),
        "camera_pose": torch.stack(camera_poses, dim=0).permute(1, 0, 2),
    }


def process_videos_on_gpu(rank, gpu_id, args, video_files_chunk, input_base_dir, output_dir):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    processed_count = 0
    skipped_count = 0
    total_files = len(video_files_chunk)
    batch_size = args.batch_size
    token_dir = Path(args.token_dir) if args.token_dir else None

    rank0_print(f"[GPU {gpu_id}] Worker started, assigned {total_files} files.")

    try:
        if not os.path.exists(args.cut3r_weights_path):
            rank0_print(f"[GPU {gpu_id}] ERROR: CUT3R weights not found: {args.cut3r_weights_path}")
            return 0, total_files

        config = Cut3rSpatialConfig(weights_path=args.cut3r_weights_path)
        cut3r = ARCroco3DStereo.from_pretrained(config.weights_path)
        cut3r.eval()
        for param in cut3r.parameters():
            param.requires_grad = False

        model_dtype = (
            torch.bfloat16
            if args.precision == "bf16"
            else torch.float16
            if args.precision == "fp16"
            else torch.float32
        )
        cut3r.to(device=device, dtype=model_dtype).eval()

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
        rank0_print(f"[GPU {gpu_id}] CUT3R point-map extractor loaded on {device}.")
    except Exception as exc:
        rank0_print(f"[GPU {gpu_id}] Error during initialization: {exc}\n{traceback.format_exc()}")
        return 0, total_files

    data_args = DataArguments(video_fps=args.video_fps, frames_upbound=args.frames_upbound)

    for i in range(0, total_files, batch_size):
        batch_paths = [Path(path) for path in video_files_chunk[i : i + batch_size]]
        batch_data = []
        skipped_in_batch = 0
        max_frames = 0

        for video_path in batch_paths:
            output_path = get_output_path(video_path, input_base_dir, output_dir)
            token_path = None
            if token_dir is not None:
                token_path = get_output_path(video_path, input_base_dir, token_dir)
                if args.require_token_file and not token_path.exists():
                    skipped_in_batch += 1
                    rank0_print(f"[GPU {gpu_id}] Missing token sidecar, skipping: {token_path}")
                    continue

            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists() and not args.overwrite:
                skipped_in_batch += 1
                continue

            preprocessed = load_and_preprocess_video_frames(
                str(video_path), data_args, image_processor, rank=gpu_id
            )
            if preprocessed is None or preprocessed.nelement() == 0:
                skipped_in_batch += 1
                rank0_print(f"[GPU {gpu_id}] Failed to load/preprocess {video_path}. Skipping.")
                continue

            batch_data.append((preprocessed, output_path, video_path, token_path))
            max_frames = max(max_frames, preprocessed.shape[0])

        processed_in_batch = 0
        if batch_data:
            padded_tensors = []
            frame_counts = []
            for preprocessed, _, _, _ in batch_data:
                frame_counts.append(preprocessed.shape[0])
                padding_needed = max_frames - preprocessed.shape[0]
                padded_tensors.append(
                    torch.nn.functional.pad(
                        preprocessed,
                        (0, 0, 0, 0, 0, 0, 0, padding_needed),
                        mode="constant",
                        value=0,
                    )
                )

            try:
                batch_tensor = torch.stack(padded_tensors, dim=0).to(
                    device=device, dtype=model_dtype
                )
                spatial_input = batch_tensor.permute(1, 0, 2, 3, 4)
                outputs = run_cut3r_pointmaps(cut3r, spatial_input)

                for idx, (_, output_path, video_path, token_path) in enumerate(batch_data):
                    try:
                        num_frames = frame_counts[idx]
                        payload = {
                            "point_maps_cam": outputs["point_maps_cam"][idx, :num_frames]
                            .detach()
                            .to(dtype=torch.float16, device="cpu"),
                            "point_maps_ref": outputs["point_maps_ref"][idx, :num_frames]
                            .detach()
                            .to(dtype=torch.float16, device="cpu"),
                            "camera_pose": outputs["camera_pose"][idx, :num_frames]
                            .detach()
                            .to(dtype=torch.float32, device="cpu"),
                            "metadata": {
                                "source_video": str(video_path),
                                "token_feature_path": str(token_path) if token_path else None,
                                "cut3r_weights_path": args.cut3r_weights_path,
                                "num_frames": int(num_frames),
                                "frames_upbound": int(args.frames_upbound),
                                "video_fps": int(args.video_fps),
                                "point_maps_cam_coordinates": "per-frame camera coordinates",
                                "point_maps_ref_coordinates": "CUT3R reference/anchor-frame coordinates",
                                "point_dtype": "float16",
                                "camera_pose_dtype": "float32",
                                "schema": "cut3r_point_maps_sidecar_v1",
                            },
                        }
                        torch.save(payload, output_path)
                        processed_in_batch += 1
                    except Exception as exc:
                        skipped_in_batch += 1
                        rank0_print(
                            f"[GPU {gpu_id}] Error saving point maps for {video_path}: "
                            f"{exc}\n{traceback.format_exc()}"
                        )
                        if output_path.exists():
                            try:
                                output_path.unlink()
                            except OSError:
                                rank0_print(f"Warning: could not remove partial file {output_path}")
            except Exception as exc:
                skipped_in_batch += len(batch_data) - processed_in_batch
                rank0_print(
                    f"[GPU {gpu_id}] Error during batch point-map inference: "
                    f"{exc}\n{traceback.format_exc()}"
                )

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

    parser = argparse.ArgumentParser(description="Extract CUT3R point-map sidecar .pt files.")
    parser.add_argument("--cut3r-weights-path", type=str, required=True)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--input-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--token-dir", type=str, default=None)
    parser.add_argument("--require-token-file", action="store_true")
    parser.add_argument("--processor-config-path", type=str, required=True)
    parser.add_argument("--gpu-ids", type=str, default="0")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
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
    rank0_print(f"Saving point-map sidecars under: {output_dir}")

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
