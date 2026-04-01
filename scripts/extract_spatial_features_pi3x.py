import argparse
import os
import sys # Added for path manipulation
import torch
import torch.nn as nn
import torch.multiprocessing as mp # Added for multiprocessing
import json
import math
from PIL import Image, ImageFile
from pathlib import Path # For path manipulation
import traceback # For error reporting in subprocesses

# Assume necessary imports from llava are available in the PYTHONPATH
# You might need to adjust imports based on your project structure
from llava.utils import process_video_with_decord # Or other video processing functions used
from llava.utils import rank0_print # Use rank0_print for controlled output
#from llava.train.train import DataArguments # Re-use DataArguments for consistency
from dataclasses import dataclass
from typing import Optional
from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor # Added direct import

# Add imports for direct spatial tower loading
from llava.model.multimodal_spatial_encoder.cut3r_spatial_encoder import Cut3rSpatialConfig, Cut3rEncoder
from llava.model.multimodal_spatial_encoder.pi3x_spatial_encoder import Pi3xSpatialConfig, Pi3xEncoder

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Constants ---
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv'] # Add more if needed

# --- Image/Video Loading and Preprocessing ---
# (Adapted from LazySupervisedDataset and encode_spatial_features)
@dataclass
class DataArguments:
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = None
    image_crop_resolution: Optional[int] = None
    image_split_resolution: Optional[int] = None
    video_fps: int = 1
    frames_upbound: int = 32
    force_sample: bool = False

def load_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        rank0_print(f"Error loading image {image_path}: {e}") # Use rank0_print
        return None

def preprocess_image_for_spatial(image, processor, target_size=(432, 432)):
    """Preprocesses a single PIL image for spatial feature extraction using the provided processor."""
    # Process using the vision model's processor
    try:
        processed_output = processor(images=image, return_tensors='pt')
        image_tensor = processed_output['pixel_values'].squeeze(0) # Shape (C, H_proc, W_proc)
    except Exception as e:
        rank0_print(f"Error processing image with processor: {e}") # Use rank0_print
        return None

    # Resize to the target size expected by the spatial tower (if different)
    c, h_proc, w_proc = image_tensor.shape
    if h_proc != target_size[0] or w_proc != target_size[1]:
        image_scaled = nn.functional.interpolate(
            image_tensor.unsqueeze(0), # Add batch dim for interpolate
            size=target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0) # Remove batch dim
    else:
        image_scaled = image_tensor

    # Return shape (1, C, H_target, W_target) - batch dim added later before spatial tower
    return image_scaled.unsqueeze(0)

    # Old manual preprocessing:
    # image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    # image_scaled = nn.functional.interpolate(image_tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
    # return image_scaled # Output shape (1, C, H, W)

def load_and_preprocess_video_frames(video_path, data_args, processor, target_size=None, rank=0):
    """
    Loads video, samples frames, preprocesses them using the processor,
    and returns a tensor of shape (F, C, H_target, W_target).
    Includes rank for logging.
    """
    try:
        # Use the same video processing function as in training
        video_frames, video_time, frame_time, num_frames_to_sample = process_video_with_decord(video_path, data_args)

        # Preprocess all frames together using the processor
        try:
            # Ensure video_frames is a list of PIL Images or compatible format
            processed_output = processor.preprocess(images=video_frames, return_tensors="pt")
            frames_tensor = processed_output['pixel_values'] # Shape (F, C, H_proc, W_proc)
        except Exception as e:
            rank0_print(f"[GPU {rank}] Error processing video frames with processor for {video_path}: {e}")
            return None

        # Resize frames to the target size if requested
        f, c, h_proc, w_proc = frames_tensor.shape
        if target_size is not None and (h_proc != target_size[0] or w_proc != target_size[1]):
            # Interpolate expects (B, C, H, W) or (C, H, W)
            # Process frame by frame or reshape if possible, let's do frame by frame for simplicity
            frames_scaled_list = []
            for i in range(f):
                frame_scaled = nn.functional.interpolate(
                    frames_tensor[i].unsqueeze(0), # Add batch dim
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0) # Remove batch dim
                frames_scaled_list.append(frame_scaled)
            frames_scaled = torch.stack(frames_scaled_list) # Shape (F, C, H_target, W_target)
        else:
            frames_scaled = frames_tensor

        # Return shape (F, C, H_target, W_target) - Batch dim added later during batching
        return frames_scaled

    except Exception as e:
        rank0_print(f"[GPU {rank}] Error processing video {video_path}: {e}")
        return None

# --- Main Extraction Logic ---

def find_video_files(input_dir):
    """Recursively finds all video files in the input directory."""
    video_files = []
    rank0_print(f"Scanning for video files in {input_dir}...")
    for ext in VIDEO_EXTENSIONS:
        # Using Path for easier relative path calculation
        input_path = Path(input_dir)
        # **/*{ext} finds files in input_dir and subdirectories
        video_files.extend(input_path.rglob(f"*{ext}"))
        # Case-insensitive search for extensions
        video_files.extend(input_path.rglob(f"*{ext.upper()}"))

    # Remove duplicates if extensions overlap (e.g., .MP4 and .mp4)
    unique_video_files = sorted(list(set(video_files)))
    rank0_print(f"Found {len(unique_video_files)} potential video files.")
    return [str(f) for f in unique_video_files] # Return as strings

def get_output_path(input_file_path: Path, input_base_dir: Path, output_base_dir: Path) -> Path:
    """Calculates the output path for a given input file."""
    try:
        relative_path = input_file_path.relative_to(input_base_dir)
    except ValueError:
        rank0_print(f"Warning: Input file {input_file_path} is not within the specified base input directory {input_base_dir}. Outputting directly to {output_base_dir}.")
        relative_path = input_file_path.name # Use only the filename

    # Change extension to .pt
    output_filename_path = output_base_dir / relative_path.with_suffix('.pt')
    return output_filename_path

def get_point_cloud_output_path(input_file_path: Path, input_base_dir: Path, point_cloud_output_base_dir: Path) -> Path:
    """Calculates the output path for the point cloud corresponding to an input file."""
    try:
        relative_path = input_file_path.relative_to(input_base_dir)
    except ValueError:
        rank0_print(f"Warning: Input file {input_file_path} is not within the specified base input directory {input_base_dir}. Outputting point cloud directly to {point_cloud_output_base_dir}.")
        relative_path = input_file_path.name # Use only the filename

    # Change extension to .ply
    output_filename_path = point_cloud_output_base_dir / relative_path.with_suffix('.ply')
    return output_filename_path

def get_preprocessed_frames_output_dir(input_file_path: Path, input_base_dir: Path, frames_output_base_dir: Path) -> Path:
    """Calculates the output directory for the preprocessed frames corresponding to an input file."""
    try:
        relative_path = input_file_path.relative_to(input_base_dir)
    except ValueError:
        rank0_print(f"Warning: Input file {input_file_path} is not within the specified base input directory {input_base_dir}. Outputting frames directly under {frames_output_base_dir}.")
        relative_path = input_file_path.stem # Use only the filename stem (no extension)

    # Create a directory based on the relative path (without extension)
    output_dir_path = frames_output_base_dir / relative_path.parent / relative_path.stem
    return output_dir_path

def get_sibling_preprocessed_frames_output_dir(feature_output_path: Path) -> Path:
    """Creates a sidecar folder next to feature .pt/.pth file."""
    return feature_output_path.parent / f"{feature_output_path.stem}_preprocessed_frames"

# --- Worker Process Function ---
def process_videos_on_gpu(rank, gpu_id, args, video_files_chunk, input_base_dir, output_dir):
    """
    Worker function executed by each process to handle a chunk of videos on a specific GPU.
    """
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device) # Ensure this process uses the assigned GPU

    processed_count = 0
    skipped_count = 0
    total_files_in_chunk = len(video_files_chunk)
    batch_size = args.batch_size

    rank0_print(f"[GPU {gpu_id}] Worker started, assigned {total_files_in_chunk} files.")

    # -- Load Model Components (within each worker process) --
    try:
        if args.spatial_encoder == "cut3r":
            rank0_print(f"[GPU {gpu_id}] Loading CUT3R Spatial Tower...")
            if not args.cut3r_weights_path or not os.path.exists(args.cut3r_weights_path):
                rank0_print(f"[GPU {gpu_id}] ERROR: CUT3R weights file not found at: {args.cut3r_weights_path}")
                return 0, total_files_in_chunk

            cut3r_config = Cut3rSpatialConfig(
                weights_path=args.cut3r_weights_path,
                export_point_cloud=args.export_point_cloud,
                point_cloud_output_dir=args.point_cloud_output_dir,
                point_cloud_voxel_size=args.point_cloud_voxel_size
            )
            spatial_tower = Cut3rEncoder(config=cut3r_config)
            target_size = (432, 432)
            rank0_print(f"[GPU {gpu_id}] CUT3R Spatial Tower initialized.")
        elif args.spatial_encoder == "pi3x":
            rank0_print(f"[GPU {gpu_id}] Loading PI3X Spatial Tower...")
            if not args.pi3x_weights_path or not os.path.exists(args.pi3x_weights_path):
                rank0_print(f"[GPU {gpu_id}] ERROR: PI3X weights path not found: {args.pi3x_weights_path}")
                return 0, total_files_in_chunk

            pi3x_config = Pi3xSpatialConfig(
                weights_path=args.pi3x_weights_path,
                input_size=args.pi3x_input_size,
            )
            spatial_tower = Pi3xEncoder(config=pi3x_config)
            # PI3X tower handles internal resize to input_size.
            target_size = None
            if args.export_point_cloud:
                rank0_print(f"[GPU {gpu_id}] Warning: --export-point-cloud is CUT3R-only and will be ignored for PI3X.")
            rank0_print(f"[GPU {gpu_id}] PI3X Spatial Tower initialized.")
        else:
            rank0_print(f"[GPU {gpu_id}] ERROR: Unsupported spatial encoder '{args.spatial_encoder}'.")
            return 0, total_files_in_chunk

        model_dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float16 if args.precision == 'fp16' else torch.float32
        spatial_tower.to(device=device, dtype=model_dtype).eval()
        rank0_print(f"[GPU {gpu_id}] Spatial tower loaded on {device}.")

        # -- Load Image Processor Config and Initialize Processor --
        rank0_print(f"[GPU {gpu_id}] Loading Processor config from {args.processor_config_path}...")
        with open(args.processor_config_path, 'r') as f:
            processor_config = json.load(f)

        image_mean_list = processor_config.get("image_mean", (0.5, 0.5, 0.5))
        image_std_list = processor_config.get("image_std", (0.5, 0.5, 0.5))
        size_config = processor_config.get("size", {"height": 384, "width": 384})
        size_tuple = (size_config["height"], size_config["width"])
        resample_val = processor_config.get("resample", 3)
        rescale_factor_val = processor_config.get("rescale_factor", 1/255.0)

        image_processor = SigLipImageProcessor(
            image_mean=image_mean_list,
            image_std=image_std_list,
            size=size_tuple,
            resample=resample_val,
            rescale_factor=rescale_factor_val
        )
        rank0_print(f"[GPU {gpu_id}] SigLipImageProcessor initialized.")

        # Convert mean/std to tensors for de-normalization if saving frames
        if args.save_preprocessed_frames:
            mean_tensor = torch.tensor(image_mean_list).view(3, 1, 1)
            std_tensor = torch.tensor(image_std_list).view(3, 1, 1)
        else:
            mean_tensor, std_tensor = None, None # Avoid unused variables

    except Exception as e:
        rank0_print(f"[GPU {gpu_id}] Error during initialization: {e}\n{traceback.format_exc()}")
        return 0, total_files_in_chunk # Indicate all files skipped due to init error

    # --- Load Dummy DataArguments ---
    data_args = DataArguments(
        video_fps=args.video_fps,
        frames_upbound=args.frames_upbound,
        force_sample=True
    )

    # --- Feature Extraction Loop for the Chunk (Batched Processing) ---
    # Use tqdm only on rank 0 or manage it externally if needed
    # Simple loop without per-process tqdm for now
    for i in range(0, total_files_in_chunk, batch_size):
        batch_paths_str = video_files_chunk[i:min(i + batch_size, total_files_in_chunk)]
        batch_paths = [Path(p) for p in batch_paths_str]

        # --- Batch Preprocessing ---
        batch_data_to_process = [] # Stores (preprocessed_input, feature_output_filename, point_cloud_output_filename, video_full_path)
        files_in_batch = 0
        skipped_in_batch = 0
        max_frames_in_batch = 0
        use_point_cloud = args.spatial_encoder == "cut3r" and args.export_point_cloud
        point_cloud_base_dir = Path(args.point_cloud_output_dir) if use_point_cloud else None
        frames_base_dir = Path(args.save_preprocessed_frames_dir) if args.save_preprocessed_frames else None

        for video_full_path in batch_paths:
            files_in_batch += 1
            feature_output_filename_path = get_output_path(video_full_path, input_base_dir, output_dir)
            feature_output_filename = str(feature_output_filename_path)

            point_cloud_output_filename = None
            if use_point_cloud and point_cloud_base_dir:
                point_cloud_output_filename_path = get_point_cloud_output_path(video_full_path, input_base_dir, point_cloud_base_dir)
                point_cloud_output_filename = str(point_cloud_output_filename_path)
                # Ensure parent dir for point cloud exists
                point_cloud_output_filename_path.parent.mkdir(parents=True, exist_ok=True)


            # Check feature file existence
            feature_output_filename_path.parent.mkdir(parents=True, exist_ok=True)
            feature_exists = os.path.exists(feature_output_filename)

            # Check point cloud existence (if exporting)
            point_cloud_exists = False
            if use_point_cloud and point_cloud_output_filename:
                point_cloud_exists = os.path.exists(point_cloud_output_filename)

            # Determine if skipping is needed
            # Skip if NOT overwriting AND (feature exists AND (point cloud exists OR not exporting point cloud))
            should_skip = not args.overwrite and (feature_exists and (point_cloud_exists or not use_point_cloud))

            if should_skip:
                skipped_in_batch += 1
                continue


            preprocessed_input = load_and_preprocess_video_frames(
                str(video_full_path),
                data_args,
                image_processor,
                target_size=target_size,
                rank=gpu_id,
            )

            if preprocessed_input is not None and preprocessed_input.nelement() > 0:
                # Store point cloud path only if exporting
                pc_path_to_store = point_cloud_output_filename if args.export_point_cloud else None

                # --- Save Preprocessed Frames (if enabled) ---
                if args.save_preprocessed_frames and frames_base_dir is not None and mean_tensor is not None and std_tensor is not None:
                    try:
                        if args.save_preprocessed_frames_layout == "sibling":
                            frames_output_dir_path = get_sibling_preprocessed_frames_output_dir(feature_output_filename_path)
                        else:
                            frames_output_dir_path = get_preprocessed_frames_output_dir(video_full_path, input_base_dir, frames_base_dir)
                        frames_output_dir_path.mkdir(parents=True, exist_ok=True)

                        # De-normalize and save each frame
                        # Move tensor to CPU for processing/saving if it isn't already
                        # Note: preprocessed_input is on CPU from load_and_preprocess...
                        preprocessed_input_float = preprocessed_input.float() # Ensure float32 for calculation
                        for frame_idx in range(preprocessed_input_float.shape[0]):
                            frame_tensor = preprocessed_input_float[frame_idx] # Shape (C, H, W)

                            # De-normalize: (tensor * std + mean) / rescale_factor
                            # Assuming rescale_factor = 1/255.0
                            # (tensor * std + mean) * 255.0
                            # Move mean/std to the frame's device (likely CPU here)
                            denormalized_frame = (frame_tensor * std_tensor.to(frame_tensor.device) + mean_tensor.to(frame_tensor.device)) / rescale_factor_val

                            # Clamp and convert to uint8
                            denormalized_frame = torch.clamp(denormalized_frame, 0, 255).to(torch.uint8)

                            # Convert to PIL Image (C, H, W) -> (H, W, C)
                            pil_image = Image.fromarray(denormalized_frame.permute(1, 2, 0).cpu().numpy())

                            # Construct filename
                            frame_filename = frames_output_dir_path / f"frame_{frame_idx:04d}.png"

                            # Save the image
                            pil_image.save(frame_filename)

                    except Exception as e_save_frame:
                        rank0_print(f"[GPU {gpu_id}] Error saving preprocessed frame for {video_full_path}, frame {frame_idx}: {e_save_frame}")
                        # Continue processing other frames/videos

                # Append data for batch inference
                batch_data_to_process.append((preprocessed_input, feature_output_filename, pc_path_to_store, video_full_path))
                max_frames_in_batch = max(max_frames_in_batch, preprocessed_input.shape[0])
            else:
                rank0_print(f"[GPU {gpu_id}] Failed to load/preprocess {video_full_path}. Skipping.")
                skipped_in_batch += 1

        # --- Batch Inference ---
        processed_in_batch = 0
        if batch_data_to_process:
            if args.spatial_encoder == "cut3r":
                padded_tensors = []
                output_info = []

                for preprocessed_input, feature_output_filename, point_cloud_output_filename, video_full_path in batch_data_to_process:
                    num_frames = preprocessed_input.shape[0]
                    padding_needed = max_frames_in_batch - num_frames
                    padded_tensor = torch.nn.functional.pad(
                        preprocessed_input,
                        (0, 0, 0, 0, 0, 0, 0, padding_needed),
                        mode='constant',
                        value=0
                    )
                    padded_tensors.append(padded_tensor)
                    output_info.append((feature_output_filename, point_cloud_output_filename, video_full_path))

                batch_tensor = torch.stack(padded_tensors, dim=0).to(device=device, dtype=model_dtype)
                point_cloud_paths_for_batch = [info[1] for info in output_info] if use_point_cloud else None

                B = batch_tensor.shape[0]
                F_max = batch_tensor.shape[1]

                try:
                    with torch.no_grad():
                        spatial_input = batch_tensor.permute(1, 0, 2, 3, 4)
                        camera_tokens_batch, patch_tokens_batch = spatial_tower(
                            spatial_input,
                            point_cloud_output_paths=point_cloud_paths_for_batch
                        )

                    expected_output_len = B * F_max
                    if camera_tokens_batch.shape[0] != expected_output_len:
                        rank0_print(
                            f"[GPU {gpu_id}] Warning: Output dimension mismatch. "
                            f"Expected first dim {expected_output_len} but got {camera_tokens_batch.shape[0]}. "
                            f"camera_tokens={tuple(camera_tokens_batch.shape)}, "
                            f"patch_tokens={tuple(patch_tokens_batch.shape)}. Skipping save for this batch."
                        )
                        skipped_in_batch += B
                    else:
                        _, token_num_cam, token_dim_cam = camera_tokens_batch.shape
                        _, token_num_patch, token_dim_patch = patch_tokens_batch.shape

                        camera_tokens_reshaped = camera_tokens_batch.view(B, F_max, token_num_cam, token_dim_cam)
                        patch_tokens_reshaped = patch_tokens_batch.view(B, F_max, token_num_patch, token_dim_patch)

                        for idx, (feature_output_filename, _point_cloud_output_filename, video_full_path) in enumerate(output_info):
                            try:
                                features_to_save = {
                                    "camera_tokens": camera_tokens_reshaped[idx].cpu(),
                                    "patch_tokens": patch_tokens_reshaped[idx].cpu()
                                }
                                torch.save(features_to_save, feature_output_filename)
                                processed_in_batch += 1

                            except Exception as e_save:
                                rank0_print(
                                    f"[GPU {gpu_id}] Error saving features for {video_full_path} "
                                    f"(point cloud saving handled separately): {e_save}"
                                )
                                skipped_in_batch += 1

                                if os.path.exists(feature_output_filename):
                                    try:
                                        os.remove(feature_output_filename)
                                    except OSError:
                                        rank0_print(
                                            f"Warning: Could not remove potentially corrupted file {feature_output_filename}"
                                        )

                except Exception as e_batch:
                    rank0_print(f"[GPU {gpu_id}] Error during CUT3R batch inference: {e_batch}\n{traceback.format_exc()}")
                    skipped_in_batch += len(batch_data_to_process) - processed_in_batch

            elif args.spatial_encoder == "pi3x":
                for preprocessed_input, feature_output_filename, _point_cloud_output_filename, video_full_path in batch_data_to_process:
                    try:
                        with torch.no_grad():
                            camera_tokens, patch_tokens = spatial_tower(
                                preprocessed_input.to(device=device, dtype=model_dtype)
                            )
                        features_to_save = {
                            "camera_tokens": camera_tokens.cpu(),
                            "patch_tokens": patch_tokens.cpu()
                        }
                        torch.save(features_to_save, feature_output_filename)
                        processed_in_batch += 1
                    except Exception as e_pi3x:
                        rank0_print(f"[GPU {gpu_id}] Error during PI3X inference/save for {video_full_path}: {e_pi3x}\n{traceback.format_exc()}")
                        skipped_in_batch += 1
                        if os.path.exists(feature_output_filename):
                            try:
                                os.remove(feature_output_filename)
                            except OSError:
                                rank0_print(f"Warning: Could not remove potentially corrupted file {feature_output_filename}")

        processed_count += processed_in_batch
        skipped_count += skipped_in_batch

        # Optional: Print progress per worker periodically
        if (i // batch_size) % 10 == 0:
            rank0_print(
                f"[GPU {gpu_id}] Progress: Batch {i//batch_size+1}/{math.ceil(total_files_in_chunk/batch_size)}, "
                f"Processed: {processed_count}, Skipped: {skipped_count}"
            )


    rank0_print(f"[GPU {gpu_id}] Worker finished. Processed: {processed_count}, Skipped: {skipped_count}")
    return processed_count, skipped_count


# --- Main Execution ---
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True) # Recommended for CUDA with multiprocessing

    parser = argparse.ArgumentParser(description="Extract Spatial Features using CUT3R or PI3X (Multi-GPU)")
    parser.add_argument("--spatial-encoder", type=str, default="cut3r", choices=["cut3r", "pi3x"], help="Spatial encoder backend to use")
    parser.add_argument("--cut3r-weights-path", type=str, default=None, help="Path to the CUT3R weights file (required when --spatial-encoder cut3r)")
    parser.add_argument("--pi3x-weights-path", type=str, default=None, help="Path to PI3X weights/checkpoint directory (required when --spatial-encoder pi3x)")
    parser.add_argument("--pi3x-input-size", type=int, default=518, help="Input size used internally by PI3X (must be multiple of 14)")
    parser.add_argument("--input-dir", type=str, default=None, help="Root directory containing video files to process recursively (ignored if --input-file is provided)")
    parser.add_argument("--input-file", type=str, default=None, help="Path to a single video file to process")
    parser.add_argument("--output-dir", type=str, required=True, help="Root directory to save the extracted features, mirroring input structure")
    parser.add_argument("--processor-config-path", type=str, required=True, help="Path to the processor_config.json file for SigLipImageProcessor")
    # parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., 'cuda:0', 'cpu')") # Replaced by gpu-ids
    parser.add_argument("--gpu-ids", type=str, default="0", help="Comma-separated list of GPU IDs to use (e.g., '0,1,2') or 'all'")
    parser.add_argument("--precision", type=str, default="fp16", choices=['fp16', 'bf16', 'fp32'], help="Computation precision")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing feature files")
    parser.add_argument("--video-fps", type=int, default=1, help="FPS for video frame sampling")
    parser.add_argument("--frames-upbound", type=int, default=10, help="Max frames to sample per video")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of videos to process per batch *per GPU*") # Clarified batch size scope
    parser.add_argument("--export-point-cloud", action="store_true", help="Export generated point cloud for each video")
    parser.add_argument("--point-cloud-output-dir", type=str, default="point_clouds", help="Directory to save exported point clouds (used if --export-point-cloud is set)")
    parser.add_argument("--point-cloud-voxel-size", type=float, default=0.01, help="Voxel size for downsampling exported point clouds")
    parser.add_argument("--save-preprocessed-frames", action="store_true", help="Save the preprocessed frames used for feature extraction as images")
    parser.add_argument("--save-preprocessed-frames-dir", type=str, default="preprocessed_frames", help="Directory to save preprocessed frames (used if --save-preprocessed-frames is set)")
    parser.add_argument("--save-preprocessed-frames-layout", type=str, default="global", choices=["global", "sibling"], help="'global': mirror structure under --save-preprocessed-frames-dir, 'sibling': save under folders next to each extracted .pt")

    args = parser.parse_args()

    # --- Validate Input Arguments ---
    if not args.input_dir and not args.input_file:
        parser.error("Either --input-dir or --input-file must be specified.")
    if args.input_dir and args.input_file:
        rank0_print("Warning: Both --input-dir and --input-file provided. --input-file will be used.")
        args.input_dir = None # Prioritize input_file

    if args.spatial_encoder == "cut3r":
        if not args.cut3r_weights_path:
            parser.error("--cut3r-weights-path is required when --spatial-encoder cut3r")
    elif args.spatial_encoder == "pi3x":
        if not args.pi3x_weights_path:
            parser.error("--pi3x-weights-path is required when --spatial-encoder pi3x")
        if args.pi3x_input_size % 14 != 0:
            parser.error("--pi3x-input-size must be a multiple of 14 for PI3X patching")

    # --- Determine GPUs to Use ---
    if args.gpu_ids.lower() == 'all':
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpu_ids = list(range(num_gpus))
            if num_gpus == 0:
                 rank0_print("Error: 'all' GPUs requested, but no CUDA devices found.")
                 sys.exit(1)
            rank0_print(f"Using all available GPUs: {gpu_ids}")
        else:
            rank0_print("Error: 'all' GPUs requested, but CUDA is not available.")
            sys.exit(1)
    else:
        try:
            gpu_ids = [int(id.strip()) for id in args.gpu_ids.split(',')]
            # Validate GPU IDs
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                for gpu_id in gpu_ids:
                    if gpu_id < 0 or gpu_id >= num_gpus:
                        rank0_print(f"Error: Invalid GPU ID {gpu_id}. Available GPUs: {list(range(num_gpus))}")
                        sys.exit(1)
            else:
                 rank0_print(f"Warning: Specified GPU IDs {gpu_ids}, but CUDA is not available. Will attempt CPU if possible (not recommended/tested).")
                 # Or exit if CPU processing is not intended/supported
                 # sys.exit(1)
            rank0_print(f"Using specified GPUs: {gpu_ids}")
        except ValueError:
            rank0_print(f"Error: Invalid format for --gpu-ids. Expected comma-separated integers (e.g., '0,1,2') or 'all'. Got: {args.gpu_ids}")
            sys.exit(1)

    if not gpu_ids:
        rank0_print("Error: No valid GPUs specified or found.")
        sys.exit(1)

    num_workers = len(gpu_ids)

    # --- Prepare Output Directory ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) # Use Path.mkdir
    if args.export_point_cloud and args.spatial_encoder == "cut3r":
        pc_output_dir = Path(args.point_cloud_output_dir)
        pc_output_dir.mkdir(parents=True, exist_ok=True)
        rank0_print(f"Point clouds will be saved to: {args.point_cloud_output_dir}")
    elif args.export_point_cloud and args.spatial_encoder != "cut3r":
        rank0_print(f"Warning: --export-point-cloud is supported only for CUT3R. Requested encoder: {args.spatial_encoder}")
    if args.save_preprocessed_frames:
        if args.save_preprocessed_frames_layout == "global":
            frames_output_dir = Path(args.save_preprocessed_frames_dir)
            frames_output_dir.mkdir(parents=True, exist_ok=True)
            rank0_print(f"Preprocessed frames will be saved to: {args.save_preprocessed_frames_dir}")
        else:
            rank0_print("Preprocessed frames will be saved in sibling folders next to each extracted feature file.")

    # --- Determine Input Files ---
    if args.input_file:
        input_file_path = Path(args.input_file)
        if not input_file_path.is_file():
            rank0_print(f"Error: Provided --input-file '{args.input_file}' not found or is not a file.")
            sys.exit(1)
        all_video_files_paths = [input_file_path]
        input_base_dir = Path(args.input_dir) if args.input_dir and input_file_path.is_relative_to(args.input_dir) else input_file_path.parent
        rank0_print(f"Processing single input file: {args.input_file}")
    elif args.input_dir:
        input_base_dir = Path(args.input_dir)
        # Find files needs Path objects
        all_video_files_paths_str = find_video_files(args.input_dir)
        all_video_files_paths = [Path(p) for p in all_video_files_paths_str] # Keep as Path objects
        if not all_video_files_paths:
            rank0_print(f"No video files found in {args.input_dir}. Exiting.")
            sys.exit(0)
        rank0_print(f"Found {len(all_video_files_paths)} video files in {args.input_dir}.")
    else:
        # This case should be caught by parser.error earlier
        rank0_print("Error: No input specified.")
        sys.exit(1)

    # --- Distribute Files Among Workers ---
    files_per_worker = [[] for _ in range(num_workers)]
    for i, file_path in enumerate(all_video_files_paths):
        worker_index = i % num_workers
        # Pass file paths as strings to worker processes
        files_per_worker[worker_index].append(str(file_path))

    # --- Launch Worker Processes ---
    rank0_print(f"Starting feature extraction with {num_workers} worker process(es) on GPUs {gpu_ids}...")
    pool_args = []
    for rank, gpu_id in enumerate(gpu_ids):
        pool_args.append(
            (rank, gpu_id, args, files_per_worker[rank], input_base_dir, output_dir)
        )

    total_processed = 0
    total_skipped = 0
    try:
        with mp.Pool(processes=num_workers) as pool:
            # Use starmap to pass multiple arguments to the worker function
            results = pool.starmap(process_videos_on_gpu, pool_args)

        # Aggregate results
        for processed, skipped in results:
            total_processed += processed
            total_skipped += skipped

    except Exception as e:
         rank0_print(f"\n--- An error occurred during multiprocessing ---")
         rank0_print(f"Error: {e}")
         rank0_print(traceback.format_exc())
         rank0_print("Feature extraction may be incomplete.")
         # Ensure totals reflect potentially partial completion if some processes finished
         if 'results' in locals():
              for processed, skipped in results:
                   total_processed += processed
                   total_skipped += skipped
         else: # If pool creation failed or starmap didn't even start
              total_skipped = len(all_video_files_paths)


    # --- Final Summary ---
    rank0_print("-" * 30)
    rank0_print(f"Feature extraction complete.")
    rank0_print(f"Successfully processed: {total_processed}")
    rank0_print(f"Skipped (exists or error): {total_skipped}")
    rank0_print(f"Total files considered: {len(all_video_files_paths)}")
    rank0_print(f"Features saved in: {args.output_dir}")
