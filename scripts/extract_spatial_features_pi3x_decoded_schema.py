"""
[中文說明]

本程式用來為每支影片預先抽取 Pi3X 的每幀 token 級 decoded 特徵，供後續多模態訓練或分析重複使用，
避免每次訓練都重新執行 Pi3X spatial encoder。

1) 作用
- 讀取影片。
- 使用與訓練一致的 decord 抽樣規則選取影片幀。
- 將抽到的幀交給 SigLipImageProcessor 做影像前處理。
- 將前處理後的幀送入 Pi3X encoder，並直接取得 decode() 輸出的 decoded_features 與 decoded_pos。
- 將每幀 decoded token 特徵與影片層級 metadata 存成一個 .pt 檔。

2) 取樣方法
- 影片抽幀規則與訓練中的 process_video_with_decord(...) 保持一致：
  - 先依 video_fps 計算步長。
  - 先做固定步長抽樣。
  - 若 frames_upbound > 0 且抽樣數超過上限，或 force_sample=True，則改為在整支影片上做 uniform sampling。
- 本檔案會額外保存 frame_idx，表示「原影片中實際被抽到的 frame indices」。
  例如 frame_idx=[0, 30, 60, 90] 代表存下來的四個 frame 分別來自原影片第 0/30/60/90 幀。

3) 前處理與資料流
- 抽樣後得到的 video_frames 會直接送進 SigLipImageProcessor.preprocess(images=video_frames, ...)，
  產生形狀為 (F, C, H_proc, W_proc) 的 tensor。
- 若 target_size 與 processor 輸出尺寸不同，會再逐幀 resize，得到 (F, C, H_target, W_target)。
- 這個 tensor 會直接作為 Pi3X encoder 的輸入，因此 Pi3X 看到的幀順序與 SigLip preprocess 的幀順序一致。
- Pi3X 內部會再把輸入 resize 到 input_size，並將數值從 [-1, 1] 轉回 [0, 1]，然後執行 encode + decode。

4) 資料維度
- F: 抽樣後的幀數
- T: 每幀的 token 數 = special/register tokens + patch tokens
- C: token feature dimension（例如 Pi3X large decoder 常見為 2048）
- decoded_features: shape = (F, T, C)
- frame_idx: shape = (F,)
- decoded_pos_template: shape = (T, 2)

5) 輸出格式
每支影片儲存成一個 .pt 檔，內容如下：

{
  "frames": {
    "decoded_features": Tensor[F, T, C],
    "frame_idx": Tensor[F]
  },
  "meta": {
    "decoded_pos_template": Tensor[T, 2],
    "num_frames": int,
    "input_size": int,
    "patch_size": int,
    "patch_start_idx": int
  }
}

6) 設計意圖
- decoded_features 保留了每幀完整 token 表示，未來可以自由切出 special tokens、patch tokens，
  或重新定義 camera token，而不必重新跑 Pi3X。
- frame_idx 保留了這些特徵對應到原影片哪幾個 frame，方便未來和 2D branch、標註、時間資訊對齊。
- decoded_pos_template 保留了 token 對應的 2D 位置模板，方便未來重建或分析 decoder 輸出。
- metadata 只存每支影片共用資訊，以節省空間。

[English Description]

This script pre-extracts Pi3X per-frame token-level decoded features for each video so downstream
training and analysis can reuse them without re-running the Pi3X spatial encoder every time.

1) Purpose
- Read a video.
- Sample frames using the same decord-based rule used by training.
- Preprocess the sampled frames with SigLipImageProcessor.
- Feed the preprocessed frames into the Pi3X encoder and directly obtain decode() outputs:
  decoded_features and decoded_pos.
- Save per-frame decoded token features together with video-level metadata into a single .pt file.

2) Sampling method
- The frame sampling rule is kept consistent with training-time process_video_with_decord(...):
  - first do fixed-step sampling based on video_fps,
  - then, if frames_upbound > 0 and the sampled count exceeds the bound, or force_sample=True,
    switch to uniform sampling across the whole video.
- This script additionally saves frame_idx, which stores the actual frame indices selected from the original video.
  For example, frame_idx=[0, 30, 60, 90] means the saved four frames come from original video frames 0/30/60/90.

3) Preprocessing and data flow
- The sampled video_frames are passed directly to
  SigLipImageProcessor.preprocess(images=video_frames, ...),
  producing a tensor of shape (F, C, H_proc, W_proc).
- If target_size differs from the processor output size, frames are resized again to
  (F, C, H_target, W_target).
- This tensor is then fed directly into the Pi3X encoder, so the frame order seen by Pi3X is the
  same as the frame order seen by the SigLip preprocessing step.
- Inside Pi3X, the input is resized to input_size, converted from [-1, 1] back to [0, 1],
  and then processed by encode + decode.

4) Tensor dimensions
- F: number of sampled frames
- T: number of tokens per frame = special/register tokens + patch tokens
- C: token feature dimension (for example, often 2048 for a large Pi3X decoder)
- decoded_features: shape = (F, T, C)
- frame_idx: shape = (F,)
- decoded_pos_template: shape = (T, 2)

5) Output format
One .pt file is saved per video, with the following structure:

{
  "frames": {
    "decoded_features": Tensor[F, T, C],
    "frame_idx": Tensor[F]
  },
  "meta": {
    "decoded_pos_template": Tensor[T, 2],
    "num_frames": int,
    "input_size": int,
    "patch_size": int,
    "patch_start_idx": int
  }
}

6) Why this format
- decoded_features preserves the full per-frame token representation, allowing future experiments
  to redefine camera tokens, special tokens, or patch token usage without re-running Pi3X.
- frame_idx preserves which original video frames these features correspond to, making it easier to
  align them with the 2D branch, annotations, or time-based metadata.
- decoded_pos_template preserves token position layout for future decoder reconstruction or analysis.
- Shared metadata is stored once per video to save disk space.
"""

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
from llava.model.multimodal_spatial_encoder.pi3x_spatial_encoder import Pi3xSpatialConfig, Pi3xEncoder

try:
    from decord import VideoReader, cpu
except Exception:
    VideoReader = None
    cpu = None

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

def compute_frame_idx_with_decord(video_path, data_args):
    """Compute sampled frame indices using the same rule as training-time process_video_with_decord."""
    if VideoReader is None or cpu is None:
        raise ImportError("decord is required to compute sampled frame indices")

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    avg_fps = round(vr.get_avg_fps() / data_args.video_fps)
    if avg_fps <= 0:
        avg_fps = 1
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]

    if data_args.frames_upbound > 0:
        if len(frame_idx) > data_args.frames_upbound or data_args.force_sample:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()

    vr.seek(0)
    return frame_idx


def load_and_preprocess_video_frames(video_path, data_args, processor, target_size=None, rank=0):
    """
    Loads video, samples frames, preprocesses them using the processor,
    and returns:
      - frames tensor of shape (F, C, H_target, W_target)
      - sampled frame indices in the original video as a LongTensor of shape (F,)
    Includes rank for logging.
    """
    try:
        sampled_frame_idx = compute_frame_idx_with_decord(video_path, data_args)

        # Use the same video processing function as in training
        video_frames, video_time, frame_time, num_frames_to_sample = process_video_with_decord(video_path, data_args)

        # Preprocess all frames together using the processor
        try:
            processed_output = processor.preprocess(images=video_frames, return_tensors="pt")
            frames_tensor = processed_output['pixel_values'] # Shape (F, C, H_proc, W_proc)
        except Exception as e:
            rank0_print(f"[GPU {rank}] Error processing video frames with processor for {video_path}: {e}")
            return None

        f, c, h_proc, w_proc = frames_tensor.shape
        if len(sampled_frame_idx) != f:
            rank0_print(
                f"[GPU {rank}] Warning: sampled_frame_idx length ({len(sampled_frame_idx)}) does not match preprocessed frame count ({f}) for {video_path}."
            )

        if target_size is not None and (h_proc != target_size[0] or w_proc != target_size[1]):
            frames_scaled_list = []
            for i in range(f):
                frame_scaled = nn.functional.interpolate(
                    frames_tensor[i].unsqueeze(0),
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                frames_scaled_list.append(frame_scaled)
            frames_scaled = torch.stack(frames_scaled_list)
        else:
            frames_scaled = frames_tensor

        return {
            "frames_tensor": frames_scaled,
            "frame_idx": torch.tensor(sampled_frame_idx, dtype=torch.long),
        }

    except Exception as e:
        rank0_print(f"[GPU {rank}] Error processing video {video_path}: {e}")
        return None

def extract_decoded_features_and_pos(spatial_tower, pixel_values):
    """Run Pi3X through decode() and return decoded features and positions."""
    input_size = spatial_tower.config.input_size
    pixel_values = nn.functional.interpolate(
        pixel_values, size=(input_size, input_size), mode='bilinear', align_corners=False
    )
    pixel_values = pixel_values * 0.5 + 0.5
    imgs = pixel_values.unsqueeze(0)  # (1, F, C, H, W)

    with torch.no_grad():
        autocast_enabled = pixel_values.is_cuda and pixel_values.dtype in (torch.float16, torch.bfloat16)
        with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=pixel_values.dtype if autocast_enabled else torch.float16):
            imgs_norm = (imgs - spatial_tower.pi3.image_mean) / spatial_tower.pi3.image_std
            B, N, C, H, W = imgs_norm.shape
            imgs_flat = imgs_norm.reshape(B * N, C, H, W)
            hidden = spatial_tower.pi3.encoder(imgs_flat, is_training=True)
            if isinstance(hidden, dict):
                hidden = hidden["x_norm_patchtokens"]
            decoded_features, decoded_pos = spatial_tower.pi3.decode(hidden, N, H, W)

    return decoded_features, decoded_pos

# --- JSON Filtering ---

def load_allowed_stems_from_jsons(json_paths: list, strict=False) -> set:
    """Load all unique video stems referenced by a list of JSON files.
    Supports both direct JSON paths and YAML data configs (with 'datasets' key).
    
    Args:
        json_paths: List of JSON or YAML file paths
        strict: If True, raise RuntimeError if any JSON fails to load (fail-fast mode).
                If False, warn and continue (permissive mode).
    
    Returns:
        Set of unique video stems
    
    Raises:
        RuntimeError: If strict=True and any JSON file fails to load
    """
    import yaml as _yaml
    resolved_json_paths = []
    for p in json_paths:
        if p.endswith('.yaml') or p.endswith('.yml'):
            with open(p, 'r') as f:
                cfg = _yaml.safe_load(f)
            for ds in cfg.get('datasets', []):
                jp = ds.get('json_path')
                if jp:
                    resolved_json_paths.append(jp)
        else:
            resolved_json_paths.append(p)

    stems = set()
    for jp in resolved_json_paths:
        try:
            with open(jp, 'r') as f:
                data = json.load(f)
            for item in data:
                vid = item.get('video')
                if vid:
                    stems.add(Path(vid).stem)
                elif 'scene_name' in item:
                    stems.add(item['scene_name'])
            rank0_print(f"  Loaded {len(data)} entries from {jp}")
        except Exception as e:
            if strict:
                rank0_print(f"  ERROR (strict mode): Could not load {jp}: {e}")
                raise RuntimeError(f"JSON loading failed in strict mode: {jp}: {e}")
            else:
                rank0_print(f"  Warning (permissive): Could not load {jp}: {e}")
    
    if strict and not stems:
        raise RuntimeError("No video stems loaded from JSON files (empty result in strict mode)")
    
    rank0_print(f"Total unique video stems from JSONs: {len(stems)}")
    return stems


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
    
    Returns:
        Tuple of (processed_count, skipped_count, succeeded_videos, failed_videos)
        where succeeded_videos and failed_videos are lists of video paths
    """
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device) # Ensure this process uses the assigned GPU

    processed_count = 0
    skipped_count = 0
    succeeded_videos = []
    failed_videos = []
    total_files_in_chunk = len(video_files_chunk)
    batch_size = args.batch_size

    rank0_print(f"[GPU {gpu_id}] Worker started, assigned {total_files_in_chunk} files.")

    # -- Load Model Components (within each worker process) --
    try:
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
        rank0_print(f"[GPU {gpu_id}] PI3X Spatial Tower initialized.")

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
        return 0, total_files_in_chunk, [], []  # Empty succeeded/failed lists

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
        batch_data_to_process = [] # Stores (preprocessed_input, feature_output_filename, video_full_path)
        files_in_batch = 0
        skipped_in_batch = 0
        frames_base_dir = Path(args.save_preprocessed_frames_dir) if args.save_preprocessed_frames else None

        for video_full_path in batch_paths:
            files_in_batch += 1
            feature_output_filename_path = get_output_path(video_full_path, input_base_dir, output_dir)
            feature_output_filename = str(feature_output_filename_path)

            # Check feature file existence
            feature_output_filename_path.parent.mkdir(parents=True, exist_ok=True)
            feature_exists = os.path.exists(feature_output_filename)

            if not args.overwrite and feature_exists:
                skipped_in_batch += 1
                continue

            preprocessed_video = load_and_preprocess_video_frames(
                str(video_full_path),
                data_args,
                image_processor,
                target_size=target_size,
                rank=gpu_id,
            )

            if preprocessed_video is not None and preprocessed_video["frames_tensor"].nelement() > 0:
                # --- Save Preprocessed Frames (if enabled) ---
                if args.save_preprocessed_frames and frames_base_dir is not None and mean_tensor is not None and std_tensor is not None:
                    try:
                        if args.save_preprocessed_frames_layout == "sibling":
                            frames_output_dir_path = get_sibling_preprocessed_frames_output_dir(feature_output_filename_path)
                        else:
                            frames_output_dir_path = get_preprocessed_frames_output_dir(video_full_path, input_base_dir, frames_base_dir)
                        frames_output_dir_path.mkdir(parents=True, exist_ok=True)

                        preprocessed_input_float = preprocessed_video["frames_tensor"].float()
                        for frame_idx in range(preprocessed_input_float.shape[0]):
                            frame_tensor = preprocessed_input_float[frame_idx]
                            denormalized_frame = (frame_tensor * std_tensor.to(frame_tensor.device) + mean_tensor.to(frame_tensor.device)) / rescale_factor_val
                            denormalized_frame = torch.clamp(denormalized_frame, 0, 255).to(torch.uint8)
                            pil_image = Image.fromarray(denormalized_frame.permute(1, 2, 0).cpu().numpy())
                            frame_filename = frames_output_dir_path / f"frame_{frame_idx:04d}.png"
                            pil_image.save(frame_filename)

                    except Exception as e_save_frame:
                        rank0_print(f"[GPU {gpu_id}] Error saving preprocessed frame for {video_full_path}, frame {frame_idx}: {e_save_frame}")

                batch_data_to_process.append((preprocessed_video, feature_output_filename, video_full_path))
            else:
                rank0_print(f"[GPU {gpu_id}] Failed to load/preprocess {video_full_path}. Skipping.")
                failed_videos.append(str(video_full_path))
                skipped_in_batch += 1

        # --- Batch Inference (PI3X processes one video at a time) ---
        processed_in_batch = 0
        if batch_data_to_process:
            for preprocessed_video, feature_output_filename, video_full_path in batch_data_to_process:
                try:
                    with torch.no_grad():
                        decoded_features, decoded_pos = extract_decoded_features_and_pos(
                            spatial_tower,
                            preprocessed_video["frames_tensor"].to(device=device, dtype=model_dtype),
                        )

                    num_frames = int(decoded_features.shape[0])
                    frame_idx = preprocessed_video["frame_idx"].clone()
                    decoded_pos_template = decoded_pos[0].detach().cpu()

                    features_to_save = {
                        "frames": {
                            "decoded_features": decoded_features.detach().cpu(),
                            "frame_idx": frame_idx,
                        },
                        "meta": {
                            "decoded_pos_template": decoded_pos_template,
                            "num_frames": num_frames,
                            "input_size": int(spatial_tower.config.input_size),
                            "patch_size": int(spatial_tower.pi3.patch_size),
                            "patch_start_idx": int(spatial_tower.pi3.patch_start_idx),
                        },
                    }
                    torch.save(features_to_save, feature_output_filename)
                    processed_in_batch += 1
                    succeeded_videos.append(str(video_full_path))
                except Exception as e_pi3x:
                    rank0_print(f"[GPU {gpu_id}] Error during PI3X inference/save for {video_full_path}: {e_pi3x}\n{traceback.format_exc()}")
                    failed_videos.append(str(video_full_path))
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
    return processed_count, skipped_count, succeeded_videos, failed_videos


# --- Main Execution ---
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True) # Recommended for CUDA with multiprocessing

    parser = argparse.ArgumentParser(description="Extract Pi3X decoded features using PI3X (Multi-GPU)")
    parser.add_argument("--pi3x-weights-path", type=str, required=True, help="Path to PI3X weights/checkpoint directory")
    parser.add_argument("--pi3x-input-size", type=int, default=518, help="Input size used internally by PI3X (must be multiple of 14)")
    parser.add_argument("--input-dir", type=str, default=None, help="Root directory containing video files to process recursively (ignored if --input-file is provided)")
    parser.add_argument("--input-file", type=str, default=None, help="Path to a single video file to process")
    parser.add_argument("--output-dir", type=str, required=True, help="Root directory to save the extracted features, mirroring input structure")
    parser.add_argument("--processor-config-path", type=str, required=True, help="Path to the processor_config.json file for SigLipImageProcessor")
    parser.add_argument("--gpu-ids", type=str, default="0", help="Comma-separated list of GPU IDs to use (e.g., '0,1,2') or 'all'")
    parser.add_argument("--precision", type=str, default="bf16", choices=['fp16', 'bf16', 'fp32'], help="Computation precision")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing feature files")
    parser.add_argument("--video-fps", type=int, default=1, help="FPS for video frame sampling")
    parser.add_argument("--frames-upbound", type=int, default=32, help="Max frames to sample per video")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of videos to process per batch *per GPU*")
    parser.add_argument("--filter-by-jsons", type=str, nargs='+', default=None,
                        help="Only extract features for videos referenced in these JSON/YAML files. "
                             "Accepts JSON paths or YAML data configs (e.g. scripts/VLM_3R/vsibench_data.yaml)")
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
    output_dir.mkdir(parents=True, exist_ok=True)
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

    # --- Filter by JSON references (if requested) - with strict mode ---
    if args.filter_by_jsons:
        rank0_print(f"Filtering videos by {len(args.filter_by_jsons)} JSON/YAML source(s) (STRICT MODE)...")
        try:
            allowed_stems = load_allowed_stems_from_jsons(args.filter_by_jsons, strict=True)
        except RuntimeError as e:
            rank0_print(f"ERROR in strict mode: {e}")
            rank0_print("Exiting due to JSON loading failure in strict mode.")
            sys.exit(1)
        
        before_count = len(all_video_files_paths)
        all_video_files_paths = [p for p in all_video_files_paths if p.stem in allowed_stems]
        rank0_print(f"Filtered: {before_count} -> {len(all_video_files_paths)} videos "
                    f"({before_count - len(all_video_files_paths)} excluded).")
        if not all_video_files_paths:
            rank0_print("ERROR (strict mode): No videos remain after filtering. This indicates a JSON mismatch.")
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
    all_succeeded_videos = []
    all_failed_videos = []
    try:
        with mp.Pool(processes=num_workers) as pool:
            # Use starmap to pass multiple arguments to the worker function
            results = pool.starmap(process_videos_on_gpu, pool_args)

        # Aggregate results
        for processed, skipped, succeeded, failed in results:
            total_processed += processed
            total_skipped += skipped
            all_succeeded_videos.extend(succeeded)
            all_failed_videos.extend(failed)

    except Exception as e:
         rank0_print(f"\n--- An error occurred during multiprocessing ---")
         rank0_print(f"Error: {e}")
         rank0_print(traceback.format_exc())
         rank0_print("Feature extraction may be incomplete.")
         # Ensure totals reflect potentially partial completion if some processes finished
         if 'results' in locals():
              for result in results:
                   if len(result) == 4:
                       processed, skipped, succeeded, failed = result
                       total_processed += processed
                       total_skipped += skipped
                       all_succeeded_videos.extend(succeeded)
                       all_failed_videos.extend(failed)
                   else:
                       # Fallback for older format (shouldn't happen with current code)
                       processed, skipped = result
                       total_processed += processed
                       total_skipped += skipped
         else: # If pool creation failed or starmap didn't even start
              total_skipped = len(all_video_files_paths)


    # --- Final Summary with Results File ---
    rank0_print("-" * 50)
    rank0_print(f"Feature extraction complete.")
    rank0_print(f"Successfully processed: {total_processed}")
    rank0_print(f"Failed: {len(all_failed_videos)}")
    rank0_print(f"Skipped (exists or preprocessing error): {total_skipped - len(all_failed_videos)}")
    rank0_print(f"Total files considered: {len(all_video_files_paths)}")
    rank0_print(f"Features saved in: {args.output_dir}")
    
    # --- Write Results to JSON File ---
    results_file = Path(args.output_dir) / "extraction_results.json"
    results_data = {
        "total_processed": total_processed,
        "total_failed": len(all_failed_videos),
        "total_skipped": total_skipped - len(all_failed_videos),
        "total_videos_considered": len(all_video_files_paths),
        "succeeded_videos": sorted(all_succeeded_videos),
        "failed_videos": sorted(all_failed_videos),
        "output_directory": args.output_dir,
    }
    
    try:
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        rank0_print(f"\nResults saved to: {results_file}")
    except Exception as e:
        rank0_print(f"Warning: Could not write results file {results_file}: {e}")
    
    # --- Print Failed Videos (if any) ---
    if all_failed_videos:
        rank0_print(f"\n⚠️  {len(all_failed_videos)} video(s) failed to process:")
        for video in sorted(all_failed_videos)[:20]:  # Print first 20
            rank0_print(f"   - {video}")
        if len(all_failed_videos) > 20:
            rank0_print(f"   ... and {len(all_failed_videos) - 20} more")
        rank0_print(f"\nTo reprocess failed videos, run:")
        rank0_print(f"  python {Path(__file__).name} --input-file <video_path> --output-dir {args.output_dir} --overwrite ...")
    
    rank0_print("-" * 50)
