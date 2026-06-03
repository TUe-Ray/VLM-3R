#!/usr/bin/env python3
"""Export a VGGT point cloud from an image folder.

This mirrors VGGT's demo_colmap depth+camera branch, but writes a lightweight
PLY and manifest that fit the presentation sample folders used in this repo.

GeoRoPE coordinate consistency rule: point-cloud export is diagnostic, but any
VGGT geometry used for train/eval must keep the same coordinate convention for
the same checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
VGGT_ROOT = REPO_ROOT / "third_party" / "VGGT"
if str(VGGT_ROOT) in sys.path:
    sys.path.remove(str(VGGT_ROOT))
sys.path.insert(0, str(VGGT_ROOT))

from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.geometry import unproject_depth_map_to_point_map  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images_square  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VGGT and export a colored point cloud PLY.")
    parser.add_argument("--image-folder", required=True, help="Folder containing ordered RGB frames.")
    parser.add_argument("--output-dir", required=True, help="Directory for VGGT outputs.")
    parser.add_argument(
        "--vggt-weights",
        default="/leonardo_work/EUHPC_D32_006/FAST/hf_models/vggt",
        help="Local VGGT checkpoint file/directory or Hugging Face model id.",
    )
    parser.add_argument("--input-size", type=int, default=518, help="VGGT inference resolution.")
    parser.add_argument("--load-size", type=int, default=1024, help="Square padded image load size.")
    parser.add_argument("--conf-threshold", type=float, default=5.0, help="Depth confidence threshold.")
    parser.add_argument("--stride", type=int, default=4, help="Pixel stride before confidence filtering.")
    parser.add_argument("--max-points", type=int, default=0, help="Optional deterministic cap; 0 disables.")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32", "auto"], default="auto")
    parser.add_argument("--device", default="cuda", help="Torch device, usually cuda or cpu.")
    parser.add_argument("--save-pt", action="store_true", help="Also save VGGT tensors as vggt_outputs.pt.")
    return parser.parse_args()


def choose_dtype(device: torch.device, precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "fp32" or device.type == "cpu":
        return torch.float32
    major, _minor = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def image_paths(image_folder: Path) -> list[Path]:
    paths = sorted(path for path in image_folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"No image files found in {image_folder}")
    return paths


def load_model(weights: str, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    weights_path = Path(weights)
    model = VGGT().to(device=device, dtype=dtype).eval()
    if weights_path.is_dir() and (weights_path / "model.safetensors").exists():
        try:
            from safetensors.torch import load_file as safe_load_file
        except ImportError as exc:
            raise RuntimeError("safetensors is required to load model.safetensors") from exc
        state = safe_load_file(str(weights_path / "model.safetensors"), device="cpu")
        model.load_state_dict(state, strict=True)
    elif weights_path.is_dir() and (weights_path / "model.pt").exists():
        state = torch.load(weights_path / "model.pt", map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=True)
    elif weights_path.exists():
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=True)
    else:
        model = VGGT.from_pretrained(weights).to(device=device, dtype=dtype).eval()

    for param in model.parameters():
        param.requires_grad_(False)
    return model


def write_ascii_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def valid_original_pixel_mask(original_coords: torch.Tensor, load_size: int, input_size: int, stride: int) -> np.ndarray:
    """Return [F,Hs,Ws] mask that excludes square-padding pixels."""
    coords = original_coords.float().numpy()[:, :4] * (float(input_size) / float(load_size))
    ys = np.arange(0, input_size, stride, dtype=np.float32)
    xs = np.arange(0, input_size, stride, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    masks = []
    for x1, y1, x2, y2 in coords:
        masks.append((xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2))
    return np.stack(masks, axis=0)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.input_size % 14 != 0:
        raise ValueError("--input-size must be divisible by 14")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    image_folder = Path(args.image_folder)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = image_paths(image_folder)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = choose_dtype(device, args.precision)
    torch.backends.cudnn.benchmark = True

    print(f"Loading {len(paths)} frames from {image_folder}")
    print(f"Loading VGGT weights from {args.vggt_weights}")
    model = load_model(args.vggt_weights, device, dtype)

    images, original_coords = load_and_preprocess_images_square([str(path) for path in paths], args.load_size)
    images = images.to(device)
    infer_images = F.interpolate(images, size=(args.input_size, args.input_size), mode="bilinear", align_corners=False)

    print(f"Running VGGT on {device} with dtype={dtype}")
    autocast_enabled = device.type == "cuda" and dtype != torch.float32
    with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=dtype):
        batch_images = infer_images.unsqueeze(0)
        aggregated_tokens_list, ps_idx = model.aggregator(batch_images)
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, batch_images.shape[-2:])
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, batch_images, ps_idx)

    extrinsic_np = extrinsic.squeeze(0).float().cpu().numpy()
    intrinsic_np = intrinsic.squeeze(0).float().cpu().numpy()
    depth_np = depth_map.squeeze(0).float().cpu().numpy()
    conf_np = depth_conf.squeeze(0).float().cpu().numpy()
    points_np = unproject_depth_map_to_point_map(depth_np, extrinsic_np, intrinsic_np)

    rgb = F.interpolate(infer_images, size=(args.input_size, args.input_size), mode="bilinear", align_corners=False)
    rgb_np = (rgb.float().cpu().numpy() * 255.0).round().astype(np.uint8).transpose(0, 2, 3, 1)

    points_s = points_np[:, :: args.stride, :: args.stride, :]
    conf_s = conf_np[:, :: args.stride, :: args.stride]
    rgb_s = rgb_np[:, :: args.stride, :: args.stride, :]
    valid_rgb_mask = valid_original_pixel_mask(original_coords, args.load_size, args.input_size, args.stride)
    finite_mask = np.isfinite(points_s).all(axis=-1) & np.isfinite(conf_s)
    conf_mask = conf_s >= args.conf_threshold
    mask = finite_mask & conf_mask & valid_rgb_mask

    selected_points = points_s[mask].reshape(-1, 3)
    selected_colors = rgb_s[mask].reshape(-1, 3)
    if args.max_points and selected_points.shape[0] > args.max_points:
        indices = np.linspace(0, selected_points.shape[0] - 1, args.max_points, dtype=np.int64)
        selected_points = selected_points[indices]
        selected_colors = selected_colors[indices]

    ply_name = f"vggt_points_depth_conf{args.conf_threshold:g}_stride{args.stride}.ply"
    ply_path = output_dir / ply_name
    write_ascii_ply(ply_path, selected_points, selected_colors)

    pt_path = None
    if args.save_pt:
        pt_path = output_dir / "vggt_outputs.pt"
        torch.save(
            {
                "extrinsic": torch.from_numpy(extrinsic_np),
                "intrinsic": torch.from_numpy(intrinsic_np),
                "depth_map": torch.from_numpy(depth_np),
                "depth_conf": torch.from_numpy(conf_np),
                "original_coords": original_coords.cpu(),
                "image_paths": [str(path) for path in paths],
            },
            pt_path,
        )

    manifest = {
        "schema": "vggt_point_cloud_export_v1",
        "image_folder": str(image_folder),
        "image_count": len(paths),
        "vggt_weights": args.vggt_weights,
        "input_size": args.input_size,
        "load_size": args.load_size,
        "conf_threshold": args.conf_threshold,
        "stride": args.stride,
        "max_points": args.max_points,
        "ply_path": str(ply_path),
        "point_count": int(selected_points.shape[0]),
        "shape_after_stride": list(points_s.shape),
        "padded_pixels_excluded": True,
        "coordinate_frame": "VGGT world coordinates from depth unprojection",
        "depth_shape": list(depth_np.shape),
        "depth_conf_min": float(np.nanmin(conf_np)),
        "depth_conf_max": float(np.nanmax(conf_np)),
        "depth_conf_mean": float(np.nanmean(conf_np)),
        "pt_path": str(pt_path) if pt_path is not None else None,
    }
    manifest_path = output_dir / "vggt_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {selected_points.shape[0]} points to {ply_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
