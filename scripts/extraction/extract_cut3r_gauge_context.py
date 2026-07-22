#!/usr/bin/env python3
"""Extract the frozen CUT3R head context required by gauge translation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
CUT3R_ROOT = REPO_ROOT / "third_party" / "CUT3R"
for value in (str(REPO_ROOT), str(CUT3R_ROOT), str(CUT3R_ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from llava.model.cut3r_gauge_translation import (  # noqa: E402
    build_teacher_mask,
    pool_points_adaptive,
    pool_points_by_positions,
    robust_scene_scale,
    validate_patch_positions,
)
from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor  # noqa: E402
from llava.model.multimodal_spatial_encoder.cut3r_spatial_encoder import prepare_input  # noqa: E402
from llava.utils import process_video_with_decord  # noqa: E402
from src.dust3r.model import ARCroco3DStereo  # noqa: E402


class VideoArguments:
    video_fps = 1
    frames_upbound = 32
    force_sample = True


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_processor(path: Path) -> SigLipImageProcessor:
    payload = json.loads(path.read_text())
    size = payload.get("size", {"height": 384, "width": 384})
    return SigLipImageProcessor(
        image_mean=payload.get("image_mean", (0.5, 0.5, 0.5)),
        image_std=payload.get("image_std", (0.5, 0.5, 0.5)),
        size=(size["height"], size["width"]),
        resample=payload.get("resample", 3),
        rescale_factor=payload.get("rescale_factor", 1 / 255.0),
    )


def load_video(path: Path, processor: SigLipImageProcessor, frames_upbound: int) -> torch.Tensor:
    args = VideoArguments()
    args.frames_upbound = int(frames_upbound)
    frames, _, _, _ = process_video_with_decord(str(path), args)
    pixels = processor.preprocess(images=frames, return_tensors="pt")["pixel_values"]
    return torch.nn.functional.interpolate(pixels, size=(432, 432), mode="bilinear", align_corners=False)


@torch.no_grad()
def run_cut3r(model: ARCroco3DStereo, pixels: torch.Tensor) -> dict[str, torch.Tensor]:
    views = prepare_input(pixels)
    shapes, feature_layers, positions = model._encode_views(views)
    features = feature_layers[-1]
    state, state_pos = model._init_state(features[0], positions[0])
    memory = model.pose_retriever.mem.expand(features[0].shape[0], -1, -1)
    initial_state, initial_memory = state.clone(), memory.clone()
    result: dict[str, list[torch.Tensor]] = {
        "dec0": [], "pos": [], "patch6": [], "patch9": [], "patch12": [], "pose12": [],
        "confidence_ref": [], "confidence_self": [], "point_maps_ref": [], "point_maps_cam": [],
    }
    for frame_idx, view in enumerate(views):
        feature, pos = features[frame_idx].to(dtype=pixels.dtype), positions[frame_idx]
        global_feature = model._get_img_level_feat(feature)
        pose_feature = (
            model.pose_token.expand(feature.shape[0], -1, -1)
            if frame_idx == 0
            else model.pose_retriever.inquire(global_feature, memory)
        )
        pose_pos = -torch.ones(feature.shape[0], 1, 2, device=feature.device, dtype=pos.dtype)
        new_state, decoder = model._recurrent_rollout(
            state, state_pos, feature, pos, pose_feature, pose_pos, initial_state,
            img_mask=view["img_mask"], reset_mask=view["reset"], update=view.get("update"),
        )
        decoder = list(decoder)
        new_memory = model.pose_retriever.update_mem(memory, global_feature, decoder[-1][:, :1])
        head_input = [decoder[0], decoder[6][:, 1:], decoder[9][:, 1:], decoder[12]]
        output = model._downstream_head(head_input, shapes[frame_idx], pos=pos)
        result["dec0"].append(decoder[0])
        result["pos"].append(pos)
        result["patch6"].append(decoder[6][:, 1:])
        result["patch9"].append(decoder[9][:, 1:])
        result["patch12"].append(decoder[12][:, 1:])
        result["pose12"].append(decoder[12][:, :1])
        result["confidence_ref"].append(output["conf"])
        result["confidence_self"].append(output["conf_self"])
        result["point_maps_ref"].append(output["pts3d_in_other_view"])
        result["point_maps_cam"].append(output["pts3d_in_self_view"])
        update = view.get("update")
        update_mask = view["img_mask"] & update if update is not None else view["img_mask"]
        update_mask = update_mask[:, None, None].to(dtype=pixels.dtype)
        state = new_state * update_mask + state * (1 - update_mask)
        memory = new_memory * update_mask + memory * (1 - update_mask)
        reset = view.get("reset")
        if reset is not None:
            reset_mask = reset[:, None, None].to(dtype=pixels.dtype)
            state = initial_state * reset_mask + state * (1 - reset_mask)
            memory = initial_memory * reset_mask + memory * (1 - reset_mask)
    return {key: torch.cat(values, dim=0) for key, values in result.items()}


def _load(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def verify_sidecars(record: dict, generated: dict, minimum_cosine: float) -> dict:
    checks = {}
    for layer in (6, 9, 12):
        payload = _load(record[f"layer{layer}_path"])
        patch = payload["patch_tokens"].float()[: generated[f"patch{layer}"].shape[0]]
        pose = payload.get("camera_tokens")
        generated_patch = generated[f"patch{layer}"].detach().cpu().float()
        if patch.shape != generated_patch.shape:
            raise RuntimeError(f"layer {layer} shape mismatch: {patch.shape} vs {generated_patch.shape}")
        cosine = torch.nn.functional.cosine_similarity(
            patch.reshape(-1, patch.shape[-1]), generated_patch.reshape(-1, generated_patch.shape[-1]), dim=-1
        ).mean()
        checks[f"layer{layer}_patch_cosine"] = float(cosine.item())
        if float(cosine) < minimum_cosine:
            raise RuntimeError(f"layer {layer} cached/generated cosine {float(cosine):.6f} < {minimum_cosine}")
        if layer == 12:
            pose = None if pose is None else pose[: generated["pose12"].shape[0]]
            if pose is None or pose.shape != generated["pose12"].cpu().shape:
                raise RuntimeError("layer-12 pose sidecar shape mismatch")
            pose_cos = torch.nn.functional.cosine_similarity(
                pose.float().reshape(-1, pose.shape[-1]), generated["pose12"].cpu().float().reshape(-1, pose.shape[-1]), dim=-1
            ).mean()
            checks["layer12_pose_cosine"] = float(pose_cos.item())
    return checks


def alignment_report(points: torch.Tensor, mask: torch.Tensor, pos: torch.Tensor, tolerance: float) -> tuple[dict, torch.Tensor, torch.Tensor]:
    position_report = validate_patch_positions(pos)
    explicit, explicit_mask = pool_points_by_positions(points, mask, pos)
    adaptive, adaptive_mask = pool_points_adaptive(points, mask)
    shared = explicit_mask & adaptive_mask
    max_abs = float((explicit[shared] - adaptive[shared]).abs().max().item()) if bool(shared.any()) else float("inf")
    value_scale = float(explicit[shared].abs().max().item()) if bool(shared.any()) else 1.0
    relative_max = max_abs / max(value_scale, 1.0)
    mask_equal = bool(torch.equal(explicit_mask, adaptive_mask))
    height, width = int(points.shape[1]), int(points.shape[2])
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    ramp = torch.stack([yy, xx, yy * width + xx], dim=-1).float().unsqueeze(0)
    ramp_mask = torch.ones(1, height, width, dtype=torch.bool)
    ramp_pos = pos[:1].cpu()
    ramp_explicit, _ = pool_points_by_positions(ramp, ramp_mask, ramp_pos)
    ramp_adaptive, _ = pool_points_adaptive(ramp, ramp_mask)
    ramp_max_abs = float((ramp_explicit - ramp_adaptive).abs().max().item())
    ramp_relative_max = ramp_max_abs / max(float(ramp_explicit.abs().max().item()), 1.0)
    passed = bool(position_report["passed"] and mask_equal and relative_max <= tolerance and ramp_relative_max <= tolerance)
    return {
        "passed": passed,
        "position": position_report,
        "explicit_adaptive_mask_equal": mask_equal,
        "explicit_adaptive_max_abs": max_abs,
        "explicit_adaptive_relative_max": relative_max,
        "synthetic_ramp_max_abs": ramp_max_abs,
        "synthetic_ramp_relative_max": ramp_relative_max,
        "tolerance": tolerance,
        "selected_method": "explicit_position_cells" if not passed else "adaptive_verified_by_explicit_positions",
    }, explicit, explicit_mask


def save_alignment_png(path: Path, pooled: torch.Tensor, valid: torch.Tensor) -> None:
    values = pooled[0, :, 2].reshape(27, 27).float()
    mask = valid[0].reshape(27, 27)
    finite = values[mask]
    low = finite.quantile(0.05) if finite.numel() else torch.tensor(0.0)
    high = finite.quantile(0.95) if finite.numel() else torch.tensor(1.0)
    normalized = ((values - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
    pixels = torch.where(mask, (normalized * 255).byte(), torch.zeros_like(normalized, dtype=torch.uint8))
    image = Image.fromarray(pixels.numpy(), mode="L").resize((432, 432), resample=Image.Resampling.NEAREST).convert("RGB")
    draw = ImageDraw.Draw(image)
    for value in range(0, 433, 16):
        draw.line((value, 0, value, 432), fill=(255, 0, 0), width=1)
        draw.line((0, value, 432, value), fill=(255, 0, 0), width=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def process_record(record: dict, model, processor, args, device, dtype, visualization: bool) -> dict:
    pixels = load_video(Path(record["video_path"]), processor, args.frames_upbound).to(device=device, dtype=dtype)
    generated = run_cut3r(model, pixels)
    sidecar_checks = verify_sidecars(record, generated, args.minimum_token_cosine)
    pointmap = _load(record["pointmap_path"])
    frame_count = int(generated["dec0"].shape[0])
    reference = pointmap["point_maps_ref"].float()[:frame_count]
    camera = pointmap["point_maps_cam"].float()[:frame_count]
    if reference.shape != generated["point_maps_ref"].cpu().shape or camera.shape != generated["point_maps_cam"].cpu().shape:
        raise RuntimeError("point-map sidecar and regenerated head output shapes differ")
    confidence_ref = generated["confidence_ref"].cpu().float()
    confidence_self = generated["confidence_self"].cpu().float()
    teacher_mask = build_teacher_mask(
        reference, camera, confidence_ref, confidence_self,
        args.reference_confidence_threshold, args.self_confidence_threshold,
    )
    scale = robust_scene_scale(reference, teacher_mask)
    report, pooled, pooled_mask = alignment_report(
        reference, teacher_mask, generated["pos"].cpu(), args.alignment_tolerance
    )
    if not report["passed"]:
        raise RuntimeError(f"token-target alignment gate failed: {report}")
    output_path = Path(record["context_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dec0": generated["dec0"].cpu().to(torch.float16),
        "pos": generated["pos"].cpu().to(torch.int64),
        "confidence_ref": confidence_ref.to(torch.float16),
        "confidence_self": confidence_self.to(torch.float16),
        "teacher_mask": teacher_mask,
        "pooled_ref_xyz": pooled,
        "pooled_valid": pooled_mask,
        "scene_scale": scale.float(),
        "metadata": {
            "schema": "cut3r_gauge_head_context_v1",
            "id": record["id"],
            "source_video": record["video_path"],
            "cut3r_checkpoint": str(args.checkpoint),
            "image_size": [432, 432],
            "patch_size": [16, 16],
            "patch_grid": [27, 27],
            "preprocessing": {"processor_config": str(args.processor_config), "final_resize": [432, 432], "crop": None, "padding": None},
            "num_frames": int(reference.shape[0]),
            "linked_sidecars": {key: record[key] for key in ("layer6_path", "layer9_path", "layer12_path", "pointmap_path")},
            "sidecar_checks": sidecar_checks,
            "alignment": report,
            "teacher_mask_definition": "finite_ref & finite_cam & cam_z>0 & original_confidence_thresholds",
            "reference_confidence_threshold": args.reference_confidence_threshold,
            "self_confidence_threshold": args.self_confidence_threshold,
        },
    }
    temporary = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, output_path)
    report_path = args.alignment_report_dir / f"{record['dataset']}_{record['stem']}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if visualization:
        save_alignment_png(args.alignment_report_dir / f"{record['dataset']}_{record['stem']}.png", pooled, pooled_mask)
    return {"id": record["id"], "context_path": str(output_path), "scene_scale": float(scale), **sidecar_checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--processor-config", type=Path, default=REPO_ROOT / "processor_config.json")
    parser.add_argument("--alignment-report-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="bf16")
    parser.add_argument("--frames-upbound", type=int, default=32)
    parser.add_argument("--reference-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--self-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--alignment-tolerance", type=float, default=3e-6)
    parser.add_argument("--minimum-token-cosine", type=float, default=0.99)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rank", type=int, default=int(os.environ.get("SLURM_PROCID", 0)))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("SLURM_NTASKS", 1)))
    parser.add_argument("--visualization-count", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("context extraction requires a GPU allocation")
    visible_device_count = torch.cuda.device_count()
    local_rank = int(os.environ.get("LOCAL_RANK", args.rank % visible_device_count)) % visible_device_count
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    model = ARCroco3DStereo.from_pretrained(str(args.checkpoint)).eval().to(device=device, dtype=dtype)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    processor = load_processor(args.processor_config)
    records = load_manifest(args.manifest)[args.rank :: args.world_size]
    completed, failed = [], []
    visualizations = 0
    for index, record in enumerate(records):
        output_path = Path(record["context_path"])
        if output_path.exists() and not args.overwrite:
            completed.append({"id": record["id"], "status": "existing"})
            continue
        try:
            completed.append(process_record(record, model, processor, args, device, dtype, visualizations < args.visualization_count))
            visualizations += 1
        except Exception as exc:
            failed.append({"id": record["id"], "error": str(exc), "traceback": traceback.format_exc()})
        print(f"[rank {args.rank}] {index+1}/{len(records)} completed={len(completed)} failed={len(failed)}", flush=True)
    summary = {"rank": args.rank, "world_size": args.world_size, "completed": completed, "failed": failed}
    args.alignment_report_dir.mkdir(parents=True, exist_ok=True)
    (args.alignment_report_dir / f"extraction_rank{args.rank}.json").write_text(json.dumps(summary, indent=2) + "\n")
    if failed:
        raise SystemExit(f"context extraction failed for {len(failed)} records")


if __name__ == "__main__":
    main()
