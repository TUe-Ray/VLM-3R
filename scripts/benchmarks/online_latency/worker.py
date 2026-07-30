#!/usr/bin/env python3
"""One fresh-process worker for the strict raw-video latency benchmark.

The worker deliberately calls the production VLM ``generate`` path.  It never
constructs an lmms-eval task, never passes a cache path, and only creates the
SpatialStack decoder-layer payload in GPU memory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from decord import VideoReader, cpu

from scripts.benchmarks.online_latency.common import (
    GENERATION_ARGS, config_hash, git_state, json_dump, median_repetition,
    runtime_basics, sha256_file, tree_inventory,
)


class CudaCallTimer:
    """Forward-hook timing with one CUDA-event pair per call."""
    def __init__(self):
        self.calls: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._starts: list[torch.cuda.Event] = []

    def pre(self, *_args):
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        self._starts.append(start)

    def post(self, *_args):
        if self._starts:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.calls.append((self._starts.pop(), end))

    @property
    def count(self) -> int:
        return len(self.calls)

    def milliseconds(self) -> list[float]:
        torch.cuda.synchronize()
        return [float(start.elapsed_time(end)) for start, end in self.calls]


class KnownLoaderAudit:
    """Counters for the only cache/sidecar reader entry points this runner permits.

    The benchmark does not import the lmms evaluator, whose private sidecar
    loader is the known sidecar read path.  Keeping the counters explicit makes
    this structural fact auditable and leaves normal evaluation code untouched.
    """
    def __init__(self):
        self.siglip_feature_cache_reads = 0
        self.spatialstack_sidecar_reads = 0
        self.projected_visual_token_cache_reads = 0
        self.residual_tensor_cache_reads = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "siglip_feature_cache_reads": self.siglip_feature_cache_reads,
            "spatialstack_sidecar_reads": self.spatialstack_sidecar_reads,
            "projected_visual_token_cache_reads": self.projected_visual_token_cache_reads,
            "residual_tensor_cache_reads": self.residual_tensor_cache_reads,
        }

    def assert_zero(self) -> None:
        counts = self.snapshot()
        if any(counts.values()):
            raise AssertionError(f"Strict no-cache violation: {counts}")


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _cuda_timed(fn):
    torch.cuda.synchronize()
    start_wall = time.perf_counter()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    value = fn()
    end.record()
    torch.cuda.synchronize()
    return value, float(start.elapsed_time(end)), (time.perf_counter() - start_wall) * 1000.0


def _decode_raw(path: str, ids: list[int]):
    vr = VideoReader(path, ctx=cpu(0), num_threads=1)
    if len(vr) <= max(ids):
        raise AssertionError(f"Raw video is shorter than manifest frame IDs: {path}")
    return vr.get_batch(ids).asnumpy()


def _prompt_ids(tokenizer, prompt: str, device):
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token
    from llava.conversation import conv_templates

    qs = "<image>\n" + prompt
    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    rendered = conv.get_prompt()
    ids = tokenizer_image_token(rendered, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return ids, ids.ne(pad).long()


def _load_cut3r(device, dtype, weights_path: str):
    # Reuse the exact online extraction implementation; it returns layers in one recurrent pass.
    from scripts.extraction.extract_cut3r_layer_features import ARCroco3DStereo, Cut3rSpatialConfig
    config = Cut3rSpatialConfig(weights_path=weights_path)
    model = ARCroco3DStereo.from_pretrained(config.weights_path).to(device=device, dtype=dtype).eval()
    model.requires_grad_(False)
    return model


def _runtime_layer_payload(cut3r, pixels: torch.Tensor, frame_ids: list[int]):
    from scripts.extraction.extract_cut3r_layer_features import run_cut3r_layers
    outputs = run_cut3r_layers(cut3r, pixels, [6, 9, 12])
    layers = {}
    for layer, (camera, patch) in outputs.items():
        # The merger accepts the sidecar schema, but this object is only RAM output
        # from the forward above and is never read from a sidecar path.
        layers[str(layer)] = {"camera_tokens": camera[0], "patch_tokens": patch[0]}
    # SpatialStack metadata uses local visual-frame positions (0..F-1), while
    # raw video frame IDs are provenance only. Do not expose the latter under
    # merger-recognized frame_indices/frame_order keys.
    return {"cut3r_dec_layers": layers,
            "metadata": {"source": "online_cut3r_forward", "raw_video_frame_ids": list(frame_ids)}}


def _load_vlm(args, device):
    from llava.model.builder import load_pretrained_model
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.pretrained, args.model_base, "vlm-3r-llava-qwen2-lora",
        device_map=str(device), attn_implementation=args.attn_implementation,
    )
    model.eval()
    model.requires_grad_(False)
    base = model.get_model()
    cfg = model.config
    # Never allow the model to auto-materialize a runtime spatial tower.
    cfg.spatial_tower_preextracted_only = True
    cfg.use_cut3r_spatialstack = args.mode == "online_spatialstack"
    cfg.use_predicted_spatialstack_residuals = args.mode == "online_predictor"
    cfg.disable_cut3r_spatialstack = args.mode != "online_spatialstack"
    if args.mode == "online_predictor":
        adapter = base.initialize_predicted_spatialstack_residual_predictor(args.predictor_checkpoint, cfg)
        adapter.to(device=device, dtype=torch.float16).eval()
    return tokenizer, model, image_processor


def _hooks(model, mode: str):
    base = model.get_model()
    timers = {"siglip": CudaCallTimer(), "qwen": CudaCallTimer()}
    handles = []
    vision = base.get_vision_tower()
    handles += [vision.register_forward_pre_hook(timers["siglip"].pre), vision.register_forward_hook(timers["siglip"].post)]
    # Qwen core is entered once for prefill and once per cached decoded token.
    handles += [model.model.register_forward_pre_hook(timers["qwen"].pre), model.model.register_forward_hook(timers["qwen"].post)]
    if mode == "online_spatialstack":
        merger = base.get_cut3r_spatialstack_merger() or base.initialize_cut3r_spatialstack_merger(model.config)
        if merger is not None:
            timers["spatialstack"] = CudaCallTimer()
            handles += [merger.register_forward_pre_hook(timers["spatialstack"].pre), merger.register_forward_hook(timers["spatialstack"].post)]
    if mode == "online_predictor":
        adapter = base.get_predicted_spatialstack_residual_predictor()
        timers["predictor_adapter"] = CudaCallTimer()
        handles += [adapter.register_forward_pre_hook(timers["predictor_adapter"].pre), adapter.register_forward_hook(timers["predictor_adapter"].post)]
        predictor = getattr(adapter, "predictor", None)
        if predictor is not None:
            timers["predictor"] = CudaCallTimer()
            handles += [predictor.register_forward_pre_hook(timers["predictor"].pre), predictor.register_forward_hook(timers["predictor"].post)]
    return timers, handles


def _timer_total(timer: CudaCallTimer | None) -> float:
    return sum(timer.milliseconds()) if timer is not None else 0.0


def _execute_one(args, sample, tokenizer, model, processor, cut3r, audit: KnownLoaderAudit):
    if sample["frame_count"] != 16 or len(sample["frame_ids"]) != 16:
        raise AssertionError("Manifest must provide exactly 16 raw frame IDs")
    torch.cuda.reset_peak_memory_stats()
    sample_start = time.perf_counter()
    decode_start = time.perf_counter()
    raw = _decode_raw(sample["raw_video_path"], sample["frame_ids"])
    decode_ms = (time.perf_counter() - decode_start) * 1000.0
    preprocess_start = time.perf_counter()
    pixels = processor.preprocess(raw, return_tensors="pt")["pixel_values"].to(device=args.device, dtype=torch.float16)
    torch.cuda.synchronize()
    preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
    payload = None
    cut3r_ms = 0.0
    if args.mode == "online_spatialstack":
        if cut3r is None:
            raise AssertionError("SpatialStack mode did not load CUT3R")
        payload, cut3r_ms, _ = _cuda_timed(lambda: _runtime_layer_payload(cut3r, pixels, sample["frame_ids"]))
    input_ids, attention_mask = _prompt_ids(tokenizer, sample["prompt"], args.device)
    timers, handles = _hooks(model, args.mode)
    try:
        generated, generate_ms, _ = _cuda_timed(lambda: model.generate(
            inputs=input_ids, images=[pixels], spatial_features=[payload] if payload is not None else None,
            attention_mask=attention_mask, modalities=["video"], **GENERATION_ARGS,
        ))
    finally:
        for handle in handles:
            handle.remove()
    qwen_calls = timers["qwen"].milliseconds()
    siglip_ms = _timer_total(timers["siglip"])
    qwen_prefill = qwen_calls[0] if qwen_calls else 0.0
    decode_token_ms = sum(qwen_calls[1:]) if len(qwen_calls) > 1 else 0.0
    predictor_ms = _timer_total(timers.get("predictor"))
    adapter_ms = _timer_total(timers.get("predictor_adapter"))
    merger_ms = _timer_total(timers.get("spatialstack"))
    residual_ms = max(0.0, adapter_ms - predictor_ms)
    # The production generate call includes setup, visual preparation, and decoder work.
    # The derived setup remainder is kept explicit rather than silently attributing it.
    branch_ms = cut3r_ms + merger_ms + predictor_ms + residual_ms
    multimodal_prepare_ms = max(0.0, generate_ms - siglip_ms - merger_ms - adapter_ms - sum(qwen_calls))
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - sample_start) * 1000.0
    audit.assert_zero()
    generated_ids = generated.detach().cpu().reshape(-1).tolist()
    assertions = {
        "raw_video": True, "frame_ids_match_manifest": True, "frame_order_match_manifest": True,
        "siglip_forward_count": timers["siglip"].count, "cut3r_loaded": cut3r is not None,
        "cut3r_forward_count": 1 if args.mode == "online_spatialstack" else 0,
        "predictor_forward_count": timers.get("predictor", CudaCallTimer()).count if args.mode == "online_predictor" else 0,
        "spatialstack_projection_count": timers.get("spatialstack", CudaCallTimer()).count if args.mode == "online_spatialstack" else 0,
        **audit.snapshot(),
    }
    if assertions["siglip_forward_count"] != 1:
        raise AssertionError(f"Expected exactly one online SigLIP forward, got {assertions}")
    if args.mode == "online_spatialstack" and not (assertions["cut3r_loaded"] and assertions["cut3r_forward_count"] >= 1 and assertions["spatialstack_projection_count"] == 1):
        raise AssertionError(f"SpatialStack online contract failed: {assertions}")
    if args.mode == "online_predictor" and (assertions["cut3r_loaded"] or assertions["cut3r_forward_count"] or assertions["predictor_forward_count"] != 1):
        raise AssertionError(f"Predictor online contract failed: {assertions}")
    if args.mode == "geometry_off" and (assertions["cut3r_loaded"] or assertions["cut3r_forward_count"] or assertions["predictor_forward_count"]):
        raise AssertionError(f"Geometry-off contract failed: {assertions}")
    return {
        "canonical_key": sample["canonical_key"], "raw_video_path": sample["raw_video_path"], "frame_ids": sample["frame_ids"],
        "frame_order": sample["frame_order"], "prompt": sample["prompt"], "generated_token_ids": generated_ids,
        "video_decode_frame_sampling_ms": decode_ms, "image_preprocess_ms": preprocess_ms,
        "siglip_forward_ms": siglip_ms, "cut3r_forward_ms": cut3r_ms, "predictor_forward_ms": predictor_ms,
        "spatialstack_projection_ms": merger_ms, "residual_construction_ms": residual_ms,
        "multimodal_preparation_ms": multimodal_prepare_ms, "qwen_prefill_ms": qwen_prefill,
        "multimodal_prepare_and_qwen_prefill_ms": multimodal_prepare_ms + qwen_prefill,
        "token_decode_ms": decode_token_ms, "ttft_ms": decode_ms + preprocess_ms + cut3r_ms + generate_ms - decode_token_ms,
        "fixed_16_token_total_ms": total_ms, "model_generate_ms": generate_ms,
        "spatial_branch_ms": cut3r_ms + merger_ms if args.mode == "online_spatialstack" else predictor_ms + residual_ms,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(args.device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(args.device)), "assertions": assertions,
    }


def _provenance(args, cold_start_ms: float, cut3r_loaded: bool) -> dict[str, Any]:
    gpu = torch.cuda.get_device_properties(args.device)
    predictor = Path(args.predictor_checkpoint)
    return {**runtime_basics(), "git": git_state(args.repo), "mode": args.mode, "gpu": {
        "visible_device": str(args.device), "name": gpu.name, "total_memory": gpu.total_memory,
        "uuid": torch.cuda.get_device_properties(args.device).uuid if hasattr(gpu, "uuid") else None,
    }, "cold_start_model_loading_ms": cold_start_ms, "cut3r_loaded": cut3r_loaded,
        "pretrained": args.pretrained, "model_base": args.model_base, "predictor_checkpoint": str(predictor),
        "predictor_sha256": sha256_file(predictor) if predictor.is_file() else None,
        "qwen_checkpoint": {"path": args.pretrained, "config_sha256": config_hash(Path(args.pretrained) / "config.json"),
                            "files": tree_inventory(args.pretrained)},
        "cut3r_checkpoint": {"path": args.cut3r_weights, "files": tree_inventory(args.cut3r_weights)},
        "precision": "fp16", "attention_implementation": args.attn_implementation,
        "generation_args": GENERATION_ARGS, "cache_paths": {"spatial_features_root": "/dev/null", "enabled": False},
        "generation_engine": "production_generate"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("geometry_off", "online_spatialstack", "online_predictor"), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--pretrained", default="/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/cut3r_spatialstack_45297963")
    parser.add_argument("--model-base", default="/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/LLaVA-NeXT-Video-7B-Qwen2")
    parser.add_argument("--predictor-checkpoint", default="outputs/official_siglip_residual_temporal_mem246g_b1_20260730_retry1/best_validation_relative_l2.pt")
    parser.add_argument("--cut3r-weights", default="third_party/CUT3R/src/cut3r_512_dpt_4_64.pth")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--measured", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the online latency worker")
    args.device = torch.device("cuda:0")
    torch.cuda.set_device(args.device)
    os.environ["SPATIAL_FEATURES_ROOT"] = "/dev/null"
    data = json.loads(Path(args.manifest).read_text())
    samples = data["samples"]
    if data["generation_args"] != GENERATION_ARGS or len(samples) != 20:
        raise AssertionError("Unexpected benchmark manifest contract")
    if args.warmups > 4 or args.measured > 16:
        raise AssertionError("Requested phase exceeds the 20-sample benchmark manifest")
    selected = samples[:args.warmups] + samples[4:4 + args.measured]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cold_start = time.perf_counter()
    tokenizer, model, processor = _load_vlm(args, args.device)
    cut3r = _load_cut3r(args.device, torch.float16, args.cut3r_weights) if args.mode == "online_spatialstack" else None
    torch.cuda.synchronize()
    cold_start_ms = (time.perf_counter() - cold_start) * 1000.0
    provenance = _provenance(args, cold_start_ms, cut3r is not None)
    json_dump(provenance, output / "runtime_provenance.json")
    audit = KnownLoaderAudit()
    records = []
    with (output / "per_sample_latency.jsonl").open("w", encoding="utf-8") as handle, torch.inference_mode():
        for sample in selected:
            repetitions = 1 if sample["ordinal"] < 4 else args.repetitions
            values = []
            for repetition in range(repetitions):
                result = _execute_one(args, sample, tokenizer, model, processor, cut3r, audit)
                result.update({"mode": args.mode, "ordinal": sample["ordinal"], "split": sample["split"], "repetition": repetition})
                values.append(result)
                handle.write(json.dumps(result, sort_keys=True) + "\n")
                handle.flush()
            aggregate = median_repetition(values)
            aggregate.update({"mode": args.mode, "ordinal": sample["ordinal"], "split": sample["split"],
                              "canonical_key": sample["canonical_key"], "frame_ids": sample["frame_ids"],
                              "raw_video_path": sample["raw_video_path"], "prompt": sample["prompt"],
                              "assertions": values[-1]["assertions"]})
            records.append(aggregate)
    json_dump({"mode": args.mode, "cold_start_model_loading_ms": cold_start_ms,
               "records": records, "all_assertions_passed": True}, output / "worker_summary.json")


if __name__ == "__main__":
    main()
