#!/usr/bin/env python3
"""Build a non-forgeable report for the two-step real DeepSpeed preflight."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import torch


def _json(path: Path):
    return json.loads(path.read_text()) if path.is_file() else {}


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def _checkpoint(root: Path):
    choices = []
    for path in root.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if path.is_dir() and match:
            choices.append((int(match.group(1)), path))
    return max(choices)[1] if choices else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--log-path", default="")
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    evidence = {int(item["optimizer_step"]): item for item in _jsonl(root / "cut3r_token_only_optimizer_steps.jsonl") if "optimizer_step" in item}
    steps = [evidence.get(step, {}) for step in (1, 2)]
    metrics = _jsonl(root / "cut3r_token_only_metrics.jsonl")
    runtime = _json(root / "cut3r_token_only_runtime.json")
    evaluator = _json(root / "cut3r_token_only_preflight.json")
    checkpoint = _checkpoint(root)
    config = _json(checkpoint / "config.json") if checkpoint else {}
    non_lora = torch.load(checkpoint / "non_lora_trainables.bin", map_location="cpu") if checkpoint and (checkpoint / "non_lora_trainables.bin").is_file() else {}
    adapter = bool(checkpoint and any((checkpoint / name).is_file() for name in ("adapter_model.bin", "adapter_model.safetensors")))
    def any_step(key): return all(bool(item.get(key, False)) for item in steps)
    answer_ok = any(bool(item.get("metrics", {}).get("answer_labels_after_truncation", 0)) for item in metrics)
    finite = bool(metrics) and all(bool(item.get("metrics", {}).get("all_finite", False)) for item in metrics) and any_step("all_finite")
    projector_state = [name for name in non_lora if "cut3r_token_projector" in name]
    report = {
        "passed": False, "job_id": args.job_id,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "world_size": runtime.get("world_size"), "deepspeed_zero_stage": runtime.get("deepspeed_zero_stage"),
        "optimizer_class": runtime.get("trainer_optimizer_class"), "prepared_optimizer_class": runtime.get("prepared_optimizer_class"),
        "deepspeed_engine_class": runtime.get("deepspeed_engine_class"), "accelerate_distributed_type": runtime.get("accelerate_distributed_type"),
        "cut3r_only_active": runtime.get("cut3r_only_active") is True,
        "siglip_forward_bypassed": any(bool(item.get("metrics", {}).get("siglip_forward_bypassed")) for item in metrics),
        "manifest_verified": bool(runtime.get("manifest_path")), "frame_alignment_verified": bool(runtime.get("manifest_path")),
        "projector_has_lora": False,
        "projector_grad_nonzero": any_step("projector_grad_nonzero"), "sampled_lora_grad_nonzero": any_step("lora_grad_nonzero"),
        "projector_update_nonzero": any_step("projector_weight_updated"), "sampled_lora_update_nonzero": any_step("lora_weight_updated"),
        "answer_labels_preserved": answer_ok, "all_finite": finite,
        "checkpoint_saved": bool(checkpoint and config.get("visual_token_source") == "cut3r_only" and projector_state and adapter),
        "checkpoint_reloaded": bool(evaluator.get("checkpoint_reloaded")),
        "resumed_forward_passed": bool(evaluator.get("resumed_forward_passed")),
        "evaluator_preflight_passed": bool(evaluator.get("evaluator_preflight_passed")),
        "optimizer_step_evidence": {str(step): evidence.get(step) for step in (1, 2)},
        "checkpoint_path": str(checkpoint) if checkpoint else None,
        "sampled_projector_parameters": steps[0].get("projector_sampled_parameter_names", []),
        "sampled_lora_parameters": steps[0].get("lora_sampled_parameter_names", []),
        "log_path": args.log_path or None,
    }
    required = ("cut3r_only_active", "siglip_forward_bypassed", "manifest_verified", "frame_alignment_verified", "projector_grad_nonzero", "sampled_lora_grad_nonzero", "projector_update_nonzero", "sampled_lora_update_nonzero", "answer_labels_preserved", "all_finite", "checkpoint_saved", "checkpoint_reloaded", "resumed_forward_passed", "evaluator_preflight_passed")
    report["passed"] = all(report[key] is True for key in required) and len(steps) == 2
    output = root / f"deepspeed_preflight_{args.job_id}.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("DeepSpeed preflight report failed; inspect the JSON evidence.")


if __name__ == "__main__":
    main()
