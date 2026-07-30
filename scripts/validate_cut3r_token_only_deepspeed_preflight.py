#!/usr/bin/env python3
"""Build evidence-backed output for the real two-step CUT3R DeepSpeed preflight."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import torch

try:
    from cut3r_token_only_checkpoint_evidence import checkpoint_delta_evidence, checkpoint_step, latest_checkpoint
except ModuleNotFoundError:
    from scripts.cut3r_token_only_checkpoint_evidence import checkpoint_delta_evidence, checkpoint_step, latest_checkpoint


def _json(path: Path):
    return json.loads(path.read_text()) if path.is_file() else {}


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def _checkpoint_state(checkpoint: Path | None):
    if checkpoint is None:
        return {}, False, False
    state_path = checkpoint / "non_lora_trainables.bin"
    if not state_path.is_file():
        return {}, False, False
    try:
        non_lora = torch.load(state_path, map_location="cpu", weights_only=True)
    except TypeError:
        non_lora = torch.load(state_path, map_location="cpu")
    adapter = any((checkpoint / name).is_file() for name in ("adapter_model.bin", "adapter_model.safetensors"))
    return non_lora, adapter, bool([name for name in non_lora if "cut3r_token_projector" in name])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--log-path", default="")
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    metrics = _jsonl(root / "cut3r_token_only_metrics.jsonl")
    runtime = _json(root / "cut3r_token_only_runtime.json")
    evaluator = _json(root / "cut3r_token_only_preflight.json")
    checkpoint = latest_checkpoint(root)
    config = _json(checkpoint / "config.json") if checkpoint else {}
    non_lora, adapter, projector_state = _checkpoint_state(checkpoint)
    delta = checkpoint_delta_evidence(root, checkpoint)
    projector_delta = delta.get("groups", {}).get("projector", {})
    lora_delta = delta.get("groups", {}).get("lora", {})
    grad_norms = [float(value) for item in metrics if isinstance((value := item.get("trainer_log", {}).get("grad_norm")), (int, float))]
    losses = [float(value) for item in metrics if isinstance((value := item.get("trainer_log", {}).get("loss")), (int, float))]
    completed_step = checkpoint_step(checkpoint) if checkpoint else None
    label_observations = []
    for item in metrics:
        values = item.get("metrics", {})
        before, after = values.get("answer_labels_before_truncation"), values.get("answer_labels_after_truncation")
        if before is None or after is None:
            continue
        before_values = before if isinstance(before, list) else [before]
        after_values = after if isinstance(after, list) else [after]
        label_observations.extend(zip(before_values, after_values))
    answer_ok = bool(label_observations) and all(int(before) > 0 and int(before) == int(after) for before, after in label_observations)
    all_finite = bool(metrics) and all(bool(item.get("metrics", {}).get("all_finite", False)) for item in metrics) and bool(grad_norms) and all(math.isfinite(value) for value in grad_norms) and all(math.isfinite(value) for value in losses) and bool(projector_delta.get("finite", False)) and bool(lora_delta.get("finite", False))
    report = {
        "passed": False,
        "job_id": args.job_id,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "world_size": runtime.get("world_size"),
        "all_four_ranks_initialized": runtime.get("world_size") == 4,
        "deepspeed_zero_stage": runtime.get("deepspeed_zero_stage"),
        "optimizer_class": runtime.get("trainer_optimizer_class"),
        "prepared_optimizer_class": runtime.get("prepared_optimizer_class"),
        "deepspeed_engine_class": runtime.get("deepspeed_engine_class"),
        "accelerate_distributed_type": runtime.get("accelerate_distributed_type"),
        "cut3r_only_active": runtime.get("cut3r_only_active") is True,
        "siglip_forward_bypassed": any(bool(item.get("metrics", {}).get("siglip_forward_bypassed")) for item in metrics),
        "manifest_verified": bool(runtime.get("manifest_path")),
        "frame_alignment_verified": bool(runtime.get("manifest_path")),
        "projector_has_lora": False,
        "global_grad_norms": grad_norms,
        "global_grad_norm_finite": bool(grad_norms) and all(math.isfinite(value) for value in grad_norms),
        "global_grad_norm_nonzero": bool(grad_norms) and any(value > 0.0 for value in grad_norms),
        "losses": losses,
        "answer_labels_preserved": answer_ok,
        "all_finite": all_finite,
        "completed_optimizer_steps": completed_step,
        "checkpoint_step_two_saved": completed_step == 2,
        "checkpoint_saved": bool(checkpoint and config.get("visual_token_source") == "cut3r_only" and projector_state and adapter),
        "checkpoint_path": str(checkpoint) if checkpoint else None,
        "checkpoint_delta_evidence": delta,
        "projector_checkpoint_delta_norm": projector_delta.get("delta_norm"),
        "projector_checkpoint_delta_nonzero": bool(projector_delta.get("nonzero", False)),
        "sampled_lora_checkpoint_delta_norm": lora_delta.get("delta_norm"),
        "sampled_lora_checkpoint_delta_nonzero": bool(lora_delta.get("nonzero", False)),
        "sampled_projector_parameters": projector_delta.get("sampled_parameter_names", []),
        "sampled_lora_parameters": lora_delta.get("sampled_parameter_names", []),
        "checkpoint_reloaded": bool(evaluator.get("checkpoint_reloaded")),
        "resumed_forward_passed": bool(evaluator.get("resumed_forward_passed")),
        "evaluator_preflight_passed": bool(evaluator.get("evaluator_preflight_passed")),
        "projector_checkpoint_values_verified": bool(evaluator.get("projector_checkpoint_values_verified")),
        "log_path": args.log_path or None,
    }
    required = (
        "all_four_ranks_initialized", "cut3r_only_active", "siglip_forward_bypassed", "manifest_verified", "frame_alignment_verified",
        "global_grad_norm_finite", "global_grad_norm_nonzero", "projector_checkpoint_delta_nonzero", "sampled_lora_checkpoint_delta_nonzero",
        "answer_labels_preserved", "all_finite", "checkpoint_step_two_saved", "checkpoint_saved", "checkpoint_reloaded",
        "resumed_forward_passed", "evaluator_preflight_passed", "projector_checkpoint_values_verified",
    )
    report["passed"] = all(report[key] is True for key in required) and bool(delta.get("complete", False)) and report["projector_has_lora"] is False
    output = root / f"deepspeed_preflight_{args.job_id}.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("DeepSpeed preflight report failed; inspect the JSON evidence.")


if __name__ == "__main__":
    main()
