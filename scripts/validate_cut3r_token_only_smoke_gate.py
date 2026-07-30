#!/usr/bin/env python3
"""Create or verify the machine-readable CUT3R-token-only smoke gate."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any

import torch
try:
    from cut3r_token_only_checkpoint_evidence import checkpoint_delta_evidence
except ModuleNotFoundError:
    from scripts.cut3r_token_only_checkpoint_evidence import checkpoint_delta_evidence


REQUIRED_BOOLEANS = (
    "siglip_forward_bypassed",
    "global_grad_norm_nonzero",
    "checkpoint_delta_evidence_complete",
    "projector_checkpoint_delta_nonzero",
    "lora_checkpoint_delta_nonzero",
    "checkpoint_saved",
    "checkpoint_reloaded",
    "resumed_forward_passed",
    "answer_labels_preserved",
    "all_finite",
    "evaluator_preflight_passed",
    "projector_checkpoint_values_verified",
)


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _read_json(path: Path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _metric_bool(records, key: str) -> bool:
    values = [record.get("metrics", {}).get(key) for record in records if key in record.get("metrics", {})]
    return bool(values) and all(bool(value) for value in values)


def _answer_labels_preserved(records) -> bool:
    observations = []
    for record in records:
        metrics = record.get("metrics", {})
        before, after = metrics.get("answer_labels_before_truncation"), metrics.get("answer_labels_after_truncation")
        if before is None or after is None:
            continue
        before_values = before if isinstance(before, list) else [before]
        after_values = after if isinstance(after, list) else [after]
        observations.extend(zip(before_values, after_values))
    return bool(observations) and all(int(before) > 0 and int(before) == int(after) for before, after in observations)


def _losses(records) -> list[float]:
    return [float(value) for record in records if isinstance((value := record.get("trainer_log", {}).get("loss")), (int, float))]


def _global_grad_norms(records) -> list[float]:
    return [float(value) for record in records if isinstance((value := record.get("trainer_log", {}).get("grad_norm")), (int, float))]


def _checkpoint_step(path: Path) -> int | None:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else None


def _latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = [(step, path) for path in run_dir.rglob("checkpoint-*")
                   if path.is_dir() and (step := _checkpoint_step(path)) is not None]
    return max(checkpoints, key=lambda item: item[0])[1] if checkpoints else None


def _checkpoint_evidence(checkpoint: Path | None) -> tuple[bool, bool, bool, bool]:
    if checkpoint is None:
        return False, False, False, False
    config = _read_json(checkpoint / "config.json", {})
    state_path = checkpoint / "non_lora_trainables.bin"
    adapter_paths = [checkpoint / name for name in ("adapter_model.bin", "adapter_model.safetensors")]
    if not state_path.is_file() or not any(path.is_file() for path in adapter_paths):
        return False, False, False, False
    try:
        state = torch.load(state_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(state_path, map_location="cpu")
    projector_keys = [key for key in state if "cut3r_token_projector" in key]
    if (checkpoint / "adapter_model.bin").is_file():
        try:
            adapter_keys = torch.load(checkpoint / "adapter_model.bin", map_location="cpu", weights_only=True).keys()
        except TypeError:
            adapter_keys = torch.load(checkpoint / "adapter_model.bin", map_location="cpu").keys()
    else:
        from safetensors import safe_open
        with safe_open(str(checkpoint / "adapter_model.safetensors"), framework="pt", device="cpu") as handle:
            adapter_keys = list(handle.keys())
    projector_has_lora = any("cut3r_token_projector" in key and "lora_" in key for key in adapter_keys)
    saved = bool(config.get("visual_token_source") == "cut3r_only" and projector_keys)
    return saved, projector_has_lora, True, bool(projector_keys)


def _preflight_evidence(run_dir: Path) -> dict[str, Any]:
    for candidate in (run_dir / "cut3r_token_only_preflight.json", run_dir / "evaluator_preflight.json", run_dir / "diagnostics" / "cut3r_token_only_preflight.json"):
        payload = _read_json(candidate)
        if isinstance(payload, dict):
            return payload
    return {}


def _build_gate(args: argparse.Namespace) -> dict[str, Any]:
    run_dir, repo = Path(args.run_dir).resolve(), Path(args.repo).resolve()
    records = _read_jsonl(run_dir / "cut3r_token_only_metrics.jsonl")
    losses, grad_norms = _losses(records), _global_grad_norms(records)
    bucket = max(1, len(losses) // 5)
    initial_median = statistics.median(losses[:bucket]) if losses else None
    final_median = statistics.median(losses[-bucket:]) if losses else None
    checkpoint = _latest_checkpoint(run_dir)
    checkpoint_saved, projector_has_lora, adapter_present, projector_present = _checkpoint_evidence(checkpoint)
    delta = checkpoint_delta_evidence(run_dir, checkpoint)
    projector_delta = delta.get("groups", {}).get("projector", {})
    lora_delta = delta.get("groups", {}).get("lora", {})
    preflight = _preflight_evidence(run_dir)
    finite_grad_norms = bool(grad_norms) and all(math.isfinite(value) for value in grad_norms)
    gate = {
        "passed": False,
        "commit": _git_commit(repo),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip(),
        "job_id": args.job_id or os.environ.get("SLURM_JOB_ID", ""),
        "run_name": args.run_name or run_dir.name,
        "visual_token_source": "cut3r_only",
        "siglip_forward_bypassed": bool(_metric_bool(records, "siglip_forward_bypassed") and preflight.get("siglip_forward_bypassed", False)),
        "projector_has_lora": projector_has_lora,
        "global_grad_norm_values": grad_norms,
        "global_grad_norm_finite": finite_grad_norms,
        "global_grad_norm_nonzero": bool(finite_grad_norms and any(value > 0.0 for value in grad_norms)),
        "checkpoint_delta_evidence": delta,
        "checkpoint_delta_evidence_complete": bool(delta.get("complete", False)),
        "projector_checkpoint_delta_norm": projector_delta.get("delta_norm"),
        "projector_checkpoint_delta_finite": bool(projector_delta.get("finite", False)),
        "projector_checkpoint_delta_nonzero": bool(projector_delta.get("nonzero", False)),
        "lora_checkpoint_delta_norm": lora_delta.get("delta_norm"),
        "lora_checkpoint_delta_finite": bool(lora_delta.get("finite", False)),
        "lora_checkpoint_delta_nonzero": bool(lora_delta.get("nonzero", False)),
        "loss_initial_median": initial_median,
        "loss_final_median": final_median,
        "loss_improved": bool(initial_median is not None and final_median is not None and final_median < initial_median),
        "loss_smoothed_slope": None if len(losses) < 2 else (losses[-1] - losses[0]) / (len(losses) - 1),
        "checkpoint_saved": checkpoint_saved,
        "checkpoint_reloaded": bool(preflight.get("checkpoint_reloaded", False)),
        "resumed_forward_passed": bool(preflight.get("resumed_forward_passed", False)),
        "answer_labels_preserved": _answer_labels_preserved(records),
        "all_finite": bool(_metric_bool(records, "all_finite") and finite_grad_norms and projector_delta.get("finite", False) and lora_delta.get("finite", False)),
        "checkpoint_path": str(checkpoint) if checkpoint else None,
        "adapter_present": adapter_present,
        "projector_state_present": projector_present,
        "metric_records": len(records),
        "loss_records": len(losses),
        "evaluator_preflight_passed": bool(preflight.get("evaluator_preflight_passed", False)),
        "projector_checkpoint_values_verified": bool(preflight.get("projector_checkpoint_values_verified", False)),
        "preflight_report": preflight.get("report_path"),
    }
    gate["passed"] = (gate["metric_records"] >= args.min_metric_records and gate["loss_records"] >= args.min_loss_records and not gate["projector_has_lora"] and all(bool(gate[key]) for key in REQUIRED_BOOLEANS))
    return gate


def _verify(gate_path: Path, expected_commit: str | None) -> int:
    gate = _read_json(gate_path)
    if not isinstance(gate, dict):
        raise RuntimeError(f"Smoke gate is missing or invalid JSON: {gate_path}")
    missing = [key for key in REQUIRED_BOOLEANS if gate.get(key) is not True]
    if gate.get("projector_has_lora") is not False:
        missing.append("projector_has_lora must be false")
    if gate.get("passed") is not True or missing:
        raise RuntimeError(f"Smoke gate did not pass: passed={gate.get('passed')}, missing={missing}")
    if expected_commit and gate.get("commit") != expected_commit:
        raise RuntimeError(f"Smoke gate commit mismatch: gate={gate.get('commit')}, expected={expected_commit}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir")
    parser.add_argument("--gate")
    parser.add_argument("--repo", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--job-id")
    parser.add_argument("--run-name")
    parser.add_argument("--min-metric-records", type=int, default=5)
    parser.add_argument("--min-loss-records", type=int, default=5)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--expected-commit")
    args = parser.parse_args()
    if args.verify:
        if not args.gate:
            parser.error("--verify requires --gate")
        return _verify(Path(args.gate).resolve(), args.expected_commit)
    if not args.run_dir:
        parser.error("gate creation requires --run-dir")
    gate = _build_gate(args)
    gate_path = Path(args.gate).resolve() if args.gate else Path(args.run_dir).resolve() / "smoke_gate.json"
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[CUT3R_TOKEN_ONLY_SMOKE_GATE] " + json.dumps(gate, sort_keys=True))
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
