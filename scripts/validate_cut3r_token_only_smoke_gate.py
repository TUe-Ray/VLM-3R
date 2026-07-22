#!/usr/bin/env python3
"""Create or verify the machine-readable CUT3R-token-only smoke gate."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any

import torch

REQUIRED_BOOLEANS = (
    "siglip_forward_bypassed",
    "projector_grad_nonzero",
    "lora_grad_nonzero",
    "projector_weight_updated",
    "lora_weight_updated",
    "loss_improved",
    "checkpoint_saved",
    "checkpoint_reloaded",
    "resumed_forward_passed",
    "answer_labels_preserved",
    "all_finite",
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
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _metric_bool(records, key: str) -> bool:
    values = [record.get("metrics", {}).get(key) for record in records if key in record.get("metrics", {})]
    return bool(values) and all(bool(value) for value in values)


def _metric_any(records, key: str) -> bool:
    return any(bool(record.get("metrics", {}).get(key, False)) for record in records)


def _answer_labels_preserved(records) -> bool:
    observations = []
    for record in records:
        metrics = record.get("metrics", {})
        before = metrics.get("answer_labels_before_truncation")
        after = metrics.get("answer_labels_after_truncation")
        if before is None or after is None:
            continue
        before_values = before if isinstance(before, list) else [before]
        after_values = after if isinstance(after, list) else [after]
        observations.extend(zip(before_values, after_values))
    return bool(observations) and all(int(before) > 0 and int(before) == int(after) for before, after in observations)


def _losses(records) -> list[float]:
    values = []
    for record in records:
        value = record.get("trainer_log", {}).get("loss")
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted((path for path in run_dir.rglob("checkpoint-*") if path.is_dir()), key=lambda path: path.name)
    return checkpoints[-1] if checkpoints else None


def _checkpoint_evidence(checkpoint: Path | None) -> tuple[bool, bool, bool, bool]:
    if checkpoint is None:
        return False, False, False, False
    config = _read_json(checkpoint / "config.json", {})
    state_path = checkpoint / "non_lora_trainables.bin"
    adapter_present = any((checkpoint / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin"))
    if not state_path.is_file() or not adapter_present:
        return False, False, False, False
    state = torch.load(state_path, map_location="cpu")
    projector_keys = [key for key in state if "cut3r_token_projector" in key]
    adapter_keys = []
    adapter_bin = checkpoint / "adapter_model.bin"
    adapter_safe = checkpoint / "adapter_model.safetensors"
    if adapter_bin.is_file():
        adapter_keys = list(torch.load(adapter_bin, map_location="cpu").keys())
    elif adapter_safe.is_file():
        from safetensors import safe_open
        with safe_open(str(adapter_safe), framework="pt", device="cpu") as handle:
            adapter_keys = list(handle.keys())
    projector_has_lora = any("cut3r_token_projector" in key and "lora_" in key for key in adapter_keys)
    saved = bool(config.get("visual_token_source") == "cut3r_only" and projector_keys)
    return saved, projector_has_lora, bool(adapter_present), bool(projector_keys)


def _preflight_evidence(run_dir: Path) -> dict[str, Any]:
    for candidate in (
        run_dir / "cut3r_token_only_preflight.json",
        run_dir / "evaluator_preflight.json",
        run_dir / "diagnostics" / "cut3r_token_only_preflight.json",
    ):
        payload = _read_json(candidate)
        if isinstance(payload, dict):
            return payload
    return {}


def _build_gate(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    repo = Path(args.repo).resolve()
    records = _read_jsonl(run_dir / "cut3r_token_only_metrics.jsonl")
    losses = _losses(records)
    bucket = max(1, len(losses) // 5)
    initial_median = statistics.median(losses[:bucket]) if losses else None
    final_median = statistics.median(losses[-bucket:]) if losses else None
    checkpoint = _latest_checkpoint(run_dir)
    checkpoint_saved, projector_has_lora, adapter_present, projector_present = _checkpoint_evidence(checkpoint)
    preflight = _preflight_evidence(run_dir)
    commit = _git_commit(repo)
    gate = {
        "passed": False,
        "commit": commit,
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip(),
        "job_id": args.job_id or os.environ.get("SLURM_JOB_ID", ""),
        "run_name": args.run_name or run_dir.name,
        "visual_token_source": "cut3r_only",
        "siglip_forward_bypassed": _metric_bool(records, "siglip_forward_bypassed"),
        "projector_has_lora": projector_has_lora,
        "projector_grad_nonzero": _metric_any(records, "projector_grad_nonzero"),
        "lora_grad_nonzero": _metric_any(records, "lora_grad_nonzero"),
        "projector_weight_updated": _metric_any(records, "projector_weight_updated"),
        "lora_weight_updated": _metric_any(records, "lora_weight_updated"),
        "loss_initial_median": initial_median,
        "loss_final_median": final_median,
        "loss_improved": bool(initial_median is not None and final_median is not None and final_median < initial_median),
        "loss_smoothed_slope": None if len(losses) < 2 else (losses[-1] - losses[0]) / (len(losses) - 1),
        "checkpoint_saved": checkpoint_saved,
        "checkpoint_reloaded": bool(preflight.get("checkpoint_reloaded", False)),
        "resumed_forward_passed": bool(preflight.get("resumed_forward_passed", False)),
        "answer_labels_preserved": _answer_labels_preserved(records),
        "all_finite": _metric_bool(records, "all_finite"),
        "checkpoint_path": str(checkpoint) if checkpoint else None,
        "adapter_present": adapter_present,
        "projector_state_present": projector_present,
        "metric_records": len(records),
        "loss_records": len(losses),
        "evaluator_preflight_passed": bool(preflight.get("evaluator_preflight_passed", False)),
        "preflight_report": preflight.get("report_path"),
    }
    gate["passed"] = (
        gate["metric_records"] >= args.min_metric_records
        and gate["loss_records"] >= args.min_loss_records
        and gate["evaluator_preflight_passed"]
        and not gate["projector_has_lora"]
        and all(bool(gate[key]) for key in REQUIRED_BOOLEANS)
    )
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
