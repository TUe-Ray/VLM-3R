"""Opt-in, allocation-free CUDA memory telemetry for controlled smoke runs."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def memory_audit_enabled() -> bool:
    return os.environ.get("DUAL_PATH_MEMORY_AUDIT", "").strip().lower() in {
        "1", "true", "yes", "y", "on",
    }


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)) or 0)


def record_cuda_memory(
    stage: str,
    *,
    extra: Optional[Dict[str, Any]] = None,
    synchronize: bool = True,
) -> None:
    """Append one rank-local JSON record without retaining tensor references."""
    if not memory_audit_enabled():
        return
    rank = _rank()
    payload: Dict[str, Any] = {
        "stage": str(stage),
        "rank": rank,
        "local_rank": int(os.environ.get("LOCAL_RANK", 0) or 0),
        "time": time.time(),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        if synchronize:
            torch.cuda.synchronize()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        payload.update(
            allocated_bytes=int(torch.cuda.memory_allocated()),
            reserved_bytes=int(torch.cuda.memory_reserved()),
            max_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            max_reserved_bytes=int(torch.cuda.max_memory_reserved()),
            free_bytes=int(free_bytes),
            total_bytes=int(total_bytes),
            device=str(torch.cuda.current_device()),
        )
    if extra:
        payload["extra"] = extra
    audit_dir = Path(os.environ.get("DUAL_PATH_MEMORY_AUDIT_DIR", "logs/diagnostics/memory_audit"))
    audit_dir.mkdir(parents=True, exist_ok=True)
    with (audit_dir / f"rank_{rank}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def reset_cuda_memory_peak(stage: str = "peak_reset") -> None:
    if not memory_audit_enabled() or not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    record_cuda_memory(stage, synchronize=False)
