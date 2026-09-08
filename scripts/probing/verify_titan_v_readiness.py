#!/usr/bin/env python
"""Verify the runner-selected CUDA device before any ScanNet probe workload."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def add_check(report: dict[str, Any], name: str, passed: bool, detail: Any) -> None:
    report["checks"].append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "schema_version": "titan_v_runner_readiness_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "physical_gpu_id": str(args.physical_gpu_id),
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "checks": [],
    }
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--id",
                str(args.physical_gpu_id),
                "--query-gpu=index,name,driver_version,memory.total,memory.used",
                "--format=csv,noheader",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        report["nvidia_smi"] = smi.stdout.strip()
        add_check(report, "physical_gpu_nvidia_smi", bool(smi.stdout.strip()), report["nvidia_smi"])
    except Exception as exc:
        add_check(report, "physical_gpu_nvidia_smi", False, str(exc))

    available = bool(torch.cuda.is_available())
    add_check(report, "torch_cuda_available", available, available)
    count = int(torch.cuda.device_count()) if available else 0
    add_check(report, "logical_device_count", count >= 1, count)
    compiled_arch_flags = str(torch._C._cuda_getArchFlags()).split()
    runtime_arch_list = torch.cuda.get_arch_list() if available else []
    report["torch_compiled_arch_flags"] = compiled_arch_flags
    report["torch_runtime_arch_list"] = runtime_arch_list
    add_check(report, "torch_build_sm70", "sm_70" in compiled_arch_flags, compiled_arch_flags)
    if available and count >= 1:
        name = torch.cuda.get_device_name(0)
        capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
        properties = torch.cuda.get_device_properties(0)
        report["logical_cuda_0"] = {
            "name": name,
            "capability": list(capability),
            "memory_total_bytes": int(properties.total_memory),
        }
        add_check(report, "logical_device_name", "titan v" in name.lower(), name)
        add_check(report, "logical_device_capability", capability == (7, 0), capability)
        try:
            left = torch.ones((32, 32), device="cuda:0", dtype=torch.float16)
            result = left @ left
            torch.cuda.synchronize(0)
            add_check(report, "fp16_matmul_and_synchronize", bool(torch.isfinite(result).all().item()), "32x32 fp16")
        except Exception as exc:
            add_check(report, "fp16_matmul_and_synchronize", False, str(exc))
    else:
        add_check(report, "logical_device_name", False, "No logical cuda:0 available")
        add_check(report, "logical_device_capability", False, "No logical cuda:0 available")
        add_check(report, "fp16_matmul_and_synchronize", False, "No logical cuda:0 available")

    report["assessment"] = "PASS" if all(check["status"] == "PASS" for check in report["checks"]) else "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["assessment"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
