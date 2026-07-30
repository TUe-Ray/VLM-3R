#!/usr/bin/env python3
"""Write the job-level provenance required for an interpolation evaluation."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path


def results_path(output_path: Path) -> str | None:
    matches = sorted(output_path.glob("**/results.json"))
    if len(matches) == 1:
        return str(matches[0].resolve())
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--execution-mode", choices=("single_gpu", "single_node_4gpu"), required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--num-processes", type=int, required=True)
    parser.add_argument("--accelerate-launch-args", required=True)
    parser.add_argument("--phase", choices=("start", "complete"), required=True)
    args = parser.parse_args()

    output_path = Path(args.output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / "execution_provenance.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    now = time.time()
    started_at = existing.get("started_at_unix", now)
    provenance = {
        **existing,
        "execution_mode": args.execution_mode,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "node_count": int(os.environ.get("SLURM_JOB_NUM_NODES", "1")),
        "gpu_count": int(os.environ.get("SLURM_GPUS_ON_NODE", args.world_size)),
        "world_size": args.world_size,
        "NUM_PROCESSES": args.num_processes,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "accelerate_launch_arguments": args.accelerate_launch_args,
        "hostname": socket.gethostname(),
        "rank_information": {
            "slurm_procid": os.environ.get("SLURM_PROCID"),
            "slurm_localid": os.environ.get("SLURM_LOCALID"),
            "slurm_nprocs": os.environ.get("SLURM_NPROCS"),
            "slurm_ntasks": os.environ.get("SLURM_NTASKS"),
        },
        "model_dtype": os.environ.get("MODEL_DTYPE", "bfloat16"),
        "generation_configuration": {
            "max_new_tokens": 16,
            "temperature": 0.0,
            "top_p": 1.0,
            "num_beams": 1,
            "do_sample": False,
        },
        "output_directory": str(output_path),
        "telemetry_directory": str((output_path / "telemetry").resolve()),
        "expected_key_manifest": os.environ.get("EXPECTED_KEY_MANIFEST"),
        "expected_key_manifest_sha256": os.environ.get("EXPECTED_KEY_MANIFEST_SHA256"),
        "started_at_unix": started_at,
    }
    if args.phase == "complete":
        provenance.update({
            "completed_at_unix": now,
            "wall_clock_seconds": now - float(started_at),
            "results_json_path": results_path(output_path),
        })
    path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
