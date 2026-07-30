from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import socket
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

MANIFEST_VERSION = 1
GENERATION_ARGS = {
    "do_sample": False,
    "min_new_tokens": 16,
    "max_new_tokens": 16,
    "num_beams": 1,
    "use_cache": True,
}


def json_dump(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_hash(path: str | Path) -> str | None:
    path = Path(path)
    return sha256_file(path) if path.is_file() else None


def tree_inventory(root: str | Path) -> list[dict[str, Any]]:
    """Cheap checkpoint provenance: names, sizes and mtimes, never content hashes."""
    root = Path(root)
    if root.is_file():
        paths = [root]
        base = root.parent
    elif root.is_dir():
        paths = sorted(path for path in root.rglob("*") if path.is_file())
        base = root
    else:
        return []
    return [{
        "path": str(path.relative_to(base)), "bytes": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
    } for path in paths]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q, method="linear"))


def bootstrap_mean_ci(values: list[float], seed: int = 20260730, draws: int = 4000) -> list[float] | None:
    if not values:
        return None
    rng = np.random.default_rng(seed)
    x = np.asarray(values, dtype=float)
    means = rng.choice(x, size=(draws, len(x)), replace=True).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(values: Iterable[float]) -> dict[str, Any]:
    x = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return {
        "n": len(x), "mean_ms": float(statistics.mean(x)) if x else None,
        "median_ms": float(statistics.median(x)) if x else None,
        "p90_ms": percentile(x, 90), "p95_ms": percentile(x, 95),
        "std_ms": float(statistics.stdev(x)) if len(x) > 1 else 0.0 if x else None,
        "bootstrap_mean_95_ci_ms": bootstrap_mean_ci(x),
    }


def median_repetition(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep scalar timing medians but conservatively retain maximum peak memory."""
    if not records:
        raise ValueError("No repetition records to aggregate")
    result = {"repetitions": len(records)}
    keys = set().union(*(record.keys() for record in records))
    for key in keys:
        values = [record.get(key) for record in records]
        numeric = [float(v) for v in values if isinstance(v, (int, float))]
        if not numeric:
            continue
        if key.startswith("peak_"):
            result[key] = max(numeric)
        else:
            result[key] = float(statistics.median(numeric))
    return result


def git_state(repo: str | Path) -> dict[str, Any]:
    def output(*args: str) -> str | None:
        try:
            return subprocess.check_output(args, cwd=repo, text=True).strip()
        except Exception:
            return None
    return {"sha": output("git", "rev-parse", "HEAD"), "branch": output("git", "branch", "--show-current"),
            "status_short": output("git", "status", "--short")}


def runtime_basics() -> dict[str, Any]:
    import torch
    return {"hostname": socket.gethostname(), "pid": os.getpid(), "python": platform.python_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "timestamp_unix": time.time()}


def csv_write(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
