#!/usr/bin/env python3
"""Validate and summarize the ScanNet depth probes for baseline replications."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from post_sft_geometry_probe_specs import (
    POST_SFT_DEPTH_FEATURE_LEVELS,
    POST_SFT_DEPTH_LAYERS,
    POST_SFT_PRE_LLM_FEATURES,
)

LEVELS = POST_SFT_DEPTH_FEATURE_LEVELS
EXPECTED_TOKENS = 75_656
REQUIRED_CHECKPOINT_FILES = (
    "adapter_model.bin",
    "non_lora_trainables.bin",
    "adapter_config.json",
    "config.json",
    "generation_config.json",
)
MIGRATED_PATH_KEYS = {"_name_or_path", "mm_vision_tower", "vision_tower", "weights_path"}


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(path: Path) -> dict[str, Any]:
    files = {}
    for name in REQUIRED_CHECKPOINT_FILES:
        item = path / name
        files[name] = {
            "exists": item.is_file(),
            "size": item.stat().st_size if item.is_file() else None,
            "sha256": sha256(item) if item.is_file() else None,
        }
    return {"path": str(path), "complete": all(v["exists"] for v in files.values()), "files": files}


def normalized_baseline_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in MIGRATED_PATH_KEYS
        and key != "transformers_version"
        and not key.startswith("eomt")
        and not key.startswith("mm_eomt")
    }


def normalized_lora_recipe(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "peft_type": config.get("peft_type"),
        "r": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
        "lora_dropout": config.get("lora_dropout"),
        "target_modules": sorted(config.get("target_modules", [])),
    }


def validate_model(
    label: str,
    checkpoint: Path,
    template: Path,
    durable_root: Path,
) -> dict[str, Any]:
    identity = checkpoint_identity(checkpoint)
    config_match = False
    lora_recipe_match = False
    disabled_eomt = True
    if identity["complete"]:
        config = read_json(checkpoint / "config.json")
        template_config = read_json(template / "config.json")
        config_match = normalized_baseline_config(config) == normalized_baseline_config(template_config)
        lora_recipe_match = normalized_lora_recipe(read_json(checkpoint / "adapter_config.json")) == normalized_lora_recipe(
            read_json(template / "adapter_config.json")
        )
        disabled_eomt = not bool(config.get("mm_eomt_enable_object_block", False)) and not bool(
            config.get("mm_eomt_selective_3d_enable", False)
        )

    present: list[str] = []
    invalid: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for level in LEVELS:
        metric_path = durable_root / "probes" / label / level / "metrics.json"
        checkpoint_path = durable_root / "probes" / label / level / "best.pt"
        history_path = durable_root / "probes" / label / level / "history.json"
        if not (metric_path.is_file() and checkpoint_path.is_file() and history_path.is_file()):
            continue
        try:
            metric = read_json(metric_path)
        except Exception as exc:  # pragma: no cover - diagnostic path
            invalid[level] = f"metrics parse error: {exc}"
            continue
        if metric.get("model_label") != label or metric.get("feature_level") != level:
            invalid[level] = "metric identity mismatch"
            continue
        if int(metric.get("num_tokens", -1)) != EXPECTED_TOKENS:
            invalid[level] = f"validation token count is {metric.get('num_tokens')}"
            continue
        present.append(level)
        rows.append(
            {
                "model_label": label,
                "feature_level": level,
                "mae": metric["mae"],
                "absrel": metric["absrel"],
                "delta125": metric["delta125"],
                "num_tokens": metric["num_tokens"],
                "metrics_path": str(metric_path),
            }
        )

    missing = [level for level in LEVELS if level not in present]
    return {
        "label": label,
        "checkpoint": identity,
        "baseline_config_match": config_match,
        "baseline_lora_recipe_match": lora_recipe_match,
        "eomt_paths_disabled": disabled_eomt,
        "present": present,
        "missing": missing,
        "invalid": invalid,
        "ready": identity["complete"] and config_match and lora_recipe_match and disabled_eomt,
        "complete": not missing and not invalid,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--durable-root", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--model", action="append", default=[], metavar="LABEL=CHECKPOINT")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()

    models = []
    for spec in args.model:
        label, separator, raw_path = spec.partition("=")
        if not separator or not label or not raw_path:
            parser.error(f"invalid --model value: {spec!r}")
        models.append(validate_model(label, Path(raw_path), args.template, args.durable_root))

    payload = {
        "schema_version": "scannet_baseline_replicates_v1",
        "levels": list(LEVELS),
        "pre_llm_features": list(POST_SFT_PRE_LLM_FEATURES),
        "requested_llm_layers": list(POST_SFT_DEPTH_LAYERS),
        "hidden_state_indexing": "requested L -> hidden_states[L + 1]",
        "target": "camera_z",
        "expected_validation_tokens": EXPECTED_TOKENS,
        "models": models,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.write_summary:
        summary_dir = args.durable_root / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        rows = [row for model in models for row in model["rows"]]
        (summary_dir / "graph_data.json").write_text(
            json.dumps({"schema_version": "scannet_baseline_replicates_graph_v1", "rows": rows}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        lines = [
            "# ScanNet baseline replication depth probes",
            "",
            "| Model | Layer | MAE | AbsRel | delta<1.25 |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row['model_label']} | {row['feature_level']} | {row['mae']:.6f} | "
                f"{row['absrel']:.6f} | {row['delta125']:.6f} |"
            )
        lines.append("")
        (summary_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    for model in models:
        print(
            f"{model['label']}: ready={model['ready']} complete={model['complete']} "
            f"present={','.join(model['present']) or '-'} missing={','.join(model['missing']) or '-'}"
        )


if __name__ == "__main__":
    main()
