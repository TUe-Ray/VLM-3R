#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_cross_attn_gamma_safe.py

Purpose:
  Inspect CUT3R SpatialStack cross-attn checkpoints without recursively reading
  DeepSpeed checkpoint subfolders.

Default behavior:
  - Only reads files directly inside each MODEL_PATH.
  - Does NOT enter checkpoint-*/global_step*/ subfolders.
  - Skips optimizer states / DeepSpeed ZeRO states / mp_rank model_states.
  - For safetensors, reads only matched tensors, not the whole model.
  - For .bin/.pt/.pth, skips large files by default to avoid OOM/Killed.

Main outputs:
  <model_path>/gamma_inspection_safe/cross_attn_gamma_report.md
  <model_path>/gamma_inspection_safe/cross_attn_param_summary.csv
  <model_path>/gamma_inspection_safe/cross_attn_gamma_summary.csv
  <model_path>/gamma_inspection_safe/cross_attn_v1_out_proj_summary.csv

Usage:
  1. Edit MODEL_PATHS below, then:
       python inspect_cross_attn_gamma_safe.py

  2. Or pass paths from CLI:
       python inspect_cross_attn_gamma_safe.py /path/to/model1 /path/to/model2

Notes:
  - V2 usually has gamma_attn / gamma_mlp.
  - V1 may not have gamma; for V1, inspect out_proj/output_proj norm.
"""

import os
import re
import csv
import math
import argparse
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ============================================================
# Change only this part if you want the simplest usage.
# ============================================================

MODEL_PATHS = [
    "/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/cut3r_spatialstack_cross_attn_v2_resize_cam_gamma_47030066",
]

# V2 expected gamma initialization.
DEFAULT_GAMMA_INIT = 0.05

# V1 old cross-attn output projection may have zero-init.
DEFAULT_V1_OUT_PROJ_INIT_NORM = 0.0

# For torch .bin/.pt/.pth files, avoid loading very large files.
# safetensors are handled safely by reading only matched tensors.
DEFAULT_MAX_TORCH_FILE_MB = 2048

# ============================================================
# Implementation
# ============================================================

SKIP_NAME_PATTERNS = [
    "optim",
    "optimizer",
    "scheduler",
    "rng_state",
    "random_state",
    "trainer_state",
    "zero_pp_rank",
    "zero_dp_rank",
    "bf16_zero",
    "fp16_zero",
    "mp_rank",
    "model_states",
    "global_step",
    "latest",
]

TORCH_EXTS = {".bin", ".pt", ".pth"}
SAFE_EXTS = {".safetensors"}
ALL_EXTS = TORCH_EXTS | SAFE_EXTS


def log(msg: str) -> None:
    print(msg, flush=True)


def is_probably_bad_checkpoint_file(path: str) -> bool:
    name = os.path.basename(path).lower()
    return any(pat in name for pat in SKIP_NAME_PATTERNS)


def is_candidate_weight_file(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    ext = os.path.splitext(path)[1].lower()
    if ext not in ALL_EXTS:
        return False
    if is_probably_bad_checkpoint_file(path):
        return False
    return True


def list_direct_weight_files(model_path: str) -> List[str]:
    """
    Only list direct files under model_path.
    Do not recursively enter checkpoint-* or global_step* folders.
    """
    if os.path.isfile(model_path):
        return [model_path] if is_candidate_weight_file(model_path) else []

    if not os.path.isdir(model_path):
        return []

    files = []
    for name in sorted(os.listdir(model_path)):
        path = os.path.join(model_path, name)
        if is_candidate_weight_file(path):
            files.append(path)

    # Prefer adapter/trainable files first.
    priority = [
        "adapter_model.safetensors",
        "adapter_model.bin",
        "non_lora_trainables.bin",
        "non_lora_trainables.safetensors",
        "pytorch_model.bin",
        "model.safetensors",
    ]

    def rank(path: str) -> Tuple[int, str]:
        base = os.path.basename(path)
        if base in priority:
            return (priority.index(base), base)
        if base.endswith(".safetensors"):
            return (100, base)
        if base.endswith(".bin"):
            return (200, base)
        if base.endswith(".pt") or base.endswith(".pth"):
            return (300, base)
        return (999, base)

    return sorted(files, key=rank)


def key_is_gamma(key: str) -> bool:
    lk = key.lower()
    return "gamma_attn" in lk or "gamma_mlp" in lk


def key_is_v1_or_cross_out_proj(key: str) -> bool:
    lk = key.lower()

    has_cross_context = (
        "cross" in lk
        or "spatialstack" in lk
        or "spatial_stack" in lk
        or "cut3r" in lk
    )

    has_out_proj = (
        "out_proj" in lk
        or "output_proj" in lk
        or "output_projection" in lk
    )

    return has_cross_context and has_out_proj


def key_is_extra_debug_param(key: str) -> bool:
    """
    Optional extra params useful for debugging whether cross-attn branch exists.
    Keep this conservative to avoid giant reports.
    """
    lk = key.lower()

    has_cross_context = (
        "cross" in lk
        or "spatialstack" in lk
        or "spatial_stack" in lk
        or "cut3r" in lk
    )

    if not has_cross_context:
        return False

    extra_terms = [
        "in_proj_weight",
        "in_proj_bias",
        "q_proj",
        "k_proj",
        "v_proj",
        "cam_proj",
        "camera_proj",
        "patch_proj",
        "mlp",
        "ffn",
    ]

    return any(term in lk for term in extra_terms)


def key_is_relevant(key: str, include_extra: bool = True) -> bool:
    if key_is_gamma(key):
        return True
    if key_is_v1_or_cross_out_proj(key):
        return True
    if include_extra and key_is_extra_debug_param(key):
        return True
    return False


def classify_key(key: str) -> str:
    lk = key.lower()
    if "gamma_attn" in lk:
        return "gamma_attn"
    if "gamma_mlp" in lk:
        return "gamma_mlp"
    if key_is_v1_or_cross_out_proj(key):
        return "out_proj_or_output_proj"
    if "in_proj" in lk:
        return "mha_in_proj"
    if "cam_proj" in lk or "camera_proj" in lk:
        return "camera_proj"
    if "patch_proj" in lk:
        return "patch_proj"
    if "mlp" in lk or "ffn" in lk:
        return "mlp_or_ffn"
    if "q_proj" in lk:
        return "q_proj"
    if "k_proj" in lk:
        return "k_proj"
    if "v_proj" in lk:
        return "v_proj"
    return "other_relevant"


def infer_layer_or_block(key: str) -> str:
    """
    Best-effort layer/block extraction from parameter name.
    This is intentionally heuristic because module names may differ.
    """
    patterns = [
        r"cross_attn_blocks\.(\d+)",
        r"cross_attention_blocks\.(\d+)",
        r"spatialstack_blocks\.(\d+)",
        r"spatial_stack_blocks\.(\d+)",
        r"blocks\.(\d+)",
        r"layers\.(\d+)",
        r"llm_layers\.(\d+)",
        r"decoder_layers\.(\d+)",
    ]

    for pat in patterns:
        m = re.search(pat, key)
        if m:
            return m.group(1)

    # Fallback: infer from dec layer names if present.
    m = re.search(r"dec(?:oder)?[_\.]?(\d+)", key.lower())
    if m:
        return "dec" + m.group(1)

    return "unknown"


def tensor_stats(tensor: Any) -> Dict[str, Any]:
    import torch

    if not torch.is_tensor(tensor):
        raise TypeError("Expected torch tensor")

    x = tensor.detach().float().cpu()
    numel = int(x.numel())

    if numel == 0:
        return {
            "shape": tuple(x.shape),
            "numel": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "norm": float("nan"),
            "value": float("nan"),
        }

    std = x.std(unbiased=False).item() if numel > 1 else 0.0
    value = x.reshape(-1)[0].item() if numel == 1 else float("nan")

    return {
        "shape": tuple(x.shape),
        "numel": numel,
        "mean": x.mean().item(),
        "std": std,
        "min": x.min().item(),
        "max": x.max().item(),
        "norm": x.norm().item(),
        "value": value,
    }


def load_relevant_from_safetensors(path: str, include_extra: bool = True) -> Iterable[Tuple[str, Any]]:
    try:
        from safetensors import safe_open
    except Exception as e:
        log(f"[warn] safetensors is not available, skip {path}: {e}")
        return

    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            matched_keys = [k for k in keys if key_is_relevant(k, include_extra=include_extra)]
            log(f"       safetensors keys={len(keys)} matched={len(matched_keys)}")

            for key in matched_keys:
                try:
                    yield key, f.get_tensor(key)
                except Exception as e:
                    log(f"[warn] failed to read tensor {key} from {path}: {e}")
    except Exception as e:
        log(f"[warn] failed to open safetensors {path}: {e}")


def unwrap_state_dict(obj: Any) -> Optional[Dict[str, Any]]:
    """
    Try common checkpoint structures.
    """
    if not isinstance(obj, dict):
        return None

    # Direct state dict.
    if any(hasattr(v, "shape") for v in obj.values()):
        return obj

    for key in [
        "state_dict",
        "model",
        "module",
        "model_state_dict",
        "trainable_params",
        "params",
    ]:
        if key in obj and isinstance(obj[key], dict):
            inner = obj[key]
            if any(hasattr(v, "shape") for v in inner.values()):
                return inner

    return None


def flatten_state_dict(sd: Dict[str, Any], prefix: str = "") -> Iterable[Tuple[str, Any]]:
    """
    Flatten nested dicts, but only yield tensors.
    """
    import torch

    for key, value in sd.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)

        if torch.is_tensor(value):
            yield full_key, value
        elif isinstance(value, dict):
            # Keep recursion shallow enough for model dicts, but optimizer states
            # should already be skipped by filename.
            for sub_key, sub_value in flatten_state_dict(value, full_key):
                yield sub_key, sub_value


def load_relevant_from_torch_file(
    path: str,
    include_extra: bool = True,
    max_torch_file_mb: int = DEFAULT_MAX_TORCH_FILE_MB,
) -> Iterable[Tuple[str, Any]]:
    import torch

    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    if file_size_mb > max_torch_file_mb:
        log(
            f"[skip] {path} is {file_size_mb:.1f} MB, larger than "
            f"--max-torch-file-mb={max_torch_file_mb}. "
            f"Use a consolidated small trainable file or increase the limit."
        )
        return

    try:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            # Older PyTorch does not have weights_only.
            obj = torch.load(path, map_location="cpu")
    except Exception as e:
        log(f"[warn] failed to torch.load {path}: {e}")
        return

    sd = unwrap_state_dict(obj)
    if sd is None:
        log(f"[warn] no tensor state_dict found in {path}")
        return

    matched = 0
    for key, value in flatten_state_dict(sd):
        if key_is_relevant(key, include_extra=include_extra):
            matched += 1
            yield key, value

    log(f"       torch file matched={matched}")


def load_relevant_tensors(
    path: str,
    include_extra: bool = True,
    max_torch_file_mb: int = DEFAULT_MAX_TORCH_FILE_MB,
) -> Iterable[Tuple[str, str, Any]]:
    """
    Yield:
      file_path, tensor_key, tensor
    """
    ext = os.path.splitext(path)[1].lower()

    log(f"[read] {path}")

    if ext in SAFE_EXTS:
        for key, tensor in load_relevant_from_safetensors(path, include_extra=include_extra):
            yield path, key, tensor
    elif ext in TORCH_EXTS:
        for key, tensor in load_relevant_from_torch_file(
            path,
            include_extra=include_extra,
            max_torch_file_mb=max_torch_file_mb,
        ):
            yield path, key, tensor
    else:
        log(f"[skip] unsupported extension: {path}")


def format_float(x: Any, digits: int = 8) -> str:
    try:
        xf = float(x)
    except Exception:
        return str(x)

    if math.isnan(xf):
        return "nan"
    if math.isinf(xf):
        return "inf" if xf > 0 else "-inf"

    return f"{xf:.{digits}g}"


def changed_flag(delta: float, atol: float) -> str:
    if math.isnan(delta):
        return "unknown"
    if abs(delta) <= atol:
        return "almost_unchanged"
    return "changed"


def inspect_one_model_path(
    model_path: str,
    gamma_init: float,
    v1_out_proj_init_norm: float,
    max_torch_file_mb: int,
    include_extra: bool,
    output_subdir_name: str = "gamma_inspection_safe",
) -> Dict[str, Any]:
    model_path = os.path.abspath(model_path)

    if os.path.isdir(model_path):
        output_dir = os.path.join(model_path, output_subdir_name)
    else:
        parent = os.path.dirname(model_path)
        stem = os.path.splitext(os.path.basename(model_path))[0]
        output_dir = os.path.join(parent, f"{stem}_{output_subdir_name}")

    os.makedirs(output_dir, exist_ok=True)

    log("=" * 100)
    log(f"[model] {model_path}")
    log(f"[output] {output_dir}")

    files = list_direct_weight_files(model_path)
    log(f"[candidate direct files] {len(files)}")

    for f in files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        log(f"  - {os.path.basename(f)} ({size_mb:.1f} MB)")

    rows: List[Dict[str, Any]] = []

    for file_path in files:
        for tensor_key, tensor in ((k, t) for _, k, t in load_relevant_tensors(
            file_path,
            include_extra=include_extra,
            max_torch_file_mb=max_torch_file_mb,
        )):
            try:
                stats = tensor_stats(tensor)
            except Exception as e:
                log(f"[warn] failed stats for {tensor_key}: {e}")
                continue

            kind = classify_key(tensor_key)
            layer = infer_layer_or_block(tensor_key)

            row = {
                "model_path": model_path,
                "file": os.path.basename(file_path),
                "key": tensor_key,
                "kind": kind,
                "layer_or_block": layer,
                "shape": str(stats["shape"]),
                "numel": stats["numel"],
                "value": stats["value"],
                "mean": stats["mean"],
                "std": stats["std"],
                "min": stats["min"],
                "max": stats["max"],
                "norm": stats["norm"],
            }
            rows.append(row)

    log(f"[matched relevant tensors] {len(rows)}")

    all_csv = os.path.join(output_dir, "cross_attn_param_summary.csv")
    write_csv(all_csv, rows)

    gamma_rows = build_gamma_summary(rows, gamma_init=gamma_init)
    gamma_csv = os.path.join(output_dir, "cross_attn_gamma_summary.csv")
    write_csv(gamma_csv, gamma_rows)

    out_proj_rows = build_out_proj_summary(rows, init_norm=v1_out_proj_init_norm)
    out_proj_csv = os.path.join(output_dir, "cross_attn_v1_out_proj_summary.csv")
    write_csv(out_proj_csv, out_proj_rows)

    report_path = os.path.join(output_dir, "cross_attn_gamma_report.md")
    write_markdown_report(
        report_path=report_path,
        model_path=model_path,
        files=files,
        rows=rows,
        gamma_rows=gamma_rows,
        out_proj_rows=out_proj_rows,
        gamma_init=gamma_init,
        v1_out_proj_init_norm=v1_out_proj_init_norm,
        max_torch_file_mb=max_torch_file_mb,
    )

    log(f"[wrote] {report_path}")
    log(f"[wrote] {all_csv}")
    log(f"[wrote] {gamma_csv}")
    log(f"[wrote] {out_proj_csv}")

    return {
        "model_path": model_path,
        "output_dir": output_dir,
        "rows": rows,
        "gamma_rows": gamma_rows,
        "out_proj_rows": out_proj_rows,
        "files": files,
    }


def build_gamma_summary(rows: List[Dict[str, Any]], gamma_init: float) -> List[Dict[str, Any]]:
    gamma_rows = []
    for r in rows:
        if r["kind"] not in {"gamma_attn", "gamma_mlp"}:
            continue

        final_value = r["value"]
        if math.isnan(float(final_value)):
            final_value = r["mean"]

        delta = float(final_value) - float(gamma_init)

        gamma_rows.append({
            "model_path": r["model_path"],
            "file": r["file"],
            "layer_or_block": r["layer_or_block"],
            "kind": r["kind"],
            "key": r["key"],
            "init_ref": gamma_init,
            "final_value": final_value,
            "delta_from_init": delta,
            "abs_delta_from_init": abs(delta),
            "changed_flag_atol_1e-4": changed_flag(delta, atol=1e-4),
            "changed_flag_atol_1e-3": changed_flag(delta, atol=1e-3),
            "norm": r["norm"],
            "shape": r["shape"],
        })

    return sorted(
        gamma_rows,
        key=lambda x: (
            str(x["layer_or_block"]),
            str(x["kind"]),
            str(x["key"]),
        ),
    )


def build_out_proj_summary(rows: List[Dict[str, Any]], init_norm: float) -> List[Dict[str, Any]]:
    out_rows = []
    for r in rows:
        if r["kind"] != "out_proj_or_output_proj":
            continue

        final_norm = float(r["norm"])
        delta_norm = final_norm - float(init_norm)

        out_rows.append({
            "model_path": r["model_path"],
            "file": r["file"],
            "layer_or_block": r["layer_or_block"],
            "kind": r["kind"],
            "key": r["key"],
            "init_norm_ref": init_norm,
            "final_norm": final_norm,
            "delta_norm_from_init": delta_norm,
            "learned_flag_norm_gt_1e-6": "yes" if final_norm > 1e-6 else "no",
            "learned_flag_norm_gt_1e-4": "yes" if final_norm > 1e-4 else "no",
            "mean": r["mean"],
            "std": r["std"],
            "min": r["min"],
            "max": r["max"],
            "shape": r["shape"],
        })

    return sorted(
        out_rows,
        key=lambda x: (
            str(x["layer_or_block"]),
            str(x["key"]),
        ),
    )


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def markdown_table(rows: List[Dict[str, Any]], columns: List[str], max_rows: int = 80) -> str:
    if not rows:
        return "_No rows._\n"

    shown = rows[:max_rows]

    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for r in shown:
        vals = []
        for col in columns:
            val = r.get(col, "")
            if isinstance(val, float):
                val = format_float(val)
            vals.append(str(val).replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")

    if len(rows) > max_rows:
        lines.append(f"\n_Showing first {max_rows} of {len(rows)} rows._")

    return "\n".join(lines) + "\n"


def write_markdown_report(
    report_path: str,
    model_path: str,
    files: List[str],
    rows: List[Dict[str, Any]],
    gamma_rows: List[Dict[str, Any]],
    out_proj_rows: List[Dict[str, Any]],
    gamma_init: float,
    v1_out_proj_init_norm: float,
    max_torch_file_mb: int,
) -> None:
    gamma_attn = [r for r in gamma_rows if r["kind"] == "gamma_attn"]
    gamma_mlp = [r for r in gamma_rows if r["kind"] == "gamma_mlp"]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Cross-Attention Gamma Inspection Report\n\n")
        f.write(f"Model path:\n\n```text\n{model_path}\n```\n\n")
        f.write(f"Gamma init reference: `{gamma_init}`\n\n")
        f.write(f"V1 out_proj init norm reference: `{v1_out_proj_init_norm}`\n\n")
        f.write(f"Max torch file size loaded: `{max_torch_file_mb} MB`\n\n")

        f.write("## Direct files inspected\n\n")
        if files:
            for p in files:
                size_mb = os.path.getsize(p) / (1024 * 1024)
                f.write(f"- `{os.path.basename(p)}` ({size_mb:.1f} MB)\n")
        else:
            f.write("_No direct candidate files found._\n")
        f.write("\n")

        f.write("## Matched tensor counts\n\n")
        f.write(f"- Total relevant tensors matched: `{len(rows)}`\n")
        f.write(f"- Gamma tensors matched: `{len(gamma_rows)}`\n")
        f.write(f"- gamma_attn tensors matched: `{len(gamma_attn)}`\n")
        f.write(f"- gamma_mlp tensors matched: `{len(gamma_mlp)}`\n")
        f.write(f"- out_proj/output_proj tensors matched: `{len(out_proj_rows)}`\n\n")

        f.write("## V2 gamma summary\n\n")
        f.write(
            markdown_table(
                gamma_rows,
                columns=[
                    "layer_or_block",
                    "kind",
                    "init_ref",
                    "final_value",
                    "delta_from_init",
                    "abs_delta_from_init",
                    "changed_flag_atol_1e-4",
                    "changed_flag_atol_1e-3",
                    "key",
                ],
                max_rows=120,
            )
        )
        f.write("\n")

        f.write("## V1 / old cross-attn out_proj summary\n\n")
        f.write(
            markdown_table(
                out_proj_rows,
                columns=[
                    "layer_or_block",
                    "init_norm_ref",
                    "final_norm",
                    "delta_norm_from_init",
                    "learned_flag_norm_gt_1e-6",
                    "learned_flag_norm_gt_1e-4",
                    "key",
                ],
                max_rows=120,
            )
        )
        f.write("\n")

        f.write("## Interpretation guide\n\n")
        f.write("- If V2 `gamma_attn` / `gamma_mlp` stay very close to `0.05`, the branch may not have learned much.\n")
        f.write("- If V2 gamma values move toward `0`, the model may be suppressing the cross-attn branch.\n")
        f.write("- If V2 gamma values grow clearly above `0.05`, the branch was trained and used.\n")
        f.write("- If V2 gamma changed but evaluation is still worse than additive Spatial Stack, the problem is likely the fusion design rather than a dead branch.\n")
        f.write("- If V1 has no gamma, use `out_proj/output_proj final_norm` as the proxy for whether the zero-init output projection learned away from zero.\n")
        f.write("\n")

        f.write("## All relevant parameter rows\n\n")
        f.write(
            markdown_table(
                rows,
                columns=[
                    "kind",
                    "layer_or_block",
                    "value",
                    "mean",
                    "std",
                    "min",
                    "max",
                    "norm",
                    "shape",
                    "key",
                ],
                max_rows=160,
            )
        )


def write_combined_report(results: List[Dict[str, Any]], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    combined_gamma = []
    combined_out = []
    combined_all = []

    for result in results:
        combined_gamma.extend(result["gamma_rows"])
        combined_out.extend(result["out_proj_rows"])
        combined_all.extend(result["rows"])

    write_csv(os.path.join(output_dir, "combined_cross_attn_gamma_summary.csv"), combined_gamma)
    write_csv(os.path.join(output_dir, "combined_cross_attn_v1_out_proj_summary.csv"), combined_out)
    write_csv(os.path.join(output_dir, "combined_cross_attn_param_summary.csv"), combined_all)

    report_path = os.path.join(output_dir, "combined_cross_attn_gamma_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Combined Cross-Attention Gamma Report\n\n")

        f.write("## Models\n\n")
        for result in results:
            f.write(f"- `{result['model_path']}`\n")
        f.write("\n")

        f.write("## Combined V2 gamma summary\n\n")
        f.write(
            markdown_table(
                combined_gamma,
                columns=[
                    "model_path",
                    "layer_or_block",
                    "kind",
                    "init_ref",
                    "final_value",
                    "delta_from_init",
                    "abs_delta_from_init",
                    "changed_flag_atol_1e-4",
                    "changed_flag_atol_1e-3",
                    "key",
                ],
                max_rows=240,
            )
        )
        f.write("\n")

        f.write("## Combined V1 / old cross-attn out_proj summary\n\n")
        f.write(
            markdown_table(
                combined_out,
                columns=[
                    "model_path",
                    "layer_or_block",
                    "init_norm_ref",
                    "final_norm",
                    "delta_norm_from_init",
                    "learned_flag_norm_gt_1e-6",
                    "learned_flag_norm_gt_1e-4",
                    "key",
                ],
                max_rows=240,
            )
        )

    log(f"[wrote] {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely inspect CUT3R SpatialStack cross-attn gamma/out_proj values."
    )

    parser.add_argument(
        "model_paths",
        nargs="*",
        help="Model/checkpoint folders or files. If omitted, uses MODEL_PATHS inside the script.",
    )
    parser.add_argument(
        "--gamma-init",
        type=float,
        default=DEFAULT_GAMMA_INIT,
        help="Reference init value for V2 gamma_attn/gamma_mlp.",
    )
    parser.add_argument(
        "--v1-out-proj-init-norm",
        type=float,
        default=DEFAULT_V1_OUT_PROJ_INIT_NORM,
        help="Reference init norm for V1 zero-initialized out_proj/output_proj.",
    )
    parser.add_argument(
        "--max-torch-file-mb",
        type=int,
        default=DEFAULT_MAX_TORCH_FILE_MB,
        help="Maximum size of .bin/.pt/.pth files to torch.load. safetensors are not affected.",
    )
    parser.add_argument(
        "--no-extra",
        action="store_true",
        help="Only collect gamma and out_proj/output_proj, not extra MHA/proj/MLP params.",
    )
    parser.add_argument(
        "--combined-output-dir",
        type=str,
        default="gamma_inspection_safe_combined",
        help="Output dir for combined report when multiple model paths are inspected.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_paths = args.model_paths if args.model_paths else MODEL_PATHS
    model_paths = [p for p in model_paths if p and p.strip()]

    if not model_paths:
        raise SystemExit("No model paths provided. Edit MODEL_PATHS or pass paths from CLI.")

    results = []
    for model_path in model_paths:
        result = inspect_one_model_path(
            model_path=model_path,
            gamma_init=args.gamma_init,
            v1_out_proj_init_norm=args.v1_out_proj_init_norm,
            max_torch_file_mb=args.max_torch_file_mb,
            include_extra=not args.no_extra,
        )
        results.append(result)

    if len(results) > 1:
        write_combined_report(results, output_dir=args.combined_output_dir)

    log("=" * 100)
    log("[done]")
    for result in results:
        log(f"Report: {os.path.join(result['output_dir'], 'cross_attn_gamma_report.md')}")


if __name__ == "__main__":
    main()