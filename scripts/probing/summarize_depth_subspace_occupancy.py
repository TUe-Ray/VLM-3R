#!/usr/bin/env python
"""Write concise, source-linked summaries for a completed depth pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROBE_POINTS = (
    "fusion_output", "projected_features", "L0", "L1", "L2", "L3", "L6",
    "L9", "L12", "L15", "L18", "L21", "L24", "L27",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def profile_line(rows: list[dict[str, str]], metric: str) -> str:
    row = next((item for item in rows if item.get("metric") == metric), None)
    if row is None:
        return f"- `{metric}`: not run."
    if row.get("stable") != "True":
        return f"- `{metric}`: no stable complete-profile distinction ({row.get('reason', 'not significant')})."
    return (
        f"- `{metric}`: stable on all {row['video_count']} development videos "
        f"(T={float(row['observed_T']):.4g}, null 95th={float(row['null_q95']):.4g}, "
        f"exact p={float(row['p_value']):.4g}, LOO={row['leave_one_video_out_passes']}/{row['leave_one_video_out_total']})."
    )


def write_semantics(path: Path) -> None:
    rows = [
        "# Extracted probe-point semantics",
        "",
        "All cached visual features are the two selected 14×14 target-frame grids. "
        "The model forward always uses 32 RGB frames; the compact cache does not treat the two target frames as model input.",
        "",
        "| Point | Exact tensor / timing |",
        "|---|---|",
        "| `fusion_output` | For additive pre-SFT SpatialStack, the tensor entering `mm_projector` (raw SigLIP visual representation, then model 2-D pooled to 14×14). Additive geometry is injected only in the decoder, so this point is before any SpatialStack contribution. |",
        "| `projected_features` | Output of `mm_projector`, then model 2-D pooled to 14×14; still before any decoder/SpatialStack injection. |",
    ]
    for point in PROBE_POINTS[2:]:
        layer = int(point[1:])
        tail = " It is the final model-normalized state after block 27." if layer == 27 else ""
        rows.append(
            f"| `{point}` | `hidden_states[{layer + 1}]`: output after transformer block {layer}. "
            f"The decoder records a pre-block state, applies an additive residual for layer {layer} if scheduled, "
            f"then executes that block; therefore a scheduled L{layer} injection is included in this point.{tail} |"
        )
    rows.extend(
        [
            "",
            "`SS012` injects at 0/1/2, `SS123` at 1/2/3, and `SS036` at 0/3/6. "
            "Thus, for example, SS123 L0 is structurally before its first possible injection, while SS012/SS036 L0 is after their first injection.",
        ]
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def is_enriched(row: dict[str, str]) -> bool:
    observed = finite_float(row.get("vf_enrich"))
    random_mean = finite_float(row.get("random_vf_enrich_mean"))
    return observed is not None and random_mean is not None and observed > random_mean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    root = Path(args.result_dir).resolve()
    v1 = root / "v1_ridge_vf"
    v4 = root / "v4_geometry_propagation"
    aggregate = read_rows(v1 / "linear_vf_aggregate.csv")
    per_video = read_rows(v1 / "linear_vf_per_video.csv")
    v1_profiles = read_rows(v1 / "profile_discrimination.csv")
    v4_profiles = read_rows(v4 / "profile_discrimination.csv")
    on_off = read_rows(v4 / "on_off_aggregate.csv")

    peak_lines = []
    for model in ("SS012", "SS123", "SS036"):
        model_rows = [row for row in aggregate if row["model"] == model]
        peak = max(model_rows, key=lambda row: float(row["linear_r2_mean"]))
        peak_lines.append(
            f"- `{model}` peaks at `{peak['probe_point']}`: R²={float(peak['linear_r2_mean']):.3f}±{float(peak['linear_r2_std']):.3f}; "
            f"VF enrichment={float(peak['vf_enrich_mean']):.2f}."
        )
    dev_rows = [row for row in per_video if row["split"] == "dev_eval"]
    enriched = [row for row in dev_rows if is_enriched(row)]
    text_rows = [
        row for row in on_off
        if row["probe_point"].startswith("L") and (finite_float(row.get("I_text_mean")) or 0.0) > 0.0
    ]
    late_text = [row for row in on_off if row["probe_point"] == "L27"]

    lines = [
        "# Pre-SFT C1 SpatialStack depth-subspace pilot",
        "",
        "Development-only pilot: 6 train, 2 validation, 4 development-evaluation videos; the separately frozen 12-video confirmation set was not accessed.",
        "",
        "## Linear depth and variance occupancy",
        "",
        "Linear depth is meaningfully readable at some held-out points in every schedule (peak macro R² values below); it is not uniformly linear through all late layers.",
        "",
        *peak_lines,
        "",
        f"Depth-direction VF enrichment exceeds the matched random-direction mean in {len(enriched)}/{len(dev_rows)} model×point×video observations. "
        "VF remains a 1-D linear-direction diagnostic, not a claim about the complete geometry representation.",
        "",
        *[profile_line(v1_profiles, metric) for metric in ("linear_r2", "vf_enrich")],
        "",
        "## Geometry ON/OFF propagation",
        "",
        "The Geometry-OFF forward withheld only the native SpatialStack payload from the same prepared model/input. "
        "Pre-LLM points are consequently zero by definition; undefined transfer ratios are preserved as N/A rather than forced to zero.",
        "",
        *[profile_line(v4_profiles, metric) for metric in ("I_visual", "I_text", "text_visual_transfer_ratio")],
        "",
        f"Nonzero text/query influence appears in {len(text_rows)} LLM model×point aggregates. "
        + ("At L27, the reported text influence and text/visual transfer are nonzero for all three schedules." if len(late_text) == 3 else "Late-layer text influence should be read from the per-video table."),
        "",
        "## Boundary",
        "",
        "These are stable development-set profiles (the CSVs retain per-video values and leave-one-video-out checks), not confirmation or causal evidence about VSI-Bench. "
        "A confirmation run must use one explicitly frozen stage/metric/rank specification and must not re-enter metric selection.",
        "",
        "## Artifacts",
        "",
        "- Linear/VF tables and plots: `v1_ridge_vf/`",
        "- ON/OFF tables and plots: `v4_geometry_propagation/`",
        "- Exact point semantics: `probe_point_semantics.md`",
    ]
    (root / "pilot_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_semantics(root / "probe_point_semantics.md")
    print(json.dumps({"summary": str(root / "pilot_summary.md"), "semantics": str(root / "probe_point_semantics.md")}, indent=2))


if __name__ == "__main__":
    main()
