#!/usr/bin/env python
"""Summarize ROI layer-comparison CSVs into compact tables."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def average_by_representation(path: Path, metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            value = row.get(metric, "")
            if value and value.lower() != "nan":
                grouped.setdefault(row["representation"], []).append(float(value))
    return {name: statistics.mean(values) for name, values in grouped.items()}


def fmt(value: float | str | None) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.4f}"


def write_csv(path: Path, header: list[str], rows: list[list[str | float]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original_dir", required=True)
    parser.add_argument("--zero_spatial_dir", required=True)
    parser.add_argument("--spatial_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    original_dir = Path(args.original_dir)
    zero_dir = Path(args.zero_spatial_dir)
    spatial_dir = Path(args.spatial_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original = average_by_representation(original_dir / "roi_similarity_summary.csv", "spearman_with_cut3r")
    zero = average_by_representation(zero_dir / "roi_similarity_summary.csv", "spearman_with_cut3r")
    spatial_cut3r = average_by_representation(
        spatial_dir / "spatial_decoder_layer_roi_summary.csv",
        "spearman_with_cut3r_dec_m1",
    )
    spatial_pi3 = average_by_representation(
        spatial_dir / "spatial_decoder_layer_roi_summary.csv",
        "spearman_with_pi3_dec_m1",
    )

    vlm_order = [
        "CUT3R teacher",
        "SigLIP",
        "Fusion",
        "LLM input",
        "H1",
        "H4",
        "H8",
        "H12",
        "H16",
        "H20",
        "H24",
        "H28",
    ]
    spatial_order = [
        "CUT3R dec -4",
        "CUT3R dec -3",
        "CUT3R dec -2",
        "CUT3R dec -1",
        "PI3 dec -4",
        "PI3 dec -3",
        "PI3 dec -2",
        "PI3 dec -1",
    ]

    vlm_rows = [[rep, original.get(rep, ""), zero.get(rep, "")] for rep in vlm_order]
    spatial_rows = [[rep, spatial_cut3r.get(rep, ""), spatial_pi3.get(rep, "")] for rep in spatial_order]
    write_csv(output_dir / "vlm_layer_cut3r_alignment.csv", ["representation", "original_vlm3r", "zero_spatial"], vlm_rows)
    write_csv(
        output_dir / "spatial_decoder_layer_alignment.csv",
        ["representation", "spearman_with_cut3r_dec_m1", "spearman_with_pi3_dec_m1"],
        spatial_rows,
    )

    lines = [
        "# ROI similarity layer comparison",
        "",
        "Spearman correlation with CUT3R teacher ROI similarity map, averaged over 3 samples, frame 16, center anchor.",
        "",
        "## VLM hidden layers",
        "",
        "| Representation | Original VLM-3R | Zero-spatial |",
        "|---|---:|---:|",
    ]
    for rep, original_value, zero_value in vlm_rows:
        lines.append(f"| {rep} | {fmt(original_value)} | {fmt(zero_value)} |")
    lines.extend([
        "",
        "## CUT3R / PI3 decoder layers",
        "",
        "| Representation | vs CUT3R dec -1 | vs PI3 dec -1 |",
        "|---|---:|---:|",
    ])
    for rep, cut3r_value, pi3_value in spatial_rows:
        lines.append(f"| {rep} | {fmt(cut3r_value)} | {fmt(pi3_value)} |")
    lines.extend([
        "",
        f"Original output: {original_dir}",
        f"Zero-spatial output: {zero_dir}",
        f"Spatial output: {spatial_dir}",
    ])
    summary_path = output_dir / "layer_roi_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_path)
    print("---")
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
