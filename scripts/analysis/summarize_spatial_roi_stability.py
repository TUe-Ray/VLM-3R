#!/usr/bin/env python3
"""Summarize spatial ROI similarity layer stability across sample shards."""

from __future__ import annotations

import argparse
import csv
import glob
import math
import random
from collections import defaultdict
from pathlib import Path


DEFAULT_METRICS = [
    "spearman_with_siglip_selected_m2",
    "spearman_with_cut3r_dec_m1",
    "spearman_with_pi3_dec_m1",
    "spearman_with_vggt_agg_m1",
    "spearman_geometry_minus_siglip",
]

DEFAULT_REPRESENTATIONS = [
    "CUT3R dec -12",
    "CUT3R dec -10",
    "CUT3R dec -8",
    "CUT3R dec -6",
    "CUT3R dec -4",
    "CUT3R dec -2",
    "CUT3R dec -1",
    "PI3 dec -12",
    "PI3 dec -10",
    "PI3 dec -8",
    "PI3 dec -6",
    "PI3 dec -4",
    "PI3 dec -2",
    "PI3 dec -1",
    "VGGT agg -12",
    "VGGT agg -10",
    "VGGT agg -8",
    "VGGT agg -6",
    "VGGT agg -4",
    "VGGT agg -2",
    "VGGT agg -1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_glob",
        required=True,
        help="Glob for shard CSVs, e.g. '/path/run*/spatial_decoder_layer_roi_summary.csv'.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--representations",
        default=",".join(DEFAULT_REPRESENTATIONS),
        help="Comma-separated representations to include. Use 'all' for every representation.",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated numeric metrics to summarize.",
    )
    parser.add_argument(
        "--target_metric",
        default="spearman_with_siglip_selected_m2",
        help="Metric used for per-sample win-rate and pairwise win-rate tables.",
    )
    parser.add_argument("--bootstrap_iters", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def to_float(value: str) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def sample_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["sample_id"], row.get("frame_idx", ""), row.get("anchor_id", ""))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def bootstrap_ci(values: list[float], iters: int, rng: random.Random) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (values[0], values[0])
    n = len(values)
    means = []
    for _ in range(iters):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo_idx = int(0.025 * (iters - 1))
    hi_idx = int(0.975 * (iters - 1))
    return means[lo_idx], means[hi_idx]


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return "nan"
    return f"{value:.4f}"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(Path(p) for p in glob.glob(args.input_glob))
    if not csv_paths:
        raise SystemExit(f"No CSV files matched: {args.input_glob}")

    requested_reps = None
    if args.representations.strip().lower() != "all":
        requested_reps = [r.strip() for r in args.representations.split(",") if r.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    rows: list[dict[str, str]] = []
    for path in csv_paths:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if requested_reps is not None and row["representation"] not in requested_reps:
                    continue
                row["source_csv"] = str(path)
                rows.append(row)
    if not rows:
        raise SystemExit("No rows left after representation filtering.")

    combined_path = out_dir / "spatial_roi_stability_combined_rows.csv"
    fieldnames = list(rows[0].keys())
    with combined_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    reps = requested_reps or sorted({row["representation"] for row in rows})
    rng = random.Random(args.seed)
    stats_rows: list[dict[str, str]] = []
    rows_by_rep = defaultdict(list)
    for row in rows:
        rows_by_rep[row["representation"]].append(row)

    for rep in reps:
        rep_rows = rows_by_rep.get(rep, [])
        if not rep_rows:
            continue
        for metric in metrics:
            values = [v for row in rep_rows if (v := to_float(row.get(metric, ""))) is not None]
            if not values:
                continue
            std = sample_std(values)
            ci_lo, ci_hi = bootstrap_ci(values, args.bootstrap_iters, rng)
            stats_rows.append(
                {
                    "representation": rep,
                    "metric": metric,
                    "n": str(len(values)),
                    "mean": fmt(mean(values)),
                    "std": fmt(std),
                    "se": fmt(std / math.sqrt(len(values))),
                    "bootstrap_ci95_low": fmt(ci_lo),
                    "bootstrap_ci95_high": fmt(ci_hi),
                }
            )

    stats_path = out_dir / "spatial_roi_stability_stats_long.csv"
    stats_fields = [
        "representation",
        "metric",
        "n",
        "mean",
        "std",
        "se",
        "bootstrap_ci95_low",
        "bootstrap_ci95_high",
    ]
    with stats_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=stats_fields)
        writer.writeheader()
        writer.writerows(stats_rows)

    values_by_key = defaultdict(dict)
    for row in rows:
        value = to_float(row.get(args.target_metric, ""))
        if value is not None:
            values_by_key[sample_key(row)][row["representation"]] = value

    win_counts = {rep: 0.0 for rep in reps}
    total_keys = 0
    for rep_values in values_by_key.values():
        candidates = {rep: rep_values[rep] for rep in reps if rep in rep_values}
        if not candidates:
            continue
        total_keys += 1
        best = max(candidates.values())
        winners = [rep for rep, value in candidates.items() if value == best]
        share = 1.0 / len(winners)
        for rep in winners:
            win_counts[rep] += share

    win_path = out_dir / f"spatial_roi_win_rates_{args.target_metric}.csv"
    with win_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["representation", "n_compared_samples", "wins", "win_rate", "target_metric"],
        )
        writer.writeheader()
        for rep in reps:
            writer.writerow(
                {
                    "representation": rep,
                    "n_compared_samples": total_keys,
                    "wins": fmt(win_counts[rep]),
                    "win_rate": fmt(win_counts[rep] / total_keys if total_keys else float("nan")),
                    "target_metric": args.target_metric,
                }
            )

    pairwise_path = out_dir / f"spatial_roi_pairwise_win_rates_{args.target_metric}.csv"
    with pairwise_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["representation_a", "representation_b", "n", "a_win_rate", "ties"],
        )
        writer.writeheader()
        for rep_a in reps:
            for rep_b in reps:
                if rep_a == rep_b:
                    continue
                n = wins = ties = 0
                for rep_values in values_by_key.values():
                    if rep_a not in rep_values or rep_b not in rep_values:
                        continue
                    n += 1
                    if rep_values[rep_a] > rep_values[rep_b]:
                        wins += 1
                    elif rep_values[rep_a] == rep_values[rep_b]:
                        ties += 1
                writer.writerow(
                    {
                        "representation_a": rep_a,
                        "representation_b": rep_b,
                        "n": n,
                        "a_win_rate": fmt(wins / n if n else float("nan")),
                        "ties": ties,
                    }
                )

    md_path = out_dir / "spatial_roi_stability_summary.md"
    target_stats = {
        row["representation"]: row
        for row in stats_rows
        if row["metric"] == args.target_metric
    }
    with md_path.open("w") as f:
        f.write("# Spatial ROI Stability Summary\n\n")
        f.write(f"Input CSVs: {len(csv_paths)}\n\n")
        f.write(f"Rows: {len(rows)}\n\n")
        f.write(f"Target metric: `{args.target_metric}`\n\n")
        f.write("| representation | n | mean | std | se | bootstrap 95% CI | win rate |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for rep in reps:
            row = target_stats.get(rep)
            if not row:
                continue
            win_rate = win_counts[rep] / total_keys if total_keys else float("nan")
            f.write(
                "| {} | {} | {} | {} | {} | [{}, {}] | {} |\n".format(
                    rep,
                    row["n"],
                    row["mean"],
                    row["std"],
                    row["se"],
                    row["bootstrap_ci95_low"],
                    row["bootstrap_ci95_high"],
                    fmt(win_rate),
                )
            )

    print(combined_path)
    print(stats_path)
    print(win_path)
    print(pairwise_path)
    print(md_path)


if __name__ == "__main__":
    main()
