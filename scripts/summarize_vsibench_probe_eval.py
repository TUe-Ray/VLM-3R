#!/usr/bin/env python
import argparse
import csv
import json
import shutil
from pathlib import Path


COLUMNS = [
    ("Overall", None),
    ("Room Size", ["room_size_estimation"]),
    ("Obj. Size", ["object_size_estimation"]),
    ("Abs. Dist.", ["object_abs_distance"]),
    ("Rel. Dist.", ["object_rel_distance"]),
    (
        "Rel. Dir.",
        [
            "object_rel_direction_easy",
            "object_rel_direction_medium",
            "object_rel_direction_hard",
        ],
    ),
    ("Route Plan", ["route_planning"]),
    ("Appearance Order", ["obj_appearance_order"]),
]


def _latest_file(root: Path, name: str) -> Path:
    if root.is_file():
        return root
    matches = sorted(set(root.rglob(name)), key=lambda path: path.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"Could not find {name} under {root}")
    return matches[0]


def _score_record(log):
    score = log.get("vsibench_score")
    if isinstance(score, dict):
        return score
    doc = log.get("doc")
    if isinstance(doc, dict):
        return doc
    return {}


def _score_value(record):
    for key in ("MRA:.5:.95:.05", "score", "accuracy"):
        value = record.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def summarize_run(label: str, path: Path):
    vsibench_path = _latest_file(path, "vsibench.json")
    results_path = _latest_file(path, "results.json")
    with vsibench_path.open("r", encoding="utf-8") as f:
        vsibench = json.load(f)
    with results_path.open("r", encoding="utf-8") as f:
        results = json.load(f)

    by_type = {}
    for log in vsibench.get("logs", []):
        record = _score_record(log)
        question_type = record.get("question_type")
        score = _score_value(record)
        if question_type is None or score is None:
            continue
        by_type.setdefault(question_type, []).append(score)

    row = {"Variant": label}
    row["Overall"] = float(results["results"]["vsibench"]["vsibench_score,none"])
    for column, question_types in COLUMNS[1:]:
        values = []
        for question_type in question_types:
            values.extend(by_type.get(question_type, []))
        row[column] = (sum(values) / len(values) * 100.0) if values else None

    return row, vsibench_path, results_path


def format_value(value):
    if value is None:
        return ""
    return f"{value:.4f}"


def write_csv(path: Path, rows):
    headers = ["Variant"] + [name for name, _ in COLUMNS]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (format_value(row.get(key)) if key != "Variant" else row[key]) for key in headers})


def _delta(probe_row, learned_row, column):
    if probe_row is None or learned_row is None:
        return None
    probe_value = probe_row.get(column)
    learned_value = learned_row.get(column)
    if probe_value is None or learned_value is None:
        return None
    return probe_value - learned_value


def interpretation(kind, rows_by_label):
    learned = rows_by_label.get("D2 learned gates original") or rows_by_label.get("D2 original learned gates")
    if kind == "intra_frame_pos_shuffle":
        probe = rows_by_label.get("D2 intra-frame position shuffle")
        deltas = [_delta(probe, learned, column) for column in ("Rel. Dir.", "Rel. Dist.", "Route Plan")]
        deltas = [value for value in deltas if value is not None]
        if not deltas:
            return "Interpretation pending: learned-gate original and intra-frame shuffle rows are both needed."
        if min(deltas) <= -2.0:
            return (
                "Intra-frame position shuffle hurts at least one geometry-sensitive score by 2+ points, "
                "so per-token 3D positions are active routing signals for this checkpoint."
            )
        return (
            "Intra-frame position shuffle is close to the learned-gate original on geometry-sensitive "
            "scores, suggesting GeoRoPE may be acting more like a generic bias than using precise "
            "within-frame spatial structure."
        )

    if kind == "geometry_shuffle":
        probe = rows_by_label.get("D2 geometry-shuffle")
        deltas = [_delta(probe, learned, column) for column in ("Rel. Dir.", "Rel. Dist.", "Route Plan")]
        deltas = [value for value in deltas if value is not None]
        if not deltas:
            return "Interpretation pending: learned-gate original and geometry-shuffle rows are both needed."
        if min(deltas) <= -2.0:
            return (
                "Geometry-shuffle hurts at least one geometry-sensitive score by 2+ points, "
                "so correct geometry positions are active routing signals for those tasks."
            )
        return (
            "Geometry-shuffle is close to the learned-gate original on the geometry-sensitive "
            "scores, suggesting geometry routing is weak or redundant for this checkpoint."
        )

    w1 = rows_by_label.get("D2 cross-frame window=1")
    if w1 is None or learned is None:
        return "Interpretation pending: learned-gate original and cross-frame window=1 rows are both needed."
    overall_delta = _delta(w1, learned, "Overall")
    route_delta = _delta(w1, learned, "Route Plan")
    order_delta = _delta(w1, learned, "Appearance Order")
    rel_dir_delta = _delta(w1, learned, "Rel. Dir.")
    improvements = [value for value in (route_delta, order_delta, rel_dir_delta) if value is not None and value >= 1.0]
    if improvements and (overall_delta is None or overall_delta >= -1.0):
        return (
            "Window=1 improves at least one target cross-frame/geometry score without a large "
            "Overall drop, which is positive zero-shot evidence for trained cross-frame fusion."
        )
    if overall_delta is not None and overall_delta <= -2.0:
        return (
            "Window=1 drops Overall by 2+ points, so eval-time cross-frame fusion is likely a "
            "distribution shift. Treat this as a probe, not evidence against a trained design."
        )
    return (
        "Window=1 is mostly flat versus the original, so this per-frame-trained checkpoint does "
        "not show strong zero-shot use of neighboring-frame geometry."
    )


def write_markdown(path: Path, rows, title: str, interp: str):
    headers = ["Variant"] + [name for name, _ in COLUMNS]
    lines = [
        f"# {title}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row["Variant"] if key == "Variant" else format_value(row.get(key)) for key in headers) + " |")
    lines.extend(["", "## Interpretation", "", interp, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["geometry_shuffle", "cross_frame", "intra_frame_pos_shuffle"], required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--learned-dir", type=Path)
    parser.add_argument("--gate0-dir", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--geometry-shuffle-dir", type=Path)
    parser.add_argument("--intra-frame-pos-shuffle-dir", type=Path)
    parser.add_argument("--cross-frame-w1-dir", type=Path)
    parser.add_argument("--cross-frame-w2-dir", type=Path)
    parser.add_argument("--primary-label")
    parser.add_argument("--csv-name", required=True)
    parser.add_argument("--md-name", required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.kind == "geometry_shuffle":
        entries = [
            ("D2 learned gates original", args.learned_dir),
            ("D2 gate=0", args.gate0_dir),
            ("D2 geometry-shuffle", args.geometry_shuffle_dir),
            ("original patch-only baseline", args.baseline_dir),
        ]
    elif args.kind == "intra_frame_pos_shuffle":
        entries = [
            ("D2 learned gates original", args.learned_dir),
            ("D2 gate=0", args.gate0_dir),
            ("D2 geometry-shuffle", args.geometry_shuffle_dir),
            ("D2 intra-frame position shuffle", args.intra_frame_pos_shuffle_dir),
            ("original patch-only baseline", args.baseline_dir),
        ]
    else:
        entries = [
            ("D2 original learned gates", args.learned_dir),
            ("D2 cross-frame window=1", args.cross_frame_w1_dir),
            ("D2 cross-frame window=2", args.cross_frame_w2_dir),
            ("D2 gate=0", args.gate0_dir),
            ("D2 geometry-shuffle", args.geometry_shuffle_dir),
            ("original patch-only baseline", args.baseline_dir),
        ]

    rows = []
    rows_by_label = {}
    primary_row = None
    for label, path in entries:
        if path is None:
            continue
        if not path.exists():
            print(f"[WARN] Skipping missing result dir for {label}: {path}")
            continue
        try:
            row, vsibench_path, results_path = summarize_run(label, path)
        except FileNotFoundError as err:
            if args.primary_label is not None and label == args.primary_label:
                raise
            print(f"[WARN] Skipping result dir for {label}: {err}")
            continue
        rows.append(row)
        rows_by_label[label] = row
        if args.primary_label is not None and label == args.primary_label:
            primary_row = row
            shutil.copy2(vsibench_path, args.output_dir / "vsibench.json")
            shutil.copy2(results_path, args.output_dir / "results.json")
        elif args.primary_label is None and args.kind == "geometry_shuffle" and label == "D2 geometry-shuffle":
            primary_row = row
            shutil.copy2(vsibench_path, args.output_dir / "vsibench.json")
            shutil.copy2(results_path, args.output_dir / "results.json")
        elif args.primary_label is None and args.kind == "intra_frame_pos_shuffle" and label == "D2 intra-frame position shuffle":
            primary_row = row
            shutil.copy2(vsibench_path, args.output_dir / "vsibench.json")
            shutil.copy2(results_path, args.output_dir / "results.json")
        elif args.primary_label is None and args.kind == "cross_frame" and label == "D2 cross-frame window=1":
            primary_row = row
            shutil.copy2(vsibench_path, args.output_dir / "vsibench.json")
            shutil.copy2(results_path, args.output_dir / "results.json")

    if primary_row is None:
        raise RuntimeError(f"No primary row was produced for kind={args.kind}.")

    if args.kind == "geometry_shuffle":
        title = "GeoRoPE Geometry-Shuffle VSiBench Summary"
    elif args.kind == "intra_frame_pos_shuffle":
        title = "GeoRoPE Intra-Frame Position-Shuffle VSiBench Summary"
    else:
        title = "GeoRoPE Cross-Frame VSiBench Summary"
    write_csv(args.output_dir / args.csv_name, rows)
    write_markdown(args.output_dir / args.md_name, rows, title, interpretation(args.kind, rows_by_label))
    print(f"[DONE] Summary CSV: {args.output_dir / args.csv_name}")
    print(f"[DONE] Summary MD: {args.output_dir / args.md_name}")


if __name__ == "__main__":
    main()
