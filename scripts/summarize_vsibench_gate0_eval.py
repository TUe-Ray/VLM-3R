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


def interpretation(gate0_row, learned_row):
    if gate0_row is None or learned_row is None:
        return "Interpretation pending: gate=0 and learned-gate rows are both needed."
    gate0_room = gate0_row.get("Room Size")
    learned_room = learned_row.get("Room Size")
    if gate0_room is None or learned_room is None:
        return "Interpretation pending: Room Size is missing for at least one row."
    delta = gate0_room - learned_room
    if delta >= 2.0:
        return (
            "Gate=0 clearly improves Room Size, so GeoRoPE Q/K routing is likely hurting "
            "Room Size at inference time. Proceed to attention analysis or a design fix."
        )
    if delta <= -2.0:
        return (
            "Gate=0 makes Room Size worse, so the RoPE gate is not the cause of Room Size "
            "degradation. The issue is likely elsewhere, such as coordinate bias, training "
            "dynamics, or global aggregation."
        )
    return (
        "Gate=0 is similar to learned-gate D2, so the Room Size drop is probably not caused "
        "by inference-time RoPE gates. Do not prioritize attention visualization."
    )


def write_markdown(path: Path, rows, interp: str):
    headers = ["Variant"] + [name for name, _ in COLUMNS]
    lines = [
        "# GeoRoPE Gate=0 VSiBench Summary",
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
    parser.add_argument("--gate0-dir", required=True, type=Path)
    parser.add_argument("--learned-dir", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--csv-name", default="gate0_eval_summary.csv")
    parser.add_argument("--md-name", default="gate0_eval_summary.md")
    args = parser.parse_args()

    output_dir = args.output_dir or args.gate0_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    learned_row = None
    gate0_row = None
    for label, path in [
        ("D2 learned gates", args.learned_dir),
        ("D2 gate=0", args.gate0_dir),
        ("original patch-only baseline", args.baseline_dir),
    ]:
        if path is None:
            continue
        row, vsibench_path, results_path = summarize_run(label, path)
        rows.append(row)
        if label == "D2 learned gates":
            learned_row = row
        elif label == "D2 gate=0":
            gate0_row = row
            shutil.copy2(vsibench_path, output_dir / "vsibench.json")
            shutil.copy2(results_path, output_dir / "results.json")

    if gate0_row is None:
        raise RuntimeError("No D2 gate=0 row was produced.")

    write_csv(output_dir / args.csv_name, rows)
    write_markdown(output_dir / args.md_name, rows, interpretation(gate0_row, learned_row))
    print(f"[DONE] Summary CSV: {output_dir / args.csv_name}")
    print(f"[DONE] Summary MD: {output_dir / args.md_name}")


if __name__ == "__main__":
    main()
