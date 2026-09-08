#!/usr/bin/env python
"""Apply one frozen development-selected depth analysis to new videos once.

This command deliberately has no automatic stage ladder.  Its input is the
immutable selection artifact written by ``analyze_depth_subspace_occupancy``
after development inspection, and it re-fits probes from only the frozen
development train/validation videos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probing.analyze_depth_subspace_occupancy import (  # noqa: E402
    PROBE_POINTS,
    any_stable,
    manifest_by_split,
    run_v1,
    run_v2,
    run_v3,
    run_v4,
    write_json,
    write_rows,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected an object in {path}")
    return payload


def clone_as_dev_eval(video: dict[str, Any]) -> dict[str, Any]:
    clone = json.loads(json.dumps(video))
    clone["split"] = "dev_eval"
    for frame in clone.get("frames", []):
        frame["split"] = "dev_eval"
    return clone


def selected_profile(rows: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    matches = []
    for row in rows:
        if str(row.get("metric")) != str(spec["metric"]):
            continue
        rank = spec.get("rank_k")
        if rank is not None and int(row.get("rank_k", -1)) != int(rank):
            continue
        matches.append(row)
    if len(matches) != 1:
        raise RuntimeError(
            "Frozen stage/metric/rank did not identify exactly one confirmation profile result: "
            f"stage={spec['stage']}, metric={spec['metric']}, rank={spec.get('rank_k')}"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-selection", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-root", default=None, help="Must match the frozen selection when specified.")
    args = parser.parse_args()

    spec_path = Path(args.frozen_selection).resolve()
    spec = load_json(spec_path)
    if spec.get("schema_version") != "depth_subspace_frozen_confirmation_selection_v1":
        raise RuntimeError(f"Unsupported frozen selection schema: {spec.get('schema_version')!r}")
    development_manifest = Path(str(spec["development_manifest"])).resolve()
    confirmation_manifest = Path(str(spec["confirmation_manifest"])).resolve()
    token_selection_path = Path(str(spec["token_selection"])).resolve()
    for path, digest_key in (
        (development_manifest, "development_manifest_sha256"),
        (confirmation_manifest, "confirmation_manifest_sha256"),
        (token_selection_path, "token_selection_sha256"),
    ):
        if not path.is_file() or sha256_file(path) != str(spec[digest_key]):
            raise RuntimeError(f"Frozen input changed or is unavailable: {path}")
    cache_root = Path(args.cache_root or spec["cache_root"]).resolve()
    if args.cache_root and cache_root != Path(str(spec["cache_root"])).resolve():
        raise RuntimeError("--cache-root must match the immutable frozen-selection cache root")
    models = [str(model) for model in spec["models"]]
    if len(models) != 3:
        raise RuntimeError("Independent confirmation requires the frozen three-model architecture comparison")

    development = manifest_by_split(load_json(development_manifest))
    confirmation = manifest_by_split(load_json(confirmation_manifest))
    if not development.get("train") or not development.get("val"):
        raise RuntimeError("Development manifest must contain the frozen train and validation videos")
    confirmation_videos = confirmation.get("confirmation", [])
    if len(confirmation_videos) < 8:
        raise RuntimeError("Confirmation manifest must contain at least eight independently frozen videos")
    development_paths = {
        str(video["video_path"])
        for split in development.values()
        for video in split
    }
    if development_paths.intersection(str(video["video_path"]) for video in confirmation_videos):
        raise RuntimeError("Confirmation videos overlap the development manifest")
    by_split = {
        "train": development["train"],
        "val": development["val"],
        "dev_eval": [clone_as_dev_eval(video) for video in confirmation_videos],
    }
    selection_payload = load_json(token_selection_path)
    selected_indices = {
        str(key): [int(index) for index in value]
        for key, value in selection_payload["selected_valid_indices"].items()
    }
    train_ids = {str(video.get("video_sample_id", video["video_path"])) for video in by_split["train"]}
    if set(selected_indices) != train_ids:
        raise RuntimeError("Frozen token selection does not match the frozen development train videos")
    config = spec["frozen_analysis_config"]
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Confirmation output directory must be new and empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    stage = str(spec["stage"])
    profiles: list[dict[str, Any]]
    if stage == "v1_ridge_vf":
        _, profiles, _ = run_v1(
            cache_root=cache_root,
            output_dir=output_dir,
            models=models,
            by_split=by_split,
            selection=selected_indices,
            alphas=tuple(float(value) for value in config["alphas"]),
            random_count=int(config["random_directions"]),
            label_permutations=int(config["label_permutations"]),
            seed=int(config["seed"]),
        )
    elif stage in {"v2_rankk_depth_subspace", "v3_alignment_transfer"}:
        rank = spec.get("rank_k")
        if rank is None:
            raise RuntimeError(f"{stage} confirmation requires a frozen rank_k")
        _, v2_profiles, fitted = run_v2(
            cache_root=cache_root,
            output_dir=output_dir,
            models=models,
            by_split=by_split,
            selection=selected_indices,
            alphas=tuple(float(value) for value in config["alphas"]),
            random_count=int(config["random_directions"]),
            seed=int(config["seed"]),
            requested_bins=int(config["depth_bins"]),
            ranks_to_report=(int(rank),),
        )
        if stage == "v2_rankk_depth_subspace":
            profiles = v2_profiles
        else:
            _, _, profiles = run_v3(
                output_dir=output_dir,
                models=models,
                by_split=by_split,
                fitted=fitted,
                ranks_to_report=(int(rank),),
            )
    elif stage == "v4_geometry_propagation":
        _, profiles = run_v4(
            cache_root=cache_root,
            output_dir=output_dir,
            models=models,
            by_split=by_split,
            seed=int(config["seed"]),
        )
    else:
        raise RuntimeError(f"Unsupported frozen stage: {stage}")

    profile = selected_profile(profiles, spec)
    result = {
        "schema_version": "depth_subspace_confirmation_result_v1",
        "frozen_selection": str(spec_path),
        "frozen_selection_sha256": sha256_file(spec_path),
        "stage": stage,
        "metric": spec["metric"],
        "rank_k": spec.get("rank_k"),
        "scientific_rationale": spec["scientific_rationale"],
        "confirmation_video_count": len(confirmation_videos),
        "probe_points": list(PROBE_POINTS),
        "profile_result": profile,
        "stable_confirmation_distinction": bool(profile.get("stable", False)),
        "selection_rule": "Confirmation evaluated one pre-frozen analysis; it did not choose a stage, metric, rank, or refinement.",
    }
    write_json(output_dir / "confirmation_result.json", result)
    write_rows(output_dir / "confirmation_profile.csv", [profile])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
