"""VSiBench task helpers for the migrated 32-frame local input cache."""

import os
from pathlib import Path

from lmms_eval.tasks.vsibench.utils import (
    process_docs,
    vsibench_aggregate_results,
    vsibench_doc_to_text,
    vsibench_process_results,
)


def vsibench_doc_to_visual(doc):
    """Return a canonical virtual MP4 identity backed by forward_frames_32_v1."""
    root_text = os.environ.get("VSI_FORWARD_FRAMES_ROOT", "").strip()
    if not root_text:
        raise RuntimeError(
            "VSI_FORWARD_FRAMES_ROOT is required by vsibench_local_cache; it must point to "
            "the authoritative forward_frames_32_v1 root."
        )
    dataset = str(doc["dataset"])
    scene_name = str(doc["scene_name"])
    cache_file = Path(root_text) / "frames" / dataset / f"{scene_name}.pt"
    if not cache_file.is_file():
        raise FileNotFoundError(f"Missing local 32-frame cache: {cache_file}")
    return [f"{dataset}/videos/{scene_name}.mp4"]
