"""VSiBench helpers for locally migrated MP4 media."""

import os
from pathlib import Path

from lmms_eval.tasks.vsibench.utils import (
    process_docs,
    vsibench_aggregate_results,
    vsibench_doc_to_text,
    vsibench_process_results,
)


def vsibench_doc_to_visual(doc):
    """Resolve a test document to its canonical locally migrated MP4."""
    root_text = os.environ.get("VSI_VIDEO_ROOT", "").strip()
    if not root_text:
        raise RuntimeError(
            "VSI_VIDEO_ROOT is required by vsibench_local_mp4; it must contain "
            "{arkitscenes,scannet,scannetpp}/{scene_name}.mp4."
        )

    video_path = Path(root_text) / str(doc["dataset"]) / f"{doc['scene_name']}.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(f"Missing local VSiBench MP4: {video_path}")
    return [str(video_path)]
