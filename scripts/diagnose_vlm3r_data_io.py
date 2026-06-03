#!/usr/bin/env python3
"""Probe VLM-3R data visibility/readability from Slurm compute nodes.

The output is JSONL so logs can be grepped or post-processed easily.
"""

import argparse
import json
import os
import socket
import time
from pathlib import Path


DEFAULT_VIDEOS = [
    "scannetpp/videos/3eba679830.mp4",
    "scannetpp/videos/712b9ae775.mp4",
    "scannet/videos/scene0148_00.mp4",
    "scannetpp/videos/c2d714d386.mp4",
    "scannet/videos/scene0630_05.mp4",
    "scannetpp/videos/816e996553.mp4",
    "arkitscenes/videos/47430414.mp4",
    "scannetpp/videos/e2caaaf5b5.mp4",
    "scannetpp/videos/469f112e38.mp4",
    "scannet/videos/scene0272_00.mp4",
    "scannetpp/videos/7a0a669e90.mp4",
    "scannetpp/videos/54b005d19d.mp4",
    "scannet/videos/scene0114_01.mp4",
    "arkitscenes/videos/42445615.mp4",
]


def emit(payload):
    base = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_procid": os.environ.get("SLURM_PROCID"),
        "slurm_nodeid": os.environ.get("SLURM_NODEID"),
        "slurm_localid": os.environ.get("SLURM_LOCALID"),
    }
    base.update(payload)
    print(json.dumps(base, sort_keys=True), flush=True)


def timed(op, **payload):
    start = time.time()
    try:
        result = op()
        emit({**payload, "ok": True, "elapsed_sec": round(time.time() - start, 6), **(result or {})})
        return True
    except Exception as exc:
        emit(
            {
                **payload,
                "ok": False,
                "elapsed_sec": round(time.time() - start, 6),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return False


def sidecar_rel_for(video_rel, subdir):
    path = Path(video_rel)
    parts = list(path.with_suffix(".pt").parts)
    if "videos" in parts:
        parts[parts.index("videos")] = subdir
    else:
        parts.insert(-1, subdir)
    return str(Path(*parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True, help="Data roots to probe, e.g. FAST and WORK mirrors.")
    parser.add_argument("--videos", nargs="*", default=DEFAULT_VIDEOS)
    parser.add_argument("--sidecar-subdir", default="spatial_features_points")
    parser.add_argument("--read-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--decord", action="store_true")
    parser.add_argument("--torch-load-one-sidecar", action="store_true")
    args = parser.parse_args()

    emit(
        {
            "event": "start",
            "cwd": os.getcwd(),
            "roots": args.roots,
            "repeat": args.repeat,
            "read_bytes": args.read_bytes,
            "decord": args.decord,
        }
    )

    decord_video_reader = None
    if args.decord:
        try:
            from decord import VideoReader, cpu

            decord_video_reader = (VideoReader, cpu)
            emit({"event": "import_decord", "ok": True})
        except Exception as exc:
            emit({"event": "import_decord", "ok": False, "error_type": type(exc).__name__, "error": str(exc)})

    torch_mod = None
    if args.torch_load_one_sidecar:
        try:
            import torch

            torch_mod = torch
            emit({"event": "import_torch", "ok": True, "torch_version": torch.__version__})
        except Exception as exc:
            emit({"event": "import_torch", "ok": False, "error_type": type(exc).__name__, "error": str(exc)})

    loaded_sidecar = False
    for round_idx in range(args.repeat):
        for root in args.roots:
            for video_rel in args.videos:
                video_path = os.path.join(root, video_rel)
                sidecar_path = os.path.join(root, sidecar_rel_for(video_rel, args.sidecar_subdir))

                def stat_video():
                    st = os.stat(video_path)
                    return {"size": st.st_size, "mtime": int(st.st_mtime)}

                timed(stat_video, event="stat_video", round=round_idx, root=root, rel=video_rel, path=video_path)

                def read_video_header():
                    with open(video_path, "rb") as f:
                        data = f.read(args.read_bytes)
                    return {"bytes_read": len(data)}

                timed(read_video_header, event="read_video_header", round=round_idx, root=root, rel=video_rel, path=video_path)

                if decord_video_reader is not None:
                    VideoReader, cpu = decord_video_reader

                    def read_decord_first_frame():
                        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
                        frame = vr.get_batch([0]).asnumpy()
                        return {"num_frames": len(vr), "frame_shape": list(frame.shape)}

                    timed(read_decord_first_frame, event="decord_first_frame", round=round_idx, root=root, rel=video_rel, path=video_path)

                def stat_sidecar():
                    st = os.stat(sidecar_path)
                    return {"size": st.st_size, "mtime": int(st.st_mtime)}

                timed(stat_sidecar, event="stat_sidecar", round=round_idx, root=root, rel=video_rel, path=sidecar_path)

                def read_sidecar_header():
                    with open(sidecar_path, "rb") as f:
                        data = f.read(args.read_bytes)
                    return {"bytes_read": len(data)}

                timed(read_sidecar_header, event="read_sidecar_header", round=round_idx, root=root, rel=video_rel, path=sidecar_path)

                if torch_mod is not None and not loaded_sidecar:

                    def load_sidecar():
                        payload = torch_mod.load(sidecar_path, map_location="cpu")
                        keys = sorted(payload.keys()) if isinstance(payload, dict) else None
                        return {"payload_type": type(payload).__name__, "keys": keys}

                    if timed(load_sidecar, event="torch_load_sidecar", round=round_idx, root=root, rel=video_rel, path=sidecar_path):
                        loaded_sidecar = True

    emit({"event": "done"})


if __name__ == "__main__":
    main()
