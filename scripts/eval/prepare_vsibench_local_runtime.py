#!/usr/bin/env python3
"""Write a local sidecar-only runtime config for a migrated VLM-3R adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--siglip-path", type=Path, required=True)
    args = parser.parse_args()

    with args.source_config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["mm_vision_tower"] = str(args.siglip_path)
    if "vision_tower" in config:
        config["vision_tower"] = str(args.siglip_path)
    # The original CUT3R token sidecars are the exact data consumed by the
    # cross-attention fusion.  Keeping the heavyweight runtime tower unloaded
    # avoids a redundant model and its unavailable historical checkpoint.
    config["spatial_tower_preextracted_only"] = True
    # Generation only consumes the final position's logits.  Avoid materializing
    # a multi-GiB full-vocabulary tensor for every visual prompt position.
    config["eval_last_token_logits_only"] = True

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    with args.output_config.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[RUNTIME CONFIG] wrote {args.output_config}")
    print("[RUNTIME CONFIG] spatial_tower_preextracted_only=True")
    print("[RUNTIME CONFIG] eval_last_token_logits_only=True")


if __name__ == "__main__":
    main()
