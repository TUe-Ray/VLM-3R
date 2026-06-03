#!/usr/bin/env python3
import argparse
import os

import torch


def unwrap_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dict/state_dict.")
    for key in ("state_dict", "model", "module"):
        nested = checkpoint.get(key)
        if isinstance(nested, dict) and any("spatial_rank_head." in str(k) for k in nested.keys()):
            return nested
    return checkpoint


def extract_spatial_rank_head(checkpoint):
    checkpoint = unwrap_checkpoint(checkpoint)
    extracted = {}
    for key, value in checkpoint.items():
        key = str(key)
        if "spatial_rank_head." not in key:
            continue
        stripped_key = key.split("spatial_rank_head.", 1)[1]
        extracted[stripped_key] = value.detach().cpu() if torch.is_tensor(value) else value
    if not extracted:
        raise ValueError("No keys containing spatial_rank_head. were found.")
    return extracted


def main():
    parser = argparse.ArgumentParser(description="Extract spatial_rank_head/P_geo weights from a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint .bin/.pt file.")
    parser.add_argument("--output", required=True, help="Output path, e.g. p_geo.bin.")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = extract_spatial_rank_head(checkpoint)
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(state, args.output)
    print(f"Saved spatial_rank_head/P_geo to: {args.output}")
    print(f"Extracted keys: {list(state.keys())}")


if __name__ == "__main__":
    main()
