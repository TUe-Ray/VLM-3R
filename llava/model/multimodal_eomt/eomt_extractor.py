"""Frozen EoMT inference wrapper for the migrated post-SFT probe assets.

This is intentionally limited to inference outputs needed by the probe cache;
it does not add EoMT modules to the VLM or implement object-token insertion.
"""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn


def _load_external_classes(repo_root: str):
    root = Path(repo_root).resolve()
    if not (root / "models" / "eomt.py").is_file():
        raise FileNotFoundError(f"EoMT source tree is missing models/eomt.py: {root}")
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    # The migrated source uses package-relative imports under a top-level
    # ``models`` package; load exactly that source tree, offline.
    from models.eomt import EoMT
    from models.vit import ViT

    return EoMT, ViT


class EoMTExtractor(nn.Module):
    """Run the frozen panoptic EoMT model and return raw mask/class logits."""

    def __init__(self, eomt_config: dict[str, Any]):
        super().__init__()
        import yaml

        self.config_path = str(eomt_config["config_path"])
        self.ckpt_path = str(eomt_config["ckpt_path"])
        self.repo_root = str(
            eomt_config.get("repo_root")
            or os.environ.get("EOMT_REPO_ROOT", "")
        )
        self.local_backbone_path = str(eomt_config.get("local_backbone_path") or "")
        self.img_size = tuple(eomt_config.get("img_size", (640, 640)))
        if not self.repo_root:
            raise RuntimeError("EOMTExtractor requires repo_root/EOMT_REPO_ROOT")
        with open(self.config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        network_cfg = config["model"]["init_args"]["network"]
        network_init = dict(network_cfg.get("init_args", {}))
        encoder_cfg = network_init.get("encoder", {})
        encoder_init = dict(encoder_cfg.get("init_args", {}))
        self.num_q = int(network_init.get("num_q", 200))
        num_blocks = int(network_init.get("num_blocks", 4))
        data_init = config.get("data", {}).get("init_args", {})
        stuff_classes = list(data_init.get("stuff_classes", []))
        self.stuff_class_ids = frozenset(int(value) for value in stuff_classes)
        self.num_classes = int(eomt_config.get("num_classes") or (max(stuff_classes) + 1 if stuff_classes else 133))
        EoMT, ViT = _load_external_classes(self.repo_root)
        # The runtime source's ViT wrapper does not expose timm's register-token
        # arguments.  Build its tiny container directly, then attach the exact
        # topology serialized by the trained EoMT checkpoint.  In particular,
        # do not substitute the migrated patch-14 DINOv2 weights for this
        # patch-16, four-register, patch-only-positional encoder.
        import timm

        encoder = ViT.__new__(ViT)
        nn.Module.__init__(encoder)
        encoder.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1))
        encoder.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1))
        encoder.backbone = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            img_size=(640, 640),
            patch_size=16,
            num_classes=0,
            reg_tokens=4,
            no_embed_class=True,
        )
        self.network = EoMT(
            encoder=encoder,
            num_classes=self.num_classes,
            num_q=self.num_q,
            num_blocks=num_blocks,
            masked_attn_enabled=False,
        )
        state = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise TypeError(f"EoMT checkpoint is not a state dict: {self.ckpt_path}")
        cleaned = {
            (str(key)[len("network."):] if str(key).startswith("network.") else str(key)): value
            for key, value in state.items()
        }
        message = self.network.load_state_dict(cleaned, strict=False)
        if len(message.missing_keys) > max(10, len(self.network.state_dict()) // 20):
            raise RuntimeError(
                f"EoMT checkpoint load left too many missing tensors: {len(message.missing_keys)}"
            )
        self.network.eval()
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)
        self.is_available = True
        self.checkpoint_load = {
            "missing_keys": list(message.missing_keys),
            "unexpected_keys": list(message.unexpected_keys),
            "state_tensor_count": len(cleaned),
        }

    def train(self, mode: bool = True):
        super().train(mode)
        self.network.eval()
        return self

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        height, width = self.img_size
        tensors = []
        for image in images:
            image = image.convert("RGB") if image.mode != "RGB" else image
            resized = image.resize((width, height), Image.Resampling.BILINEAR)
            tensor = torch.from_numpy(np.asarray(resized, dtype=np.float32)).permute(2, 0, 1)
            tensors.append(tensor)
        return torch.stack(tensors, dim=0)

    @torch.no_grad()
    def forward(self, images: list[Image.Image], frame_meta=None) -> dict[str, Any]:
        batch = self.preprocess(images)
        device = next(self.network.parameters()).device
        batch = batch.to(device=device)
        dtype = next(self.network.parameters()).dtype
        context = nullcontext()
        if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
            context = torch.autocast(device_type="cuda", dtype=dtype)
        with context:
            mask_layers, class_layers = self.network(batch / 255.0)
        mask_logits = mask_layers[-1]
        class_logits = class_layers[-1]
        return {
            "mask_logits": mask_logits,
            "class_logits": class_logits,
            "mask_resolution": tuple(int(value) for value in mask_logits.shape[-2:]),
            "query_count": int(self.num_q),
            "frame_meta": frame_meta,
            "stuff_class_ids": self.stuff_class_ids,
            "is_available": True,
        }
