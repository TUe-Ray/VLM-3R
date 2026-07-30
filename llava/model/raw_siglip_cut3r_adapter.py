"""Online raw-SigLIP adapter for predicted CUT3R SpatialStack residuals."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch
from torch import nn

from llava.model.raw_siglip_cut3r import (
    FrozenSpatialStackPostprocessor,
    QWEN_LAYERS,
    SOURCE_LAYERS,
    load_raw_predictor_checkpoint,
)


class RawSigLIPCut3RResidualAdapter(nn.Module):
    """Predict raw CUT3R grids and inject only visual 14x14 patch positions."""

    _EXCLUDED_METADATA_KEYS = (
        "newline_token_indices", "padding_token_indices", "answer_token_indices",
        "text_token_indices", "special_token_indices", "camera_prefix_token_indices",
        "cut3r_camera_token_indices", "spatial_bridge_token_indices",
    )

    def __init__(self, predictor, postprocessor, source_layers=SOURCE_LAYERS, llm_layers=QWEN_LAYERS, gamma_layers=None):
        super().__init__()
        self.predictor = predictor
        self.postprocessor = postprocessor
        self.source_layers = tuple(int(x) for x in source_layers)
        self.llm_layers = tuple(int(x) for x in llm_layers)
        self.gamma_layers = tuple(float(x) for x in (gamma_layers or [1.0] * len(self.llm_layers)))
        if self.source_layers != tuple(getattr(predictor, "source_layers", ())):
            raise RuntimeError("Raw predictor source layers do not match raw adapter mapping.")
        if len(self.source_layers) != len(self.llm_layers) or len(self.llm_layers) != len(self.gamma_layers):
            raise ValueError("Raw adapter mappings/gammas must have equal lengths.")
        self.last_debug: Dict[str, object] = {}

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, config):
        predictor, checkpoint = load_raw_predictor_checkpoint(checkpoint_path)
        source_layers = tuple(int(x) for x in checkpoint["architecture"]["source_layers"])
        configured_source = tuple(int(x.strip()) for x in str(getattr(config, "cut3r_spatialstack_layers", "6,9,12")).split(","))
        configured_llm = tuple(int(x.strip()) for x in str(getattr(config, "cut3r_spatialstack_llm_layers", "0,1,2")).split(","))
        if source_layers != configured_source or configured_llm != QWEN_LAYERS:
            raise RuntimeError(f"Raw predictor mapping {source_layers}->{QWEN_LAYERS} conflicts with config {configured_source}->{configured_llm}.")
        teacher_path = checkpoint.get("teacher_checkpoint") or getattr(config, "raw_cut3r_teacher_checkpoint", None)
        if not teacher_path:
            raise RuntimeError("Raw predictor checkpoint has no teacher_checkpoint.")
        postprocessor = FrozenSpatialStackPostprocessor.from_teacher_checkpoint(teacher_path)
        gammas = [float(getattr(config, f"raw_cut3r_gamma_layer{i}", 1.0)) for i in range(len(configured_llm))]
        adapter = cls(predictor, postprocessor, source_layers, configured_llm, gammas)
        adapter.checkpoint_metadata = {
            "path": str(checkpoint_path), "architecture": dict(checkpoint["architecture"]),
            "teacher_checkpoint": str(teacher_path),
        }
        return adapter

    @staticmethod
    def _items(visual_metadata) -> List[Mapping[str, object]]:
        if isinstance(visual_metadata, Mapping):
            return [visual_metadata]
        if isinstance(visual_metadata, (list, tuple)):
            return list(visual_metadata)
        raise RuntimeError("Raw predicted SpatialStack needs visual metadata.")

    def _positions(self, inputs_embeds, batch_index: int, metadata: Mapping[str, object]):
        visual = metadata.get("visual_token_indices")
        frame_ids = metadata.get("visual_frame_ids")
        if not isinstance(visual, torch.Tensor) or not isinstance(frame_ids, torch.Tensor):
            raise RuntimeError("Raw predicted SpatialStack metadata lacks visual token/frame IDs.")
        visual = visual.to(device=inputs_embeds.device, dtype=torch.long)
        frame_ids = frame_ids.to(device=inputs_embeds.device, dtype=torch.long)
        if visual.numel() == 0 or visual.numel() != frame_ids.numel():
            raise RuntimeError("Raw predicted SpatialStack has empty/inconsistent visual metadata.")
        if int(visual.min()) < 0 or int(visual.max()) >= int(inputs_embeds.shape[1]):
            raise RuntimeError("Raw predicted SpatialStack visual positions are out of sequence bounds.")
        excluded = [
            value.to(device=inputs_embeds.device, dtype=torch.long)
            for key in self._EXCLUDED_METADATA_KEYS
            if isinstance((value := metadata.get(key)), torch.Tensor) and value.numel()
        ]
        if excluded and torch.isin(visual, torch.cat(excluded)).any():
            raise RuntimeError("Raw predicted SpatialStack visual positions overlap non-visual positions.")
        order = metadata.get("frame_order")
        if order is None:
            order = list(dict.fromkeys(int(x) for x in frame_ids.detach().cpu().tolist()))
        order = [int(x) for x in order]
        positions = []
        for frame in order:
            current = visual[frame_ids == frame]
            if current.numel() != 196:
                raise RuntimeError(f"Raw predicted SpatialStack needs 196 patch tokens/frame; frame {frame} has {current.numel()}.")
            positions.append(current)
        return order, torch.cat(positions)

    def forward(self, inputs_embeds: torch.Tensor, visual_metadata) -> Dict[int, torch.Tensor]:
        items = self._items(visual_metadata)
        if len(items) != inputs_embeds.shape[0]:
            raise RuntimeError("Raw predicted SpatialStack metadata batch size mismatch.")
        residuals = {layer: torch.zeros_like(inputs_embeds) for layer in self.llm_layers}
        debug = []
        for index, metadata in enumerate(items):
            raw = metadata.get("raw_siglip_features")
            if not isinstance(raw, torch.Tensor):
                raise RuntimeError("Raw predicted SpatialStack requires raw_siglip_features from the vision tower.")
            order, positions = self._positions(inputs_embeds, index, metadata)
            if raw.dim() != 3 or tuple(raw.shape[1:]) != (729, 1152) or raw.shape[0] != len(order):
                raise RuntimeError(f"Raw SigLIP metadata must be [F,729,1152] aligned to visual frames, got {tuple(raw.shape)}.")
            parameter = next(self.predictor.parameters())
            if parameter.device != inputs_embeds.device:
                self.to(device=inputs_embeds.device)
                parameter = next(self.predictor.parameters())
            raw = raw.to(device=parameter.device, dtype=parameter.dtype)
            mask = torch.ones(1, raw.shape[0], dtype=torch.bool, device=raw.device)
            predicted = self.predictor(raw.unsqueeze(0), mask)
            predicted_residuals = self.postprocessor(predicted)
            for source, llm, gamma in zip(self.source_layers, self.llm_layers, self.gamma_layers):
                value = predicted_residuals[source][0].to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                residuals[llm][index].index_copy_(0, positions, value.reshape(-1, value.shape[-1]) * gamma)
            debug.append({"sample_index": index, "frames": len(order), "raw_shape": list(raw.shape), "visual_patch_positions": int(positions.numel())})
        self.last_debug = {
            "source": "raw_siglip_predicted_cut3r", "cut3r_loaded": False, "cut3r_executed": False,
            "source_layers": list(self.source_layers), "llm_layers": list(self.llm_layers), "samples": debug,
            "finite": all(torch.isfinite(value).all().item() for value in residuals.values()),
        }
        if not getattr(self, "_logged_runtime_contract", False):
            print(
                "[RAW_SIGLIP_CUT3R] CUT3R_loaded=false CUT3R_executed=false "
                f"predictor={getattr(self.predictor, 'architecture_name', type(self.predictor).__name__)} "
                f"raw_shapes={[entry['raw_shape'] for entry in debug]} "
                f"residual_shape={list(next(iter(residuals.values())).shape)} "
                f"injection_layers={list(self.llm_layers)} injected_tokens="
                f"{[entry['visual_patch_positions'] for entry in debug]} finite={self.last_debug['finite']}",
                flush=True,
            )
            self._logged_runtime_contract = True
        return residuals
