import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_bool_config(value, default=False):
    if value is None:
        return default
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _parse_int_list(value, name):
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    try:
        parsed = [int(item) for item in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a comma-separated list of integers, got {value!r}.") from exc
    if not parsed:
        raise ValueError(f"{name} must contain at least one layer.")
    return parsed


def _empty_long(device):
    return torch.empty(0, dtype=torch.long, device=device)


class Cut3RSpatialStackBranch(nn.Module):
    def __init__(self, feature_dim: int, hidden_size: int, zero_init: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(int(feature_dim))
        self.proj_in = nn.Linear(int(feature_dim), int(hidden_size))
        self.act = nn.GELU()
        self.proj_out = nn.Linear(int(hidden_size), int(hidden_size))
        if zero_init:
            nn.init.zeros_(self.proj_out.weight)
            nn.init.zeros_(self.proj_out.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj_out(self.act(self.proj_in(self.norm(tokens))))


class Cut3RSpatialStackMerger(nn.Module):
    """Build dense LLM residuals from pre-extracted CUT3R decoder-layer sidecars."""

    EXCLUDED_METADATA_KEYS = (
        "newline_token_indices",
        "padding_token_indices",
        "answer_token_indices",
        "text_token_indices",
        "special_token_indices",
        "camera_prefix_token_indices",
    )

    def __init__(self, config):
        super().__init__()
        self.cut3r_layers = _parse_int_list(
            getattr(config, "cut3r_spatialstack_layers", "6,9,12"),
            "cut3r_spatialstack_layers",
        )
        self.llm_layers = _parse_int_list(
            getattr(config, "cut3r_spatialstack_llm_layers", "0,1,2"),
            "cut3r_spatialstack_llm_layers",
        )
        if len(self.cut3r_layers) != len(self.llm_layers):
            raise ValueError(
                "cut3r_spatialstack_layers and cut3r_spatialstack_llm_layers must have the same length, "
                f"got {self.cut3r_layers} and {self.llm_layers}."
            )
        feature_dim = getattr(config, "cut3r_spatialstack_feature_dim", None)
        if feature_dim is None:
            feature_dim = getattr(config, "spatial_feature_dim", None)
        if feature_dim is None:
            raise ValueError(
                "use_cut3r_spatialstack=True requires cut3r_spatialstack_feature_dim "
                "or spatial_feature_dim so trainable merger parameters exist before optimizer creation."
            )
        self.feature_dim = int(feature_dim)
        self.hidden_size = int(getattr(config, "hidden_size"))
        self.feature_key = str(getattr(config, "cut3r_spatialstack_feature_key", "cut3r_dec_layers"))
        self.patch_only = _as_bool_config(getattr(config, "cut3r_spatialstack_patch_only", True), True)
        self.zero_init = _as_bool_config(getattr(config, "cut3r_spatialstack_zero_init", True), True)
        self.log_first_n = int(getattr(config, "cut3r_spatialstack_log_first_n", 3) or 0)
        self.layer_map = {int(llm_layer): int(cut3r_layer) for cut3r_layer, llm_layer in zip(self.cut3r_layers, self.llm_layers)}
        self.branches = nn.ModuleDict(
            {
                str(cut3r_layer): Cut3RSpatialStackBranch(
                    self.feature_dim,
                    self.hidden_size,
                    zero_init=self.zero_init,
                )
                for cut3r_layer in self.cut3r_layers
            }
        )
        self.last_debug = {}

    @staticmethod
    def resize_square_grid(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
        if tokens.dim() != 2:
            raise ValueError(f"CUT3R frame tokens must be [tokens, dim], got {tuple(tokens.shape)}.")
        source_tokens = int(tokens.shape[0])
        target_tokens = int(target_tokens)
        source_side = int(math.isqrt(source_tokens))
        target_side = int(math.isqrt(target_tokens))
        if source_side * source_side != source_tokens:
            raise ValueError(f"CUT3R source token count must be a square grid, got {source_tokens}.")
        if target_side * target_side != target_tokens:
            raise ValueError(f"Target visual token count must be a square grid, got {target_tokens}.")
        if source_tokens == target_tokens:
            return tokens
        grid = tokens.reshape(source_side, source_side, tokens.shape[-1]).permute(2, 0, 1).unsqueeze(0)
        resized = F.interpolate(
            grid.float(),
            size=(target_side, target_side),
            mode="bilinear",
            align_corners=False,
        )
        return resized[0].permute(1, 2, 0).reshape(target_tokens, tokens.shape[-1]).to(dtype=tokens.dtype)

    @staticmethod
    def _metadata_items(visual_metadata):
        if isinstance(visual_metadata, dict):
            return [visual_metadata]
        if isinstance(visual_metadata, (list, tuple)):
            return list(visual_metadata)
        raise RuntimeError(
            "CUT3R SpatialStack requires visual_metadata from prepare_inputs_labels_for_multimodal(); "
            f"got {type(visual_metadata).__name__}."
        )

    @staticmethod
    def _feature_items(spatial_features, batch_size: int) -> List[dict]:
        if spatial_features is None:
            raise RuntimeError("use_cut3r_spatialstack=True requires pre-extracted CUT3R spatial_features sidecars.")
        if isinstance(spatial_features, dict):
            if batch_size != 1:
                raise RuntimeError(
                    "A single spatial_features dict can only be used with batch_size=1 for CUT3R SpatialStack; "
                    f"got batch_size={batch_size}."
                )
            return [spatial_features]
        if isinstance(spatial_features, (list, tuple)):
            if len(spatial_features) != batch_size:
                raise RuntimeError(
                    "CUT3R SpatialStack spatial_features batch mismatch: "
                    f"features={len(spatial_features)}, visual_metadata={batch_size}."
                )
            return list(spatial_features)
        raise RuntimeError(f"Unsupported spatial_features type for CUT3R SpatialStack: {type(spatial_features).__name__}.")

    def _extract_layer_tokens(self, sidecar: dict, cut3r_layer: int) -> torch.Tensor:
        if not isinstance(sidecar, dict):
            raise RuntimeError(f"CUT3R SpatialStack sidecar must be a dict, got {type(sidecar).__name__}.")
        layer_key = str(int(cut3r_layer))
        if self.feature_key in sidecar:
            layer_payloads = sidecar[self.feature_key]
            if not isinstance(layer_payloads, dict):
                raise RuntimeError(
                    f"CUT3R SpatialStack sidecar[{self.feature_key!r}] must be a dict keyed by decoder layer."
                )
            if layer_key not in layer_payloads and int(cut3r_layer) not in layer_payloads:
                raise RuntimeError(
                    f"CUT3R SpatialStack sidecar is missing decoder layer {cut3r_layer}; "
                    f"available keys={sorted(str(k) for k in layer_payloads.keys())}."
                )
            payload = layer_payloads.get(layer_key, layer_payloads.get(int(cut3r_layer)))
            if isinstance(payload, dict):
                if "patch_tokens" not in payload:
                    raise RuntimeError(f"CUT3R decoder layer {cut3r_layer} payload lacks 'patch_tokens'.")
                tokens = payload["patch_tokens"]
            else:
                tokens = payload
        elif "patch_tokens" in sidecar:
            if len(self.cut3r_layers) != 1:
                raise RuntimeError(
                    "Legacy CUT3R sidecar schema with top-level 'patch_tokens' is only valid when exactly "
                    f"one cut3r_spatialstack_layer is configured; got {self.cut3r_layers}."
                )
            tokens = sidecar["patch_tokens"]
        else:
            raise RuntimeError(
                f"CUT3R SpatialStack sidecar must contain {self.feature_key!r} or legacy 'patch_tokens'; "
                f"got keys={sorted(sidecar.keys())}."
            )
        if not isinstance(tokens, torch.Tensor):
            raise RuntimeError(f"CUT3R layer {cut3r_layer} patch tokens must be a tensor, got {type(tokens).__name__}.")
        if tokens.dim() == 4 and int(tokens.shape[0]) == 1:
            tokens = tokens[0]
        if tokens.dim() != 3:
            raise RuntimeError(
                f"CUT3R layer {cut3r_layer} patch tokens must be [frames,tokens,dim], got {tuple(tokens.shape)}."
            )
        if int(tokens.shape[-1]) != self.feature_dim:
            raise RuntimeError(
                f"CUT3R layer {cut3r_layer} feature dim mismatch: sidecar dim={int(tokens.shape[-1])}, "
                f"configured cut3r_spatialstack_feature_dim={self.feature_dim}."
            )
        return tokens.detach()

    @staticmethod
    def _sidecar_frame_indices(sidecar: dict) -> Optional[List[int]]:
        for key in ("frame_indices", "frame_order"):
            if key in sidecar:
                value = sidecar[key]
                if isinstance(value, torch.Tensor):
                    return [int(x) for x in value.detach().cpu().flatten().tolist()]
                return [int(x) for x in value]
        metadata = sidecar.get("metadata") if isinstance(sidecar, dict) else None
        if isinstance(metadata, dict):
            for key in ("frame_indices", "frame_order"):
                if key in metadata:
                    value = metadata[key]
                    if isinstance(value, torch.Tensor):
                        return [int(x) for x in value.detach().cpu().flatten().tolist()]
                    return [int(x) for x in value]
        return None

    @staticmethod
    def _grid_shape_at(metadata: dict, local_frame_idx: int) -> Optional[Tuple[int, int]]:
        shapes = metadata.get("visual_grid_shapes", None)
        if not isinstance(shapes, (list, tuple)) or local_frame_idx >= len(shapes):
            return None
        shape = shapes[local_frame_idx]
        if isinstance(shape, torch.Tensor):
            shape = shape.detach().cpu().flatten().tolist()
        if not isinstance(shape, (list, tuple)) or len(shape) < 2:
            return None
        return int(shape[0]), int(shape[1])

    def _validate_visual_metadata(self, metadata: dict, batch_idx: int, device) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        visual_indices = metadata.get("visual_token_indices", None)
        frame_ids = metadata.get("visual_frame_ids", None)
        if not isinstance(visual_indices, torch.Tensor) or not isinstance(frame_ids, torch.Tensor):
            raise RuntimeError(f"CUT3R SpatialStack metadata[{batch_idx}] is missing visual_token_indices/visual_frame_ids.")
        visual_indices = visual_indices.to(device=device, dtype=torch.long)
        frame_ids = frame_ids.to(device=device, dtype=torch.long)
        if visual_indices.numel() != frame_ids.numel():
            raise RuntimeError(
                f"CUT3R SpatialStack metadata[{batch_idx}] visual_token_indices and visual_frame_ids length mismatch: "
                f"{visual_indices.numel()} vs {frame_ids.numel()}."
            )
        excluded = []
        for key in self.EXCLUDED_METADATA_KEYS:
            value = metadata.get(key, _empty_long(device))
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                excluded.append(value.to(device=device, dtype=torch.long))
        if excluded and visual_indices.numel() > 0:
            excluded_indices = torch.cat(excluded)
            overlap = torch.isin(visual_indices, excluded_indices)
            if bool(overlap.any().item()):
                bad = visual_indices[overlap][:16].detach().cpu().tolist()
                raise RuntimeError(
                    f"CUT3R SpatialStack metadata[{batch_idx}] visual tokens overlap excluded token positions: {bad}."
                )
        frame_order = metadata.get("frame_order", None)
        if frame_order is None:
            frame_order = list(dict.fromkeys(int(x) for x in frame_ids.detach().cpu().tolist()))
        else:
            frame_order = [int(x) for x in frame_order]
        return visual_indices, frame_ids, frame_order

    def _ensure_module_dtype(self, device, dtype):
        param = next(self.parameters(), None)
        if param is not None and (param.device != device or param.dtype != dtype):
            self.to(device=device, dtype=dtype)

    def forward(
        self,
        spatial_features,
        visual_metadata,
        *,
        seq_len: int,
        device,
        dtype,
    ) -> Dict[int, torch.Tensor]:
        metadata_items = self._metadata_items(visual_metadata)
        batch_size = len(metadata_items)
        feature_items = self._feature_items(spatial_features, batch_size)
        self._ensure_module_dtype(device, dtype)

        residuals = {
            int(llm_layer): torch.zeros(
                batch_size,
                int(seq_len),
                self.hidden_size,
                device=device,
                dtype=dtype,
            )
            for llm_layer in self.llm_layers
        }
        debug = {
            "selected_cut3r_layers": list(self.cut3r_layers),
            "selected_llm_layers": list(self.llm_layers),
            "feature_dim": int(self.feature_dim),
            "hidden_size": int(self.hidden_size),
            "zero_init": bool(self.zero_init),
            "samples": [],
            "layers": {},
        }

        for batch_idx, (metadata, sidecar) in enumerate(zip(metadata_items, feature_items)):
            visual_indices, frame_ids, frame_order = self._validate_visual_metadata(metadata, batch_idx, device)
            if int(visual_indices.numel()) == 0:
                continue
            if int(visual_indices.min().item()) < 0 or int(visual_indices.max().item()) >= int(seq_len):
                bad_min = int(visual_indices.min().item())
                bad_max = int(visual_indices.max().item())
                raise RuntimeError(
                    f"CUT3R SpatialStack metadata[{batch_idx}] visual_token_indices out of bounds for "
                    f"seq_len={int(seq_len)}: min={bad_min}, max={bad_max}."
                )
            sidecar_frame_indices = self._sidecar_frame_indices(sidecar)
            if sidecar_frame_indices is not None and sidecar_frame_indices != frame_order:
                raise RuntimeError(
                    f"CUT3R SpatialStack frame_indices mismatch for sample {batch_idx}: "
                    f"visual frame_order={frame_order}, sidecar frame_indices={sidecar_frame_indices}."
                )
            if sidecar_frame_indices is None and frame_order != list(range(len(frame_order))):
                raise RuntimeError(
                    f"CUT3R SpatialStack sidecar for sample {batch_idx} lacks frame_indices/frame_order, "
                    f"but visual frame_order={frame_order} is not the unambiguous default order."
                )
            should_debug_sample = self.log_first_n < 0 or len(debug["samples"]) < self.log_first_n
            if should_debug_sample:
                debug["samples"].append(
                    {
                        "sample_id": int(batch_idx),
                        "visual_token_count": int(visual_indices.numel()),
                        "frame_order": list(frame_order),
                    }
                )

            for llm_layer, cut3r_layer in self.layer_map.items():
                patch_tokens = self._extract_layer_tokens(sidecar, cut3r_layer).to(device=device, dtype=dtype)
                if int(patch_tokens.shape[0]) != len(frame_order):
                    raise RuntimeError(
                        f"CUT3R SpatialStack frame count mismatch for sample {batch_idx}, layer {cut3r_layer}: "
                        f"sidecar frames={int(patch_tokens.shape[0])}, visual frame_order={frame_order}."
                    )
                aligned_frames = []
                aligned_indices = []
                raw_counts = []
                aligned_counts = []
                for local_frame_idx, frame_id in enumerate(frame_order):
                    frame_mask = frame_ids == int(frame_id)
                    frame_visual_indices = visual_indices[frame_mask]
                    target_count = int(frame_visual_indices.numel())
                    if target_count == 0:
                        continue
                    grid_shape = self._grid_shape_at(metadata, local_frame_idx)
                    if grid_shape is not None:
                        grid_h, grid_w = grid_shape
                        if grid_h != grid_w:
                            raise RuntimeError(
                                f"CUT3R SpatialStack requires square visual grids for sample {batch_idx}, "
                                f"frame {frame_id}; got visual_grid_shapes[{local_frame_idx}]={grid_shape}."
                            )
                        if grid_h * grid_w != target_count:
                            raise RuntimeError(
                                f"CUT3R SpatialStack visual token count mismatch for sample {batch_idx}, "
                                f"frame {frame_id}: visual_grid_shape={grid_shape} implies {grid_h * grid_w} "
                                f"tokens, but visual metadata has {target_count} positions."
                            )
                    raw_frame_tokens = patch_tokens[local_frame_idx]
                    aligned = self.resize_square_grid(raw_frame_tokens, target_count)
                    aligned_frames.append(aligned)
                    aligned_indices.append(frame_visual_indices)
                    raw_counts.append(int(raw_frame_tokens.shape[0]))
                    aligned_counts.append(int(aligned.shape[0]))
                if not aligned_frames:
                    continue
                aligned_tokens = torch.cat(aligned_frames, dim=0)
                target_indices = torch.cat(aligned_indices, dim=0)
                projected = self.branches[str(cut3r_layer)](aligned_tokens)
                residuals[int(llm_layer)][batch_idx, target_indices] = projected
                if should_debug_sample:
                    debug["layers"].setdefault(str(llm_layer), []).append(
                        {
                            "sample_id": int(batch_idx),
                            "cut3r_layer": int(cut3r_layer),
                            "raw_token_counts": raw_counts,
                            "aligned_token_counts": aligned_counts,
                            "residual_norm": float(projected.detach().float().norm().item()),
                        }
                    )

        self.last_debug = debug
        return residuals
