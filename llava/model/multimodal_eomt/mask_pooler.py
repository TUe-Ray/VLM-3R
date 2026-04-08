"""Mask-guided pooling for EoMT side outputs.

This module converts dense visual patch/grid features into object-centric tokens
using EoMT soft masks. It is intentionally standalone and does not modify the
main fusion path.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskGuidedPooler(nn.Module):
    """Pool visual grid tokens with selected EoMT soft masks.

    Inputs:
        soft_masks: (B, Q, Hm, Wm)
        visual_features: (B, T, D), where T is typically a square grid token count.
        class_logits: optional (B, Q, C+1), used by class_confidence selection.

    Output:
        Dict containing pooled tokens and selection metadata.
    """

    SUPPORTED_SELECTIONS = {
        "class_confidence",
        "mean_mask_confidence",
        "mask_area",
        "mask_energy",
    }

    def __init__(
        self,
        default_top_k: int = 5,
        default_selection: str = "class_confidence",
        default_mask_area_threshold: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        if default_selection not in self.SUPPORTED_SELECTIONS:
            raise ValueError(
                f"Unsupported default_selection '{default_selection}'. "
                f"Supported: {sorted(self.SUPPORTED_SELECTIONS)}"
            )
        self.default_top_k = int(default_top_k)
        self.default_selection = default_selection
        self.default_mask_area_threshold = float(default_mask_area_threshold)
        self.eps = float(eps)

    def _infer_grid_shape(
        self,
        token_count: int,
        grid_size: Optional[Tuple[int, int]] = None,
        prefix_tokens: Optional[int] = None,
    ) -> Tuple[int, int, int]:
        """Infer prefix token count and spatial grid size from token count."""
        if grid_size is not None:
            gh, gw = int(grid_size[0]), int(grid_size[1])
            if gh <= 0 or gw <= 0:
                raise ValueError(f"Invalid grid_size: {grid_size}")
            grid_tokens = gh * gw
            if prefix_tokens is None:
                inferred_prefix = token_count - grid_tokens
                if inferred_prefix < 0:
                    raise ValueError(
                        f"token_count={token_count} smaller than grid tokens {grid_tokens}"
                    )
                return inferred_prefix, gh, gw
            if prefix_tokens + grid_tokens != token_count:
                raise ValueError(
                    f"prefix_tokens + grid_size mismatch: {prefix_tokens} + {grid_tokens} != {token_count}"
                )
            return int(prefix_tokens), gh, gw

        if prefix_tokens is not None:
            remainder = token_count - int(prefix_tokens)
            side = int(math.isqrt(remainder))
            if side * side != remainder:
                raise ValueError(
                    f"Cannot infer square grid from token_count={token_count} with prefix={prefix_tokens}"
                )
            return int(prefix_tokens), side, side

        side = int(math.isqrt(token_count))
        if side * side == token_count:
            return 0, side, side

        for pref in range(1, token_count):
            remainder = token_count - pref
            side = int(math.isqrt(remainder))
            if side * side == remainder:
                return pref, side, side

        raise ValueError(
            f"Unable to infer prefix+square-grid decomposition for token_count={token_count}"
        )

    def _compute_query_scores(
        self,
        soft_masks: torch.Tensor,
        class_logits: Optional[torch.Tensor],
        selection: str,
        mask_area_threshold: float,
    ) -> torch.Tensor:
        if selection == "class_confidence":
            if class_logits is None:
                raise ValueError("class_confidence selection requires class_logits")
            if class_logits.shape[-1] <= 1:
                raise ValueError(
                    "class_logits must include at least one foreground class and one no-object class"
                )
            class_probs = torch.softmax(class_logits.float(), dim=-1)
            foreground_probs = class_probs[:, :, :-1]
            scores, _ = foreground_probs.max(dim=-1)
            return scores

        if selection == "mean_mask_confidence":
            return soft_masks.float().mean(dim=(-1, -2))

        if selection == "mask_area":
            return (soft_masks.float() >= float(mask_area_threshold)).float().mean(dim=(-1, -2))

        if selection == "mask_energy":
            return soft_masks.float().pow(2).mean(dim=(-1, -2))

        raise ValueError(
            f"Unsupported selection '{selection}'. Supported: {sorted(self.SUPPORTED_SELECTIONS)}"
        )

    def select_topk_queries(
        self,
        soft_masks: torch.Tensor,
        class_logits: Optional[torch.Tensor],
        top_k: int,
        selection: str,
        mask_area_threshold: float,
    ) -> Dict[str, torch.Tensor]:
        """Select top-k query indices per frame according to a heuristic score."""
        if soft_masks.ndim != 4:
            raise ValueError(f"soft_masks must be 4D (B,Q,H,W), got shape {tuple(soft_masks.shape)}")

        bsz, num_queries = soft_masks.shape[0], soft_masks.shape[1]
        if num_queries == 0:
            empty_idx = soft_masks.new_zeros((bsz, 0), dtype=torch.long)
            empty_scores = soft_masks.new_zeros((bsz, 0), dtype=torch.float32)
            return {
                "topk_indices": empty_idx,
                "topk_scores": empty_scores,
                "topk_class_ids": empty_idx,
                "all_scores": soft_masks.new_zeros((bsz, 0), dtype=torch.float32),
            }

        effective_k = max(1, min(int(top_k), num_queries))
        scores = self._compute_query_scores(
            soft_masks=soft_masks,
            class_logits=class_logits,
            selection=selection,
            mask_area_threshold=mask_area_threshold,
        )
        topk_scores, topk_indices = scores.topk(effective_k, dim=1, largest=True)

        if class_logits is not None and class_logits.shape[-1] > 1:
            class_probs = torch.softmax(class_logits.float(), dim=-1)
            fg_probs = class_probs[:, :, :-1]
            _, max_class_ids = fg_probs.max(dim=-1)
            topk_class_ids = torch.gather(max_class_ids, dim=1, index=topk_indices)
        else:
            topk_class_ids = torch.full_like(topk_indices, fill_value=-1)

        return {
            "topk_indices": topk_indices,
            "topk_scores": topk_scores,
            "topk_class_ids": topk_class_ids,
            "all_scores": scores,
        }

    def forward(
        self,
        soft_masks: torch.Tensor,
        visual_features: torch.Tensor,
        class_logits: Optional[torch.Tensor] = None,
        top_k: Optional[int] = None,
        selection: Optional[str] = None,
        grid_size: Optional[Tuple[int, int]] = None,
        prefix_tokens: Optional[int] = None,
        mask_area_threshold: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        if soft_masks.ndim != 4:
            raise ValueError(f"soft_masks must be 4D (B,Q,H,W), got shape {tuple(soft_masks.shape)}")
        if visual_features.ndim != 3:
            raise ValueError(
                f"visual_features must be 3D (B,T,D), got shape {tuple(visual_features.shape)}"
            )

        selection = selection or self.default_selection
        if selection not in self.SUPPORTED_SELECTIONS:
            raise ValueError(
                f"Unsupported selection '{selection}'. Supported: {sorted(self.SUPPORTED_SELECTIONS)}"
            )
        top_k = int(self.default_top_k if top_k is None else top_k)
        mask_area_threshold = (
            self.default_mask_area_threshold
            if mask_area_threshold is None
            else float(mask_area_threshold)
        )

        b_masks, _, _, _ = soft_masks.shape
        b_vis, token_count, feat_dim = visual_features.shape
        batch_size = min(b_masks, b_vis)
        if batch_size <= 0:
            raise ValueError(
                f"Empty batch for pooling: soft_masks batch={b_masks}, visual_features batch={b_vis}"
            )

        soft_masks = soft_masks[:batch_size]
        visual_features = visual_features[:batch_size]
        if class_logits is not None:
            class_logits = class_logits[:batch_size]

        prefix_count, grid_h, grid_w = self._infer_grid_shape(
            token_count=token_count,
            grid_size=grid_size,
            prefix_tokens=prefix_tokens,
        )

        grid_tokens = visual_features[:, prefix_count:, :]
        if grid_tokens.shape[1] != grid_h * grid_w:
            raise ValueError(
                f"Grid token count mismatch: expected {grid_h * grid_w}, got {grid_tokens.shape[1]}"
            )

        resized_masks = F.interpolate(
            soft_masks.float(),
            size=(grid_h, grid_w),
            mode="bilinear",
            align_corners=False,
        )

        selected = self.select_topk_queries(
            soft_masks=resized_masks,
            class_logits=class_logits,
            top_k=top_k,
            selection=selection,
            mask_area_threshold=mask_area_threshold,
        )

        topk_indices = selected["topk_indices"]
        topk_scores = selected["topk_scores"]
        topk_class_ids = selected["topk_class_ids"]
        k = topk_indices.shape[1]

        idx_expanded = topk_indices.unsqueeze(-1).unsqueeze(-1).expand(batch_size, k, grid_h, grid_w)
        selected_masks = torch.gather(resized_masks, dim=1, index=idx_expanded)

        # Convert selected masks into normalized spatial weights.
        weights = selected_masks.clamp(min=0.0)
        weights = weights / weights.sum(dim=(-1, -2), keepdim=True).clamp_min(self.eps)

        grid_tokens_flat = grid_tokens.view(batch_size, grid_h * grid_w, feat_dim)
        weights_flat = weights.view(batch_size, k, grid_h * grid_w)
        pooled_tokens = torch.einsum("bkh,bhd->bkd", weights_flat, grid_tokens_flat)

        return {
            "pooled_tokens": pooled_tokens,
            "selected_indices": topk_indices,
            "selected_scores": topk_scores,
            "selected_class_ids": topk_class_ids,
            "selection_method": selection,
            "all_scores": selected["all_scores"],
            "selected_masks": selected_masks,
            "weights": weights,
            "grid_size": torch.tensor([grid_h, grid_w], device=pooled_tokens.device),
            "prefix_token_count": torch.tensor([prefix_count], device=pooled_tokens.device),
            "batch_mismatch": torch.tensor([b_masks - b_vis], device=pooled_tokens.device),
        }
