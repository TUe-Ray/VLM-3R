from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


TARGET_DIMS = {
    "depth": 1,
    "log_depth": 1,
    "log_distance": 1,
    "azimuth": 1,
    "elevation": 1,
    "xyz": 3,
}


class AuxiliaryGeometryHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        target_names: Optional[Iterable[str]] = None,
        loss_type: str = "smooth_l1",
        allow_missing_targets: bool = False,
    ):
        super().__init__()
        self.target_names = list(target_names or ["azimuth", "elevation", "log_distance"])
        for target_name in self.target_names:
            if target_name not in TARGET_DIMS:
                raise ValueError(f"Unsupported auxiliary geometry target: {target_name}")
        if loss_type not in {"smooth_l1", "mse", "l1"}:
            raise ValueError(f"Unsupported geometry_loss_type: {loss_type}")
        self.loss_type = loss_type
        self.allow_missing_targets = bool(allow_missing_targets)
        self.target_dims = {name: TARGET_DIMS[name] for name in self.target_names}
        out_dim = sum(self.target_dims.values())
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_dim),
        )

    def forward(self, hidden_states: torch.Tensor):
        raw = self.mlp(hidden_states)
        outputs = {}
        start = 0
        for target_name in self.target_names:
            dim = self.target_dims[target_name]
            outputs[target_name] = raw[..., start:start + dim]
            start += dim
        return outputs

    def _element_loss(self, target_name: str, pred: torch.Tensor, target: torch.Tensor):
        if target_name == "azimuth":
            error = torch.atan2(torch.sin(pred - target), torch.cos(pred - target))
            zero = torch.zeros_like(error)
            if self.loss_type == "smooth_l1":
                return F.smooth_l1_loss(error, zero, reduction="none")
            if self.loss_type == "mse":
                return F.mse_loss(error, zero, reduction="none")
            return F.l1_loss(error, zero, reduction="none")
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred, target, reduction="none")
        if self.loss_type == "mse":
            return F.mse_loss(pred, target, reduction="none")
        return F.l1_loss(pred, target, reduction="none")

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        geometry_mask: Optional[torch.Tensor],
    ):
        numerator = None
        denominator = None
        device = next(iter(predictions.values())).device
        dtype = next(iter(predictions.values())).dtype
        missing = []
        for target_name in self.target_names:
            if target_name not in predictions or target_name not in targets:
                missing.append(target_name)
                continue
            pred = predictions[target_name]
            target = targets[target_name].to(device=pred.device, dtype=pred.dtype)
            if target.shape != pred.shape:
                raise ValueError(
                    f"Geometry target shape mismatch for {target_name}: "
                    f"pred={tuple(pred.shape)} target={tuple(target.shape)}"
                )
            valid = torch.isfinite(pred) & torch.isfinite(target)
            if geometry_mask is not None:
                valid = valid & geometry_mask.to(device=pred.device, dtype=torch.bool).unsqueeze(-1)
            element_loss = self._element_loss(target_name, pred.float(), target.float())
            element_loss = torch.nan_to_num(element_loss, nan=0.0, posinf=0.0, neginf=0.0)
            valid = valid.to(dtype=element_loss.dtype)
            cur_num = (element_loss * valid).sum()
            cur_den = valid.sum()
            numerator = cur_num if numerator is None else numerator + cur_num
            denominator = cur_den if denominator is None else denominator + cur_den

        if missing and not self.allow_missing_targets:
            raise ValueError(
                f"Missing auxiliary geometry targets for {missing}. "
                f"Available targets: {sorted(targets.keys())}. "
                "Either change aux_geometry_targets or set allow_missing_geometry_targets=True."
            )
        if numerator is None or denominator is None or denominator.detach().item() <= 0:
            return torch.zeros((), device=device, dtype=dtype)
        return numerator / denominator.clamp_min(1.0)
