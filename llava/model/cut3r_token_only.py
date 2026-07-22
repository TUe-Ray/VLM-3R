"""Project final-layer CUT3R patch-token sidecars into LLM visual space."""

import torch
import torch.nn as nn


def extract_cut3r_patch_tokens(sidecar, frame_count, feature_dim=768, sidecar_key="patch_tokens", sample_index=0):
    """Strictly validate the one CUT3R sidecar assigned to a sampled video."""
    if not isinstance(sidecar, dict):
        raise RuntimeError(f"CUT3R-token-only sidecar {sample_index} must be a dict, got {type(sidecar).__name__}.")
    if sidecar_key != "patch_tokens":
        raise ValueError("CUT3R-token-only supports only the patch_tokens sidecar key.")
    if sidecar_key not in sidecar:
        raise KeyError(f"CUT3R-token-only sidecar {sample_index} is missing {sidecar_key!r}.")
    tokens = sidecar[sidecar_key]
    if not isinstance(tokens, torch.Tensor):
        raise TypeError(f"CUT3R-token-only sidecar {sample_index} {sidecar_key!r} must be a tensor.")
    expected_shape = (int(frame_count), 729, int(feature_dim))
    if tuple(tokens.shape) != expected_shape:
        raise RuntimeError(f"CUT3R-token-only sidecar {sample_index} shape mismatch: got {tuple(tokens.shape)}, expected {expected_shape}.")
    if not torch.isfinite(tokens).all():
        raise RuntimeError(f"CUT3R-token-only sidecar {sample_index} contains non-finite patch_tokens.")
    return tokens


class Cut3RTokenOnlyProjector(nn.Module):
    """Checkpoint-identifiable LayerNorm -> MLP projector for CUT3R tokens."""

    def __init__(self, feature_dim: int = 768, hidden_size: int = 3584):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_size = int(hidden_size)
        self.norm = nn.LayerNorm(self.feature_dim)
        self.proj_in = nn.Linear(self.feature_dim, self.hidden_size)
        self.act = nn.GELU()
        self.proj_out = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"CUT3R patch tokens must be [F,T,C], got {tuple(tokens.shape)}.")
        if int(tokens.shape[-1]) != self.feature_dim:
            raise ValueError(
                f"CUT3R patch-token channel size must be {self.feature_dim}, got {int(tokens.shape[-1])}."
            )
        return self.proj_out(self.act(self.proj_in(self.norm(tokens))))
