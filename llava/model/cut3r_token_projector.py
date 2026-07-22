"""CUT3R-only projector."""

import torch
import torch.nn as nn


class Cut3RTokenProjector(nn.Module):
    """Independent replica of the reference ``mlp2x_gelu`` projector."""

    def __init__(self, feature_dim: int, hidden_size: int, use_layernorm: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim) if use_layernorm else nn.Identity()
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected CUT3R tokens [F,T,C], got {tuple(tokens.shape)}")
        return self.layers(self.norm(tokens))
