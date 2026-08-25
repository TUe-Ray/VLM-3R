"""CPU contracts for the opt-in depth-subspace diagnostic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBING_DIR = REPO_ROOT / "scripts" / "probing"
if str(PROBING_DIR) not in sys.path:
    sys.path.insert(0, str(PROBING_DIR))

from analyze_depth_subspace_occupancy import profile_permutation_test, variance_fraction, variance_fractions_for_bases  # noqa: E402
from extract_depth_probe_features import cleaned_text_token_indices, register_pre_llm_hooks  # noqa: E402


class _Base(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mm_projector = torch.nn.Linear(3, 4, bias=False)

    def get_fusion_block(self):
        return None


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = _Base()

    def get_model(self):
        return self.base


def test_additive_pre_sft_fusion_output_is_projector_input() -> None:
    model = _Model()
    captured: dict[str, torch.Tensor] = {}
    handles = register_pre_llm_hooks(model, "SS012", ["fusion_output", "projected_features"], captured)
    value = torch.randn(2, 196, 3)
    try:
        projected = model.get_model().mm_projector(value)
    finally:
        for handle in handles:
            handle.remove()
    assert torch.equal(captured["fusion_output"], value)
    assert torch.equal(captured["projected_features"], projected)


def test_text_mask_excludes_structural_tokens() -> None:
    metadata = {
        "text_token_indices": torch.tensor([0, 1, 2, 3, 4, 5]),
        "newline_token_indices": torch.tensor([1]),
        "padding_token_indices": torch.tensor([2]),
        "special_token_indices": torch.tensor([3]),
        "camera_prefix_token_indices": torch.tensor([4]),
    }
    assert cleaned_text_token_indices(metadata, seq_len=6, device=torch.device("cpu")).tolist() == [0, 5]


def test_variance_fraction_uses_raw_coordinate_direction() -> None:
    features = np.asarray([[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [-2.0, 0.0]])
    assert variance_fraction(features, np.asarray([[1.0], [0.0]])) == 1.0
    assert variance_fraction(features, np.asarray([[0.0], [1.0]])) == 0.0


def test_batched_variance_fractions_match_individual_calculation() -> None:
    rng = np.random.default_rng(5)
    features = rng.normal(size=(17, 5))
    bases = []
    for _ in range(3):
        bases.append(np.linalg.qr(rng.normal(size=(5, 2)), mode="reduced")[0])
    batched = variance_fractions_for_bases(features, np.stack(bases))
    expected = np.asarray([variance_fraction(features, basis) for basis in bases])
    assert np.allclose(batched, expected)


def test_profile_permutation_keeps_layer_profile_together() -> None:
    # Four videos, three architectures, two correlated probe points.  The
    # architecture ordering is stable only when each video's full profile is
    # retained as a unit during label permutation.
    values = np.asarray(
        [
            [[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]],
            [[0.1, 0.1], [1.1, 1.1], [3.1, 3.1]],
            [[-0.1, -0.1], [0.9, 0.9], [2.9, 2.9]],
            [[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]],
        ]
    )
    result = profile_permutation_test(values, seed=0)
    assert result["permutation_exact"] is True
    assert result["observed_T"] > result["null_q95"]
    assert result["stable"] is True
