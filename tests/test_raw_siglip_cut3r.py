"""CPU checks for raw SigLIP/CUT3R distillation primitives."""

from __future__ import annotations

import torch
from torch import nn

from llava.model.raw_siglip_cut3r import (
    FrozenSpatialStackPostprocessor,
    PatchCoordinateResampler,
    RawSpatialTemporalPredictor,
    RawTokenMLPPredictor,
    load_raw_predictor_checkpoint,
    raw_predictor_checkpoint_payload,
)


class TinyMerger(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleDict({str(layer): nn.Linear(768, 11) for layer in (6, 9, 12)})
        self.residual_scale = 0.7

    @staticmethod
    def resize_square_grid(tokens, target_tokens):
        assert target_tokens == 196
        # A differentiable selection is sufficient to verify the postprocess
        # autograd contract without allocating a full oracle projector.
        return tokens[:196]


def _tokens(batch=1, frames=1):
    return torch.randn(batch, frames, 729, 1152)


def test_residual_only_loss_backpropagates_through_frozen_postprocessor():
    predictor = RawTokenMLPPredictor(hidden_dim=8, residual_blocks=1)
    postprocessor = FrozenSpatialStackPostprocessor(TinyMerger())
    predicted = predictor(_tokens())
    residual = postprocessor(predicted)
    loss = sum(value.square().mean() for value in residual.values())
    loss.backward()
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad) for parameter in predictor.parameters())
    assert all(parameter.grad is None for parameter in postprocessor.parameters())


def test_full_objective_reaches_every_trainable_predictor_parameter():
    predictor = RawTokenMLPPredictor(hidden_dim=8, residual_blocks=1)
    postprocessor = FrozenSpatialStackPostprocessor(TinyMerger())
    targets = {layer: torch.randn(1, 1, 729, 768) for layer in (6, 9, 12)}
    predicted = predictor(_tokens())
    predicted_residual = postprocessor(predicted)
    target_residual = postprocessor.targets(targets)
    raw_loss = sum((predicted[layer] - targets[layer]).square().mean() for layer in predicted)
    residual_loss = sum((predicted_residual[layer] - target_residual[layer]).square().mean() for layer in predicted)
    (raw_loss + residual_loss).backward()
    missing = [
        name for name, parameter in predictor.named_parameters()
        if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all() or not torch.count_nonzero(parameter.grad))
    ]
    assert not missing, f"Full raw/residual objective missed predictor gradients: {missing}"
    assert all(parameter.grad is None for parameter in postprocessor.parameters())


def test_oracle_postprocessor_matches_resize_branch_and_single_scale():
    merger = TinyMerger()
    postprocessor = FrozenSpatialStackPostprocessor(merger)
    features = {layer: torch.randn(1, 2, 729, 768) for layer in (6, 9, 12)}
    output = postprocessor.targets(features)
    for layer in (6, 9, 12):
        expected = torch.stack(
            [merger.branches[str(layer)](merger.resize_square_grid(frame, 196)) * merger.residual_scale
             for frame in features[layer].reshape(-1, 729, 768)],
        ).reshape(1, 2, 196, 11)
        torch.testing.assert_close(output[layer], expected)


def test_raw_spatial_temporal_mask_and_shapes():
    predictor = RawSpatialTemporalPredictor(
        hidden_dim=12, spatial_blocks=1, temporal_layers=1, temporal_heads=3,
        temporal_ffn_dim=24, adapter_dim=4, temporal_max_frames=4,
    )
    tokens = _tokens(batch=1, frames=2)
    mask = torch.tensor([[True, False]])
    output = predictor(tokens, mask)
    assert set(output) == {6, 9, 12}
    assert all(tuple(value.shape) == (1, 2, 729, 768) for value in output.values())
    assert all(torch.count_nonzero(value[~mask]) == 0 for value in output.values())


def test_alignment_identity_and_row_major_landmarks():
    identity = PatchCoordinateResampler()
    values = torch.arange(729, dtype=torch.float32).reshape(1, 1, 729, 1).expand(-1, -1, -1, 1152)
    torch.testing.assert_close(identity(values), values)
    assert values[0, 0, 0, 0] == 0
    assert values[0, 0, 26, 0] == 26
    assert values[0, 0, 27, 0] == 27


def test_raw_checkpoint_roundtrip(tmp_path):
    predictor = RawTokenMLPPredictor(hidden_dim=8)
    path = tmp_path / "raw.pt"
    torch.save(raw_predictor_checkpoint_payload(predictor, teacher_checkpoint="teacher"), path)
    restored, checkpoint = load_raw_predictor_checkpoint(path)
    assert checkpoint["teacher_checkpoint"] == "teacher"
    inputs = _tokens(frames=1)
    for layer in (6, 9, 12):
        torch.testing.assert_close(predictor(inputs)[layer], restored(inputs)[layer])


if __name__ == "__main__":
    import tempfile

    test_residual_only_loss_backpropagates_through_frozen_postprocessor()
    test_full_objective_reaches_every_trainable_predictor_parameter()
    test_oracle_postprocessor_matches_resize_branch_and_single_scale()
    test_raw_spatial_temporal_mask_and_shapes()
    test_alignment_identity_and_row_major_landmarks()
    with tempfile.TemporaryDirectory() as directory:
        test_raw_checkpoint_roundtrip(__import__("pathlib").Path(directory))
    print("raw SigLIP/CUT3R CPU checks passed")
