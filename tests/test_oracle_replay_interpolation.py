from types import SimpleNamespace

import torch

from llava.model.oracle_replay_interpolation import (
    build_independent_oracle_payloads,
    effective_residual_scales,
    interpolate_payloads,
)


class DummyMerger:
    llm_layers = (0, 1, 2)
    residual_scale = 0.75

    def __call__(self, features, metadata, *, seq_len, device, dtype):
        # New tensors are deliberately constructed on each invocation.
        base = features["value"].to(device=device, dtype=dtype)
        return {layer: base.new_zeros((1, seq_len, base.shape[-1])) + base.mean() * self.residual_scale for layer in self.llm_layers}


def test_loaded_scales_are_reported_without_defaulting():
    assert effective_residual_scales(DummyMerger()) == {0: 0.75, 1: 0.75, 2: 0.75}
    try:
        effective_residual_scales(SimpleNamespace(llm_layers=(0,)))
    except RuntimeError as error:
        assert "residual_scale" in str(error)
    else:
        raise AssertionError("missing loaded scale must fail")


def test_oracle_and_replay_payloads_have_independent_storage():
    merger = DummyMerger()
    oracle, replay, provenance = build_independent_oracle_payloads(
        merger, {"value": torch.arange(8, dtype=torch.float32).reshape(2, 4)}, [{"x": 1}],
        seq_len=5, device=torch.device("cpu"), dtype=torch.float32,
    )
    for layer in oracle:
        torch.testing.assert_close(oracle[layer], replay[layer])
        assert oracle[layer] is not replay[layer]
        assert oracle[layer].untyped_storage().data_ptr() != replay[layer].untyped_storage().data_ptr()
    assert provenance["oracle"]["effective_residual_scales"] == {0: 0.75, 1: 0.75, 2: 0.75}


def test_interpolation_endpoints_and_formula():
    teacher = {0: torch.ones(1, 2, 3), 1: torch.ones(1, 2, 3) * 2}
    predicted = {0: torch.ones(1, 2, 3) * 5, 1: torch.ones(1, 2, 3) * 10}
    assert interpolate_payloads(teacher, predicted, 0.0)[0] is teacher[0]
    assert interpolate_payloads(teacher, predicted, 1.0)[1] is predicted[1]
    middle = interpolate_payloads(teacher, predicted, 0.25)
    torch.testing.assert_close(middle[0] - teacher[0], 0.25 * (predicted[0] - teacher[0]))
