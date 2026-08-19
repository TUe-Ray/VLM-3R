import torch

from llava.model.c1_structured_isometry import canonical_linear_weight, canonical_square


def test_c1_structured_isometry_is_rng_independent_and_orthogonal():
    torch.manual_seed(7)
    first = canonical_square(12).clone()
    torch.manual_seed(99991)
    second = canonical_square(12).clone()
    assert torch.equal(first, second)
    identity = torch.eye(12)
    assert torch.allclose(first.transpose(0, 1) @ first, identity, atol=2e-6, rtol=2e-6)


def test_c1_structured_semi_isometries_have_expected_orientation():
    expansion = canonical_linear_weight(12, 20)
    contraction = canonical_linear_weight(20, 12)
    assert expansion.shape == (20, 12)
    assert contraction.shape == (12, 20)
    assert torch.allclose(expansion.transpose(0, 1) @ expansion, torch.eye(12), atol=2e-6, rtol=2e-6)
    assert torch.allclose(contraction @ contraction.transpose(0, 1), torch.eye(12), atol=2e-6, rtol=2e-6)
