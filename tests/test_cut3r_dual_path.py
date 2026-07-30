import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "llava" / "model" / "cut3r_dual_path.py"
spec = importlib.util.spec_from_file_location("_cut3r_dual_path", MODULE_PATH)
dual = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dual
spec.loader.exec_module(dual)


def test_hybrid_mask_blocks_text_to_spatial_and_preserves_causality():
    text = torch.tensor([[True, True, True]])
    spatial = torch.tensor([[True, True, True, False]])
    frames = torch.tensor([[0, 0, 1, -1]])
    allow = dual.build_hybrid_attention_allow_mask(text, spatial, frames, "frame_local")
    assert torch.equal(allow[0, :3, :3], torch.tril(torch.ones(3, 3, dtype=torch.bool)))
    assert not allow[0, :3, 3:].any()
    assert allow[0, 3, :3].all()
    assert allow[0, 3, 4] and not allow[0, 3, 5]
    assert not allow[0, 6].any()


def test_global_hybrid_mask_allows_cross_frame_spatial_attention():
    allow = dual.build_hybrid_attention_allow_mask(
        torch.tensor([[True]]), torch.tensor([[True, True]]), torch.tensor([[0, 1]]), "global"
    )
    assert allow[0, 1, 2] and allow[0, 2, 1]


def test_writeback_frame_local_leaves_unselected_visual_state_unchanged():
    torch.manual_seed(0)
    module = dual.DualPathWriteback(hidden_size=8, num_heads=2, output_init_std=1e-3)
    states = torch.randn(1, 3, 8)
    memory = torch.randn(1, 4, 8)
    cache = dual.DualPathSpatialCache(
        memory, torch.ones(1, 4, dtype=torch.bool), torch.tensor([[0, 0, 1, 1]]), torch.tensor([0])
    )
    branch = type("Branch", (), {"query_scope": "text_only", "writeback_visibility": "frame_local", "writeback": module, "last_debug": {}})()
    # Exercise the public implementation without constructing Qwen blocks.
    result = dual.Cut3RDualPathSpatialBranch.apply_writeback(
        branch,
        states,
        cache,
        torch.ones(1, 3, dtype=torch.bool),
        torch.tensor([[False, True, True]]),
        torch.tensor([[-1, 0, 1]]),
    )
    assert not torch.equal(result[:, 0], states[:, 0])
    assert torch.equal(result[:, 1:], states[:, 1:])


def test_cache_repeat_reorder_and_dtype_device_validation():
    cache = dual.DualPathSpatialCache(
        torch.randn(2, 3, 4), torch.ones(2, 3, dtype=torch.bool), torch.tensor([[0, 0, 0], [1, 1, 1]]), torch.tensor([10, 20])
    )
    expanded = cache.batch_repeat_interleave(2)
    assert expanded.source_batch_ids.tolist() == [10, 10, 20, 20]
    reordered = expanded.reorder_cache(torch.tensor([3, 1, 2, 0]))
    assert reordered.source_batch_ids.tolist() == [20, 10, 20, 10]
    reordered.validate(4, reordered.states.device, reordered.states.dtype)


def test_writeback_visibility_allows_text_all_frames_and_visual_same_frame():
    allow = dual.build_writeback_allow_mask(
        query_valid=torch.tensor([[True, True]]),
        query_is_visual=torch.tensor([[False, True]]),
        query_frame_ids=torch.tensor([[-1, 1]]),
        spatial_valid=torch.tensor([[True, True, True]]),
        spatial_frame_ids=torch.tensor([[0, 1, 1]]),
        visibility="frame_local",
    )
    assert allow[0, 0].all()
    assert allow[0, 1].tolist() == [False, True, True]


def test_protected_early_canonical_lora_names_remain_frozen():
    assert not dual.is_trainable_downstream_lora_parameter(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    )
    assert not dual.is_trainable_downstream_lora_parameter(
        "base_model.model.model.layers.2.mlp.up_proj.lora_B.default.weight"
    )
    assert dual.is_trainable_downstream_lora_parameter(
        "base_model.model.model.layers.3.self_attn.q_proj.lora_A.default.weight"
    )
    assert dual.is_trainable_downstream_lora_parameter("base_model.model.lm_head.lora_A.default.weight")


def test_merged_donor_state_maps_base_layers_and_retains_only_target_lora_factors():
    target_state = {
        "self_attn.q_proj.base_layer.weight": torch.zeros(2, 3),
        "self_attn.q_proj.base_layer.bias": torch.zeros(2),
        "self_attn.q_proj.lora_A.default.weight": torch.zeros(1, 3),
        "self_attn.q_proj.lora_B.default.weight": torch.zeros(2, 1),
    }
    source_state = {
        "self_attn.q_proj.weight": torch.full((2, 3), 7.0),
        "self_attn.q_proj.bias": torch.full((2,), 11.0),
    }
    prepared, retained = dual.prepare_merged_peft_donor_state(target_state, source_state, "tiny donor")
    assert set(prepared) == {
        "self_attn.q_proj.base_layer.weight",
        "self_attn.q_proj.base_layer.bias",
    }
    assert torch.equal(prepared["self_attn.q_proj.base_layer.weight"], source_state["self_attn.q_proj.weight"])
    assert retained == [
        "self_attn.q_proj.lora_A.default.weight",
        "self_attn.q_proj.lora_B.default.weight",
    ]


def test_merged_donor_state_rejects_non_lora_shape_mismatch():
    target_state = {"mlp.down_proj.base_layer.weight": torch.zeros(2, 3)}
    source_state = {"mlp.down_proj.weight": torch.zeros(3, 2)}
    try:
        dual.prepare_merged_peft_donor_state(target_state, source_state, "tiny donor")
    except RuntimeError as error:
        assert "shape_mismatched" in str(error)
    else:
        raise AssertionError("Expected a donor shape mismatch to be rejected.")


class _TinySwiGLU(nn.Module):
    def __init__(self, hidden_size=8, intermediate_size=15):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def forward(self, states):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(states)) * self.up_proj(states))


def test_tokenwise_mlp_chunking_matches_output_and_gradients():
    torch.manual_seed(7)
    reference_mlp = _TinySwiGLU()
    chunked_mlp = deepcopy(reference_mlp)
    reference_input = torch.randn(1, 11, 8, requires_grad=True)
    chunked_input = reference_input.detach().clone().requires_grad_(True)

    reference_output = dual.Cut3RDualPathSpatialBranch._run_tokenwise_mlp(reference_mlp, reference_input, 0)
    chunked_output = dual.Cut3RDualPathSpatialBranch._run_tokenwise_mlp(chunked_mlp, chunked_input, 3)
    reference_output.square().mean().backward()
    chunked_output.square().mean().backward()

    torch.testing.assert_close(chunked_output, reference_output, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(chunked_input.grad, reference_input.grad, rtol=1e-6, atol=1e-7)
    for (_, reference_parameter), (_, chunked_parameter) in zip(reference_mlp.named_parameters(), chunked_mlp.named_parameters()):
        torch.testing.assert_close(chunked_parameter.grad, reference_parameter.grad, rtol=1e-6, atol=1e-7)


def test_chunked_writeback_matches_output_and_gradients():
    torch.manual_seed(11)
    reference_writeback = dual.DualPathWriteback(hidden_size=8, num_heads=2, output_init_std=1e-3)
    chunked_writeback = deepcopy(reference_writeback)
    reference_queries = torch.randn(1, 7, 8, requires_grad=True)
    chunked_queries = reference_queries.detach().clone().requires_grad_(True)
    reference_memory = torch.randn(1, 5, 8, requires_grad=True)
    chunked_memory = reference_memory.detach().clone().requires_grad_(True)
    query_valid = torch.tensor([[True, True, False, True, True, True, False]])
    query_is_visual = torch.tensor([[False, True, True, False, True, True, False]])
    query_frame_ids = torch.tensor([[-1, 0, 0, -1, 1, 1, -1]])
    spatial_valid = torch.tensor([[True, True, False, True, True]])
    spatial_frame_ids = torch.tensor([[0, 0, -1, 1, 1]])
    allow = dual.build_writeback_allow_mask(
        query_valid, query_is_visual, query_frame_ids, spatial_valid, spatial_frame_ids, "frame_local"
    )

    reference_output = reference_writeback(reference_queries, reference_memory, allow)
    chunked_output, valid_ratio = chunked_writeback.forward_chunked(
        chunked_queries,
        chunked_memory,
        query_valid,
        query_is_visual,
        query_frame_ids,
        spatial_valid,
        spatial_frame_ids,
        "frame_local",
        query_chunk_size=3,
    )
    reference_output.square().mean().backward()
    chunked_output.square().mean().backward()

    torch.testing.assert_close(chunked_output, reference_output, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(chunked_queries.grad, reference_queries.grad, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(chunked_memory.grad, reference_memory.grad, rtol=1e-6, atol=1e-7)
    for (_, reference_parameter), (_, chunked_parameter) in zip(reference_writeback.named_parameters(), chunked_writeback.named_parameters()):
        torch.testing.assert_close(chunked_parameter.grad, reference_parameter.grad, rtol=1e-6, atol=1e-7)
    assert valid_ratio == float(allow.sum().item()) / float(allow.numel())
