import importlib.util
import sys
from pathlib import Path

import torch


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
