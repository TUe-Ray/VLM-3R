import importlib.util
import pathlib
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CUT3R_PATH = REPO_ROOT / "llava" / "model" / "cut3r_spatialstack.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cut3r_mod = load_module("_cut3r_spatialstack", CUT3R_PATH)
Cut3RSpatialStackMerger = cut3r_mod.Cut3RSpatialStackMerger
Cut3RSpatialStackPreAggregator = cut3r_mod.Cut3RSpatialStackPreAggregator
Cut3RSpatialStackCrossAttentionBlock = cut3r_mod.Cut3RSpatialStackCrossAttentionBlock
Cut3RSpatialStackCrossAttentionBlockV2 = cut3r_mod.Cut3RSpatialStackCrossAttentionBlockV2
Cut3RCameraTokenProjector = cut3r_mod.Cut3RCameraTokenProjector


def test_zero3_safe_layer_payload_mapping_preserves_integer_key_contract():
    payloads = {0: torch.randn(1, 2, 3), 2: torch.randn(1, 2, 3)}
    wrapped = cut3r_mod.SpatialStackLayerPayloads(payloads)
    assert list(wrapped) == [0, 2]
    assert set(wrapped.keys()) == {0, 2}
    assert wrapped[0] is payloads[0]
    assert isinstance(wrapped.__dict__["_payloads"], dict)


def _config(**overrides):
    values = {
        "hidden_size": 8,
        "cut3r_spatialstack_layers": "6",
        "cut3r_spatialstack_llm_layers": "0",
        "cut3r_spatialstack_feature_dim": 4,
        "cut3r_spatialstack_feature_key": "cut3r_dec_layers",
        "cut3r_spatialstack_zero_init": False,
        "cut3r_spatialstack_log_first_n": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _metadata(visual_indices=None, frame_ids=None, frame_order=None, **extra):
    if visual_indices is None:
        visual_indices = [1, 2, 4, 5]
    if frame_ids is None:
        frame_ids = [0 for _ in visual_indices]
    if frame_order is None:
        frame_order = list(dict.fromkeys(int(x) for x in frame_ids))
    meta = {
        "visual_token_indices": torch.tensor(visual_indices, dtype=torch.long),
        "visual_frame_ids": torch.tensor(frame_ids, dtype=torch.long),
        "frame_order": frame_order,
        "visual_grid_shapes": [(2, 2) for _ in frame_order],
        "raw_visual_grid_shapes": [(2, 2) for _ in frame_order],
        "newline_token_indices": torch.empty(0, dtype=torch.long),
        "padding_token_indices": torch.empty(0, dtype=torch.long),
        "answer_token_indices": torch.empty(0, dtype=torch.long),
        "text_token_indices": torch.empty(0, dtype=torch.long),
        "special_token_indices": torch.empty(0, dtype=torch.long),
        "camera_prefix_token_indices": torch.empty(0, dtype=torch.long),
    }
    meta.update(extra)
    return meta


def _tokens(frames=1, tokens=4, dim=4):
    return torch.arange(frames * tokens * dim, dtype=torch.float32).reshape(frames, tokens, dim)


def _camera_tokens(frames=1, dim=4):
    return torch.arange(frames * dim, dtype=torch.float32).reshape(frames, 1, dim)


def _preagg_sidecar(frames=1, tokens=4, dim=4):
    base = _tokens(frames=frames, tokens=tokens, dim=dim)
    return {
        "cut3r_dec_layers": {
            "6": base,
            "9": base + 100.0,
            "12": base + 200.0,
        }
    }


def test_layer_map_length_mismatch_has_clear_error():
    with pytest.raises(ValueError, match="same length"):
        Cut3RSpatialStackMerger(
            _config(cut3r_spatialstack_layers="6,9", cut3r_spatialstack_llm_layers="0")
        )


def test_sidecar_parsing_supports_layer_dict_tensor_and_payload():
    merger = Cut3RSpatialStackMerger(
        _config(
            cut3r_spatialstack_layers="6,9",
            cut3r_spatialstack_llm_layers="0,1",
            hidden_size=6,
        )
    )
    sidecar = {
        "cut3r_dec_layers": {
            "6": _tokens(),
            "9": {"patch_tokens": _tokens()},
        }
    }
    residuals = merger(sidecar, [_metadata()], seq_len=8, device=torch.device("cpu"), dtype=torch.float32)
    assert sorted(residuals.keys()) == [0, 1]
    assert residuals[0].shape == (1, 8, 6)
    assert residuals[1].shape == (1, 8, 6)
    assert merger.fusion_type == "add"
    assert sorted(merger.branches.keys()) == ["6", "9"]


def test_preaggregator_weighted_sum_shape_and_equal_initial_weights():
    aggregator = Cut3RSpatialStackPreAggregator([6, 9, 12], feature_dim=4, preagg_type="weighted_sum")
    features = {
        6: torch.randn(2, 4, 4),
        9: torch.randn(2, 4, 4),
        12: torch.randn(2, 4, 4),
    }
    out = aggregator(features)
    weights = torch.softmax(aggregator.scalar_logits.detach(), dim=0)
    assert out.shape == features[6].shape
    assert torch.allclose(weights, torch.full((3,), 1.0 / 3.0))


def test_preaggregator_concat_linear_shape_and_shape_mismatch_error():
    aggregator = Cut3RSpatialStackPreAggregator([6, 9, 12], feature_dim=4, preagg_type="concat_linear")
    features = {
        6: torch.randn(2, 4, 4),
        9: torch.randn(2, 4, 4),
        12: torch.randn(2, 4, 4),
    }
    out = aggregator(features)
    assert out.shape == features[6].shape
    assert aggregator.concat_proj.in_features == 12
    assert aggregator.concat_proj.out_features == 4
    features[12] = torch.randn(2, 5, 4)
    with pytest.raises(RuntimeError, match="identical feature shapes"):
        aggregator(features)


def test_preagg_weighted_sum_shared_projector_targets_are_configurable_and_backward():
    merger = Cut3RSpatialStackMerger(
        _config(
            hidden_size=6,
            cut3r_spatialstack_llm_layers="1,2,3",
            cut3r_spatialstack_preagg_enable=True,
            cut3r_spatialstack_preagg_layers="6,9,12",
            cut3r_spatialstack_preagg_type="weighted_sum",
            cut3r_spatialstack_preagg_projector_sharing="shared",
            cut3r_spatialstack_zero_init=False,
        )
    )
    residuals = merger(_preagg_sidecar(), [_metadata()], seq_len=8, device=torch.device("cpu"), dtype=torch.float32)
    assert sorted(residuals.keys()) == [1, 2, 3]
    assert list(merger.branches.keys()) == ["shared"]
    assert residuals[1].shape == (1, 8, 6)
    assert merger.last_debug["preagg_aggregated_feature_shape"] == [1, 4, 4]
    loss = sum(residual.float().pow(2).mean() for residual in residuals.values())
    assert torch.isfinite(loss)
    loss.backward()
    assert merger.preaggregator.scalar_logits.grad is not None
    assert torch.isfinite(merger.preaggregator.scalar_logits.grad).all()


def test_preagg_concat_linear_layer_specific_projectors_targets_are_configurable_and_backward():
    merger = Cut3RSpatialStackMerger(
        _config(
            hidden_size=6,
            cut3r_spatialstack_llm_layers="0,1,2",
            cut3r_spatialstack_preagg_enable=True,
            cut3r_spatialstack_preagg_layers="6,9,12",
            cut3r_spatialstack_preagg_type="concat_linear",
            cut3r_spatialstack_preagg_projector_sharing="layer_specific",
            cut3r_spatialstack_zero_init=False,
        )
    )
    residuals = merger(_preagg_sidecar(), [_metadata()], seq_len=8, device=torch.device("cpu"), dtype=torch.float32)
    assert sorted(residuals.keys()) == [0, 1, 2]
    assert sorted(merger.branches.keys()) == ["0", "1", "2"]
    assert len({id(branch) for branch in merger.branches.values()}) == 3
    assert all(residual.shape == (1, 8, 6) for residual in residuals.values())
    loss = sum(residual.float().pow(2).mean() for residual in residuals.values())
    assert torch.isfinite(loss)
    loss.backward()
    assert merger.preaggregator.concat_proj.weight.grad is not None
    assert torch.isfinite(merger.preaggregator.concat_proj.weight.grad).all()
    assert all(gamma.grad is not None for gamma in merger.preagg_layer_gammas.values())


def test_camera_token_extraction_supports_layer_payloads():
    merger = Cut3RSpatialStackMerger(_config(hidden_size=6))
    sidecar = {
        "cut3r_dec_layers": {
            "6": {
                "camera_tokens": torch.arange(2 * 1 * 4, dtype=torch.float32).reshape(2, 1, 4),
                "patch_tokens": _tokens(frames=2),
            }
        }
    }
    cams = merger._extract_layer_camera_tokens(sidecar, 6)
    assert cams.shape == (2, 1, 4)
    assert torch.allclose(cams[1, 0], torch.tensor([4.0, 5.0, 6.0, 7.0]))


def test_camera_token_projector_has_learnable_scale_and_hidden_shape():
    projector = Cut3RCameraTokenProjector(feature_dim=4, hidden_size=6, init_scale=0.5)
    out = projector(torch.randn(3, 4))
    assert out.shape == (3, 6)
    assert projector.gamma.requires_grad
    assert torch.allclose(projector.gamma.detach(), torch.tensor(0.5))


def test_legacy_patch_tokens_schema_is_single_layer_only():
    merger = Cut3RSpatialStackMerger(_config())
    residuals = merger({"patch_tokens": _tokens()}, [_metadata()], seq_len=8, device=torch.device("cpu"), dtype=torch.float32)
    assert sorted(residuals.keys()) == [0]

    multi_layer = Cut3RSpatialStackMerger(
        _config(cut3r_spatialstack_layers="6,9", cut3r_spatialstack_llm_layers="0,1")
    )
    with pytest.raises(RuntimeError, match="Legacy CUT3R sidecar schema"):
        multi_layer({"patch_tokens": _tokens()}, [_metadata()], seq_len=8, device=torch.device("cpu"), dtype=torch.float32)


def test_square_grid_resize_is_general_and_keeps_729_to_196_as_case():
    resized = Cut3RSpatialStackMerger.resize_square_grid(torch.randn(9, 3), 4)
    assert resized.shape == (4, 3)

    resized_729 = Cut3RSpatialStackMerger.resize_square_grid(torch.randn(729, 5), 196)
    assert resized_729.shape == (196, 5)


def test_visual_metadata_overlap_with_excluded_tokens_fails():
    merger = Cut3RSpatialStackMerger(_config())
    metadata = _metadata(newline_token_indices=torch.tensor([2], dtype=torch.long))
    sidecar = {"cut3r_dec_layers": {"6": _tokens()}}
    with pytest.raises(RuntimeError, match="overlap excluded"):
        merger(sidecar, [metadata], seq_len=8, device=torch.device("cpu"), dtype=torch.float32)


def test_frame_order_mismatch_fails():
    merger = Cut3RSpatialStackMerger(_config())
    sidecar = {
        "frame_indices": [1],
        "cut3r_dec_layers": {"6": _tokens()},
    }
    with pytest.raises(RuntimeError, match="frame_indices mismatch"):
        merger(sidecar, [_metadata(frame_order=[0])], seq_len=8, device=torch.device("cpu"), dtype=torch.float32)


def test_cross_attn_config_activates_cross_attn_blocks():
    merger = Cut3RSpatialStackMerger(
        _config(
            cut3r_spatialstack_fusion_type="cross_attn",
            cut3r_spatialstack_cross_attn_heads=2,
            cut3r_spatialstack_cross_attn_dropout=0.0,
            cut3r_spatialstack_cross_attn_zero_init=True,
        )
    )
    assert merger.fusion_type == "cross_attn"
    assert list(merger.branches.keys()) == []
    assert list(merger.cross_attn_blocks.keys()) == ["0"]
    assert merger.cross_attn_blocks["0"].num_heads == 2


def test_cross_attn_block_shape_checks():
    block = Cut3RSpatialStackCrossAttentionBlock(feature_dim=4, hidden_size=8, num_heads=2)
    visual_hidden = torch.randn(5, 8)
    geometry_tokens = torch.randn(7, 4)
    delta = block(visual_hidden, geometry_tokens)
    assert delta.shape == visual_hidden.shape


def test_cross_attn_prepares_same_frame_geometry_payload():
    metadata = _metadata(
        visual_indices=[1, 2, 3, 4, 6, 7, 8, 9],
        frame_ids=[0, 0, 0, 0, 1, 1, 1, 1],
        frame_order=[0, 1],
        visual_grid_shapes=[(2, 2), (2, 2)],
    )
    sidecar = {"cut3r_dec_layers": {"6": _tokens(frames=2, tokens=4, dim=4)}}
    merger = Cut3RSpatialStackMerger(_config(cut3r_spatialstack_fusion_type="cross_attn"))

    payload = merger(sidecar, [metadata], seq_len=12, device=torch.device("cpu"), dtype=torch.float32)

    assert sorted(payload.keys()) == [0]
    assert payload[0]["same_frame_only"] is True
    assert [entry["frame_id"] for entry in payload[0]["frames"]] == [0, 1]
    assert payload[0]["frames"][0]["visual_indices"].tolist() == [1, 2, 3, 4]
    assert payload[0]["frames"][1]["visual_indices"].tolist() == [6, 7, 8, 9]
    assert payload[0]["frames"][0]["geometry_tokens"].shape == (4, 4)
    assert payload[0]["frames"][1]["geometry_tokens"].shape == (4, 4)
    assert merger.last_debug["fusion_type"] == "cross_attn"


def test_cross_attn_frame_order_mismatch_fails():
    merger = Cut3RSpatialStackMerger(_config(cut3r_spatialstack_fusion_type="cross_attn"))
    sidecar = {
        "frame_indices": [1],
        "cut3r_dec_layers": {"6": _tokens()},
    }
    with pytest.raises(RuntimeError, match="frame_indices mismatch"):
        merger(sidecar, [_metadata(frame_order=[0])], seq_len=8, device=torch.device("cpu"), dtype=torch.float32)


def test_cross_attn_updates_only_visual_tokens():
    merger = Cut3RSpatialStackMerger(
        _config(
            hidden_size=8,
            cut3r_spatialstack_fusion_type="cross_attn",
            cut3r_spatialstack_cross_attn_heads=2,
            cut3r_spatialstack_cross_attn_zero_init=True,
        )
    )
    metadata = _metadata(visual_indices=[1, 2, 4, 5])
    sidecar = {"cut3r_dec_layers": {"6": _tokens(dim=4)}}
    payload = merger(sidecar, [metadata], seq_len=8, device=torch.device("cpu"), dtype=torch.float32)
    with torch.no_grad():
        merger.cross_attn_blocks["0"].out_proj.bias.fill_(0.25)

    hidden = torch.randn(1, 8, 8)
    updated, stat = merger.apply_cross_attn_layer(hidden, 0, payload[0])

    visual = torch.tensor([1, 2, 4, 5])
    non_visual = torch.tensor([0, 3, 6, 7])
    assert stat["fusion_type"] == "cross_attn"
    assert not torch.allclose(updated[0, visual], hidden[0, visual])
    assert torch.allclose(updated[0, non_visual], hidden[0, non_visual])


def test_cross_attn_zero_init_is_noop_on_hidden_states():
    merger = Cut3RSpatialStackMerger(
        _config(
            hidden_size=8,
            cut3r_spatialstack_fusion_type="cross_attn",
            cut3r_spatialstack_cross_attn_heads=2,
            cut3r_spatialstack_cross_attn_zero_init=True,
        )
    )
    metadata = _metadata(visual_indices=[1, 2, 4, 5])
    sidecar = {"cut3r_dec_layers": {"6": _tokens(dim=4)}}
    payload = merger(sidecar, [metadata], seq_len=8, device=torch.device("cpu"), dtype=torch.float32)
    hidden = torch.randn(1, 8, 8)

    updated, stat = merger.apply_cross_attn_layer(hidden, 0, payload[0])

    assert stat["cross_attn_output_norm"] == 0.0
    assert torch.allclose(updated, hidden, atol=0.0, rtol=0.0)


def test_cross_attn_v2_resize_uses_camera_memory_and_updates_only_visual_tokens():
    merger = Cut3RSpatialStackMerger(
        _config(
            hidden_size=8,
            cut3r_spatialstack_fusion_type="cross_attn_v2",
            cut3r_spatialstack_projector_hidden_dim=16,
            cut3r_spatialstack_cross_attn_heads=2,
            cut3r_spatialstack_cross_attn_patch_align="resize",
            cut3r_spatialstack_cross_attn_use_camera_tokens=True,
            cut3r_spatialstack_require_camera_tokens=True,
            cut3r_spatialstack_cross_attn_zero_init=False,
        )
    )
    metadata = _metadata(visual_indices=[1, 2, 4, 5], visual_grid_shapes=[(2, 2)])
    sidecar = {
        "cut3r_dec_layers": {
            "6": {
                "camera_tokens": _camera_tokens(),
                "patch_tokens": _tokens(tokens=9),
            }
        }
    }
    payload = merger(sidecar, [metadata], seq_len=8, device=torch.device("cpu"), dtype=torch.float32)
    frame = payload[0]["frames"][0]
    assert frame["geometry_tokens"].shape == (4, 4)
    assert frame["camera_tokens"].shape == (1, 4)
    assert merger.last_debug["fusion_type"] == "cross_attn_v2"
    assert merger.last_debug["layers"]["0"][0]["camera_tokens_present"] is True

    hidden = torch.randn(1, 8, 8)
    updated, stat = merger.apply_cross_attn_layer(hidden, 0, payload[0])
    visual = torch.tensor([1, 2, 4, 5])
    non_visual = torch.tensor([0, 3, 6, 7])
    assert stat["fusion_type"] == "cross_attn_v2"
    assert stat["patch_align"] == "resize"
    assert stat["use_camera_tokens"] is True
    assert stat["geo_memory_shapes"] == [[1, 5, 8]]
    assert not torch.allclose(updated[0, visual], hidden[0, visual])
    assert torch.allclose(updated[0, non_visual], hidden[0, non_visual])


def test_cross_attn_v2_merge_alignment_shapes_and_forward():
    merger = Cut3RSpatialStackMerger(
        _config(
            hidden_size=8,
            cut3r_spatialstack_fusion_type="cross_attn_v2",
            cut3r_spatialstack_projector_hidden_dim=16,
            cut3r_spatialstack_merge_size=2,
            cut3r_spatialstack_cross_attn_heads=2,
            cut3r_spatialstack_cross_attn_patch_align="merge",
            cut3r_spatialstack_cross_attn_use_camera_tokens=True,
            cut3r_spatialstack_require_camera_tokens=True,
            cut3r_spatialstack_cross_attn_zero_init=False,
        )
    )
    metadata = _metadata(visual_indices=[1, 2, 4, 5], visual_grid_shapes=[(2, 2)])
    sidecar = {
        "cut3r_dec_layers": {
            "6": {
                "camera_tokens": _camera_tokens(),
                "patch_tokens": _tokens(tokens=16),
            }
        }
    }
    payload = merger(sidecar, [metadata], seq_len=8, device=torch.device("cpu"), dtype=torch.float32)
    frame = payload[0]["frames"][0]
    assert frame["geometry_tokens"].shape == (4, 16)
    hidden = torch.randn(1, 8, 8)
    updated, stat = merger.apply_cross_attn_layer(hidden, 0, payload[0])
    assert updated.shape == hidden.shape
    assert stat["patch_align"] == "merge"
    assert stat["geo_memory_shapes"] == [[1, 5, 8]]


def test_cross_attn_v2_required_camera_tokens_missing_fails():
    merger = Cut3RSpatialStackMerger(
        _config(
            hidden_size=8,
            cut3r_spatialstack_fusion_type="cross_attn_v2",
            cut3r_spatialstack_projector_hidden_dim=16,
            cut3r_spatialstack_cross_attn_heads=2,
            cut3r_spatialstack_cross_attn_use_camera_tokens=True,
            cut3r_spatialstack_require_camera_tokens=True,
        )
    )
    sidecar = {"cut3r_dec_layers": {"6": {"patch_tokens": _tokens()}}}
    with pytest.raises(RuntimeError, match="camera_tokens"):
        merger(sidecar, [_metadata()], seq_len=8, device=torch.device("cpu"), dtype=torch.float32)


def test_cross_attn_v2_positive_gammas_nonzero_weights_and_backward():
    block = Cut3RSpatialStackCrossAttentionBlockV2(
        feature_dim=4,
        hidden_size=8,
        num_heads=2,
        projector_hidden_dim=16,
        gamma_attn_init=0.05,
        gamma_mlp_init=0.05,
    )
    assert torch.allclose(block.gamma_attn.detach(), torch.tensor(0.05))
    assert torch.allclose(block.gamma_mlp.detach(), torch.tensor(0.05))
    assert block.cross_attention.in_proj_weight.detach().abs().sum() > 0
    assert block.cross_attention.out_proj.weight.detach().abs().sum() > 0
    assert block.camera_proj[2].weight.detach().abs().sum() > 0
    assert block.patch_proj[2].weight.detach().abs().sum() > 0
    assert block.ffn[3].weight.detach().abs().sum() > 0

    visual_hidden = torch.randn(2, 4, 8, requires_grad=True)
    patch_tokens = torch.randn(2, 4, 4, requires_grad=True)
    camera_tokens = torch.randn(2, 1, 4, requires_grad=True)
    delta, stats = block(
        visual_hidden,
        patch_tokens,
        camera_tokens,
        visual_grid_shape=(2, 2),
        geometry_grid_shape=(2, 2),
        return_stats=True,
    )
    assert delta.shape == visual_hidden.shape
    assert stats["geo_memory_shape"] == [2, 5, 8]
    loss = delta.float().pow(2).mean()
    assert torch.isfinite(loss)
    loss.backward()
    for param in (
        block.cross_attention.in_proj_weight,
        block.cross_attention.out_proj.weight,
        block.camera_proj[2].weight,
        block.patch_proj[2].weight,
        block.ffn[3].weight,
        block.gamma_attn,
        block.gamma_mlp,
    ):
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        assert param.grad.detach().abs().sum() > 0


def test_cross_attn_v2_force_zero_gamma_at_eval_preserves_learned_params_and_zeroes_delta():
    torch.manual_seed(0)
    block = Cut3RSpatialStackCrossAttentionBlockV2(
        feature_dim=4,
        hidden_size=8,
        num_heads=2,
        projector_hidden_dim=16,
        gamma_attn_init=0.05,
        gamma_mlp_init=0.07,
        force_zero_gamma_at_eval=True,
    )
    visual_hidden = torch.randn(2, 4, 8)
    patch_tokens = torch.randn(2, 4, 4)
    camera_tokens = torch.randn(2, 1, 4)

    block.eval()
    delta, stats = block(
        visual_hidden,
        patch_tokens,
        camera_tokens,
        visual_grid_shape=(2, 2),
        geometry_grid_shape=(2, 2),
        return_stats=True,
    )
    assert torch.equal(delta, torch.zeros_like(delta))
    assert stats["cross_attn_v2_force_zero_gamma_at_eval"] is True
    assert stats["learned_gamma_attn"] == pytest.approx(0.05)
    assert stats["learned_gamma_mlp"] == pytest.approx(0.07)
    assert stats["effective_gamma_attn"] == 0.0
    assert stats["effective_gamma_mlp"] == 0.0
    assert stats["delta_norm"] == 0.0
    assert torch.allclose(block.gamma_attn.detach(), torch.tensor(0.05))
    assert torch.allclose(block.gamma_mlp.detach(), torch.tensor(0.07))

    block.train()
    train_delta, train_stats = block(
        visual_hidden,
        patch_tokens,
        camera_tokens,
        visual_grid_shape=(2, 2),
        geometry_grid_shape=(2, 2),
        return_stats=True,
    )
    assert train_stats["effective_gamma_attn"] == pytest.approx(0.05)
    assert train_stats["effective_gamma_mlp"] == pytest.approx(0.07)
    assert train_stats["delta_norm"] > 0.0
    assert train_delta.detach().abs().sum() > 0


def test_dense_residuals_are_zero_at_non_visual_positions():
    merger = Cut3RSpatialStackMerger(_config(hidden_size=5))
    sidecar = {"cut3r_dec_layers": {"6": _tokens(dim=4)}}
    residuals = merger(sidecar, [_metadata()], seq_len=8, device=torch.device("cpu"), dtype=torch.float32)
    residual = residuals[0]
    visual = torch.tensor([1, 2, 4, 5])
    non_visual = torch.tensor([0, 3, 6, 7])
    assert residual.shape == (1, 8, 5)
    assert torch.all(residual[0, non_visual] == 0)
    assert residual[0, visual].abs().sum() > 0


def test_residual_scale_controls_cut3r_spatialstack_residual_strength():
    metadata = _metadata()
    sidecar = {"cut3r_dec_layers": {"6": _tokens(dim=4)}}
    torch.manual_seed(31)
    baseline = Cut3RSpatialStackMerger(_config(hidden_size=5))
    half = Cut3RSpatialStackMerger(_config(hidden_size=5, cut3r_spatialstack_residual_scale=0.5))
    zero = Cut3RSpatialStackMerger(_config(hidden_size=5, cut3r_spatialstack_residual_scale=0.0))
    half.load_state_dict(baseline.state_dict())
    zero.load_state_dict(baseline.state_dict())

    base_residual = baseline(sidecar, [metadata], seq_len=8, device=torch.device("cpu"), dtype=torch.float32)[0]
    half_residual = half(sidecar, [metadata], seq_len=8, device=torch.device("cpu"), dtype=torch.float32)[0]
    zero_residual = zero(sidecar, [metadata], seq_len=8, device=torch.device("cpu"), dtype=torch.float32)[0]

    assert torch.allclose(half_residual, base_residual * 0.5)
    assert torch.all(zero_residual == 0)


def test_frame_shuffle_swaps_only_cut3r_spatialstack_source_frames():
    metadata = _metadata(
        visual_indices=[1, 2, 3, 4, 6, 7, 8, 9],
        frame_ids=[0, 0, 0, 0, 1, 1, 1, 1],
        frame_order=[0, 1],
        visual_grid_shapes=[(2, 2), (2, 2)],
    )
    sidecar = {"cut3r_dec_layers": {"6": _tokens(frames=2, tokens=4, dim=4)}}
    torch.manual_seed(23)
    baseline = Cut3RSpatialStackMerger(_config(hidden_size=5))
    shuffled = Cut3RSpatialStackMerger(
        _config(
            hidden_size=5,
            cut3r_spatialstack_frame_shuffle=True,
            cut3r_spatialstack_frame_shuffle_mode="reverse",
        )
    )
    shuffled.load_state_dict(baseline.state_dict())

    base_residual = baseline(sidecar, [metadata], seq_len=12, device=torch.device("cpu"), dtype=torch.float32)[0]
    shuffled_residual = shuffled(sidecar, [metadata], seq_len=12, device=torch.device("cpu"), dtype=torch.float32)[0]

    frame0 = torch.tensor([1, 2, 3, 4])
    frame1 = torch.tensor([6, 7, 8, 9])
    assert torch.allclose(shuffled_residual[0, frame0], base_residual[0, frame1])
    assert torch.allclose(shuffled_residual[0, frame1], base_residual[0, frame0])


def test_token_shuffle_reorders_only_cut3r_spatialstack_tokens_within_frame():
    visual = torch.tensor([1, 2, 4, 5])
    metadata = _metadata(visual_indices=visual.tolist(), frame_ids=[0, 0, 0, 0])
    sidecar = {"cut3r_dec_layers": {"6": _tokens(frames=1, tokens=4, dim=4)}}
    torch.manual_seed(29)
    baseline = Cut3RSpatialStackMerger(_config(hidden_size=5))
    shuffled = Cut3RSpatialStackMerger(
        _config(
            hidden_size=5,
            cut3r_spatialstack_token_shuffle=True,
            cut3r_spatialstack_token_shuffle_mode="reverse",
        )
    )
    shuffled.load_state_dict(baseline.state_dict())

    base_residual = baseline(sidecar, [metadata], seq_len=8, device=torch.device("cpu"), dtype=torch.float32)[0]
    shuffled_residual = shuffled(sidecar, [metadata], seq_len=8, device=torch.device("cpu"), dtype=torch.float32)[0]

    assert torch.allclose(shuffled_residual[0, visual], base_residual[0, visual.flip(0)])



def test_per_frame_token_mean_removes_spatial_variation_only_at_eval():
    visual = torch.tensor([1, 2, 4, 5])
    metadata = _metadata(visual_indices=visual.tolist(), frame_ids=[0, 0, 0, 0])
    sidecar = {"cut3r_dec_layers": {"6": _tokens(frames=1, tokens=4, dim=4)}}
    torch.manual_seed(31)
    mean_pooled = Cut3RSpatialStackMerger(
        _config(hidden_size=5, cut3r_spatialstack_per_frame_token_mean=True)
    )

    train_residual = mean_pooled(sidecar, [metadata], seq_len=8, device=torch.device("cpu"), dtype=torch.float32)[0]
    assert not torch.allclose(train_residual[0, visual], train_residual[0, visual[:1]].expand_as(train_residual[0, visual]))

    mean_pooled.eval()
    eval_residual = mean_pooled(sidecar, [metadata], seq_len=8, device=torch.device("cpu"), dtype=torch.float32)[0]
    assert torch.allclose(eval_residual[0, visual], eval_residual[0, visual[:1]].expand_as(eval_residual[0, visual]))


def test_per_frame_token_mean_rejects_token_shuffle_combination():
    with pytest.raises(ValueError, match="mutually exclusive"):
        Cut3RSpatialStackMerger(
            _config(
                cut3r_spatialstack_token_shuffle=True,
                cut3r_spatialstack_per_frame_token_mean=True,
            )
        )

def test_cpu_sidecar_inputs_are_cast_inside_merger():
    merger = Cut3RSpatialStackMerger(_config(hidden_size=5))
    sidecar = {"cut3r_dec_layers": {"6": _tokens(dim=4).cpu()}}
    residuals = merger(sidecar, [_metadata()], seq_len=8, device=torch.device("cpu"), dtype=torch.float64)
    assert residuals[0].dtype == torch.float64
    assert next(merger.parameters()).dtype == torch.float64


def _import_llava_qwen():
    pytest.importorskip("transformers")
    try:
        from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM
    except Exception as exc:
        pytest.skip(f"LlavaQwen import unavailable in this environment: {exc}")
    return LlavaQwenConfig, LlavaQwenForCausalLM


def _tiny_qwen_config(use_cut3r_spatialstack, fusion_type="add"):
    LlavaQwenConfig, _ = _import_llava_qwen()
    config = LlavaQwenConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=True,
    )
    config._attn_implementation = "eager"
    config.use_cut3r_spatialstack = bool(use_cut3r_spatialstack)
    config.cut3r_spatialstack_layers = "6"
    config.cut3r_spatialstack_llm_layers = "0"
    config.cut3r_spatialstack_feature_dim = 4
    config.cut3r_spatialstack_feature_key = "cut3r_dec_layers"
    config.cut3r_spatialstack_zero_init = True
    config.cut3r_spatialstack_log_first_n = 3
    config.cut3r_spatialstack_fusion_type = fusion_type
    config.cut3r_spatialstack_cross_attn_heads = 4
    config.cut3r_spatialstack_cross_attn_dropout = 0.0
    config.cut3r_spatialstack_cross_attn_zero_init = True
    config.cut3r_spatialstack_cross_attn_same_frame_only = True
    config.llm_visual_3d_rope_enable = False
    return config


def _make_zero_init_residuals(model, input_ids):
    metadata = _metadata(visual_indices=[1, 2, 3, 4])
    sidecar = {"cut3r_dec_layers": {"6": _tokens(tokens=4, dim=4)}}
    return model.model.cut3r_spatialstack_merger(
        sidecar,
        [metadata],
        seq_len=int(input_ids.shape[1]),
        device=input_ids.device,
        dtype=model.model.embed_tokens.weight.dtype,
    )


def _make_cross_attn_payload(model, input_ids):
    metadata = _metadata(visual_indices=[1, 2, 3, 4])
    sidecar = {"cut3r_dec_layers": {"6": _tokens(tokens=4, dim=4)}}
    return model.model.cut3r_spatialstack_merger(
        sidecar,
        [metadata],
        seq_len=int(input_ids.shape[1]),
        device=input_ids.device,
        dtype=model.model.embed_tokens.weight.dtype,
    )


def test_zero_init_cut3r_spatialstack_path_matches_disabled_logits():
    _, LlavaQwenForCausalLM = _import_llava_qwen()
    torch.manual_seed(11)
    disabled = LlavaQwenForCausalLM(_tiny_qwen_config(False))
    torch.manual_seed(11)
    enabled = LlavaQwenForCausalLM(_tiny_qwen_config(True))
    missing, unexpected = enabled.load_state_dict(disabled.state_dict(), strict=False)
    assert unexpected == []
    assert missing and all("cut3r_spatialstack_merger" in name for name in missing)

    input_ids = torch.tensor([[3, 4, 5, 6, 7, 8]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    residuals = _make_zero_init_residuals(enabled, input_ids)
    assert torch.all(residuals[0] == 0)

    disabled.eval()
    enabled.eval()
    with torch.no_grad():
        disabled_logits = disabled(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).logits
        enabled_logits = enabled(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            spatialstack_residuals_by_layer=residuals,
        ).logits
    assert torch.allclose(enabled_logits, disabled_logits, atol=1e-6, rtol=1e-6)


def test_nonzero_projection_injects_on_prefill_and_cached_decode_skips():
    _, LlavaQwenForCausalLM = _import_llava_qwen()
    torch.manual_seed(13)
    model = LlavaQwenForCausalLM(_tiny_qwen_config(True))
    branch = model.model.cut3r_spatialstack_merger.branches["6"]
    with torch.no_grad():
        branch.proj_out.weight.fill_(0.05)
        branch.proj_out.bias.fill_(0.01)

    input_ids = torch.tensor([[3, 4, 5, 6, 7, 8]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    residuals = _make_zero_init_residuals(model, input_ids)
    assert residuals[0].abs().sum() > 0

    model.eval()
    with torch.no_grad():
        no_injection = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            spatialstack_residuals_by_layer={},
        ).logits
        injected = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            spatialstack_residuals_by_layer=residuals,
        ).logits
        assert not torch.allclose(injected, no_injection)
        assert model.model._last_cut3r_spatialstack_injection_stats[0]["layer_idx"] == 0

        prefill = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
            spatialstack_residuals_by_layer=residuals,
        )
        decode_ids = torch.tensor([[9]], dtype=torch.long)
        decode_attention = torch.ones(1, input_ids.shape[1] + 1, dtype=torch.long)
        bad_prefill_shape_residual = {0: torch.zeros(1, input_ids.shape[1], model.config.hidden_size)}
        model(
            input_ids=decode_ids,
            attention_mask=decode_attention,
            past_key_values=prefill.past_key_values,
            use_cache=True,
            return_dict=True,
            spatialstack_residuals_by_layer=bad_prefill_shape_residual,
        )
    assert model.model._last_cut3r_spatialstack_injection_stats == []


def test_zero_init_cross_attn_path_matches_disabled_logits():
    _, LlavaQwenForCausalLM = _import_llava_qwen()
    torch.manual_seed(31)
    disabled = LlavaQwenForCausalLM(_tiny_qwen_config(False))
    torch.manual_seed(31)
    enabled = LlavaQwenForCausalLM(_tiny_qwen_config(True, fusion_type="cross_attn"))
    missing, unexpected = enabled.load_state_dict(disabled.state_dict(), strict=False)
    assert unexpected == []
    assert missing and all("cut3r_spatialstack_merger" in name for name in missing)

    input_ids = torch.tensor([[3, 4, 5, 6, 7, 8]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    payload = _make_cross_attn_payload(enabled, input_ids)

    disabled.eval()
    enabled.eval()
    with torch.no_grad():
        disabled_logits = disabled(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).logits
        enabled_logits = enabled(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            spatialstack_cross_attn_inputs_by_layer=payload,
        ).logits
    assert torch.allclose(enabled_logits, disabled_logits, atol=1e-6, rtol=1e-6)


def test_cross_attn_cached_decode_skips_payload():
    _, LlavaQwenForCausalLM = _import_llava_qwen()
    torch.manual_seed(37)
    model = LlavaQwenForCausalLM(_tiny_qwen_config(True, fusion_type="cross_attn"))

    input_ids = torch.tensor([[3, 4, 5, 6, 7, 8]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    payload = _make_cross_attn_payload(model, input_ids)

    model.eval()
    with torch.no_grad():
        prefill = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
            spatialstack_cross_attn_inputs_by_layer=payload,
        )
        decode_ids = torch.tensor([[9]], dtype=torch.long)
        decode_attention = torch.ones(1, input_ids.shape[1] + 1, dtype=torch.long)
        bad_decode_payload = {
            0: {
                "cut3r_layer": 6,
                "same_frame_only": True,
                "frames": [
                    {
                        "batch_idx": 0,
                        "frame_id": 0,
                        "visual_indices": torch.tensor([4], dtype=torch.long),
                        "geometry_tokens": torch.randn(4, 4),
                    }
                ],
            }
        }
        model(
            input_ids=decode_ids,
            attention_mask=decode_attention,
            past_key_values=prefill.past_key_values,
            use_cache=True,
            return_dict=True,
            spatialstack_cross_attn_inputs_by_layer=bad_decode_payload,
        )
    assert model.model._last_cut3r_spatialstack_injection_stats == []
    assert model.model._cut3r_spatialstack_cached_decode_skip_count >= 1
