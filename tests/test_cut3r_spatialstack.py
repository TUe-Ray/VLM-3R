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
Cut3RSpatialStackCrossAttentionBlock = cut3r_mod.Cut3RSpatialStackCrossAttentionBlock


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
