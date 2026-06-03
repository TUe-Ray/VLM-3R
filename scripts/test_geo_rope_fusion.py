import torch

try:
    from llava.model.llava_arch import pool_point_map_to_tokens
    from llava.model.multimodal_fusion_block.builder import CrossAttentionFusion, GeoRoPEFusionCrossAttention, GeoRoPEFusionRotary
except ImportError:
    import math
    import pathlib
    import sys
    import torch.nn.functional as F

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "llava" / "model"))
    from multimodal_fusion_block.builder import CrossAttentionFusion, GeoRoPEFusionCrossAttention, GeoRoPEFusionRotary

    def pool_point_map_to_tokens(point_maps, target_num_tokens):
        if not isinstance(point_maps, torch.Tensor):
            raise TypeError(f"point_maps must be a torch.Tensor, got {type(point_maps)}")
        side = int(math.isqrt(int(target_num_tokens)))
        if side * side != int(target_num_tokens):
            raise ValueError(f"target_num_tokens must be square, got {target_num_tokens}")
        if point_maps.shape[-1] == 3:
            pm = point_maps.permute(0, 3, 1, 2)
        elif point_maps.shape[1] == 3:
            pm = point_maps
        else:
            raise ValueError(f"Unexpected point_maps shape: {tuple(point_maps.shape)}")
        pm = F.interpolate(pm.float(), size=(side, side), mode="bilinear", align_corners=False)
        return pm.permute(0, 2, 3, 1).reshape(pm.shape[0], side * side, 3)


def copy_baseline_weights(baseline, geo_rope_fusion_block):
    geo_rope_fusion_block.clip_norm.load_state_dict(baseline.clip_norm.state_dict())
    geo_rope_fusion_block.spatial_encoder_norm.load_state_dict(baseline.spatial_encoder_norm.state_dict())
    geo_rope_fusion_block.clip_query_proj.load_state_dict(baseline.clip_query_proj.state_dict())
    geo_rope_fusion_block.spatial_encoder_key_proj.load_state_dict(baseline.spatial_encoder_key_proj.state_dict())
    geo_rope_fusion_block.spatial_encoder_value_proj.load_state_dict(baseline.spatial_encoder_value_proj.state_dict())
    geo_rope_fusion_block.out_proj.load_state_dict(baseline.out_proj.state_dict())
    geo_rope_fusion_block.out_norm.load_state_dict(baseline.out_norm.state_dict())

    in_weight = baseline.cross_attention.in_proj_weight
    in_bias = baseline.cross_attention.in_proj_bias
    d_attn = geo_rope_fusion_block.d_attn
    geo_rope_fusion_block.attn_query_proj.weight.data.copy_(in_weight[:d_attn])
    geo_rope_fusion_block.attn_key_proj.weight.data.copy_(in_weight[d_attn:2 * d_attn])
    geo_rope_fusion_block.attn_value_proj.weight.data.copy_(in_weight[2 * d_attn:])
    geo_rope_fusion_block.attn_query_proj.bias.data.copy_(in_bias[:d_attn])
    geo_rope_fusion_block.attn_key_proj.bias.data.copy_(in_bias[d_attn:2 * d_attn])
    geo_rope_fusion_block.attn_value_proj.bias.data.copy_(in_bias[2 * d_attn:])
    geo_rope_fusion_block.attn_out_proj.load_state_dict(baseline.cross_attention.out_proj.state_dict())


def check_mode(mode, group_split):
    torch.manual_seed(7)
    baseline = CrossAttentionFusion(
        d_clip=54,
        d_spatial_encoder=40,
        d_attn=54,
        num_heads=3,
    )
    block = GeoRoPEFusionCrossAttention(
        d_clip=54,
        d_spatial_encoder=40,
        d_attn=54,
        num_heads=3,
        geo_rope_fusion_mode=mode,
        group_split=group_split,
        log_stats=True,
        dropout_rate=0.0,
    )
    copy_baseline_weights(baseline, block)
    baseline.eval()
    block.eval()

    clip_features = torch.randn(2, 16, 54)
    spatial_features = torch.randn(2, 25, 40)
    pos_clip = torch.randn(2, 16, 3)
    pos_spatial = torch.randn(2, 25, 3)

    assert block.geo_rope_fusion_gate_q.item() == 0.0
    assert block.geo_rope_fusion_gate_k.item() == 0.0

    out_a, attn = block(clip_features, spatial_features, pos_clip, pos_spatial)
    out_b, _ = block(clip_features, spatial_features, pos_clip + 10.0, pos_spatial - 10.0)
    out_base, _ = baseline(clip_features, spatial_features)

    assert out_a.shape == clip_features.shape
    assert attn.shape == (2, 16, 25)
    assert torch.allclose(out_a, out_b, atol=0.0, rtol=0.0)
    assert torch.allclose(out_a, out_base, atol=1e-5, rtol=1e-5)
    assert block.last_geo_rope_fusion_stats["q_geo_rope_fusion_minus_q"]["std"] > 0.0
    assert block.last_geo_rope_fusion_stats["mean_abs_rope_delta_q"] > 0.0
    assert block.last_geo_rope_fusion_stats["group_split"] == group_split


def check_forced_variant():
    torch.manual_seed(11)
    block = GeoRoPEFusionCrossAttention(
        d_clip=54,
        d_spatial_encoder=40,
        d_attn=54,
        num_heads=3,
        geo_rope_fusion_mode="spherical",
        group_split="2,1,2",
        gate_type="forced",
        log_stats=True,
        dropout_rate=0.0,
    )
    assert not any("gate" in name for name, _ in block.named_parameters())

    clip_features = torch.randn(2, 16, 54)
    spatial_features = torch.randn(2, 25, 40)
    pos_clip = torch.randn(2, 16, 3)
    pos_spatial = torch.randn(2, 25, 3)

    out, attn = block(clip_features, spatial_features, pos_clip, pos_spatial)
    stats = block.last_geo_rope_fusion_stats
    assert out.shape == clip_features.shape
    assert attn.shape == (2, 16, 25)
    assert stats["gate_type"] == "forced"
    assert stats["mean_abs_rope_delta_q"] > 0.0
    assert stats["mean_abs_rope_delta_k"] > 0.0
    assert abs(stats["effective_rope_delta_q"] - stats["mean_abs_rope_delta_q"]) < 1e-6
    assert abs(stats["effective_rope_delta_k"] - stats["mean_abs_rope_delta_k"]) < 1e-6


def check_per_head_variant():
    torch.manual_seed(13)
    block = GeoRoPEFusionCrossAttention(
        d_clip=54,
        d_spatial_encoder=40,
        d_attn=54,
        num_heads=3,
        geo_rope_fusion_mode="spherical",
        group_split="2,1,2",
        gate_type="per_head",
        head_gate_init=0.99,
        log_stats=True,
        dropout_rate=0.0,
    )
    assert not hasattr(block, "geo_rope_fusion_gate_q")
    assert not hasattr(block, "geo_rope_fusion_gate_k")
    assert block.geo_rope_fusion_head_gate_logit.shape == (3,)
    gate = torch.sigmoid(block.geo_rope_fusion_head_gate_logit.detach())
    assert torch.allclose(gate, torch.full((3,), 0.99), atol=1e-5, rtol=0.0)

    clip_features = torch.randn(2, 16, 54, requires_grad=True)
    spatial_features = torch.randn(2, 25, 40, requires_grad=True)
    pos_clip = torch.randn(2, 16, 3)
    pos_spatial = torch.randn(2, 25, 3)

    out, attn = block(clip_features, spatial_features, pos_clip, pos_spatial)
    loss = out.float().square().mean() + attn.float().square().mean()
    loss.backward()

    stats = block.last_geo_rope_fusion_stats
    assert out.shape == clip_features.shape
    assert attn.shape == (2, 16, 25)
    assert stats["gate_type"] == "per_head"
    assert stats["head_gate_shape"] == [1, 3, 1, 1]
    assert abs(stats["gate_mean"] - 0.99) < 1e-5
    assert stats["gate_min"] > 0.98
    assert len(stats["gate_per_head_values"]) == 3
    assert stats["effective_rope_delta_q"] > 0.0
    assert stats["effective_rope_delta_k"] > 0.0
    assert block.geo_rope_fusion_head_gate_logit.grad is not None
    assert torch.isfinite(block.geo_rope_fusion_head_gate_logit.grad).all()
    assert block._last_head_gate_grad_norm is not None
    assert torch.isfinite(torch.tensor(block._last_head_gate_grad_norm))


def expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def main():
    point_maps = torch.randn(2, 8, 8, 3)
    token_pos = pool_point_map_to_tokens(point_maps, 16)
    assert token_pos.shape == (2, 16, 3)

    point_maps_chw = point_maps.permute(0, 3, 1, 2)
    token_pos_chw = pool_point_map_to_tokens(point_maps_chw, 16)
    assert torch.allclose(token_pos, token_pos_chw)

    valid_cases = (
        ("depth", "1"),
        ("xyz", "1,1,1"),
        ("xyz", "2,1,2"),
        ("spherical", "1,1,1"),
        ("spherical", "2,1,2"),
        ("spherical", "3,1,3"),
    )
    for mode, group_split in valid_cases:
        check_mode(mode, group_split)
    check_forced_variant()
    check_per_head_variant()

    expect_value_error(lambda: GeoRoPEFusionRotary(head_dim=18, mode="depth", group_split="2,1,2"))
    expect_value_error(lambda: GeoRoPEFusionRotary(head_dim=18, mode="xyz", group_split="1"))
    expect_value_error(lambda: GeoRoPEFusionRotary(head_dim=18, mode="spherical", group_split="1,1"))
    expect_value_error(lambda: GeoRoPEFusionRotary(head_dim=18, mode="xyz", group_split="1,0,1"))
    expect_value_error(lambda: GeoRoPEFusionRotary(head_dim=18, mode="spherical", group_split="-1,1,1"))
    print("GeoRoPE Fusion sanity checks passed.")


if __name__ == "__main__":
    main()
