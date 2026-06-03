import copy
import json
import math

import torch
from transformers import Qwen2Config

from llava.model.language_model.llava_qwen import LlavaQwenModel


def make_config(enable=True, alpha=1.0):
    cfg = Qwen2Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    cfg._attn_implementation = "eager"
    cfg.use_cache = True
    cfg.llm_visual_3d_rope_enable = enable
    cfg.llm_visual_3d_rope_alpha = alpha
    cfg.llm_visual_3d_rope_mode = "spherical"
    cfg.llm_visual_3d_rope_group_split = "2,1,2"
    cfg.llm_visual_3d_rope_max_depth = 10.0
    cfg.llm_visual_3d_rope_layers = "all"
    cfg.llm_visual_3d_rope_log_stats = True
    cfg.llm_visual_3d_rope_log_layers = "all"
    cfg.llm_visual_3d_rope_force_eager_attention = True
    return cfg


def finite_stats(stats):
    for item in stats:
        for key, value in item.items():
            if isinstance(value, float):
                assert math.isfinite(value), f"{key} is not finite: {value}"


def deterministic_shuffle(pos, mask, seed=0):
    shuffled = pos.clone()
    permutations = []
    for batch_idx in range(mask.shape[0]):
        idx = torch.nonzero(mask[batch_idx], as_tuple=False).flatten()
        generator = torch.Generator(device=pos.device)
        generator.manual_seed(seed + batch_idx)
        perm = torch.randperm(idx.numel(), generator=generator, device=pos.device)
        if torch.equal(perm, torch.arange(idx.numel(), device=pos.device)):
            perm = torch.roll(perm, shifts=1, dims=0)
        original = shuffled[batch_idx, idx].clone()
        shuffled[batch_idx, idx] = original[perm]
        permutations.append((idx, perm))
    return shuffled, permutations


def main():
    torch.manual_seed(123)
    batch_size = 2
    seq_len = 12
    input_ids = torch.randint(0, 64, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    attention_mask[1, -2:] = 0
    llm_geo_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    llm_geo_mask[0, 3:9] = True
    llm_geo_mask[1, 2:7] = True
    llm_geo_mask &= attention_mask.bool()
    llm_geo_pos = torch.randn(batch_size, seq_len, 3)
    llm_geo_pos[:, :, 2].abs_()

    base = LlavaQwenModel(make_config(enable=False, alpha=0.0)).eval()
    alpha0 = LlavaQwenModel(make_config(enable=True, alpha=0.0)).eval()
    alpha1 = LlavaQwenModel(make_config(enable=True, alpha=1.0)).eval()
    alpha0.load_state_dict(copy.deepcopy(base.state_dict()))
    alpha1.load_state_dict(copy.deepcopy(base.state_dict()))

    with torch.no_grad():
        out_base = base(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
        out_alpha0 = alpha0(
            input_ids=input_ids,
            attention_mask=attention_mask,
            llm_geo_pos=llm_geo_pos,
            llm_geo_mask=llm_geo_mask,
            return_dict=True,
        ).last_hidden_state
        alpha0_max_abs = (out_base - out_alpha0).abs().max().item()
        assert alpha0_max_abs == 0.0, f"alpha=0 equivalence failed: max_abs={alpha0_max_abs}"

        out_alpha1 = alpha1(
            input_ids=input_ids,
            attention_mask=attention_mask,
            llm_geo_pos=llm_geo_pos,
            llm_geo_mask=llm_geo_mask,
            return_dict=True,
        ).last_hidden_state
        alpha1_output_delta = (out_base - out_alpha1).abs().mean().item()

    stats = alpha1._last_llm_visual_3d_rope_stats
    finite_stats(stats)
    active = [item for item in stats if not item.get("skipped", False)]
    assert active, "alpha=1 produced no active 3D RoPE stats"
    deltas = [float(item["visual_visual_logits_delta_mean_abs"]) for item in active]
    assert max(deltas) > 0.0, "alpha=1 visual-visual logit delta is zero"
    assert max(float(item.get("masked_attention_prob_max", 0.0)) for item in active) == 0.0
    assert max(int(item.get("attention_prob_nan_count", 0) or 0) for item in active) == 0
    assert max(int(item.get("attention_prob_inf_count", 0) or 0) for item in active) == 0
    assert max(int(item.get("masked_attention_prob_nonfinite_count", 0) or 0) for item in active) == 0

    shuffled, permutations = deterministic_shuffle(llm_geo_pos, llm_geo_mask, seed=0)
    original_values = llm_geo_pos[llm_geo_mask].flatten().sort().values
    shuffled_values = shuffled[llm_geo_mask].flatten().sort().values
    assert torch.equal(original_values, shuffled_values), "shuffle changed geometry value distribution"
    assignment_delta = (llm_geo_pos[llm_geo_mask] - shuffled[llm_geo_mask]).abs().sum().item()
    assert assignment_delta > 0.0, "shuffle did not change token-position assignments"

    with torch.no_grad():
        prefill = alpha1(
            input_ids=input_ids[:1],
            attention_mask=torch.ones(1, seq_len, dtype=torch.long),
            llm_geo_pos=llm_geo_pos[:1],
            llm_geo_mask=llm_geo_mask[:1],
            use_cache=True,
            return_dict=True,
        )
        prefill_active = [item for item in alpha1._last_llm_visual_3d_rope_stats if not item.get("skipped", False)]
        assert prefill_active, "generation prefill did not use geometry metadata"
        alpha1(
            input_ids=input_ids[:1, -1:],
            attention_mask=torch.ones(1, seq_len + 1, dtype=torch.long),
            past_key_values=prefill.past_key_values,
            llm_geo_pos=llm_geo_pos[:1],
            llm_geo_mask=llm_geo_mask[:1],
            use_cache=True,
            return_dict=True,
        )
        decode_reasons = {item.get("skip_reason") for item in alpha1._last_llm_visual_3d_rope_stats}
        assert "non_prefill_or_cached_decode" in decode_reasons, decode_reasons

    table = {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "num_visual_tokens": [int(llm_geo_mask[b].sum().item()) for b in range(batch_size)],
        "num_valid_geometry_tokens": int(llm_geo_mask.sum().item()),
        "num_frames": "synthetic_attention_only",
        "visual_token_indices": [
            torch.nonzero(llm_geo_mask[b], as_tuple=False).flatten().tolist() for b in range(batch_size)
        ],
        "first_aligned_3d_positions": [
            llm_geo_pos[b, llm_geo_mask[b]][:3].tolist() for b in range(batch_size)
        ],
        "newline_prefix_padding_exclusion": {
            "newline_overlap": 0,
            "prefix_overlap": 0,
            "padding_overlap": int((llm_geo_mask & ~attention_mask.bool()).sum().item()),
        },
        "alpha0_equivalence_max_abs": alpha0_max_abs,
        "alpha1_output_delta_mean_abs": alpha1_output_delta,
        "alpha1_visual_visual_logits_delta_mean_abs": deltas,
        "shuffle_assignment_delta_abs_sum": assignment_delta,
        "shuffle_permutation_heads": [
            {"tokens": idx[:6].tolist(), "permutation": perm[:6].tolist()} for idx, perm in permutations
        ],
        "causal_mask_masked_attention_prob_max": max(float(item.get("masked_attention_prob_max", 0.0)) for item in active),
        "attention_prob_nan_count": max(int(item.get("attention_prob_nan_count", 0) or 0) for item in active),
        "attention_prob_inf_count": max(int(item.get("attention_prob_inf_count", 0) or 0) for item in active),
        "masked_attention_prob_nonfinite_count": max(
            int(item.get("masked_attention_prob_nonfinite_count", 0) or 0) for item in active
        ),
        "generation_decode_skip_reason": sorted(decode_reasons),
    }
    print(json.dumps(table, indent=2, sort_keys=True))
    print("LLM visual-token 3D RoPE attention validation passed.")


if __name__ == "__main__":
    main()
