import math
import unittest

import torch

from llava.model.c1_structured_isometry import SCHEME_VERSION
from llava.model.c2_cca_qk import PairedQKObserver, c1_stat_contract, compose_c2_qk
from llava.model.cut3r_spatialstack import Cut3RSpatialStackCrossAttentionBlock


def c1_reference(target_std=1.0):
    return {
        "architecture": "spatialstack_cross_attn_v1",
        "canonicalization_scheme_version": SCHEME_VERSION,
        "qk_basis_mode": "shared_canonical",
        "r0": 0.1,
        "qk_logit_calibration": {
            "statistic": "population_std_over_all_same_frame_attention_logits",
            "variance": "population_E_x2_minus_E_x_squared",
            "target_std": target_std,
            "qk_scale_application": "multiply_both_q_and_k",
        },
        "residual_calibration": {
            "sample_statistic": "rms_delta_over_rms_pre_injection_hidden_at_visual_tokens",
            "sample_aggregation": "median_over_samples",
            "target": "base_artifact_r0",
            "target_scope": "per_injection_site",
        },
    }


class C2CcaQkTests(unittest.TestCase):
    def test_per_head_cca_composes_qk_only(self):
        torch.manual_seed(4)
        block = Cut3RSpatialStackCrossAttentionBlock(6, 8, 2, zero_init=False)
        visual = torch.randn(3, 9, 8)
        geometry = torch.randn(3, 9, 6)
        with PairedQKObserver(block, collect_cca=True, collect_logits=True, chunk_size=4) as observer:
            block(visual, geometry)
        state = observer.fit_cca(1e-3)
        self.assertEqual(tuple(state["a"].shape), (2, 4, 4))
        self.assertEqual(tuple(state["mu_q"].shape), (2, 4))
        self.assertEqual(observer.pair_count, 27)
        self.assertTrue(torch.isfinite(state["canonical_correlations"]).all())
        q_before = block.q_proj.weight.detach().clone()
        k_before = block.k_proj.weight.detach().clone()
        v_before = block.v_proj.state_dict()
        o_before = block.out_proj.state_dict()
        compose_c2_qk(block, state)
        self.assertFalse(torch.equal(q_before, block.q_proj.weight))
        self.assertFalse(torch.equal(k_before, block.k_proj.weight))
        self.assertTrue(all(torch.equal(value, block.v_proj.state_dict()[key]) for key, value in v_before.items()))
        self.assertTrue(all(torch.equal(value, block.out_proj.state_dict()[key]) for key, value in o_before.items()))

    def test_generic_logit_target_uses_square_root(self):
        target, raw = 2.5, 0.4
        scale = math.sqrt(target / raw)
        self.assertAlmostEqual(scale * scale * raw, target)
        qk, residual = c1_stat_contract(c1_reference(target))
        self.assertEqual(qk["target_std"], target)
        self.assertEqual(residual["sample_aggregation"], "median_over_samples")

    def test_legacy_c1_contract_is_rejected(self):
        legacy = c1_reference()
        legacy.pop("qk_logit_calibration")
        with self.assertRaises(ValueError):
            c1_stat_contract(legacy)


if __name__ == "__main__":
    unittest.main()
