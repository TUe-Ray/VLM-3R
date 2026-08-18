import pathlib
import sys
from types import SimpleNamespace

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBING_DIR = REPO_ROOT / "scripts" / "probing"
if str(PROBING_DIR) not in sys.path:
    sys.path.insert(0, str(PROBING_DIR))

from aggregate_pre_sft_fusion_multiseed import summarize  # noqa: E402
from scripts.diagnose_layerwise_spatial_hidden_scan import seeded_fusion_initialization  # noqa: E402
from llava.model.multimodal_fusion_block.builder import build_multimodal_fusion_block  # noqa: E402


def _row(variant, feature, fusion_seed, mae):
    return {
        "experiment_variant": variant,
        "feature_level": feature,
        "fusion_init_seed": fusion_seed,
        "probe_seed": 0,
        "mae": mae,
        "absrel": mae / 10,
        "delta125": 1 - mae / 10,
        "metrics_path": f"/{variant}/{feature}/{fusion_seed}.json",
    }


def test_multiseed_summary_reports_scores_mean_and_sample_std():
    rows = [
        _row("ss_identity", "layer_0", 0, 2.0),
        _row("ss_identity", "layer_0", 1, 4.0),
        _row("vlm3r_native", "layer_0", 0, 3.0),
        _row("vlm3r_native", "layer_0", 1, 5.0),
        _row("vlm3r_native", "fusion_output", 0, 1.0),
        _row("vlm3r_native", "fusion_output", 1, 3.0),
    ]
    summaries, issues = summarize(
        rows,
        variants=["ss_identity", "vlm3r_native"],
        seeds=[0, 1],
        probe_seed=0,
    )
    assert issues == []
    ss = next(row for row in summaries if row["experiment_variant"] == "ss_identity")
    assert ss["seed_scores"] == [
        {"fusion_init_seed": 0, "mae": 2.0, "absrel": 0.2, "delta125": 0.8},
        {"fusion_init_seed": 1, "mae": 4.0, "absrel": 0.4, "delta125": 0.6},
    ]
    assert ss["mae_mean"] == 3.0
    assert round(ss["mae_std"], 6) == round(2**0.5, 6)
    diagnostic = next(row for row in summaries if row["feature_level"] == "fusion_output")
    assert diagnostic["diagnostic_only"] is True


def test_native_vlm3r_fusion_seed_changes_only_fresh_attention_weights():
    config = SimpleNamespace(mm_hidden_size=18, hidden_size=32, spatial_feature_dim=4, fusion_block="cross_attention")
    with seeded_fusion_initialization(0):
        seed_zero = build_multimodal_fusion_block(config)
    with seeded_fusion_initialization(1):
        seed_one = build_multimodal_fusion_block(config)
    assert not torch.equal(seed_zero.clip_query_proj.weight, seed_one.clip_query_proj.weight)
