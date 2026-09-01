"""CPU-only unit tests for zero-cost proxy arithmetic and ranking orientation."""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/probing/evaluate_pre_sft_zero_cost_proxies.py"
spec = importlib.util.spec_from_file_location("zero_cost_proxies", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_proxy_scores_use_standard_gradnorm_snip_and_diagonal_fisher_definitions():
    parameter = torch.nn.Parameter(torch.tensor([2.0, -3.0]))
    gradient = torch.tensor([4.0, -5.0])
    scores = module.proxy_scores([parameter], [gradient])
    assert math.isclose(scores["gradnorm"], (4.0**2 + (-5.0) ** 2) ** 0.5, rel_tol=1e-6)
    assert math.isclose(scores["snip"], 23.0, rel_tol=1e-6)
    assert math.isclose(scores["fisher"], 41.0, rel_tol=1e-6)
    assert scores["parameters_with_gradient"] == 2


def test_cost_proxy_is_oriented_lower_is_better_for_vsi_spearman():
    rows = [
        {"candidate": "a", "vsi_avg": 3.0, "total_params": 1.0},
        {"candidate": "b", "vsi_avg": 2.0, "total_params": 2.0},
        {"candidate": "c", "vsi_avg": 1.0, "total_params": 3.0},
    ]
    correlations = module.correlation_rows(rows)
    total = next(item for item in correlations if item["proxy"] == "total_params")
    assert total["expected_orientation"] == "lower_is_better"
    assert math.isclose(total["spearman_vs_vsi_avg"], 1.0, rel_tol=1e-12)
