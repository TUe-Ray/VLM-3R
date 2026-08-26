"""CPU-only checks for the post-SFT EoMT consumer-grid cache schema."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from scripts.probing.eomt_consumer_grid_cache import consumer_masks, tensor_summary


class EoMTConsumerGridCacheTest(unittest.TestCase):
    def test_sigmoid_then_bilinear_resize_matches_consumer_contract(self) -> None:
        logits = torch.tensor([[[[-2.0, 0.0], [1.0, 3.0]]]], dtype=torch.float32)
        expected = F.interpolate(
            torch.sigmoid(logits), size=(3, 3), mode="bilinear", align_corners=False
        )
        actual = consumer_masks(logits, (3, 3))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(actual, expected))

    def test_zero_variance_is_recorded_not_rejected(self) -> None:
        zeros = torch.zeros((32, 200, 14, 14), dtype=torch.float32)
        summary = tensor_summary(zeros, name="zeros", shape=(32, 200, 14, 14))
        self.assertTrue(summary["finite"])
        self.assertEqual(summary["variance"], 0.0)
        self.assertFalse(summary["nontrivial_variance"])


if __name__ == "__main__":
    unittest.main()
