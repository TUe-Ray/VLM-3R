import unittest

import torch

from llava.model.cut3r_token_only import Cut3RTokenOnlyProjector, extract_cut3r_patch_tokens


class Cut3RTokenOnlyTest(unittest.TestCase):
    def test_projector_shape_finite_and_update(self):
        projector = Cut3RTokenOnlyProjector(768, 32)
        tokens = torch.randn(2, 729, 768)
        before = projector.proj_out.weight.detach().clone()
        output = projector(tokens)
        self.assertEqual(tuple(output.shape), (2, 729, 32))
        self.assertTrue(torch.isfinite(output).all())
        optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-3)
        output.square().mean().backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in projector.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, projector.proj_out.weight.detach()))

    def test_sidecar_rejections(self):
        valid = {"patch_tokens": torch.zeros(2, 729, 768)}
        self.assertEqual(tuple(extract_cut3r_patch_tokens(valid, 2).shape), (2, 729, 768))
        with self.assertRaises(KeyError):
            extract_cut3r_patch_tokens({}, 2)
        with self.assertRaises(ValueError):
            extract_cut3r_patch_tokens(valid, 2, sidecar_key="camera_tokens")
        with self.assertRaises(RuntimeError):
            extract_cut3r_patch_tokens({"patch_tokens": torch.zeros(1, 729, 768)}, 2)
        with self.assertRaises(RuntimeError):
            extract_cut3r_patch_tokens({"patch_tokens": torch.zeros(2, 728, 768)}, 2)
        with self.assertRaises(RuntimeError):
            extract_cut3r_patch_tokens({"patch_tokens": torch.zeros(2, 729, 767)}, 2)
        bad = torch.zeros(2, 729, 768)
        bad[0, 0, 0] = float("nan")
        with self.assertRaises(RuntimeError):
            extract_cut3r_patch_tokens({"patch_tokens": bad}, 2)


if __name__ == "__main__":
    unittest.main()
