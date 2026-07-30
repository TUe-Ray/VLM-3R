import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from llava.model.cut3r_token_only import (
    Cut3RTokenOnlyProjector,
    assert_cut3r_token_projector_checkpoint_values,
)
from llava.train.llava_trainer import LLaVATrainer
from llava.train.train import requires_cut3r_token_only_frame_indices
from llava.train.train import write_cut3r_token_only_initial_weight_samples


class _TelemetryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(visual_token_source="cut3r_only")
        self.cut3r_token_projector = nn.Linear(1, 1, bias=False)
        self.language_model = nn.Module()
        self.language_model.register_parameter("lora_A", nn.Parameter(torch.ones(1, 1)))
        self._cut3r_token_only_last_metrics = {"source": "cut3r_only", "siglip_forward_bypassed": True}

    def get_model(self):
        return self



class _FakeDeepSpeedEngine:
    """Minimal engine modelled on Accelerate's backward -> engine.step lifecycle."""

    def __init__(self, model):
        self.model = model
        self._boundary = True
        self._step_applied = False

    def is_gradient_accumulation_boundary(self):
        return self._boundary

    def step(self, *args, **kwargs):
        self._step_applied = False
        if not self._boundary:
            return None
        with torch.no_grad():
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-0.1)
                    parameter.grad = None
        self._step_applied = True
        return None

class Cut3RTokenOnlyTelemetryTest(unittest.TestCase):
    def _trainer_harness(self, directory, *, smoke=True, rank0=True):
        trainer = object.__new__(LLaVATrainer)
        trainer.model = _TelemetryModel()
        trainer.args = SimpleNamespace(
            output_dir=directory,
            cut3r_token_smoke_telemetry=smoke,
            cut3r_token_smoke_full_scan_steps=2,
        )
        trainer.state = SimpleNamespace(global_step=0)
        trainer.accelerator = SimpleNamespace(optimizer_step_was_skipped=False)
        trainer.is_world_process_zero = lambda: rank0
        trainer._cut3r_token_only_optimizer_hook_installed = False
        trainer._cut3r_token_only_pending_scans = {}
        trainer._cut3r_token_only_optimizer_evidence = {}
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
        return trainer

    @staticmethod
    def _backward(model):
        loss = model.cut3r_token_projector.weight.square().sum() + model.language_model.lora_A.square().sum()
        loss.backward()

    def test_group_stats_is_bound_through_trainer_instance(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self._trainer_harness(directory)
            stats = trainer._cut3r_group_stats(list(trainer.model.named_parameters()))
        self.assertGreater(stats["parameter_norm"], 0.0)
        self.assertTrue(stats["parameter_finite"])

    def test_post_optimizer_evidence_runs_only_for_rank0_smoke_steps_one_and_two(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self._trainer_harness(directory, smoke=True, rank0=True)
            trainer._install_cut3r_token_only_optimizer_hook(trainer.optimizer)
            self._backward(trainer.model)
            self.assertEqual(trainer._cut3r_token_only_optimizer_evidence, {})
            trainer.optimizer.step()
            first = trainer._cut3r_token_only_optimizer_evidence[1]
            self.assertGreater(first["projector_grad_norm"], 0.0)
            self.assertGreater(first["lora_grad_norm"], 0.0)
            self.assertGreater(first["projector_update_delta_norm"], 0.0)
            self.assertGreater(first["lora_update_delta_norm"], 0.0)
            self.assertTrue(first["all_finite"])
            trainer.optimizer.zero_grad()
            trainer.state.global_step = 1
            self._backward(trainer.model)
            trainer.optimizer.step()
            self.assertIn(2, trainer._cut3r_token_only_optimizer_evidence)
            trainer.optimizer.zero_grad()
            trainer.state.global_step = 2
            self._backward(trainer.model)
            trainer.optimizer.step()
            self.assertEqual(sorted(trainer._cut3r_token_only_optimizer_evidence), [1, 2])

    def test_deepspeed_engine_hook_captures_before_engine_zero_grad(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self._trainer_harness(directory, smoke=True, rank0=True)
            trainer.deepspeed = _FakeDeepSpeedEngine(trainer.model)
            trainer._install_cut3r_token_only_optimizer_hook(trainer.optimizer)
            self._backward(trainer.model)
            trainer.deepspeed.step()
            evidence = trainer._cut3r_token_only_optimizer_evidence[1]
            self.assertTrue(evidence["optimizer_was_run"])
            self.assertGreater(evidence["projector_grad_norm"], 0.0)
            self.assertGreater(evidence["lora_grad_norm"], 0.0)
            self.assertGreater(evidence["projector_update_delta_norm"], 0.0)
            self.assertGreater(evidence["lora_update_delta_norm"], 0.0)
            self.assertIsNone(trainer.model.cut3r_token_projector.weight.grad)
            self.assertIsNone(trainer.model.language_model.lora_A.grad)

    def test_non_smoke_or_nonzero_rank_does_not_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            for smoke, rank0 in ((False, True), (True, False)):
                trainer = self._trainer_harness(directory, smoke=smoke, rank0=rank0)
                trainer._install_cut3r_token_only_optimizer_hook(trainer.optimizer)
                self._backward(trainer.model)
                trainer.optimizer.step()
                self.assertEqual(trainer._cut3r_token_only_optimizer_evidence, {})

    def test_frame_index_policy_is_cut3r_only(self):
        self.assertTrue(requires_cut3r_token_only_frame_indices(SimpleNamespace(visual_token_source="cut3r_only")))
        self.assertFalse(requires_cut3r_token_only_frame_indices(SimpleNamespace(visual_token_source="siglip_only")))
        self.assertFalse(requires_cut3r_token_only_frame_indices(SimpleNamespace(visual_token_source=None)))

    def test_projector_checkpoint_value_verification(self):
        projector = Cut3RTokenOnlyProjector(768, 8)
        raw_state = {
            f"base_model.model.cut3r_token_projector.{key}": value.detach().clone()
            for key, value in projector.state_dict().items()
        }
        keys = assert_cut3r_token_projector_checkpoint_values(projector, raw_state)
        self.assertEqual(keys, sorted(projector.state_dict()))
        missing = dict(raw_state)
        missing.pop(next(iter(missing)))
        with self.assertRaises(RuntimeError):
            assert_cut3r_token_projector_checkpoint_values(projector, missing)
        unexpected = dict(raw_state)
        unexpected["model.cut3r_token_projector.extra"] = torch.zeros(1)
        with self.assertRaises(RuntimeError):
            assert_cut3r_token_projector_checkpoint_values(projector, unexpected)
        different = dict(raw_state)
        key = next(iter(different))
        different[key] = different[key] + 1.0
        with self.assertRaises(AssertionError):
            assert_cut3r_token_projector_checkpoint_values(projector, different)

    def test_checkpoint_delta_uses_only_bounded_saved_samples(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "cut3r_token_only_checkpoint_evidence.py"
        spec = importlib.util.spec_from_file_location("cut3r_checkpoint_delta", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = _TelemetryModel()
            args = SimpleNamespace(output_dir=str(root), cut3r_token_checkpoint_delta_validation=True)
            old_rank = os.environ.get("RANK")
            os.environ["RANK"] = "0"
            try:
                sample_path = write_cut3r_token_only_initial_weight_samples(model, args, width=1)
            finally:
                if old_rank is None:
                    os.environ.pop("RANK", None)
                else:
                    os.environ["RANK"] = old_rank
            checkpoint = root / "checkpoint-2"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text('{"visual_token_source": "cut3r_only"}')
            torch.save({"base_model.model.cut3r_token_projector.weight": model.cut3r_token_projector.weight.detach() + 1.0}, checkpoint / "non_lora_trainables.bin")
            torch.save({"base_model.model.language_model.lora_A": model.language_model.lora_A.detach() + 1.0}, checkpoint / "adapter_model.bin")
            self.assertTrue(sample_path.is_file())
            evidence = module.checkpoint_delta_evidence(root, checkpoint)
            self.assertTrue(evidence["complete"])
            self.assertTrue(evidence["groups"]["projector"]["nonzero"])
            self.assertTrue(evidence["groups"]["lora"]["nonzero"])

    def test_numeric_checkpoint_selection(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "validate_cut3r_token_only_smoke_gate.py"
        spec = importlib.util.spec_from_file_location("cut3r_smoke_gate", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("checkpoint-9", "checkpoint-60", "checkpoint-100", "checkpoint-bad"):
                (root / name).mkdir()
            self.assertEqual(module._latest_checkpoint(root).name, "checkpoint-100")


if __name__ == "__main__":
    unittest.main()
