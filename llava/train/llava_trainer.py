import os
import json
import torch
import torch.nn as nn
import datetime
import time

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, GradientAccumulationPlugin
from torch.utils.data import Dataset, Sampler, DataLoader

from trl.trainer import DPOTrainer
from trl.trainer.utils import DPODataCollatorWithPadding

from transformers import Trainer, TrainerCallback, TrainerState, TrainerControl
from transformers.trainer import is_sagemaker_mp_enabled, get_parameter_names, has_length, ALL_LAYERNORM_LAYERS, logger, is_accelerate_available, is_datasets_available, GradientAccumulationPlugin
from transformers.trainer_utils import seed_worker
from transformers.trainer_pt_utils import get_length_grouped_indices as get_length_grouped_indices_hf
from transformers.trainer_pt_utils import AcceleratorConfig
from typing import List, Optional
from datetime import timedelta

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches, InitProcessGroupKwargs

if is_datasets_available():
    import datasets

from llava.utils import rank0_print


class ProgressLoggerCallback(TrainerCallback):
    """Progress bar friendly for SLURM .out files with ETA (no carriage returns)."""

    def __init__(self):
        super().__init__()
        self.start_time = None
        self.start_step = 0

    @staticmethod
    def _format_time(seconds):
        """Format seconds as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Record the start time when training begins."""
        import time
        self.start_time = time.time()
        self.start_step = int(state.global_step or 0)

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if not state.is_world_process_zero or logs is None or self.start_time is None:
            return

        import time
        elapsed = time.time() - self.start_time
        step = state.global_step
        max_steps = state.max_steps or 1

        # Avoid division by zero
        if step == 0:
            return

        pct = 100.0 * step / max_steps
        bar_len = 30
        filled = int(bar_len * step / max_steps)
        bar = "█" * filled + "░" * (bar_len - filled)
        epoch = state.epoch or 0.0
        loss = logs.get("loss", logs.get("train_loss"))
        lr = logs.get("learning_rate")

        # Calculate timing from steps completed in this process. On resume,
        # state.global_step includes checkpointed steps from the previous run.
        completed_this_run = max(step - self.start_step, 1)
        avg_time_per_step = elapsed / completed_this_run
        remaining_steps = max_steps - step
        eta_seconds = avg_time_per_step * remaining_steps

        elapsed_str = self._format_time(elapsed)
        eta_str = self._format_time(eta_seconds)
        speed_str = f"{avg_time_per_step:.2f}s/it"

        loss_str = f" | loss={loss:.4f}" if isinstance(loss, float) else ""
        lr_str = f" | lr={lr:.2e}" if isinstance(lr, float) else ""
        rank_loss = logs.get("spatial_rank_loss")
        rank_acc = logs.get("spatial_rank_accuracy")
        rank_str = ""
        if isinstance(rank_loss, float):
            rank_str += f" | L_rank={rank_loss:.4f}"
        if isinstance(rank_acc, float):
            rank_str += f" | rank_acc={rank_acc:.3f}"
        bev_loss = logs.get("loss_bev")
        bev_mae = logs.get("bev_mae_meter")
        bev_valid = logs.get("valid_bev_token_ratio")
        if isinstance(bev_loss, float):
            rank_str += f" | L_bev={bev_loss:.4f}"
        if isinstance(bev_mae, float):
            rank_str += f" | bev_mae={bev_mae:.3f}m"
        if isinstance(bev_valid, float):
            rank_str += f" | bev_valid={bev_valid:.3f}"
        depth_loss = logs.get("loss_depth")
        depth_mae = logs.get("depth_mae_meter")
        depth_valid = logs.get("valid_depth_token_ratio")
        if isinstance(depth_loss, float):
            rank_str += f" | L_depth={depth_loss:.4f}"
        if isinstance(depth_mae, float):
            rank_str += f" | depth_mae={depth_mae:.3f}m"
        if isinstance(depth_valid, float):
            rank_str += f" | depth_valid={depth_valid:.3f}"
        pointmap_loss = logs.get("loss_pointmap")
        pointmap_mae = logs.get("pointmap_mean_abs_error_meter", logs.get("pointmap_mae_meter"))
        pointmap_valid = logs.get("valid_pointmap_token_ratio")
        if isinstance(pointmap_loss, float):
            rank_str += f" | L_pm={pointmap_loss:.4f}"
        if isinstance(pointmap_mae, float):
            rank_str += f" | pm_mae={pointmap_mae:.3f}m"
        if isinstance(pointmap_valid, float):
            rank_str += f" | pm_valid={pointmap_valid:.3f}"

        print(f"[{bar}] {step}/{max_steps} ({pct:.1f}%) [{elapsed_str}<{eta_str}, {speed_str}] | epoch={epoch:.3f}{loss_str}{lr_str}{rank_str}", flush=True)


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_variable_length_grouped_indices(lengths, batch_size, world_size, megabatch_mult=8, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    megabatch_size = world_size * batch_size * megabatch_mult
    megabatches = [sorted_indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: indices[i], reverse=True) for megabatch in megabatches]
    shuffled_indices = [i for megabatch in megabatches for i in megabatch]
    world_batch_size = world_size * batch_size
    batches = [shuffled_indices[i : i + world_batch_size] for i in range(0, len(lengths), world_batch_size)]
    batch_indices = torch.randperm(len(batches), generator=generator)
    batches = [batches[i] for i in batch_indices]

    return [i for batch in batches for i in batch]


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - reorder by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=generator)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=generator)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - reorder by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=None):
    indices = get_length_grouped_indices_hf(lengths, batch_size * world_size, generator=generator)

    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    batch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in batch_indices]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_modality_length_grouped_indices_auto(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices_auto_single(mm_lengths, batch_size, world_size, generator=generator)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices_auto_single(lang_lengths, batch_size, world_size, generator=generator)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    # FIXME: Hard code to avoid last batch mixed with different modalities
    # if len(additional_batch) > 0:
    #     megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        variable_length: bool = False,
        group_by_modality: bool = False,
        group_by_modality_auto: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.variable_length = variable_length
        self.group_by_modality = group_by_modality
        self.group_by_modality_auto = group_by_modality_auto

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.variable_length:
            assert not self.group_by_modality, "Variable length grouping is not supported with modality grouping."
            indices = get_variable_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            if self.group_by_modality:
                indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
            elif self.group_by_modality_auto:
                indices = get_modality_length_grouped_indices_auto(self.lengths, self.batch_size, self.world_size, generator=self.generator)
            else:
                indices = get_length_grouped_indices_auto_single(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class Cut3RTokenOnlyOptimizerTelemetryCallback(TrainerCallback):
    """Install the smoke-only scan around the prepared optimizer's real step."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.trainer._install_cut3r_token_only_optimizer_hook(kwargs.get("optimizer"))
        return control



class LLaVATrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cut3r_token_only_optimizer_hook_installed = False
        self._cut3r_token_only_pending_scans = {}
        self._cut3r_token_only_optimizer_evidence = {}
        self.add_callback(Cut3RTokenOnlyOptimizerTelemetryCallback(self))

    def _cut3r_token_only_base_model(self):
        return self.model.get_model() if hasattr(self.model, "get_model") else getattr(self.model, "model", None)

    def _cut3r_token_only_active(self):
        base_model = self._cut3r_token_only_base_model()
        return bool(
            base_model is not None
            and str(getattr(base_model.config, "visual_token_source", "siglip_only") or "siglip_only").lower()
            == "cut3r_only"
        )

    def _cut3r_token_only_smoke_scan_steps(self):
        if not self._cut3r_token_only_active() or not bool(getattr(self.args, "cut3r_token_smoke_telemetry", False)):
            return ()
        if not self.is_world_process_zero():
            return ()
        count = min(2, max(0, int(getattr(self.args, "cut3r_token_smoke_full_scan_steps", 2))))
        return tuple(range(1, count + 1))

    def _cut3r_token_only_named_parameter_groups(self):
        """Return a small deterministic smoke sample; never snapshot every LoRA tensor."""
        named_parameters = sorted(self.model.named_parameters(), key=lambda item: item[0])
        projector = [(name, parameter) for name, parameter in named_parameters if "cut3r_token_projector" in name]
        preferred_projector = [
            item for item in projector
            if item[0].endswith(("proj_in.weight", "proj_out.weight"))
        ]
        projector = preferred_projector or projector[:2]
        all_lora = [(name, parameter) for name, parameter in named_parameters if "lora_" in name]
        if len(all_lora) <= 6:
            lora = all_lora
        else:
            middle = len(all_lora) // 2
            picked = [*all_lora[:2], *all_lora[middle:middle + 2], *all_lora[-2:]]
            seen = set()
            lora = [item for item in picked if not (item[0] in seen or seen.add(item[0]))]
        return {"projector": projector, "lora": lora}

    @staticmethod
    def _cut3r_sample(parameter, width=256):
        flat = parameter.detach().reshape(-1)
        return flat[:min(int(flat.numel()), int(width))].float()

    def _spatial_rank_metrics(self):
        for module in self.model.modules():
            metrics = getattr(module, "_spatial_rank_last_metrics", None)
            if metrics:
                return metrics
        return None

    def _cut3r_token_only_metrics(self):
        for module in self.model.modules():
            metrics = getattr(module, "_cut3r_token_only_last_metrics", None)
            if metrics:
                return dict(metrics)
        return None

    @staticmethod
    def _cut3r_group_stats(named_parameters, before=None, include_grad=True):
        """Summarise deterministic small parameter slices with bounded transfers."""
        parameter_sq, parameter_finite = [], []
        gradient_sq, gradient_finite, gradient_nonzero = [], [], []
        delta_sq, delta_finite = [], []
        before = before or {}
        for name, parameter in named_parameters:
            value = LLaVATrainer._cut3r_sample(parameter)
            parameter_sq.append(value.square().sum())
            parameter_finite.append(torch.isfinite(value).all())
            if name in before:
                delta = value - before[name].to(device=value.device, dtype=value.dtype)
                delta_sq.append(delta.square().sum())
                delta_finite.append(torch.isfinite(delta).all())
            if include_grad and parameter.grad is not None:
                gradient = parameter.grad.detach().reshape(-1)[:value.numel()].float()
                gradient_sq.append(gradient.square().sum())
                gradient_finite.append(torch.isfinite(gradient).all())
                gradient_nonzero.append(gradient.abs().sum() > 0)

        def _norm(values):
            return float(torch.stack(values).sum().sqrt().item()) if values else 0.0
        def _all(values):
            return bool(torch.stack(values).all().item()) if values else True
        def _any(values):
            return bool(torch.stack(values).any().item()) if values else False
        return {
            "parameter_norm": _norm(parameter_sq), "parameter_finite": _all(parameter_finite),
            "gradient_norm": _norm(gradient_sq), "gradient_finite": _all(gradient_finite),
            "gradient_nonzero": _any(gradient_nonzero), "update_delta_norm": _norm(delta_sq),
            "update_finite": _all(delta_finite), "update_nonzero": _norm(delta_sq) > 0.0,
        }

    @staticmethod
    def _cut3r_group_snapshot(named_parameters):
        return {name: LLaVATrainer._cut3r_sample(parameter).cpu().clone() for name, parameter in named_parameters}

    def _write_cut3r_token_only_optimizer_evidence(self, evidence):
        if not self.is_world_process_zero():
            return
        step = int(evidence["optimizer_step"])
        self._cut3r_token_only_optimizer_evidence[step] = dict(evidence)
        line = json.dumps(self._jsonable(evidence), sort_keys=True)
        rank0_print(f"[CUT3R_TOKEN_ONLY][OPTIMIZER_STEP] {line}")
        try:
            os.makedirs(self.args.output_dir, exist_ok=True)
            with open(os.path.join(self.args.output_dir, "cut3r_token_only_optimizer_steps.jsonl"), "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            rank0_print(f"[CUT3R_TOKEN_ONLY][OPTIMIZER_STEP][WARN] failed to write JSONL: {exc}")

    def _capture_cut3r_token_only_pre_update(self, optimizer_step):
        groups = self._cut3r_token_only_named_parameter_groups()
        return {
            name: {
                "before": self._cut3r_group_snapshot(parameters),
                "stats": self._cut3r_group_stats(parameters, include_grad=True),
            }
            for name, parameters in groups.items()
        }

    def _capture_cut3r_token_only_post_update(self, optimizer_step, before, optimizer_was_run):
        groups = self._cut3r_token_only_named_parameter_groups()
        stats = {
            name: self._cut3r_group_stats(parameters, before=before[name]["before"], include_grad=False)
            for name, parameters in groups.items()
        }
        evidence = {"optimizer_step": int(optimizer_step), "optimizer_was_run": bool(optimizer_was_run)}
        for name in ("projector", "lora"):
            pre, post = before[name]["stats"], stats[name]
            evidence.update({
                f"{name}_parameter_norm_before": pre["parameter_norm"],
                f"{name}_parameter_norm_after": post["parameter_norm"],
                f"{name}_parameter_finite": bool(pre["parameter_finite"] and post["parameter_finite"]),
                f"{name}_grad_norm": pre["gradient_norm"],
                f"{name}_grad_finite": bool(pre["gradient_finite"]),
                f"{name}_grad_nonzero": bool(pre["gradient_nonzero"]),
                f"{name}_update_delta_norm": post["update_delta_norm"],
                f"{name}_update_finite": bool(post["update_finite"]),
                f"{name}_weight_updated": bool(optimizer_was_run and post["update_nonzero"]),
                f"{name}_sampled_parameter_names": [parameter_name for parameter_name, _ in groups[name]],
            })
        evidence["all_finite"] = all(bool(evidence[key]) for key in (
            "projector_parameter_finite", "projector_grad_finite", "projector_update_finite",
            "lora_parameter_finite", "lora_grad_finite", "lora_update_finite",
        ))
        base_model = self._cut3r_token_only_base_model()
        if base_model is not None:
            metrics = dict(getattr(base_model, "_cut3r_token_only_last_metrics", {}))
            metrics.update(evidence)
            base_model._cut3r_token_only_last_metrics = metrics
        self._write_cut3r_token_only_optimizer_evidence(evidence)

    def _install_cut3r_token_only_optimizer_hook(self, optimizer=None):
        if not self._cut3r_token_only_active():
            return
        scan_steps = self._cut3r_token_only_smoke_scan_steps()
        if self.is_world_process_zero():
            rank0_print(
                "[CUT3R_TOKEN_ONLY][TELEMETRY] "
                f"smoke_mode={bool(getattr(self.args, 'cut3r_token_smoke_telemetry', False)).__str__().lower()} "
                f"sampled_update_optimizer_steps={list(scan_steps)}"
            )
        optimizer = optimizer or self.optimizer
        if self.is_world_process_zero():
            ds_config = getattr(getattr(self.args, "hf_deepspeed_config", None), "config", {}) or {}
            runtime = {
                "cut3r_only_active": True,
                "trainer_optimizer_class": type(self.optimizer).__name__ if self.optimizer is not None else None,
                "prepared_optimizer_class": type(optimizer).__name__ if optimizer is not None else None,
                "deepspeed_engine_class": type(getattr(self, "deepspeed", None)).__name__ if getattr(self, "deepspeed", None) is not None else None,
                "accelerate_distributed_type": str(getattr(self.accelerator, "distributed_type", None)),
                "world_size": int(getattr(self.args, "world_size", 1)),
                "global_rank": int(getattr(self.args, "process_index", 0)),
                "local_rank": int(getattr(self.args, "local_rank", 0)),
                "gradient_accumulation_steps": int(getattr(self.args, "gradient_accumulation_steps", 1)),
                "deepspeed_zero_stage": (ds_config.get("zero_optimization") or {}).get("stage"),
                "manifest_path": os.environ.get("CUT3R_TOKEN_SIDECAR_MANIFEST", ""),
            }
            rank0_print("[CUT3R_TOKEN_ONLY][RUNTIME] " + json.dumps(runtime, sort_keys=True))
            try:
                os.makedirs(self.args.output_dir, exist_ok=True)
                with open(os.path.join(self.args.output_dir, "cut3r_token_only_runtime.json"), "w", encoding="utf-8") as handle:
                    json.dump(runtime, handle, indent=2, sort_keys=True)
                    handle.write("\n")
            except OSError as exc:
                rank0_print(f"[CUT3R_TOKEN_ONLY][RUNTIME][WARN] failed to write runtime JSON: {exc}")
        if optimizer is None or not scan_steps or self._cut3r_token_only_optimizer_hook_installed:
            return
        original_step = optimizer.step

        def wrapped_step(*args, **kwargs):
            optimizer_step = int(getattr(self.state, "global_step", 0) or 0) + 1
            before = None
            if optimizer_step in scan_steps:
                before = self._capture_cut3r_token_only_pre_update(optimizer_step)
                self._cut3r_token_only_pending_scans[optimizer_step] = before
            result = original_step(*args, **kwargs)
            if before is not None:
                optimizer_was_run = not bool(getattr(self.accelerator, "optimizer_step_was_skipped", False))
                self._capture_cut3r_token_only_post_update(optimizer_step, before, optimizer_was_run)
                self._cut3r_token_only_pending_scans.pop(optimizer_step, None)
            return result

        optimizer.step = wrapped_step
        self._cut3r_token_only_optimizer_hook_installed = True

    def training_step(self, model, inputs):
        started = time.monotonic()
        loss = super().training_step(model, inputs)
        base_model = model.get_model() if hasattr(model, "get_model") else getattr(model, "model", None)
        if base_model is not None and getattr(base_model.config, "visual_token_source", "siglip_only") == "cut3r_only":
            metrics = dict(getattr(base_model, "_cut3r_token_only_last_metrics", {}))
            metrics.update({
                "all_finite": bool(torch.isfinite(loss.detach()).all().item()),
                "step_time_s": time.monotonic() - started,
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
                "smoke_mode": bool(getattr(self.args, "cut3r_token_smoke_telemetry", False)),
                "sampled_update_optimizer_steps": list(self._cut3r_token_only_smoke_scan_steps()),
            })
            base_model._cut3r_token_only_last_metrics = metrics
        return loss

    def _bev_metrics(self):
        for module in self.model.modules():
            metrics = getattr(module, "_bev_last_metrics", None)
            if metrics:
                return metrics
        return None

    def _depth_metrics(self):
        for module in self.model.modules():
            metrics = getattr(module, "_depth_last_metrics", None)
            if metrics:
                return metrics
        return None

    def _pointmap_metrics(self):
        for module in self.model.modules():
            metrics = getattr(module, "_pointmap_last_metrics", None)
            if metrics:
                return metrics
        return None

    @staticmethod
    def _jsonable(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().float().item())
            return value.detach().float().cpu().tolist()
        if isinstance(value, dict):
            return {str(k): LLaVATrainer._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [LLaVATrainer._jsonable(v) for v in value]
        return value

    @staticmethod
    def _flatten_numeric(prefix, value, out):
        if isinstance(value, dict):
            for key, item in value.items():
                LLaVATrainer._flatten_numeric(f"{prefix}/{key}" if prefix else str(key), item, out)
            return
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                out[prefix] = float(value.detach().float().item())
            return
        if isinstance(value, bool):
            out[prefix] = float(value)
            return
        if isinstance(value, (int, float)) and value is not None:
            out[prefix] = float(value)

    def _geo_rope_fusion_stats(self):
        for module in self.model.modules():
            stats = getattr(module, "last_geo_rope_fusion_stats", None)
            if not stats or "mean_abs_rope_delta_q" not in stats:
                continue
            stats = dict(stats)
            grad_norm = getattr(module, "_last_head_gate_grad_norm", None)
            if grad_norm is not None:
                stats["gate_logit_grad_norm"] = grad_norm
            return stats
        return None

    def _geo_rope_fusion_metrics(self):
        stats = self._geo_rope_fusion_stats()
        if not stats:
            return None, None
        metrics = {}
        self._flatten_numeric("geo_rope", stats, metrics)
        return metrics, self._jsonable(stats)

    def _write_cut3r_token_only_metrics(self, metrics, logs):
        if not metrics or not self.is_world_process_zero():
            return
        step = int(getattr(self.state, "global_step", 0) or 0)
        if getattr(self, "_last_cut3r_token_only_metrics_step", None) == step:
            return
        self._last_cut3r_token_only_metrics_step = step
        payload = {"step": step, "trainer_log": self._jsonable(logs), "metrics": self._jsonable(metrics)}
        line = json.dumps(payload, sort_keys=True)
        rank0_print(f"[CUT3R_TOKEN_ONLY_METRICS] {line}")
        try:
            os.makedirs(self.args.output_dir, exist_ok=True)
            with open(os.path.join(self.args.output_dir, "cut3r_token_only_metrics.jsonl"), "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            rank0_print(f"[CUT3R_TOKEN_ONLY_METRICS][WARN] failed to write JSONL: {exc}")

    def _write_geo_rope_fusion_stats(self, stats):
        if not stats or not self.is_world_process_zero():
            return
        step = int(getattr(self.state, "global_step", 0) or 0)
        if getattr(self, "_last_geo_rope_stats_logged_step", None) == step:
            return
        self._last_geo_rope_stats_logged_step = step
        payload = {
            "step": step,
            "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
            **stats,
        }
        line = json.dumps(payload, sort_keys=True)
        rank0_print(f"[GEO_ROPE_STATS] {line}")
        try:
            os.makedirs(self.args.output_dir, exist_ok=True)
            with open(os.path.join(self.args.output_dir, "geo_rope_fusion_stats.jsonl"), "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as exc:
            rank0_print(f"[GEO_ROPE_STATS][WARN] failed to write JSONL: {exc}")

    def _llm_visual_3d_rope_stats(self):
        for module in self.model.modules():
            stats = getattr(module, "_last_llm_visual_3d_rope_stats", None)
            if stats:
                return stats
        return None

    def _llm_visual_3d_rope_debug(self):
        for module in self.model.modules():
            debug = getattr(module, "_last_llm_geo_debug", None)
            if debug:
                return debug
        return None

    def _llm_visual_3d_rope_metrics(self):
        stats = self._llm_visual_3d_rope_stats()
        if not stats:
            return None, None
        aggregate = {
            "num_logged_layers": len(stats),
            "attention_delta_mean_abs": 0.0,
            "visual_visual_logits_delta_mean_abs": 0.0,
            "num_valid_geo_tokens": 0,
        }
        deltas = [
            float(item.get("attention_delta_mean_abs", 0.0) or 0.0)
            for item in stats
            if not item.get("skipped", False)
        ]
        vv_deltas = [
            float(item.get("visual_visual_logits_delta_mean_abs", 0.0) or 0.0)
            for item in stats
            if not item.get("skipped", False)
        ]
        if deltas:
            aggregate["attention_delta_mean_abs"] = sum(deltas) / len(deltas)
        if vv_deltas:
            aggregate["visual_visual_logits_delta_mean_abs"] = sum(vv_deltas) / len(vv_deltas)
        for item in stats:
            aggregate["num_valid_geo_tokens"] = max(
                aggregate["num_valid_geo_tokens"],
                int(item.get("num_valid_geo_tokens", 0) or 0),
            )
        metrics = {}
        self._flatten_numeric("llm_visual_3d_rope", aggregate, metrics)
        return metrics, {"aggregate": aggregate, "layers": self._jsonable(stats), "metadata": self._jsonable(self._llm_visual_3d_rope_debug())}

    def _write_llm_visual_3d_rope_stats(self, stats):
        if not stats or not self.is_world_process_zero():
            return
        step = int(getattr(self.state, "global_step", 0) or 0)
        if getattr(self, "_last_llm_visual_3d_rope_stats_logged_step", None) == step:
            return
        self._last_llm_visual_3d_rope_stats_logged_step = step
        payload = {"step": step, **stats}
        line = json.dumps(payload, sort_keys=True)
        rank0_print(f"[LLM_VISUAL_3D_ROPE_STATS] {line}")
        try:
            os.makedirs(self.args.output_dir, exist_ok=True)
            with open(os.path.join(self.args.output_dir, "llm_visual_3d_rope_stats.jsonl"), "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as exc:
            rank0_print(f"[LLM_VISUAL_3D_ROPE_STATS][WARN] failed to write JSONL: {exc}")

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        if (
            torch.cuda.is_available()
            and (
                getattr(self.control, "should_log", False)
                or getattr(self.control, "should_evaluate", False)
                or getattr(self.control, "should_save", False)
            )
        ):
            torch.cuda.empty_cache()
        return super()._maybe_log_save_evaluate(*args, **kwargs)

    def log(self, logs, *args, **kwargs):
        metrics = self._spatial_rank_metrics()
        bev_metrics = self._bev_metrics()
        depth_metrics = self._depth_metrics()
        pointmap_metrics = self._pointmap_metrics()
        cut3r_token_metrics = self._cut3r_token_only_metrics()
        geo_rope_metrics, geo_rope_stats = self._geo_rope_fusion_metrics()
        llm_rope_metrics, llm_rope_stats = self._llm_visual_3d_rope_metrics()
        if cut3r_token_metrics:
            numeric_cut3r_metrics = {}
            self._flatten_numeric("cut3r_token_only", cut3r_token_metrics, numeric_cut3r_metrics)
            logs = dict(logs)
            logs.update(numeric_cut3r_metrics)
            self._write_cut3r_token_only_metrics(cut3r_token_metrics, logs)
        if metrics:
            logs = dict(logs)
            logs.update(metrics)
        if bev_metrics:
            numeric_bev_metrics = {}
            self._flatten_numeric("", bev_metrics, numeric_bev_metrics)
            logs = dict(logs)
            logs.update(numeric_bev_metrics)
        if depth_metrics:
            numeric_depth_metrics = {}
            self._flatten_numeric("", depth_metrics, numeric_depth_metrics)
            logs = dict(logs)
            logs.update(numeric_depth_metrics)
            for key in (
                "depth_point_map_key",
                "depth_head_source",
                "depth_shuffle_mode",
                "depth_point_map_key_used",
                "depth_target_space",
            ):
                if key in depth_metrics:
                    logs[key] = str(depth_metrics[key])
        if pointmap_metrics:
            numeric_pointmap_metrics = {}
            self._flatten_numeric("", pointmap_metrics, numeric_pointmap_metrics)
            logs = dict(logs)
            logs.update(numeric_pointmap_metrics)
            for key in (
                "pointmap_point_map_key",
                "pointmap_head_source",
                "pointmap_point_map_key_used",
                "pointmap_target_space",
            ):
                if key in pointmap_metrics:
                    logs[key] = str(pointmap_metrics[key])
        if geo_rope_metrics:
            logs = dict(logs)
            logs.update(geo_rope_metrics)
            self._write_geo_rope_fusion_stats(geo_rope_stats)
        if llm_rope_metrics:
            logs = dict(logs)
            logs.update(llm_rope_metrics)
            self._write_llm_visual_3d_rope_stats(llm_rope_stats)
        return super().log(logs, *args, **kwargs)

    def _build_sampler_generator(self):
        seed = getattr(self.args, "data_seed", None)
        if seed is None:
            seed = getattr(self.args, "seed", 42)

        generator = torch.Generator()
        generator.manual_seed(int(seed))
        return generator

    def create_accelerator_and_postprocess(self):
        grad_acc_kwargs = {"num_steps": self.args.gradient_accumulation_steps}
        grad_acc_kwargs["sync_with_dataloader"] = False
        gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        rank0_print("Setting NCCL timeout to INF to avoid running errors.")

        # create accelerator object
        self.accelerator = Accelerator(
            dispatch_batches=self.args.dispatch_batches, split_batches=self.args.split_batches, deepspeed_plugin=self.args.deepspeed_plugin, gradient_accumulation_plugin=gradient_accumulation_plugin, kwargs_handlers=[accelerator_kwargs]
        )
        # some Trainer classes need to use `gather` instead of `gather_for_metrics`, thus we store a flag
        self.gather_function = self.accelerator.gather_for_metrics

        # deepspeed and accelerate flags covering both trainer args and accelerate launcher
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None

        # post accelerator creation setup
        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            fsdp_plugin.limit_all_gathers = self.args.fsdp_config.get("limit_all_gathers", fsdp_plugin.limit_all_gathers)
            if is_accelerate_available("0.23.0"):
                fsdp_plugin.activation_checkpointing = self.args.fsdp_config.get("activation_checkpointing", fsdp_plugin.activation_checkpointing)
                if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                    raise ValueError("The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg " "can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic " "when using FSDP.")

        if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
            self.propagate_args_to_deepspeed()

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        sampler_generator = self._build_sampler_generator()

        if self.args.group_by_length:
            lengths = self.train_dataset.lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                generator=sampler_generator,
            )
        elif self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                group_by_modality=True,
                generator=sampler_generator,
            )
        elif self.args.group_by_modality_length_auto:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                group_by_modality_auto=True,
                generator=sampler_generator,
            )
        elif self.args.group_by_varlen:
            lengths = self.train_dataset.lengths
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                # self.args.train_batch_size, # TODO: seems that we should have gradient_accumulation_steps
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                variable_length=True,
                generator=sampler_generator,
            )
        else:
            return super()._get_train_sampler()

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            train_sampler = self._get_train_sampler()
            if train_sampler is not None:
                dataloader_params["sampler"] = train_sampler
            else:
                dataloader_params["shuffle"] = True
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_num_workers * 2 if self.args.dataloader_num_workers != 0 else None

        dataloader = self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

        return dataloader

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            if self.args.mm_projector_lr is not None:
                lr_mapper["mm_projector"] = self.args.mm_projector_lr
            if self.args.mm_vision_tower_lr is not None:
                lr_mapper["vision_tower"] = self.args.mm_vision_tower_lr
            if self.args.fusion_block_lr is not None:
                lr_mapper["fusion_block"] = self.args.fusion_block_lr
            if len(lr_mapper) > 0:
                special_lr_parameters = [name for name, _ in opt_model.named_parameters() if any(module_keyword in name for module_keyword in lr_mapper)]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                for module_keyword, lr in lr_mapper.items():
                    module_parameters = [name for name, _ in opt_model.named_parameters() if module_keyword in name]
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if self.args.lora_enable:
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            # 获取checkpoint路径
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # 分离并保存LoRA参数
            base_model = model.module if hasattr(model, "module") else model
            state_dict = get_peft_state_maybe_zero_3(base_model.named_parameters(), self.args.lora_bias)
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(base_model.named_parameters())

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                os.makedirs(output_dir, exist_ok=True)
                if hasattr(base_model, "config"):
                    base_model.config.save_pretrained(output_dir)
                if hasattr(base_model, "generation_config"):
                    base_model.generation_config.save_pretrained(output_dir)
                base_model.save_pretrained(output_dir, state_dict=state_dict)
                torch.save(non_lora_state_dict, os.path.join(output_dir, "non_lora_trainables.bin"))

        elif getattr(self.args, "tune_mm_mlp_adapter", False) or (
            getattr(self.args, "tune_fusion_block", False)) or (
            getattr(self.args, "tune_cut3r_spatialstack", False)) or (
            hasattr(self.args, "mm_tunable_parts") and (len(self.args.mm_tunable_parts.split(",")) == 1 and ("mm_mlp_adapter" in self.args.mm_tunable_parts or "mm_vision_resampler" in self.args.mm_tunable_parts or "cut3r_spatialstack" in self.args.mm_tunable_parts))
        ):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ["mm_projector", "vision_resampler", "fusion_block", "cut3r_spatialstack", "cut3r_camera_token_projector", "cut3r_token_projector", "bev_head", "depth_head", "pointmap_head", "spatial_bridge_tokens"]
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(["embed_tokens", "embed_in"])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))

        # 保存其他训练状态（优化器状态等）
        super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)


class LLaVADPOTrainer(DPOTrainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False) or (
            getattr(self.args, "tune_cut3r_spatialstack", False)) or (
            hasattr(self.args, "mm_tunable_parts") and (len(self.args.mm_tunable_parts.split(",")) == 1 and ("mm_mlp_adapter" in self.args.mm_tunable_parts or "mm_vision_resampler" in self.args.mm_tunable_parts or "cut3r_spatialstack" in self.args.mm_tunable_parts))
        ):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ["mm_projector", "vision_resampler", "cut3r_spatialstack", "cut3r_camera_token_projector", "cut3r_token_projector", "bev_head", "depth_head", "pointmap_head", "spatial_bridge_tokens"]
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(["embed_tokens", "embed_in"])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
        else:
            # super(LLaVADPOTrainer, self)._save_checkpoint(model, trial, metrics)
            # print(type(model))
            # from transformers.modeling_utils import unwrap_model
            # print(type(unwrap_model(model)))
            # print(unwrap_model(model).config)
            if self.args.lora_enable:
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(run_dir, checkpoint_folder)
                from transformers.modeling_utils import unwrap_model

                unwrapped_model = unwrap_model(model)
                self.save_my_lora_ckpt(output_dir, self.args, unwrapped_model)
            else:
                super(LLaVADPOTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            pass
        else:
            super(LLaVADPOTrainer, self)._save(output_dir, state_dict)
