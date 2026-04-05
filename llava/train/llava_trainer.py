import os
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

        # Calculate timing
        avg_time_per_step = elapsed / step
        remaining_steps = max_steps - step
        eta_seconds = avg_time_per_step * remaining_steps

        elapsed_str = self._format_time(elapsed)
        eta_str = self._format_time(eta_seconds)
        speed_str = f"{avg_time_per_step:.2f}s/it"

        loss_str = f" | loss={loss:.4f}" if isinstance(loss, float) else ""
        lr_str = f" | lr={lr:.2e}" if isinstance(lr, float) else ""

        print(f"[{bar}] {step}/{max_steps} ({pct:.1f}%) [{elapsed_str}<{eta_str}, {speed_str}] | epoch={epoch:.3f}{loss_str}{lr_str}", flush=True)


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

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
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

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices_auto_single(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices_auto_single(lang_lengths, batch_size, world_size, generator=None)]
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


class LLaVATrainer(Trainer):

    def _init_stage_profiler(self):
        if hasattr(self, "_stage_profiler"):
            return
        self._stage_profiler = {
            "steps": 0,
            "data_wait_s": 0.0,
            "sample_total_s": 0.0,
            "sample_core_s": 0.0,
            "media_s": 0.0,
            "text_s": 0.0,
            "spatial_load_s": 0.0,
            "forward_s": 0.0,
            "backward_s": 0.0,
            "step_total_s": 0.0,
        }
        self._stage_profile_last_step_end = None
        self._stage_profile_last_forward_s = 0.0

    def _stage_profiler_enabled(self):
        return bool(getattr(self.args, "profile_training_stages", False))

    def _nvtx_enabled(self):
        return bool(
            getattr(self.args, "enable_nvtx_ranges", False)
            and torch.cuda.is_available()
            and hasattr(torch.cuda, "nvtx")
        )

    def _nvtx_push(self, name: str):
        if self._nvtx_enabled():
            torch.cuda.nvtx.range_push(name)

    def _nvtx_pop(self):
        if self._nvtx_enabled():
            torch.cuda.nvtx.range_pop()

    def _nsys_delayed_capture_enabled(self):
        return bool(getattr(self.args, "nsys_capture_after_warmup", False) and torch.cuda.is_available())

    def _try_start_nsys_capture(self, current_step: int):
        if not self._nsys_delayed_capture_enabled():
            return
        if getattr(self, "_nsys_capture_started", False):
            return
        if getattr(self, "_nsys_capture_start_failed", False):
            return

        warmup_steps = max(0, int(getattr(self.args, "profile_warmup_steps", 5)))
        if current_step <= warmup_steps:
            return

        try:
            # Synchronize before starting CUDA profiler capture for cleaner boundaries.
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            self._nsys_capture_started = True
            if self.is_world_process_zero:
                rank0_print(f"[NSYS] Delayed capture started at global_step={current_step} (warmup_steps={warmup_steps}).")
        except Exception as exc:
            self._nsys_capture_start_failed = True
            if self.is_world_process_zero:
                rank0_print(f"[NSYS][WARN] Failed to start delayed capture: {exc}")

    def _try_stop_nsys_capture(self):
        if not getattr(self, "_nsys_capture_started", False):
            return
        try:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            if self.is_world_process_zero:
                rank0_print("[NSYS] Delayed capture stopped at train end.")
        except Exception as exc:
            if self.is_world_process_zero:
                rank0_print(f"[NSYS][WARN] Failed to stop delayed capture: {exc}")
        finally:
            self._nsys_capture_started = False

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def train(self, *args, **kwargs):
        try:
            return super().train(*args, **kwargs)
        finally:
            self._try_stop_nsys_capture()

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

        if self.args.group_by_length:
            lengths = self.train_dataset.lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
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

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        profiling_on = self._stage_profiler_enabled()
        nvtx_on = self._nvtx_enabled()
        if not profiling_on and not nvtx_on:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        if profiling_on:
            self._init_stage_profiler()
            forward_start = time.perf_counter()
        if nvtx_on:
            self._nvtx_push("forward_loss")
        try:
            outputs = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        finally:
            if nvtx_on:
                self._nvtx_pop()
        if profiling_on:
            self._stage_profile_last_forward_s = time.perf_counter() - forward_start
        return outputs

    def training_step(self, model: nn.Module, inputs, num_items_in_batch=None):
        profiling_on = self._stage_profiler_enabled()
        nvtx_on = self._nvtx_enabled()
        nsys_on = self._nsys_delayed_capture_enabled()
        current_step = self.state.global_step + 1
        if nsys_on:
            self._try_start_nsys_capture(current_step)

        if not profiling_on and not nvtx_on and not nsys_on:
            if num_items_in_batch is None:
                return super().training_step(model, inputs)
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        if profiling_on:
            self._init_stage_profiler()
            step_start = time.perf_counter()
            warmup_steps = max(0, int(getattr(self.args, "profile_warmup_steps", 5)))
            collect_this_step = current_step > warmup_steps
        else:
            collect_this_step = False

        if collect_this_step and self._stage_profile_last_step_end is not None:
            if nvtx_on:
                self._nvtx_push("data_wait")
            self._stage_profiler["data_wait_s"] += max(0.0, step_start - self._stage_profile_last_step_end)
            if nvtx_on:
                self._nvtx_pop()

        sample_total_s = self._safe_float(inputs.pop("_profile_sample_total_s", 0.0))
        sample_core_s = self._safe_float(inputs.pop("_profile_sample_core_s", 0.0))
        media_s = self._safe_float(inputs.pop("_profile_media_s", 0.0))
        text_s = self._safe_float(inputs.pop("_profile_text_s", 0.0))
        spatial_load_s = self._safe_float(inputs.pop("_profile_spatial_load_s", 0.0))

        if profiling_on:
            self._stage_profile_last_forward_s = 0.0
            fwbw_start = time.perf_counter()
        if nvtx_on:
            self._nvtx_push("train_step_fwd_bwd")
        try:
            if num_items_in_batch is None:
                loss = super().training_step(model, inputs)
            else:
                loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        finally:
            if nvtx_on:
                self._nvtx_pop()
        if profiling_on:
            step_end = time.perf_counter()

        if collect_this_step:
            fwd_s = max(0.0, self._stage_profile_last_forward_s)
            fwbw_s = max(0.0, step_end - fwbw_start)
            self._stage_profiler["steps"] += 1
            self._stage_profiler["sample_total_s"] += sample_total_s
            self._stage_profiler["sample_core_s"] += sample_core_s
            self._stage_profiler["media_s"] += media_s
            self._stage_profiler["text_s"] += text_s
            self._stage_profiler["spatial_load_s"] += spatial_load_s
            self._stage_profiler["forward_s"] += fwd_s
            self._stage_profiler["backward_s"] += max(0.0, fwbw_s - fwd_s)
            self._stage_profiler["step_total_s"] += max(0.0, step_end - step_start)

        if profiling_on:
            self._stage_profile_last_step_end = step_end
        return loss

    def log(self, logs):
        if self._stage_profiler_enabled() and hasattr(self, "_stage_profiler"):
            prof_steps = max(0, int(self._stage_profiler.get("steps", 0)))
            if prof_steps > 0:
                prof_avg = {k: v / prof_steps for k, v in self._stage_profiler.items() if k.endswith("_s")}
                logs = dict(logs)
                logs.update(
                    {
                        "prof_data_wait_s": prof_avg.get("data_wait_s", 0.0),
                        "prof_sample_total_s": prof_avg.get("sample_total_s", 0.0),
                        "prof_sample_core_s": prof_avg.get("sample_core_s", 0.0),
                        "prof_media_s": prof_avg.get("media_s", 0.0),
                        "prof_text_s": prof_avg.get("text_s", 0.0),
                        "prof_spatial_load_s": prof_avg.get("spatial_load_s", 0.0),
                        "prof_forward_s": prof_avg.get("forward_s", 0.0),
                        "prof_backward_s": prof_avg.get("backward_s", 0.0),
                        "prof_step_total_s": prof_avg.get("step_total_s", 0.0),
                        "prof_steps_count": prof_steps,
                    }
                )
        return super().log(logs)

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
            hasattr(self.args, "mm_tunable_parts") and (len(self.args.mm_tunable_parts.split(",")) == 1 and ("mm_mlp_adapter" in self.args.mm_tunable_parts or "mm_vision_resampler" in self.args.mm_tunable_parts))
        ):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ["mm_projector", "vision_resampler", "fusion_block"]
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
            hasattr(self.args, "mm_tunable_parts") and (len(self.args.mm_tunable_parts.split(",")) == 1 and ("mm_mlp_adapter" in self.args.mm_tunable_parts or "mm_vision_resampler" in self.args.mm_tunable_parts))
        ):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ["mm_projector", "vision_resampler"]
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
