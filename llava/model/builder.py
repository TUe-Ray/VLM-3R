#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.utils import rank0_print


def _force_config_attn_implementation(config, attn_implementation):
    if config is None or not attn_implementation:
        return
    for attr in ("_attn_implementation", "_attn_implementation_internal", "attn_implementation"):
        try:
            setattr(config, attr, attn_implementation)
        except Exception:
            pass


def _prepare_multimodal_token_embeddings(tokenizer, model):
    """Apply LLaVA special tokens before an Accelerate CPU-offload dispatch."""
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", torch_dtype="float16",attn_implementation="flash_attention_2", customized_config=None, overwrite_config=None, **kwargs):
    kwargs["device_map"] = device_map
    token_embeddings_prepared_before_dispatch = False

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    elif torch_dtype == "float16":
        kwargs["torch_dtype"] = torch.float16
    elif torch_dtype == "bfloat16":
        kwargs["torch_dtype"] = torch.bfloat16
    else:
        import pdb;pdb.set_trace()

    if customized_config is not None:
        kwargs["config"] = customized_config

    if "multimodal" in kwargs:
        if kwargs["multimodal"] is True:
            is_multimodal = True
            kwargs.pop("multimodal")
    else:
        is_multimodal = False

    if "llava" in model_name.lower() or is_multimodal:
        # Load LLaVA model
        if "lora" in model_name.lower() and model_base is None:
            warnings.warn(
                "There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged."
            )
        if "lora" in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            rank0_print("Loading LLaVA from base model...")
            if "mixtral" in model_name.lower():
                from llava.model.language_model.llava_mixtral import LlavaMixtralConfig

                lora_cfg_pretrained = LlavaMixtralConfig.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                model = LlavaMixtralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            elif "mistral" in model_name.lower():
                from llava.model.language_model.llava_mistral import LlavaMistralConfig

                lora_cfg_pretrained = LlavaMistralConfig.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                model = LlavaMistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            elif "gemma" in model_name.lower():
                from llava.model.language_model.llava_gemma import LlavaGemmaConfig

                lora_cfg_pretrained = LlavaGemmaConfig.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                model = LlavaGemmaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            elif "qwen" in model_name.lower():
                from llava.model.language_model.llava_qwen import LlavaQwenConfig
                additional_config = {
                    "tie_word_embeddings": False,
                    "use_cache": True,
                    "vocab_size": 152064
                }
                lora_cfg_pretrained = LlavaQwenConfig.from_pretrained(model_path)
                if overwrite_config is not None:
                    overwrite_config.update(additional_config)
                    rank0_print(f"Overwriting config with {overwrite_config}")
                    for k, v in overwrite_config.items():
                        setattr(lora_cfg_pretrained, k, v)
                    _force_config_attn_implementation(lora_cfg_pretrained, attn_implementation)
                    load_kwargs = dict(kwargs)
                    cpu_merge = os.environ.get("SPATIALFOCUS_CPU_MERGE_LORA") == "1" and kwargs.get("device_map") == "auto"
                    if cpu_merge:
                        load_kwargs.pop("device_map", None)
                        load_kwargs.pop("max_memory", None)
                        load_kwargs.pop("offload_buffers", None)
                    model = LlavaQwenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=not cpu_merge and kwargs.get("device_map") is not None, attn_implementation=attn_implementation, config=lora_cfg_pretrained, **load_kwargs)
                else:
                    overwrite_config = additional_config
                    for k, v in overwrite_config.items():
                        setattr(lora_cfg_pretrained, k, v)
                    _force_config_attn_implementation(lora_cfg_pretrained, attn_implementation)
                    load_kwargs = dict(kwargs)
                    cpu_merge = os.environ.get("SPATIALFOCUS_CPU_MERGE_LORA") == "1" and kwargs.get("device_map") == "auto"
                    if cpu_merge:
                        load_kwargs.pop("device_map", None)
                        load_kwargs.pop("max_memory", None)
                        load_kwargs.pop("offload_buffers", None)
                    model = LlavaQwenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=not cpu_merge and kwargs.get("device_map") is not None, attn_implementation=attn_implementation, config=lora_cfg_pretrained, **load_kwargs)
                # model.to(device="cuda", dtype=torch.float16)

            else:
                from llava.model.language_model.llava_llama import LlavaConfig

                lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, attn_implementation=attn_implementation, **kwargs)

            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            rank0_print("Loading additional LLaVA weights...")
            if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
                non_lora_trainables = torch.load(os.path.join(model_path, "non_lora_trainables.bin"), map_location="cpu")
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download

                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder)
                    return torch.load(cache_file, map_location="cpu")

                non_lora_trainables = load_from_hf(model_path, "non_lora_trainables.bin")
            non_lora_trainables = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith("model.model.") for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()}
            # Backward compat: remap old weight key names from before the GeoRoPE Fusion rename.
            _geo_rope_key_remap = {
                "model.fusion_block.rope_gate_q": "model.fusion_block.geo_rope_fusion_gate_q",
                "model.fusion_block.rope_gate_k": "model.fusion_block.geo_rope_fusion_gate_k",
            }
            non_lora_trainables = {_geo_rope_key_remap.get(k, k): v for k, v in non_lora_trainables.items()}
            msg = model.load_state_dict(non_lora_trainables, strict=False)
            rank0_print(f"[DEBUG] non_lora_trainables loaded: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")

            if len(msg.missing_keys) > 0:
                rank0_print("[DEBUG] first missing keys:")
                for k in msg.missing_keys[:50]:
                    rank0_print(f"  MISSING: {k}")

            if len(msg.unexpected_keys) > 0:
                rank0_print("[DEBUG] first unexpected keys:")
                for k in msg.unexpected_keys[:50]:
                    rank0_print(f"  UNEXPECTED: {k}")

            from peft import PeftModel

            rank0_print("Loading LoRA weights...")
            cpu_merge = os.environ.get("SPATIALFOCUS_CPU_MERGE_LORA") == "1" and kwargs.get("device_map") == "auto"
            if cpu_merge:
                # ``dispatch_model`` records the original module tensor
                # shapes.  Resizing after dispatch leaves its CPU-offload
                # state dict at the old vocabulary size, which only fails on
                # the first generated token.  Prepare the tokenizer and both
                # language embedding matrices before dispatch instead.
                _prepare_multimodal_token_embeddings(tokenizer, model)
                token_embeddings_prepared_before_dispatch = True
                model = model.to(device="cpu", dtype=torch.float16)
            model = PeftModel.from_pretrained(model, model_path)
            rank0_print("Merging LoRA weights...")
            model = model.merge_and_unload()
            if cpu_merge:
                from accelerate import dispatch_model, infer_auto_device_map

                gpu_count = max(torch.cuda.device_count(), 1)
                # Reserve roughly 4 GiB per card for visual activations,
                # allocator fragmentation, and cross-device transfers during
                # the 32-frame forward.  The merged fp16 model remains
                # sharded across both TITAN Vs, with CPU available as a
                # safety spillover.
                gpu_budget = os.environ.get("SPATIALFOCUS_CPU_MERGE_GPU_BUDGET", "8GiB")
                cpu_budget = os.environ.get("SPATIALFOCUS_CPU_MERGE_CPU_BUDGET", "40GiB")
                # A 32-frame SigLIP pass needs substantially more activation
                # headroom on GPU 0 than Qwen decoder layers do on GPU 1.
                # Keep the historical uniform budget as the default, while
                # permitting sidecar-only evaluation wrappers to set, for
                # example, ``6GiB,10GiB``.
                per_gpu_budgets = os.environ.get("SPATIALFOCUS_CPU_MERGE_GPU_BUDGETS", "").strip()
                if per_gpu_budgets:
                    parsed_budgets = [value.strip() for value in per_gpu_budgets.split(",") if value.strip()]
                    if len(parsed_budgets) != gpu_count:
                        raise ValueError(
                            "SPATIALFOCUS_CPU_MERGE_GPU_BUDGETS must provide one budget for every visible GPU; "
                            f"got {parsed_budgets!r} for {gpu_count} GPU(s)."
                        )
                    max_memory = {index: parsed_budgets[index] for index in range(gpu_count)}
                else:
                    max_memory = {index: gpu_budget for index in range(gpu_count)}
                max_memory["cpu"] = cpu_budget
                device_map = infer_auto_device_map(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=["Qwen2DecoderLayer"],
                )
                # The SigLIP wrapper owns a nested pretrained module. Keep the
                # complete vision tower on one card; splitting its nested
                # convolutional weights across Accelerate devices causes the
                # historical list-of-frame forward to mix cuda:0/cuda:1.
                # Keep the complete visual/fusion path together.  The
                # fusion block's forward receives SigLIP activations from
                # this card and cannot safely move a dispatched/meta module
                # at runtime; the projector and resampler are small enough
                # to remain beside it.
                for module_name in (
                    "model.vision_tower",
                    "model.fusion_block",
                    "model.geometry_aware_projection",
                    "model.vision_resampler",
                    "model.mm_projector",
                    # SpatialStack constructs residuals from the visual
                    # embedding device and deliberately self-aligns its
                    # module at forward time.  CPU offload would replace its
                    # weights with meta tensors, making that alignment
                    # impossible, so it belongs with the visual path.
                    "model.cut3r_spatialstack_merger",
                ):
                    # ``infer_auto_device_map`` can expose nested vision
                    # children rather than their wrapper when a tighter GPU
                    # budget spills the remainder to CPU.  Replacing only a
                    # direct parent key then fails to uphold the one-device
                    # vision contract and later ``vision_tower.to(cuda)``
                    # encounters meta tensors.  Collapse every descendant
                    # before pinning the intended multimodal component.
                    nested_keys = [
                        key for key in device_map
                        if key.startswith(f"{module_name}.")
                    ]
                    if module_name in device_map or nested_keys:
                        for key in nested_keys:
                            del device_map[key]
                        device_map[module_name] = 0
                # Leave activation headroom for the 32-frame SigLIP pass on
                # the card that hosts the vision tower.
                for layer_name in (
                    "model.layers.10",
                    "model.layers.11",
                    "model.layers.12",
                    "model.layers.13",
                    "model.layers.14",
                    "model.layers.15",
                    "model.layers.16",
                ):
                    if layer_name in device_map:
                        device_map[layer_name] = 1
                rank0_print(f"Dispatching merged model with device map: {device_map}")
                # KV cache is a per-layer collection.  Accelerate's default
                # input/output alignment would move the entire collection to
                # GPU 0 at each generate step, corrupting a decoder split
                # across GPU 0 and GPU 1.  Keep cache entries on their owning
                # decoder devices instead.
                model = dispatch_model(
                    model,
                    device_map=device_map,
                    offload_buffers=True,
                    skip_keys=["past_key_values", "past_key_value"],
                )
            rank0_print("Model is loaded...")
            # The Qwen LoRA branch removes device_map before constructing the
            # custom LLaVA class.  Place the merged model explicitly for
            # single-rank evaluator calls, whose device_map is normally auto.
            if isinstance(device_map, str) and (device_map == "auto" or device_map.startswith("cuda")) and torch.cuda.is_available():
                target_device = "cuda" if device_map == "auto" else device_map
                model.to(device=target_device, dtype=torch.float16)
        elif model_base is not None:  # this may be mm projector only, loading projector with preset language mdoel
            rank0_print(f"Loading LLaVA from base model {model_base}...")
            if "mixtral" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaMixtralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            elif "mistral" in model_name.lower() or "zephyr" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            elif "gemma" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaGemmaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            elif (
                "wizardlm-2" in model_name.lower()
                and "vicuna" in model_name.lower()
                or "llama" in model_name.lower()
                or "yi" in model_name.lower()
                or "nous-hermes" in model_name.lower()
                or "llava-v1.6-34b" in model_name.lower()
                or "llava-v1.5" in model_name.lower()
            ):
                from llava.model.language_model.llava_llama import LlavaConfig

                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                if customized_config is None:
                    llava_cfg = LlavaConfig.from_pretrained(model_path)
                    if "v1.5" in model_name.lower():
                        llava_cfg.delay_load = True  # a workaround for correctly loading v1.5 models
                else:
                    llava_cfg = customized_config

                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                llava_cfg = LlavaConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=llava_cfg, **kwargs)
            elif "qwen" in model_name.lower() or "quyen" in model_name.lower():
                from llava.model.language_model.llava_qwen import LlavaQwenConfig
                tokenizer = AutoTokenizer.from_pretrained(model_base)
                additional_config = {
                    "tie_word_embeddings": False,
                    "use_cache": True,
                    "vocab_size": 152064
                }
                if overwrite_config is not None:
                    overwrite_config.update(additional_config)
                    cfg_pretrained = LlavaQwenConfig.from_pretrained(model_path)
                    rank0_print(f"Overwriting config with {overwrite_config}")
                    for k, v in overwrite_config.items():
                        setattr(cfg_pretrained, k, v)
                    _force_config_attn_implementation(cfg_pretrained, attn_implementation)
                    del kwargs["device_map"]
                    model = LlavaQwenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=False, attn_implementation=attn_implementation, config=cfg_pretrained, **kwargs)
                    model.to(device="cuda", dtype=torch.float16)
                else:
                    overwrite_config = additional_config
                    model = LlavaQwenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=False, attn_implementation=attn_implementation, **kwargs)
            else:
                raise ValueError(f"Model {model_name} not supported")

            mm_projector_weights = torch.load(os.path.join(model_path, "mm_projector.bin"), map_location="cpu")
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            rank0_print(f"Loaded LLaVA model: {model_path}")
            if "mixtral" in model_name.lower():
                from llava.model.language_model.llava_mixtral import LlavaMixtralConfig

                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                if customized_config is None:
                    llava_cfg = LlavaMixtralConfig.from_pretrained(model_path)
                else:
                    llava_cfg = customized_config

                if overwrite_config is not None:
                    rank0_print(f"Overwriting config with {overwrite_config}")
                    for k, v in overwrite_config.items():
                        setattr(llava_cfg, k, v)

                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMixtralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=llava_cfg, **kwargs)

            elif "mistral" in model_name.lower() or "zephyr" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, **kwargs)
            elif (
                "wizardlm-2" in model_name.lower()
                and "vicuna" in model_name.lower()
                or "llama" in model_name.lower()
                or "yi" in model_name.lower()
                or "nous-hermes" in model_name.lower()
                or "llava-v1.6-34b" in model_name.lower()
                or "llava-v1.5" in model_name.lower()
            ):
                from llava.model.language_model.llava_llama import LlavaConfig

                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                if customized_config is None:
                    llava_cfg = LlavaConfig.from_pretrained(model_path)
                    if "v1.5" in model_name.lower():
                        llava_cfg.delay_load = True  # a workaround for correctly loading v1.5 models
                else:
                    llava_cfg = customized_config

                if overwrite_config is not None:
                    rank0_print(f"Overwriting config with {overwrite_config}")
                    for k, v in overwrite_config.items():
                        setattr(llava_cfg, k, v)

                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=llava_cfg, **kwargs)

            elif "qwen" in model_name.lower() or "quyen" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                if "moe" in model_name.lower() or "A14B" in model_name.lower():
                    from llava.model.language_model.llava_qwen_moe import LlavaQwenMoeConfig
                    if overwrite_config is not None:
                        llava_cfg = LlavaQwenMoeConfig.from_pretrained(model_path)
                        rank0_print(f"Overwriting config with {overwrite_config}")
                        for k, v in overwrite_config.items():
                            setattr(llava_cfg, k, v)
                        _force_config_attn_implementation(llava_cfg, attn_implementation)
                        model = LlavaQwenMoeForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=llava_cfg, **kwargs)
                    else:
                        model = LlavaQwenMoeForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, **kwargs)

                else:
                    from llava.model.language_model.llava_qwen import LlavaQwenConfig
                    additional_config = {
                        "tie_word_embeddings": False,
                        "use_cache": True,
                        "vocab_size": 152064
                    }
                    del kwargs["device_map"]
                    if overwrite_config is not None:
                        overwrite_config.update(additional_config)
                        llava_cfg = LlavaQwenConfig.from_pretrained(model_path)
                        rank0_print(f"Overwriting config with {overwrite_config}")
                        for k, v in overwrite_config.items():
                            setattr(llava_cfg, k, v)
                        _force_config_attn_implementation(llava_cfg, attn_implementation)
                        model = LlavaQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=False, attn_implementation=attn_implementation, config=llava_cfg, **kwargs)
                        model.to(device="cuda", dtype=torch.float16)
                    else:
                        overwrite_config = additional_config
                        model = LlavaQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=False, attn_implementation=attn_implementation, **kwargs)

            elif "gemma" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaGemmaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            else:
                try:
                    from llava.model.language_model.llava_llama import LlavaConfig

                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                    if customized_config is None:
                        llava_cfg = LlavaConfig.from_pretrained(model_path)
                        if "v1.5" in model_path.lower():
                            llava_cfg.delay_load = True  # a workaround for correctly loading v1.5 models
                    else:
                        llava_cfg = customized_config

                    if overwrite_config is not None:
                        rank0_print(f"Overwriting config with {overwrite_config}")
                        for k, v in overwrite_config.items():
                            setattr(llava_cfg, k, v)
                    model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=llava_cfg, **kwargs)
                except:
                    raise ValueError(f"Model {model_name} not supported")

    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch.float16)
        else:
            use_fast = False
            if "mpt" in model_name.lower().replace("prompt", ""):
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    rank0_print(f"Model Class: {model.__class__.__name__}")
    image_processor = None

    if "llava" in model_name.lower() or is_multimodal:
        if not token_embeddings_prepared_before_dispatch:
            _prepare_multimodal_token_embeddings(tokenizer, model)

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != "auto":
            vision_tower.to(device="cuda", dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        context_len = model.config.max_position_embeddings
    elif hasattr(model.config, "tokenizer_model_max_length"):
        context_len = model.config.tokenizer_model_max_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
