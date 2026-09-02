# Agent Instructions

SpatialFocus is used on more than one machine. Apply these project-wide rules
everywhere, then dispatch by hostname before using runtime/storage instructions:

- Hostname mps-edu-06: read and follow docs/agents/mps-edu-06.md.
- Hostname matching *.snellius.surf.nl: read and follow docs/agents/snellius.md.

Machine documents override only runtime, storage, hardware, and
machine-environment instructions. They never override this shared scientific
contract, checkpoint provenance rules, or repository hygiene.

## Repository Rules

- Check git status --short before edits. Do not overwrite unrelated user
  changes or edit third_party/ unless explicitly requested.
- Do not put model weights, datasets, generated feature caches, or Hugging
  Face caches in Git.
- Prefer small experiment-specific wrappers/configuration over broad
  training-script changes.
- Do not install, remove, or upgrade dependencies without user approval unless
  explicitly instructed.
- Report missing assets, manifests, data, or initialization instead of silently
  substituting another checkpoint or dataset.

## Pre-SFT Proxy Scientific Contract

The primary research question is pre-SFT architecture screening:

> Can low-cost signals available before architecture-specific SFT predict the
> final post-SFT VSI-Bench ranking of 3D-information injection mechanisms?

 The formal proxy state is:

    pretrained base VLM
    + pretrained spatial encoder / required spatial inputs
    + candidate-specific verified pre-SFT initialization or C1 calibration
    + fresh trainable modules/LoRA required by the proxy protocol
    + supervised QA calibration loss
    + forward/backward
    + no optimizer update

Allowed: base pretrained VLM, pretrained CUT3R/spatial encoder, deterministic
candidate initialization, C1 calibration, QA labels, forward/backward, and
gradient measurement.

Forbidden: candidate post-SFT adapter_model.bin, post-SFT
non_lora_trainables.bin, trained LoRA, trained fusion/projector parameters,
candidate-specific SFT checkpoint state, and any optimizer update.

Pre-SFT refers to checkpoint/model state, not to absence of labels or
gradients. A trained VLM3R reproduction must never silently become the
pre-SFT Baseline. A trained-checkpoint score is only a separately labelled
post-SFT diagnostic and must not be pooled with the formal proxy study.

## Checkpoint Provenance Guard

Before every formal proxy run, verify and record:

- base-model identity;
- candidate initialization/C1 identity and hash;
- no post-SFT adapter, trained non-LoRA state, trained LoRA, or trained
  fusion/projector state was loaded;
- no optimizer was constructed or stepped;
- current Git commit; and
- fixed calibration manifest/sample identity.

If exact pre-SFT initialization is unavailable or ambiguous, mark the
candidate unavailable and ask for direction. Do not infer/reconstruct it from
trained state or a similarly named run. Historical post-SFT defaults never
override this rule.

## Current Formal Proxy and Calibration

The authoritative migration audit is:

- docs/snellius_proxy_migration_manifest.md
- docs/snellius_proxy_migration_files.txt

Do not change its scientific conclusions without a concrete error. The formal
C1 evaluator currently has exactly five candidates: C1 VLM3R Baseline;
SpatialStack additive 0/1/2, 1/2/3, and 0/3/6; and SpatialStack
cross-attention 0/1/2. Base/zero-spatial, selective fusion, geometry, and
post-SFT candidates must not be silently added.

The audited Baseline smoke uses scene0384_00, 32 RGB frames, input shape
[1,424], 13 supervised labels, seed 42, and PEFT 0.4.0. Its SFT-trainable
scope is 349,689,856 parameters: LoRA 322,961,408; C1 VLM3R fusion 9,747,456;
and mm_projector 16,980,992. Use these audited values, not prior estimates.

Every candidate comparison uses the same fixed calibration preprocessing and
manifest. Model forward uses 32 RGB frames; compact two-frame probe targets
are not model-forward geometry. If an architecture requires full point maps,
verify the full-frame sidecars rather than substituting compact targets.

## Proxy Loss, Definitions, and Gradient Coverage

The primary architecture-only protocol uses the same supervised QA backward:

    L_proxy = L_QA

A score using L_QA + lambda * L_depth is a separately labelled
recipe-matched/supervision-specific analysis and must not be mixed into the
primary architecture-only benchmark.

- GradNorm: sum over selected parameter tensors of gradient L2 norm.
- SNIP: sum over selected scalar parameters of absolute(theta * gradient).
- Fisher: sum over selected scalar parameters of gradient squared.

No optimizer step is performed. Every selected scope must report selected
parameter count, gradient-covered parameter count, and uncovered intended
groups. A formal run requires expected gradient coverage in every selected
group. PEFT version is scientifically relevant because it can alter LoRA
target/module construction, initialization, and counts.

## Candidate Representation and Probe Conventions

- A candidate changing pre-SFT forward representation requires its own pre-SFT
  representation probe.
- Reuse of a representation probe for an auxiliary-head/loss-only change is
  allowed only after numerical forward equivalence. Its gradient proxies may
  still differ under a changed loss.
- New pre-SFT depth probes must follow the project layer policy in
  scripts/probing/probe_layer_policy.py. Historical partial caches are not
  silently promoted to formal coverage.
- Preserve LLM cache indexing: requested layer L maps to hidden_states[L + 1].

## Smoke Before Formal Sweep

Run exactly one minimal migration validation before the formal candidate sweep,
starting with the VLM3R Baseline and using the machine-specific executor:

- one fixed calibration minibatch;
- one supervised QA forward/backward;
- no optimizer;
- finite GradNorm, SNIP, and Fisher;
- audited selected-parameter count and expected gradient coverage;
- proof of no post-SFT state and no unintended CPU/meta trainable parameter;
- peak accelerator memory and runtime.

Do not turn this into a large test suite. After the Baseline smoke passes,
apply the same fixed protocol to the remaining formal candidates.

## Logging, Verification, and Git

Formal proxy provenance records timestamp/run ID, Git commit, hostname/cluster,
job/process identifier, GPU model, CUDA/PyTorch/Transformers/PEFT/Accelerate
versions, dtype, TF32 state, attention/checkpointing/offload state, base and
C1 identities, calibration IDs, seed, loss definition, selected and
gradient-covered counts, GradNorm/SNIP/Fisher, peak memory, runtime, and zero
optimizer steps. Keep results compact; do not dump large tensors or per-layer
logs without a debugging need.

- Python changes: python -m py_compile <changed files>.
- Shell changes: bash -n <changed scripts>.
- Use the machine-specific minimal smoke for GPU/wrapper validation; never use
  a login node for GPU work.
- Preserve unrelated changes. Stage/commit only files changed by this chat,
  use Conventional Commits, and do not commit migration artifacts or
  machine-specific data. When the user requests review before commit, leave
  the working diff uncommitted.
### Hardware

- Host: `mps-edu-06`
- OS: Ubuntu 22.04
- CPU: AMD Ryzen Threadripper 2950X, 16 physical cores / 32 logical threads
- RAM: approximately 62 GiB
- Swap: approximately 119 GiB
- GPUs: 2 x NVIDIA TITAN V, 12 GiB VRAM each
- Root/NVMe device: Samsung SSD 970 EVO 2TB
- Data SSD: Samsung SSD 860 EVO 2TB mounted at `/mnt/DATA_SSD`

### Direct Execution; No Slurm

- Do not use `sbatch`, `srun`, Slurm partitions, QoS settings, or Leonardo job directives on `mps-edu-06`.
- Run commands directly from the shell.
- Use `CUDA_VISIBLE_DEVICES` to select GPUs when useful.
- If a supported workload needs multiple GPUs, use the repository's local distributed launcher or `torchrun` as appropriate; do not assume old Leonardo multi-GPU settings are valid here.
- Existing `.sbatch` and Leonardo Slurm wrappers are historical command/config references. Do not submit them directly. Reuse the underlying command or create a small local wrapper.
- Before a new or materially changed training/evaluation/extraction configuration, run a small local smoke test first. Confirm imports, paths, data loading, tensor shapes, output writing, and at least the first expected progress signal before launching the full run.
- Before starting a long run, state the intended command, GPU selection, and output/log path.
- Write experiment logs under `logs/` or another explicit experiment-specific output directory.
- Do not stop, kill, or otherwise interrupt a process that the agent did not start unless the user explicitly authorizes it.
- Infrastructure failures can still occur locally. Distinguish transient filesystem/GPU/process failures from code bugs before editing code.

### GPU Compatibility

- TITAN V is a Volta-generation GPU with 12 GiB VRAM. Do not blindly reuse Leonardo assumptions about GPU count, VRAM, mixed precision, or batch size.
- In particular, verify dtype support and memory use before full runs. Do not assume a Leonardo `bf16`, 4-GPU, or large-batch configuration will run unchanged.
- Prefer a minimal single-GPU smoke test before scaling to both GPUs.

### Proven local two-GPU VLM extraction

- The working local setup uses one sharded extraction process, not dataset sharding. The process iterates through the complete manifest sequentially while the model weights are split across both GPUs.
- Expose both physical GPUs during model loading:

  ```bash
  CUDA_VISIBLE_DEVICES=0,1 SPATIALFOCUS_CPU_MERGE_LORA=1
  ```

- `SPATIALFOCUS_CPU_MERGE_LORA=1` makes the loader load the base model on CPU, merge the LoRA checkpoint on CPU, then dispatch the merged model with Accelerate. The successful placement was approximately:
  - `model.vision_tower` and decoder layers `0-9`: logical GPU `0`
  - decoder layers `10-27` and `spatial_tower`/`fusion_block`/`vision_resampler`/`mm_projector`: logical GPU `1`
  - `lm_head` and any overflow layers: CPU
- Keep the complete vision tower on one GPU; splitting its nested weights can break the list-of-frames forward path. CPU offload is intentional and provides activation/allocator headroom.
- `GPU=0` in local wrappers identifies the physical GPU used for readiness checks. After `CUDA_VISIBLE_DEVICES=1`, that physical GPU is logical `cuda:0` inside PyTorch. Never use a device map referring to logical GPU `1` while exposing only `CUDA_VISIBLE_DEVICES=0`.
- Before a long run, verify the selected physical GPU with `nvidia-smi --id="$GPU"`, then verify inside PyTorch: TITAN V, capability `(7,0)`, `sm_70` in `torch.cuda.get_arch_list()`, and a small fp16 allocation/matmul/synchronize test.
- The migrated input adapter still sends 32 RGB frames to the model and writes only the 2 selected probe-frame outputs. It must not construct MP4 paths from `forward_frames_32_v1`, and it must not use the compact 2-frame target bundle as model-forward geometry.
- For multiple requested LLM layers, collect them from one model forward and preserve the convention `requested L -> hidden_states[L + 1]`. Save each layer in a separate cache namespace/path.
- Independent probe trainings can run one per physical GPU. With `CUDA_VISIBLE_DEVICES=1`, pass `--device cuda:0` because the selected physical GPU is remapped to logical device zero.
- After feature extraction completes, always use parallel GPU probe training when there are independent probe jobs (for example, separate model/layer combinations): launch one worker per available physical GPU, pin each worker with `CUDA_VISIBLE_DEVICES`, and pass `--device cuda:0` inside each worker's remapped environment. Keep outputs isolated per worker and aggregate only after all workers finish.

## Environment

- Use the existing project environment unless instructed otherwise.
- Do not install, remove, or upgrade dependencies without asking.
- If a documented command uses the `vlm3r` conda environment, first verify that the environment exists locally rather than recreating it automatically.

## Local Storage Layout

### Code and Active Working Cache: NVMe / Root Filesystem

- Repository:
  `/home/shaoruei/SpatialFocus`
- Recommended regenerated probing/hidden-feature cache:
  `/home/shaoruei/probe_cache`
- Recommended small/derived probe outputs:
  `/home/shaoruei/probe_outputs`

`/home/shaoruei` is on the root NVMe filesystem (`/dev/nvme0n1p3`). It is fast and appropriate for active probing caches, but it also contains the operating system. Do not fill it to capacity. Keep substantial system headroom; as a working rule, leave roughly 200-300 GiB free unless the user explicitly decides otherwise.

### Canonical Migrated Inputs: SATA SSD

Use `/mnt/DATA_SSD/shaoruei` for persistent migrated models and input data.

Current layout:

```text
/mnt/DATA_SSD/shaoruei/
├── models/
│   ├── base/
│   └── vlm3r_runs/
└── probing_data/
    ├── probe_targets_2f_v1/
    ├── forward_frames_32_v1/
    └── cut3r_features/
```

### Base Models

- LLaVA base model:
  `/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2`
- SigLIP vision tower:
  `/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384`

### Experiment Checkpoints

Experiment checkpoints live under:

`/mnt/DATA_SSD/shaoruei/models/vlm3r_runs`

The default VLM3R baseline model checkpoint is:

`/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/Reproduction_2`

Unless an experiment explicitly names a different baseline replicate, treat
`Reproduction_2` as the canonical VLM3R baseline checkpoint.  Do not silently
substitute `baseline_apr05_reproduction`, `selec_100pct_baseline_40390735`, or
another checkpoint when a task refers only to the VLM3R baseline.

Migrated runs currently include:

- `Reproduction_2`
- `zero_spatial_features`
- `cut3r_depth_loss_43817021`
- `cut3r_bev_loss_8n32g_42837152`
- `cut3r_spatialstack_44323703`
- `cut3r_spatialstack_cross_attn_45303862`
- `cut3r_spatialstack_d2_pointmap_45457911`
- `cut3r_spatialstack_token_mlp_dec6_9_12_llm0_3_6_47029970`
- `cut3r_spatialstack_token_mlp_dec6_9_12_llm1_2_3_4n`

Each migrated run contains the five required root files:
`adapter_model.bin`, `non_lora_trainables.bin`, `adapter_config.json`, `config.json`, and `generation_config.json`.

Do not assume other historical Leonardo checkpoints are present locally. In particular, the historical Geo-RoPE run was not migrated because its expected source directory was missing on Leonardo.

## Probe Dataset Semantics

The authoritative probe set contains 2,400 videos:

- ScanNet: 1,199
- ScanNet++: 854
- ARKitScenes: 347

The model forward pass uses 32 sampled RGB frames per video, while the current probe target/evaluation uses only 2 selected frame positions per video. Do not conflate these two quantities.

### 32-Frame RGB Forward Cache

Local root:

`/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1`

- Contains exactly 2,400 cache files.
- Each video cache represents the exact 32 decoded RGB frames used by the authoritative probing preprocessing.
- RGB frames are stored losslessly as `uint8` arrays/tensors.
- This is a decoded-frame cache, not the original MP4 dataset tree.
- Old code that directly constructs `VideoReader(...mp4...)` paths cannot simply be pointed at this directory. Use or implement the cache-aware loading path when running from these migrated inputs.

### 2-Frame Probe Targets

Local root:

`/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1`

- Contains exactly 2,400 compact target files / 4,800 selected target frames.
- These files contain only the 2 selected probe frames per video.
- Typical compact schema includes:
  `point_maps_cam`, `point_maps_ref`, `camera_pose`, `selected_frame_indices`, dataset/source metadata, and provenance.
- These are probing/evaluation targets. They are **not** a replacement for a full 32-frame `spatial_features_points` sidecar when a model architecture itself requires point-map geometry as an input.
- Before running a checkpoint whose forward path requires full point-map sidecars, verify that the required full-resolution/full-frame geometry input has been migrated. It is not part of `probe_targets_2f_v1`.

## Preextracted Spatial Sidecars

- Extraction utilities live under `scripts/extraction/`, for example:
  `python scripts/extraction/extract_cut3r_point_maps.py`.
- The provenance scripts are mainly under `logs/chore/` and `logs/chore/archived/`. Treat those as historical records of what was extracted.

### CUT3R Decoder-Layer Features

All migrated CUT3R token sidecars share the expected `.pt` dictionary schema with `camera_tokens`, `patch_tokens`, and sometimes `metadata`.

- `camera_tokens`: frame-level CUT3R camera-token data.
- `patch_tokens`: 729-token spatial patch grid.
- Feature dimension: 768.

Local root:

`/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features`

For each of ScanNet, ScanNet++, and ARKitScenes, the migrated 2,400-video probe subset uses:

- decoder layer 6: `spatial_features_dec_6`
- decoder layer 9: `spatial_features_dec_9`
- decoder layer 12/final: `spatial_features`

Verified local counts:

- dec6: 2,400 `.pt` files
- dec9: 2,400 `.pt` files
- dec12/final: 2,400 `.pt` files

The local tree is therefore conceptually:

```text
/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features/
├── scannet/
│   ├── spatial_features_dec_6/
│   ├── spatial_features_dec_9/
│   └── spatial_features/
├── scannetpp/
│   ├── spatial_features_dec_6/
│   ├── spatial_features_dec_9/
│   └── spatial_features/
└── arkitscenes/
    ├── spatial_features_dec_6/
    ├── spatial_features_dec_9/
    └── spatial_features/
```

For local SpatialStack runs that need layers 6, 9, and 12, prefer a single local root with a mapping equivalent to:

`6:spatial_features_dec_6;9:spatial_features_dec_9;12:spatial_features`

Do not assume decoder `-2` or `-4` features are present locally. They existed in historical Leonardo storage but were not part of the migrated probe subset described here.

### CUT3R Point Maps

Historical CUT3R point-map sidecars use subdir `spatial_features_points` with a schema containing `point_maps_ref`, `point_maps_cam`, `camera_pose`, and `metadata`.

- `point_maps_ref` / `pts3d_in_other_view`: CUT3R reference/anchor-frame coordinates.
- `point_maps_cam` / `pts3d_in_self_view`: per-frame camera coordinates.
- Keep the coordinate source identical between training and evaluation for a checkpoint.

Important local-server distinction:

- The compact local probe target bundle under `probe_targets_2f_v1` contains only the 2 selected probe frames.
- The original full 32-frame point-map sidecar tree from Leonardo has **not** been established as a migrated local input in this file.
- Do not point `GEOMETRY_SPATIAL_FEATURES_ROOT` at the compact 2-frame target bundle unless the code path is explicitly designed for probe targets rather than model-forward geometry.
- If a model-forward path requires full point-map geometry, stop and verify/migrate the correct input rather than silently substituting the compact target bundle.

### Pi3X and VGGT Sidecars

- This file intentionally keeps only CUT3R sidecar details.
- Before modifying Pi3X or VGGT decoded features, point maps, schemas, or loader settings, read `docs/data-sidecars.md`.
- Do not assume historical Pi3X/VGGT sidecars were migrated to this server.
- The archived Pi3X geometry wrapper is:
  `scripts/archived/old_files/old_bash/train/rope/train_geo_rope_fusion_cut3r_pi3x_pos.sh`.

### Spatial Rank Head / P_geo

- `scripts/extraction/extract_spatial_rank_head.py` does not produce dataset sidecars. It extracts `spatial_rank_head.*` weights from a trained checkpoint into a small state dict, often called `p_geo.bin`.
- Use:
  `python scripts/extraction/extract_spatial_rank_head.py --checkpoint <ckpt> --output <p_geo.bin>`.

## Probing Feature Extraction and Cache Policy

### Mandatory feature set for new pre-SFT probes

Every **new** pre-SFT depth probe must extract and evaluate the complete
representation set below in one run:

```text
siglip_output
fusion_output
projected_features
layer_0
layer_1
layer_2
layer_3
layer_6
layer_9
layer_12
layer_15
layer_18
layer_21
layer_24
layer_27
```

This is the canonical pre-SFT policy in
`scripts/probing/probe_layer_policy.py` and is enforced by
`extract_depth_probe_features.py`. Existing caches/results are historical and
are not retroactively repaired. An intentional missing-layer completion or
other legacy partial diagnostic must pass
`--allow-incomplete-pre-sft-features` explicitly and document why.

### Retained Post-SFT Depth-Probe Feature Caches

The long post-SFT depth-probe extractions were completed already.  Their
retained feature tensors are intentionally stored under `/home/shaoruei/probe_outputs/`
(not only under the recommended transient `/home/shaoruei/probe_cache/` root).
Before scheduling any post-SFT re-extraction, inspect and reuse these roots.
Each complete root below contains the authoritative 1,199-video ScanNet probe
population (2,398 selected-frame tensors), with `gt_depth/`, `metadata/`, and
per-feature tensors at `features/<model-label>/<feature-level>/`:

- EoMT object: `/home/shaoruei/probe_outputs/post_sft_eomt_object_full_20260825`
  (`eomt_object`; `fusion_output`, `projected_features`, and L0/1/2/3/6/9/12/15/18/21/24/27).
- EoMT selective, authoritative checkpoint-exact v2:
  `/home/shaoruei/probe_outputs/post_sft_eomt_selective_full_v2_20260831`
  (`eomt_selective`; the same 14 feature levels).  The older
  `/home/shaoruei/probe_outputs/post_sft_eomt_selective_full_20260825` root
  retains only provenance, targets, and probe results; its feature tensors were
  purged.  It must not be used for new formal comparisons.
- Geo-RoPE fusion: `/home/shaoruei/probe_outputs/post_sft_geo_rope_fusion_full_20260823`
  (`geo_rope_fusion`; the same 14 feature levels).
- Visual 3D-RoPE: `/home/shaoruei/probe_outputs/post_sft_visual_3d_rope_full_20260823`
  (`visual_3d_rope`; the same 14 feature levels).

The post-SFT SpatialStack L0/3/6 model
`cut3r_spatialstack_token_mlp_dec6_9_12_llm0_3_6_47029970` is retained across
two compatible cache roots:

- `/home/shaoruei/probe_cache/scannet_ss_add_036_post_sft_all_layers_v1/full`
  contains `siglip_output`, `projected_features`, and L0/1/2/3/6/9/15/21/27.
- `/home/shaoruei/probe_cache/scannet_ss_add_036_post_sft_v1/full` contains the
  complementary L12/18/24 tensors (and duplicate pre-LLM tensors).

The post-SFT zero-spatial checkpoint is also retained at:

- `/home/shaoruei/probe_cache/scannet_depth_layers_v1/full`
  (`zero_spatial`, checkpoint `zero_spatial_features`; `siglip_output`,
  `projected_features`, and L1/2/12/18/24).  Each level contains all 2,398
  selected-frame tensors and uses the authoritative ScanNet split SHA-256
  `d478cb684958dfc25066821ec83d5216469577c9e282e33bdf87d3c88b200d8e`.

The corresponding post-SFT `probes/`, full-data result CSV/JSON, completeness
reports, and extraction provenance remain beside their feature cache roots.
The official ScanNet split is still fixed at 1,006 train videos and 193
validation videos; subsampling experiments must vary only the 1,006-video
training partition at video granularity.

Several other post-SFT experiments retain durable probe metrics/provenance but
not a reusable full feature cache.  Do not mistake their saved `best.pt` or
`metrics.json` files for feature tensors; locate or regenerate a cache only if
the requested work genuinely needs new subsampling fits.

### Retained Pre-SFT Depth-Probe Feature Caches

All complete pre-SFT roots below use the same authoritative 1,199-video
ScanNet population (1,006 train / 193 validation), contain 2,398 selected-frame
tensors per listed feature level, and use split SHA-256
`d478cb684958dfc25066821ec83d5216469577c9e282e33bdf87d3c88b200d8e`.
They use the base checkpoint
`/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2` without a
post-SFT adapter.

- Plain pre-SFT base VLM:
  `/home/shaoruei/probe_cache/scannet_depth_layers_v1/full`
  (`pre_sft_base_vlm`; `siglip_output`, `projected_features`, and
  L0/1/2/3/6/9/12/15/18/21/24/27).
- C1 VLM3R:
  `/home/shaoruei/probe_cache/c1_vlm3r_v1/full`
  (`c1_vlm3r`; L0/1/2/3/6/9/15/21/27).
- C1 SpatialStack additive L0/1/2:
  `/home/shaoruei/probe_cache/c1_additive_v1/full`
  (`c1_spatialstack_add`; L0/1/2/3/6/9/15/21/27).
- C1 SpatialStack additive L0/3/6:
  `/home/shaoruei/probe_cache/c1_ss_add_036/full`
  (`c1_spatialstack_add_036`; L0/1/2/3/6/9/15/21/27).
- C1 SpatialStack additive L1/2/3:
  `/home/shaoruei/probe_cache/c1_ss_add_123/full`
  (`c1_spatialstack_add_123`; L0/1/2/3/6/9/15/21/27).
- C1 SpatialStack cross-attention:
  `/home/shaoruei/probe_cache/c1_ss_cross_attn_v1/full`
  (`c1_spatialstack_cross_attn_v1`; L0/1/2/3/6/9/15/21/27).

The specialized pre-SFT depth-subspace occupancy cache at
`/home/shaoruei/probe_cache/depth_subspace_occupancy_v1` is not a full formal
ScanNet cache.  Its `SS012`, `SS036`, and `SS123` namespaces contain only 48
selected-frame tensors (24 videos) per level and must not be substituted for
the complete roots above.

- New research targets include model/layer combinations that were never extracted on Leonardo. Do not assume historical probing caches cover the current experiment.
- Leonardo GPU budget should not be assumed available for new extraction. New hidden/fusion feature extraction should run on `mps-edu-06` unless the user explicitly changes this plan.
- Historical probing caches serialized only the 2 selected probe-frame outputs even though the forward pass used 32 frames. Preserve this storage-efficient behavior unless an experiment explicitly requires all 32 hidden-state outputs.
- A complete 2,400-video cache for one LLM hidden layer with shape `14 x 14 x 3584` in fp16 is approximately 6.75 GB decimal (about 6.29 GiB).
- A complete 2,400-video fusion-output cache with shape `14 x 14 x 1152` in fp16 is approximately 2.17 GB decimal.
- Prefer rolling/batched probing caches rather than permanently storing every model x every layer combination.
- Recommended working cache root:
  `/home/shaoruei/probe_cache`.
- Keep durable metrics, probe checkpoints, manifests, and compact results; delete or recycle regeneratable hidden-feature caches when space becomes tight.
- Isolate smoke and formal artifacts, for example under `scannet_depth_layers_v1/smoke/` and `scannet_depth_layers_v1/full/`. Never let a formal `--resume` extraction reuse smoke features or smoke provenance.
- A ScanNet L6 parity marker is an execution gate for baseline missing layers. Require matching split identity, selected-frame mapping, checkpoint evidence, layer identity, and exactly `75,656` validation tokens. MAE must be within 5%; AbsRel or δ<1.25 outside 5% produces `PASS_WITH_WARNING` and must not unlock missing-layer execution.
- Preserve the authoritative ScanNet split identity when available: 1,199 videos, 1,006 train, 193 validation, split SHA-256 `d478cb684958dfc25066821ec83d5216469577c9e282e33bdf87d3c88b200d8e`.
- A baseline-layer completion marker is an operational prerequisite for zero-spatial missing-layer jobs; it is not a scientific parity claim about the zero-spatial checkpoint.

## Training / Evaluation Scripts

- The old `scripts/archived/old_files/old_bash/train/train_vsi*.sh` wrappers are legacy entry points.
- The current shared base wrappers are `train_cut3r_Baseline.sh` and `train_cut3r_spatialstack.sh`; avoid changing them unless the task actually requires it.
- Prefer creating or editing dedicated local wrapper scripts for new experiments.
- Keep train/eval wrapper names descriptive, for example:
  `train_<feature>_<backbone>.sh`
  `eval_<feature>_<benchmark>.sh`
- When adapting a Leonardo wrapper, remove scheduler directives and replace old absolute paths with the local paths in this file. Do not change experiment semantics merely to make a wrapper local.
- Do not assume old GPU-count, batch-size, precision, or distributed-launch settings remain valid on 2 x TITAN V 12 GiB.

## Geometry / RoPE Design

- Before modifying geometry projection, GeoRoPE, or their training/evaluation configuration, read `docs/designs.md`.

## Successful Local VSiBench Run

The local VSiBench evaluator runs directly from the migrated MP4 files. Use the
`vsibench` conda environment, not the historical Leonardo/Slurm wrappers:

```bash
cd /home/shaoruei/SpatialFocus
CONDA_ENV=vsibench \
RUN_NAME=vsibench_reproduction2_mp4_full_YYYYMMDD_HHMM \
scripts/eval/run_vsibench_local_mp4.sh
```

Before a full run, use a one-sample smoke test:

```bash
CONDA_ENV=vsibench LIMIT=1 \
RUN_NAME=vsibench_mp4_smoke \
scripts/eval/run_vsibench_local_mp4.sh
```

The known-good configuration exposes both TITAN V GPUs (`CUDA_VISIBLE_DEVICES=0,1`)
but launches one sharded Accelerate process (`--num_processes 1`), with
`--batch_size 1`, `max_frames_num=32`, and PyAV MP4 decoding. The loader uses
`SPATIALFOCUS_CPU_MERGE_LORA=1` and the tested asymmetric CPU-merge budgets
(`6GiB,10GiB`). Model weights are split across the two GPUs; this is not a
dataset-sharded/DDP run.

The evaluator expects 288 MP4s at
`/mnt/DATA_SSD/shaoruei/probing_data/vsibench_test/{arkitscenes,scannet,scannetpp}/`
and CUT3R token sidecars at
`/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features/{dataset}/spatial_features/`.
The preflight should report 5,130 prompts, 288 unique videos, and zero missing
MP4s or sidecars. Runtime preparation enables sidecar-only CUT3R fusion and
last-token-only logits, so a local CUT3R source checkout and full-vocabulary
logits for every video token are not required.

Logs are written to `logs/vsibench_local/<run-name>/launch.log` (and the
lmms-eval output directory below that run directory). A successful three-sample
timing smoke test averaged 6.612 seconds per prompt (3.741 seconds MP4 loading
and 2.863 seconds generation), projecting about 9.42 hours for all 5,130
prompts. Reserve roughly 10--12 hours for a full run, since cold startup and
CPU LoRA merging add a few minutes.

The repository does not require a local VSiBench run for routine work. Treat
`/home/shaoruei/SpatialFocus/VSI result.csv` as the primary static reference
for model VSiBench scores, and consult it when a model's score is needed.
`/home/shaoruei/SpatialFocus/post-sft-result-for-codex.xlsx` is supplementary
reference material. Do not infer or fabricate scores from unrelated logs, and
do not update these reference files unless the user explicitly provides or
requests a score update.

## Verification

- For Python changes, run:
  `python -m py_compile <changed files>`.
- For geometry projection changes, run:
  `conda run -n vlm3r python tests/test_metric_grounded_geometry_projection.py`, if the `vlm3r` environment and required local assets are available.
- For shell wrapper changes, run:
  `bash -n <changed scripts>`.
- For local GPU/extraction changes, perform a minimal 1-3 sample smoke test before the full 2,400-video run.
- If a dependency, environment, model input, or migrated asset is unavailable, report that clearly instead of silently skipping or substituting a different source.

## Git

- Never revert user changes unless explicitly asked.
- Before beginning implementation work, update the current branch with `git pull --rebase`. If uncommitted work prevents a safe rebase, do not stash, discard, or overwrite it; report the blocker and resolve it safely before implementing.
- Do not commit every small edit. After completing a substantial, coherent implementation phase—for example, an end-to-end pipeline that has passed its relevant verification/smoke tests—stage only the files changed by this chat, create one Conventional Commit, and push it to the current upstream branch.
- Stage and commit only files changed by this chat; do not include pre-existing user or other-agent changes.
- When asked to commit, commit relevant files together and split unrelated changes into separate commits.
- Use Conventional Commits:
  `<type>[optional scope]: <description>`.

## Legacy Leonardo Provenance

Historical configs, logs, and scripts may still contain paths beginning with `/leonardo_work`, `/leonardo_scratch`, or `/leonardo/home`. Treat these as provenance unless the user explicitly asks to work on Leonardo again.

Common historical-to-local mappings include:

```text
Leonardo base/checkpoint storage
    -> /mnt/DATA_SSD/shaoruei/models/

Leonardo selected probe point-map targets
    -> /mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1/

Leonardo selected video inputs / decoded forward frames
    -> /mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1/

Leonardo CUT3R dec6/dec9/final sidecars for the 2,400 probe videos
    -> /mnt/DATA_SSD/shaoruei/probing_data/cut3r_features/
```

Do not use old Leonardo paths as active runtime defaults on `mps-edu-06`. Resolve them to the local paths above or explicitly report that the required asset was not migrated.

## Unmounted Storage

- `/dev/sdb1` is an approximately 7.3 TiB ext4 partition on an 8 TB Seagate drive and is currently not part of the approved project storage layout.
- Do not mount, format, repartition, write to, or otherwise use `/dev/sdb1` unless the user explicitly requests it and its intended ownership/content has been verified.
