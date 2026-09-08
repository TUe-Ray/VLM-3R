# mps-edu-06 Runtime and Storage Instructions

Read this file when hostname is exactly mps-edu-06. It supplies
machine-specific runtime and storage instructions and never relaxes the shared
scientific contract in the root AGENTS.md.

This preserves the detailed mps-edu-06 operational content that was formerly
in the root AGENTS.md before the Snellius documentation split.

## Local Server Runtime

### Hardware

- Host: mps-edu-06; OS: Ubuntu 22.04.
- CPU: AMD Ryzen Threadripper 2950X, 16 physical / 32 logical cores.
- RAM: about 62 GiB; swap: about 119 GiB.
- GPUs: 2 x NVIDIA TITAN V, 12 GiB VRAM each.
- Root/NVMe: Samsung SSD 970 EVO 2 TB; data SSD: Samsung SSD 860 EVO 2 TB at
  /mnt/DATA_SSD.

### Direct execution; no Slurm

- Do not use sbatch, srun, Slurm partitions, QoS settings, or Leonardo job
  directives. Run commands directly from the shell.
- Use CUDA_VISIBLE_DEVICES to select GPUs. Use the local distributed launcher
  or torchrun only when needed; do not reuse Leonardo GPU-count assumptions.
- Existing sbatch and Leonardo wrappers are historical references. Do not
  submit them directly: reuse their underlying command or make a small local
  wrapper.
- Before material training/evaluation/extraction changes, use a small smoke.
  Confirm imports, paths, data loading, tensor shapes, output writing, and the
  first expected progress signal.
- Before a long run state the command, GPU selection, output and log path.
  Write explicit experiment logs. Do not stop a process you did not start
  without user authorization. Diagnose transient filesystem/GPU/process
  failures before editing code.

### TITAN V compatibility and two-GPU placement

- TITAN V is Volta with 12 GiB VRAM. Do not assume Leonardo bf16, four-GPU, or
  large-batch settings work. Begin with a one-GPU smoke before scaling.
- The proven VLM extraction is one sharded process over the whole manifest
  with weights split across both GPUs, not dataset sharding. Use:

      CUDA_VISIBLE_DEVICES=0,1 SPATIALFOCUS_CPU_MERGE_LORA=1

- CPU merge loads the base and merges LoRA on CPU before Accelerate dispatch.
  Known-good placement: vision tower and decoder 0-9 on logical GPU 0; decoder
  10-27 plus spatial_tower, fusion_block, vision_resampler, and mm_projector
  on logical GPU 1; lm_head and overflow on CPU.
- Keep the complete vision tower on one GPU: nested splitting can break
  list-of-frames forward. CPU offload is intentional on this machine.
- GPU=0 in a wrapper is physical. With CUDA_VISIBLE_DEVICES=1, that physical
  GPU is PyTorch cuda:0; never use logical cuda:1 while only physical GPU 0 is
  exposed.
- Before a long run check nvidia-smi --id="$GPU", then TITAN V, capability
  (7,0), sm_70 in torch.cuda.get_arch_list(), and a small fp16
  allocation/matmul/synchronize test.
- Independent probe fits may use one worker per physical GPU. Pin with
  CUDA_VISIBLE_DEVICES, pass --device cuda:0 after remapping, isolate output,
  and aggregate only after workers finish.

## Environment and Storage

- Use the existing project environment. Do not install, remove, or upgrade
  dependencies without approval. Verify vlm3r exists before recreating it.
- Repository: /home/shaoruei/SpatialFocus.
- Regeneratable hidden/probing cache: /home/shaoruei/probe_cache.
- Derived probe outputs: /home/shaoruei/probe_outputs.
- Home is root NVMe with the OS. Leave about 200-300 GiB free unless the user
  decides otherwise.

Persistent migrated models and inputs are under /mnt/DATA_SSD/shaoruei:

    models/
      base/
      vlm3r_runs/
    probing_data/
      probe_targets_2f_v1/
      forward_frames_32_v1/
      cut3r_features/

Base paths:

- /mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2
- /mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384

Checkpoints are under /mnt/DATA_SSD/shaoruei/models/vlm3r_runs. For an
unspecified ordinary VLM3R baseline, Reproduction_2 is canonical; do not
silently substitute baseline_apr05_reproduction,
selec_100pct_baseline_40390735, or another run. Migrated runs are:

- Reproduction_2; zero_spatial_features; cut3r_depth_loss_43817021;
  cut3r_bev_loss_8n32g_42837152.
- cut3r_spatialstack_44323703; cut3r_spatialstack_cross_attn_45303862;
  cut3r_spatialstack_d2_pointmap_45457911.
- cut3r_spatialstack_token_mlp_dec6_9_12_llm0_3_6_47029970 and
  cut3r_spatialstack_token_mlp_dec6_9_12_llm1_2_3_4n.

Each migrated run has adapter_model.bin, non_lora_trainables.bin,
adapter_config.json, config.json, and generation_config.json. These are
post-SFT artifacts; their permissibility is governed by the root scientific
contract. Do not assume other historical Leonardo checkpoints, including
Geo-RoPE, were migrated.

## Probe Inputs and Sidecars

The authoritative probe set is 2,400 videos: ScanNet 1,199, ScanNet++ 854,
and ARKitScenes 347. Model forward uses 32 RGB frames; targets use two selected
frame positions. Never conflate them.

- /mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1 has exactly 2,400
  lossless uint8 32-frame decoded RGB caches. It is not an MP4 tree; old
  VideoReader mp4 code requires the cache-aware loader.
- /mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1 has 2,400 compact
  files / 4,800 selected target frames. Typical contents are point_maps_cam,
  point_maps_ref, camera_pose, selected indices, source metadata, and
  provenance.
- Compact targets are evaluation/probe targets, not a full 32-frame
  spatial_features_points input. Never point GEOMETRY_SPATIAL_FEATURES_ROOT
  there for model-forward geometry.

### CUT3R

- Extraction utilities are in scripts/extraction. Logs under logs/chore and
  logs/chore/archived are historical extraction provenance.
- CUT3R token pt dictionaries contain camera_tokens, patch_tokens, and
  sometimes metadata; patch grid is 729 tokens, feature dimension 768.
- Root: /mnt/DATA_SSD/shaoruei/probing_data/cut3r_features.
- Across ScanNet, ScanNet++, and ARKitScenes, the migrated subset contains
  2,400 videos total: ScanNet 1,199, ScanNet++ 854, and ARKitScenes 347.
  For each CUT3R decoder feature level—decoder 6 in spatial_features_dec_6,
  decoder 9 in spatial_features_dec_9, and final decoder 12 in
  spatial_features—there are 2,400 pt files total across all three datasets.
- For local SpatialStack 6/9/12 use mapping
  6:spatial_features_dec_6;9:spatial_features_dec_9;12:spatial_features.
  Decoder -2/-4 did not migrate.
- Historical point maps use spatial_features_points with point_maps_ref
  (pts3d_in_other_view), point_maps_cam (pts3d_in_self_view), camera_pose, and
  metadata. Keep coordinate source equal between training and evaluation.
- The full 32-frame point-map tree is not established locally. A forward path
  needing it must stop and verify/migrate it, never substitute compact targets.

Before modifying Pi3X/VGGT sidecars, maps, schemas, or loaders, read
docs/data-sidecars.md. Do not assume their historical sidecars are available.
The archived Pi3X geometry wrapper is
scripts/archived/old_files/old_bash/train/rope/train_geo_rope_fusion_cut3r_pi3x_pos.sh.

The spatial rank-head extraction writes a small p_geo.bin:

    python scripts/extraction/extract_spatial_rank_head.py --checkpoint <ckpt> --output <p_geo.bin>

## Probing Feature Cache Policy

Every new pre-SFT depth probe must extract and evaluate all of:

    siglip_output
    fusion_output
    projected_features
    layer_0 layer_1 layer_2 layer_3 layer_6 layer_9 layer_12
    layer_15 layer_18 layer_21 layer_24 layer_27

This is enforced by scripts/probing/probe_layer_policy.py and
extract_depth_probe_features.py. A legacy partial diagnostic needs
--allow-incomplete-pre-sft-features and an explanation. Preserve requested
layer L -> hidden_states[L + 1], use one forward for multiple layers, and save
separate cache namespaces.

Complete listed caches share 1,199 ScanNet videos, 1,006 train / 193
validation, 2,398 selected-frame tensors, and split SHA-256
d478cb684958dfc25066821ec83d5216469577c9e282e33bdf87d3c88b200d8e.
Subsampling changes only train videos at video granularity.

### Retained post-SFT caches

- /home/shaoruei/probe_outputs/post_sft_eomt_object_full_20260825:
  eomt_object, fusion/projected and L0/1/2/3/6/9/12/15/18/21/24/27.
- /home/shaoruei/probe_outputs/post_sft_eomt_selective_full_v2_20260831:
  checkpoint-exact eomt_selective, same levels. The older
  post_sft_eomt_selective_full_20260825 has no retained features and is not
  valid for new comparisons.
- /home/shaoruei/probe_outputs/post_sft_geo_rope_fusion_full_20260823 and
  post_sft_visual_3d_rope_full_20260823, same coverage.
- SpatialStack L0/3/6 uses
  /home/shaoruei/probe_cache/scannet_ss_add_036_post_sft_all_layers_v1/full
  for SigLIP/projected/L0/1/2/3/6/9/15/21/27 and
  /home/shaoruei/probe_cache/scannet_ss_add_036_post_sft_v1/full for
  L12/18/24.
- /home/shaoruei/probe_cache/scannet_depth_layers_v1/full has post-SFT
  zero_spatial: SigLIP/projected/L1/2/12/18/24.

### Retained pre-SFT caches

All use plain base without post-SFT adapter:

- /home/shaoruei/probe_cache/scannet_depth_layers_v1/full:
  pre_sft_base_vlm, SigLIP/projected and L0/1/2/3/6/9/12/15/18/21/24/27.
- /home/shaoruei/probe_cache/c1_vlm3r_v1/full: c1_vlm3r,
  L0/1/2/3/6/9/15/21/27.
- /home/shaoruei/probe_cache/c1_additive_v1/full:
  c1_spatialstack_add, same C1 coverage.
- /home/shaoruei/probe_cache/c1_ss_add_036/full:
  c1_spatialstack_add_036, same coverage.
- /home/shaoruei/probe_cache/c1_ss_add_123/full:
  c1_spatialstack_add_123, same coverage.
- /home/shaoruei/probe_cache/c1_ss_cross_attn_v1/full:
  c1_spatialstack_cross_attn_v1, same coverage.

depth_subspace_occupancy_v1 is not formal: SS012/SS036/SS123 contain only 48
tensors / 24 videos. New targets may lack historical caches. New hidden/fusion
extraction runs on mps unless changed by the user. Store only two selected
outputs from the 32-frame forward unless needed otherwise. A full 2,400-video
fp16 LLM [14,14,3584] cache is 6.75 GB decimal (6.29 GiB); fusion [14,14,1152]
is 2.17 GB. Prefer rolling caches; retain metrics/manifests/checkpoints;
recycle regeneratable data; isolate smoke and formal artifacts.

The ScanNet L6 parity gate needs matching split, frame mapping, checkpoint,
layer, exactly 75,656 validation tokens, and MAE within 5%. AbsRel or
delta<1.25 outside 5% is PASS_WITH_WARNING and does not unlock missing layers.
A baseline-layer completion marker is operational, not a zero-spatial
scientific parity claim.

## Training, Geometry, and VSiBench

- Legacy scripts/archived/old_files/old_bash/train/train_vsi*.sh wrappers are
  not current. Shared wrappers are train_cut3r_Baseline.sh and
  train_cut3r_spatialstack.sh; avoid changing them unless necessary. Make
  dedicated descriptive local wrappers. When adapting Leonardo, remove
  scheduler directives and map paths without changing semantics.
- Before geometry projection, GeoRoPE, or related configuration changes, read
  docs/designs.md.
- For geometry projection changes, run the following when the vlm3r
  environment and required local assets are available:

      conda run -n vlm3r python tests/test_metric_grounded_geometry_projection.py

Use vsibench and migrated MP4s:

    cd /home/shaoruei/SpatialFocus
    CONDA_ENV=vsibench RUN_NAME=vsibench_reproduction2_mp4_full_YYYYMMDD_HHMM scripts/eval/run_vsibench_local_mp4.sh
    CONDA_ENV=vsibench LIMIT=1 RUN_NAME=vsibench_mp4_smoke scripts/eval/run_vsibench_local_mp4.sh

Known-good VSiBench is one sharded Accelerate process over both TITAN V GPUs,
batch 1, 32 frames, PyAV, CPU LoRA merge and 6GiB,10GiB budgets; it is not DDP.
It expects 288 MP4s under
/mnt/DATA_SSD/shaoruei/probing_data/vsibench_test/{arkitscenes,scannet,scannetpp}/
and final CUT3R sidecars under cut3r_features/{dataset}/spatial_features.
Preflight: 5,130 prompts, 288 videos, no missing input. Logs go under
logs/vsibench_local/<run-name>/launch.log. Three-sample timing was 6.612
seconds/prompt (about 9.42 hours); reserve 10-12 hours.

For routine score lookup use /home/shaoruei/SpatialFocus/VSI result.csv;
post-sft-result-for-codex.xlsx is supplementary. Never fabricate or update
scores without explicit user input.

## mps Historical Provenance and Unmounted Storage

Leonardo paths (/leonardo_work, /leonardo_scratch, /leonardo/home) are
provenance only. Map base/checkpoints to /mnt/DATA_SSD/shaoruei/models/,
targets to probe_targets_2f_v1/, frames to forward_frames_32_v1/, and CUT3R
6/9/final to cut3r_features/. Do not use Leonardo paths as active defaults.

/dev/sdb1 is an approximately 7.3 TB ext4 partition on an 8 TB Seagate drive,
outside approved project storage. Do not mount, format, repartition, write, or
otherwise use it without explicit authorization and verified ownership.
