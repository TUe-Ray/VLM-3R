# Snellius Runtime and Storage Instructions

Read this file when hostname matches *.snellius.surf.nl. It supplies
machine-specific runtime/storage instructions and never relaxes the shared
scientific contract in the root AGENTS.md.

## Slurm Runtime

- GPU work must use Slurm. Do not run model forward/backward, training,
  evaluation, extraction, or a proxy smoke on login nodes.
- Snellius login hosts match *.local.snellius.surf.nl (for example int5 or
  int6); do not depend on a specific login-node hostname. Login nodes need not
  expose nvidia-smi. Discover and record the actual allocated GPU model and
  VRAM inside every GPU job instead.
- The current full-A100 proxy experiment uses the gpu_a100 partition. Request
  its project/account fields only when supplied by the user or required by
  current Slurm policy; do not invent an account. gpu_a100, gpu_h100, gpu_mig,
  and gpu_vis were visible at inspection time, but they are not interchangeable
  experimental defaults.
- Before a material GPU configuration, submit the smallest meaningful Slurm
  smoke. State the intended sbatch command, requested resources, output root,
  and log location before a long run.
- Start the proxy migration with one GPU and one process. Do not add DDP,
  data parallelism, or model sharding merely because multiple GPUs exist.
- Do not reuse mps/Leonardo scheduler directives, TITAN V VRAM assumptions,
  CUDA_VISIBLE_DEVICES=0,1 sharding, CPU LoRA merge, or Accelerate CPU/meta
  offload placement.

## Intended Home Layout

The repository is code only. Do not put datasets, models, C1 artifacts, proxy
outputs, feature caches, or Hugging Face caches inside it.

    SPATIALFOCUS_ROOT=/home/shuang/SpatialFocus
    MODEL_ROOT=/home/shuang/models
    DATA_ROOT=/home/shuang/probing_data
    C1_ROOT=/home/shuang/c1_artifacts
    PROXY_OUTPUT_ROOT=/home/shuang/proxy_outputs
    HF_HOME=/home/shuang/hf_cache

Roles:

- SPATIALFOCUS_ROOT is the checkout only.
- MODEL_ROOT holds immutable base VLM and SigLIP model directories.
- DATA_ROOT holds transferred calibration RGB/CUT3R/QA/provenance inputs.
- C1_ROOT holds byte-identical C1 calibration artifacts and their provenance.
- PROXY_OUTPUT_ROOT holds durable results, metadata, logs, and compact reports.
- HF_HOME is the Hugging Face cache and must not be the only copy of a
  scientific artifact.

At inspection, only /home/shuang/SpatialFocus existed; the five intended
asset/output directories were absent and must not be populated during routine
documentation work. /home/shuang resolves to /gpfs/home3/shuang. The
/scratch-shared root resolves to /gpfs/scratch1/shared, but shared scratch is
for regeneratable temporary data and not the only copy of durable output.

The mandatory one-minibatch five-candidate migration is 18.59 GiB according
to docs/snellius_proxy_migration_manifest.md. The available quota command
returned no user-specific quota information and lfs/mmlsquota helpers were
not available. GPFS-wide df capacity is not evidence of personal quota.
Before transferring the 18.59 GiB set, obtain/record a usable home quota and
current usage, then confirm enough headroom for inputs, outputs, and HF cache.
Do not create or transfer large assets until that gate passes.

## Authoritative Migration Audit

docs/snellius_proxy_migration_manifest.md and
docs/snellius_proxy_migration_files.txt are authoritative mps-to-Snellius
migration audit records. Preserve them and do not alter their scientific
conclusions without a concrete error.

- Use the manifest transfer table and path-remapping instructions; do not
  invent an alternate Hub source or omit model shards.
- Preserve C1 JSON byte-for-byte. Map/configure its location rather than
  editing absolute provenance strings inside scientific artifacts.
- The compact target tree is not an input to the supported proxy: the
  current code validates that its root exists. Follow the manifest rather than
  transferring the full target tree.
- Never transfer/load post-SFT adapter_model.bin, non_lora_trainables.bin,
  trained fusion/projector state, trained LoRA, or candidate SFT checkpoints
  for the formal pre-SFT proxy.

## Numerical Migration Smoke

The initial gpu_a100 Baseline smoke should change only device placement from
the validated mps result. Record:

- GPU model and VRAM; hostname, Slurm job ID, and allocation;
- PyTorch, CUDA runtime, Transformers, PEFT, and Accelerate versions;
- model dtype; torch CUDA matmul/cudnn TF32 states; attention implementation;
  gradient-checkpointing state; and CPU/meta offload state;
- base-model, C1 artifact, calibration manifest identities, Git commit, seed,
  selected parameter count, gradient-covered count, peak memory, runtime,
  proxy loss, and confirmation of zero optimizer steps.

Do not opportunistically switch to BF16, TF32, FlashAttention, torch.compile,
a different attention backend, or different gradient checkpointing before the
Baseline validates. After validation, any common setting must be documented
and held fixed across candidates.

For the first one-GPU smoke, every intended trainable parameter must be
GPU-resident. Do not inherit the TITAN-V auto device map or CPU/meta offload
strategy. If memory requires offload later, stop and assess its impact on
gradient coverage and proxy semantics before proceeding.

## Verified Proxy Facts

The migration manifest establishes the following, rather than historical
estimates:

- Formal C1 roster: exactly five candidates: C1 VLM3R Baseline; SpatialStack
  additive 0/1/2, 1/2/3, 0/3/6; and SpatialStack cross-attention 0/1/2.
  Do not silently add Base/zero-spatial, selective fusion, geometry, or
  post-SFT candidates to the formal evaluator.
- Baseline smoke calibration: ScanNet scene0384_00; 32 cached RGB frames;
  input shape [1,424]; 13 supervised labels; seed 42.
- PEFT version 0.4.0.
- Baseline SFT-trainable scope: 349,689,856 parameters, made of LoRA
  322,961,408, C1 VLM3R fusion 9,747,456, and mm_projector 16,980,992.

Use the fixed manifest/calibration input for every candidate in a comparison.
The model forward uses 32 RGB frames; compact two-frame targets are not
model-forward geometry. Full point-map sidecars are required when an
architecture forward path asks for them.

## Environment

Use the existing project environment. Modules, Conda base, and Conda vsibench
were visible on the inspected login node; vlm3r was not. Do not recreate,
install, remove, or upgrade dependencies automatically. Consult the migration
manifest environment section before recreating vlm3r. PEFT 0.4.0 is
scientifically relevant because it controls LoRA target/module construction,
initialization, and the selected trainable count.

The manifest reports mps source versions including Python 3.10, Torch 2.1.1,
Transformers 4.40.0.dev0, PEFT 0.4.0, and Accelerate 0.29.1. Treat them as
migration comparison values, not as a license to install or change the
Snellius environment without approval.
