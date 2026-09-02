# Agent Instructions

This checkout is operated on **Snellius**, not `mps-edu-06`. These Snellius
rules take precedence over historical local-server and Leonardo wrappers.

## Repository Rules

- Check `git status --short` before editing. Do not overwrite unrelated user
  changes or edit `third_party/` unless explicitly asked.
- Do not put model weights, datasets, Hugging Face caches, or regenerated
  feature caches in Git.
- Prefer small, experiment-specific wrappers/configuration over invasive
  training-script changes.
- Do not install, remove, or upgrade dependencies without user approval unless
  explicitly instructed.
- Report a missing asset, manifest, or initialization rather than silently
  substituting another dataset or checkpoint.

## Snellius Runtime

- GPU work must be submitted through Slurm. Do not run model
  forward/backward, training, evaluation, or extraction on a login node.
- The login host observed for this checkout is `int5.local.snellius.surf.nl`;
  it did not expose `nvidia-smi`. Discover the allocated GPU from inside the
  job instead of treating the login node as a GPU capability check.
- `gpu_a100`, `gpu_h100`, `gpu_mig`, and `gpu_vis` were visible through
  `sinfo`. At inspection time, `gpu_a100`, `gpu_h100`, and `gpu_vis`
  had their correspondingly named partition QoS. Do not assume that this
  grants a particular project allocation, account, or QoS entitlement: obtain
  project-specific submission fields from the user or current Slurm policy.
- Do not reuse mps, Leonardo, or another cluster's scheduler, GPU-count,
  VRAM, precision, device-map, or launcher assumptions. In particular, do not
  inherit TITAN-V two-GPU sharding, `CUDA_VISIBLE_DEVICES=0,1`, or CPU LoRA
  merge/offload recipes.
- Before a materially new GPU configuration, submit the smallest meaningful
  Slurm smoke job. Record actual GPU model/VRAM inside it. Start the current
  proxy with one GPU and one process; add DDP or model sharding only when the
  experiment explicitly requires it.
- State the intended `sbatch` command, requested resources, output directory,
  and log path before a long run. Do not stop a job you did not start without
  explicit user authorization.

## Storage Layout

Resolve paths in the job environment; do not copy legacy absolute paths into
new wrappers.

- `SPATIALFOCUS_ROOT`: code checkout. Derive it with `git rev-parse
  --show-toplevel`; the inspected checkout resolved to
  `/home/shuang/SpatialFocus`.
- `MODEL_ROOT` and `DATA_ROOT`: verified persistent project storage for base
  models, sidecars, and immutable inputs. Their project-specific locations
  were not established by this checkout, so require them as configuration.
- `C1_ROOT`: persistent immutable C1 calibration artifacts and manifests.
- `PROXY_OUTPUT_ROOT`: persistent durable results, provenance, compact
  reports, and checkpoints.
- `SCRATCH_ROOT`: job-local/shared scratch for regeneratable intermediates,
  feature caches, temporary working directories, and Slurm logs. The
  `/scratch-shared` root resolves to `/gpfs/scratch1/shared`; an existing
  project-named directory was observed at `/gpfs/scratch1/shared/fshuang`,
  but validate access and retention policy before use.
- `HF_HOME`: scratch-backed Hugging Face cache, preferably under
  `${SCRATCH_ROOT}/hf_cache`; it must not be the only copy of an important
  artifact.

The `/project` root was not present in this login environment. `/home` maps
to the GPFS admin filesystem and `/scratch-shared` to shared scratch. Keep
code and small durable metadata out of large scratch caches; never assume
scratch is the sole durable location. Do not transfer data or models merely to
satisfy a path convention.

## Pre-SFT Proxy Scientific Contract

The current research question is **pre-SFT architecture screening**:

> Can low-cost signals available before architecture-specific SFT predict the
> final post-SFT VSI-Bench ranking of 3D-information injection mechanisms?

The formal proxy state is:

```text
pretrained base VLM
+ pretrained spatial encoder / required spatial inputs
+ candidate-specific verified pre-SFT initialization or C1 calibration
+ fresh trainable modules/LoRA required by the current proxy protocol
+ supervised QA calibration loss
+ forward/backward
+ no optimizer update
```

Allowed: base pretrained VLM, pretrained CUT3R/spatial encoder, deterministic
candidate initialization, C1 calibration, QA labels, forward/backward passes,
and gradient measurement.

Forbidden: candidate post-SFT `adapter_model.bin`, post-SFT
`non_lora_trainables.bin`, trained LoRA, trained fusion/projector parameters,
candidate-specific SFT checkpoint state, and any optimizer update.

**“Pre-SFT” refers to checkpoint/model state, not to absence of labels or
gradients.** A trained VLM3R reproduction must never silently stand in for the
pre-SFT Baseline. A checkpoint diagnostic may be retained only under a clearly
separate post-SFT label and must not be pooled with the formal proxy study.

## Checkpoint Provenance Guard

Before every formal proxy run, record and verify:

- base-model identity;
- candidate initialization/C1 artifact identity and hash;
- that no post-SFT adapter or trained non-LoRA state was loaded;
- that no optimizer was created or stepped;
- current Git commit; and
- fixed calibration sample/manifest identity.

If a candidate's exact pre-SFT initialization is unavailable or ambiguous,
mark it unavailable and ask for direction. Do not reconstruct it from trained
weights or guess a plausible substitute. Historical post-SFT checkpoint
defaults are overridden by this section for proxy work.

## Proxy Loss, Definitions, and Scope

The primary architecture-only comparison uses the same supervised QA backward
for every candidate:

```text
L_proxy = L_QA
```

An auxiliary-loss score, `L_QA + lambda * L_depth`, is a separately labelled
recipe-matched/supervision-specific analysis. Do not silently mix it into the
architecture-only benchmark.

- GradNorm: sum over selected parameter tensors of `||gradient||_2`.
- SNIP: sum over selected scalar parameters of `|theta * gradient|`.
- Fisher: sum over selected scalar parameters of `gradient^2`.

No optimizer step is performed. For each selected scope, report both the
selected parameter count and the gradient-covered parameter count (plus any
uncovered group). A formal run requires expected gradient coverage for every
intended selected group.

The current pre-SFT script dynamically derives rank-128 historical LoRA scope
and candidate fusion scope; the old approximate counts (LoRA 322,961,408,
fusion 9,747,456, projector 16,980,992, total 349,689,856) were not verified
against this Snellius checkout and are not acceptance values. The Baseline
smoke must establish and record actual selected and gradient-covered counts.

## Numerical Migration and Offload Rules

The first migration smoke changes as little as possible beyond device
placement. Record GPU model, GPU VRAM, PyTorch, CUDA runtime, Transformers,
PEFT, and Accelerate versions; model dtype; TF32 states
(`torch.backends.cuda.matmul.allow_tf32` and
`torch.backends.cudnn.allow_tf32`); attention implementation; gradient
checkpointing; and any CPU/meta offload.

Do not opportunistically switch to BF16, TF32, FlashAttention, `torch.compile`,
a different attention backend, or different gradient checkpointing before
validating the migrated Baseline. Afterwards, a common documented setting may
be adopted for all candidates.

For the initial one-GPU proxy smoke, the intended trainable parameter set must
be GPU-resident. Do not inherit an Accelerate CPU/meta-offload device map from
the TITAN-V setup. If offload is genuinely required, stop and explicitly
assess its effect on gradient coverage and proxy semantics before continuing.

## Calibration, Architecture, and Probe Conventions

- Reuse the exact calibration sample, preprocessing, seed, and manifest from
  the validated migration smoke. Every candidate in a comparison uses the same
  fixed calibration set.
- A historical reference names `scene0384_00`, 32 RGB frames, and seed 42,
  but use those values only when the mps migration manifest verifies them.
  Do not invent tokenization, label, or shape statistics.
- The pilot roster is: Base VLM/zero-spatial; VLM3R Baseline; SpatialStack
  0/1/2, 1/2/3, and 0/3/6; SpatialStack cross-attention; and selective fusion.
  It remains a pilot until all intended mechanisms are included.
- A candidate that changes the pre-SFT forward representation needs its own
  pre-SFT representation probe. Representation-probe reuse is allowed for an
  auxiliary-head/loss-only change only after numerical forward equivalence;
  its gradient proxies may still differ under a changed loss.
- Preserve project-wide input semantics: the model forward uses 32 RGB frames;
  compact two-frame targets are not model-forward geometry. A geometry model
  requiring point maps must receive verified full-frame sidecars. For retained
  hidden-state caches, requested LLM layer `L` maps to `hidden_states[L + 1]`.

## Smoke Before Formal Sweep

Run exactly one minimal Slurm migration validation before the seven-model
sweep, starting with the VLM3R Baseline:

- one fixed calibration minibatch and one supervised QA forward/backward;
- no optimizer;
- finite GradNorm, SNIP, and Fisher;
- actual expected selected-parameter and gradient-coverage counts;
- evidence of no post-SFT state and no unintended CPU/meta trainable tensor;
- peak GPU memory and runtime.

Do not turn this into a large test suite. Once this smoke passes, apply the
same protocol to the remaining candidates.

## Logging and Environment

Every formal result records timestamp/run ID, Git commit, hostname/cluster,
Slurm job ID, GPU model, software versions, dtype, TF32 state, base and C1
identities, calibration IDs, seed, loss definition, selected and
gradient-covered parameter counts, all three proxy scores, peak memory,
runtime, and confirmation of zero optimizer steps. Keep logs/provenance
compact; do not dump large tensors or per-layer diagnostics without a debugging
need.

Use the existing project environment if it exists. Modules and Conda were
available on the inspected login node; visible Conda environments were `base`
and `vsibench`, not `vlm3r`. Inspect the migration environment manifest
before recreating anything. Treat PEFT version as scientifically relevant
because it can alter LoRA target/module construction and selected counts.

## Verification and Git

- Python changes: `python -m py_compile <changed files>`.
- Shell changes: `bash -n <changed scripts>`.
- Cluster-specific wrappers: validate with the minimal Slurm smoke, never GPU
  code on the login node.
- Preserve unrelated changes. Do not automatically commit machine-specific
  data or migration artifacts. If this file is the only intended change, keep
  the diff limited to `AGENTS.md`.

## Other-machine Provenance

Old mps/TITAN-V, direct-execution, `/mnt/DATA_SSD`, CPU-LoRA-merge, and local
VSiBench instructions are historical provenance only. They are not active
Snellius runtime guidance.
