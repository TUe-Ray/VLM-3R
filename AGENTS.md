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
