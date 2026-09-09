# Controlled-fusion B/C/D/E/H pre-SFT extension

This experiment applies the existing C1 and depth-probe protocol to the five
controlled-fusion designs used by the Snellius SFT campaign. It is a separate
extension and does not change or pool with the authoritative five-candidate
formal proxy roster.

## Exact architecture mapping

| ID | CUT3R source | Injection site | Fusion | Projector binding |
|---|---|---|---|---|
| B | decoder 12 patches | before `mm_projector` | additive in 1152-D SigLIP space | source-specific |
| C | decoder 12 patches | before LLM layer 0 | cross-attention V1 | source-specific |
| D | decoder 12 patches | before LLM layer 0 | additive token MLP | source-specific |
| E | decoder 12 patches reused at L0/L1/L2 | before LLM layers 0, 1, 2 | additive token MLP | independent site-specific projectors |
| H | decoder 12 patches reused at L0/L1/L2 | before LLM layers 0, 1, 2 | independent cross-attention V1 blocks | source-specific source mapping |

The canonical definitions live in
`llava/model/controlled_fusion_pre_sft.py`. The pre-SFT path constructs these
modules fresh on the plain base model and never loads a controlled-fusion SFT
checkpoint.

## C1 initialization

All five candidates use the existing fixed 32-video calibration manifest and
base `r0`. Dense maps are deterministic structured isometries, independent of
RNG state. Variant B calibrates its residual in native pre-projector SigLIP
space; C/D/E/H calibrate each native LLM injection site sequentially. Every
artifact records the exact topology and is locked by SHA-256 before probing.

The official artifact validator requires:

- exactly 32 calibration videos;
- the fixed base-r0 and calibration-manifest hashes;
- a median calibrated residual ratio within 5% of `r0` at every site;
- `no_training=true`; and
- explicit proof that no post-SFT checkpoint supplied initialization.

## Complete representation policy

Every new depth probe extracts all 15 required representations in one model
forward:

```text
siglip_output
fusion_output
projected_features
layer_0 layer_1 layer_2 layer_3 layer_6 layer_9 layer_12
layer_15 layer_18 layer_21 layer_24 layer_27
```

The forward consumes 32 RGB frames and CUT3R decoder-12 patch sidecars. Only
the two authoritative selected target frames are serialized. Probe training
uses the fixed 1,006-train/193-validation ScanNet split and seed zero.

## Local execution

On `mps-edu-06`, use the dedicated direct-execution wrapper:

```bash
scripts/probing/run_controlled_fusion_pre_sft_local.sh preflight
scripts/probing/run_controlled_fusion_pre_sft_local.sh calibrate
scripts/probing/run_controlled_fusion_pre_sft_local.sh smoke
scripts/probing/run_controlled_fusion_pre_sft_local.sh full
```

The smoke runs the plain pre-SFT Baseline first, then B/C/D/E/H, with one
train and one validation video. A verified smoke marker gates the full sweep.
The full sweep also starts with a fresh same-commit Baseline, processes one
candidate at a time, trains independent probe levels on both physical GPUs,
preserves compact metrics/provenance, and recycles regenerated feature tensors
after successful fits to maintain NVMe headroom.

Defaults:

- working cache: `/home/shaoruei/probe_cache/controlled_fusion_pre_sft_v1`
- durable artifacts/results: `/home/shaoruei/probe_outputs/controlled_fusion_pre_sft_v1`
- logs: `logs/controlled_fusion_pre_sft_v1`

Do not use the Snellius SFT checkpoints as input. They are downstream targets
and remain forbidden by the shared pre-SFT scientific contract.
