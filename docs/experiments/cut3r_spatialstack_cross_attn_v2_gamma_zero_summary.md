# CUT3R SpatialStack V2 Gamma-Zero Eval Ablation

## What changed

- Added `cut3r_spatialstack_cross_attn_v2_force_zero_gamma_at_eval`.
- Default is `False`, so normal V2 training and inference behavior is unchanged.
- When the flag is `True` and the V2 block is in eval mode, the learned
  `gamma_attn` and `gamma_mlp` parameters remain intact but the runtime
  effective values are both `0.0`, making the V2 residual delta zero.
- Additive SpatialStack and old `cross_attn` do not read this flag.

## How to run

Normal V2 eval uses the existing V2 checkpoint/config with the flag disabled.

Gamma-zero eval:

```bash
sbatch eval_cut3r_spatialstack_cross_attn_v2_gamma_zero_vsibench.sh
```

Override the checkpoint if needed:

```bash
TRAIN_OUTPUT_DIR=/path/to/v2/final/checkpoint \
PRETRAINED_LOCAL=auto \
sbatch eval_cut3r_spatialstack_cross_attn_v2_gamma_zero_vsibench.sh
```

## Comparison enabled

- Baseline checkpoint: original model.
- V2 normal: trained V2 checkpoint with normal learned gamma at inference.
- V2 gamma-zero: same trained V2 checkpoint with V2 inference injection disabled.

This isolates the contribution of inference-time V2 geometry injection from
training-side changes in the V2 checkpoint.

## Logging

When V2 stats are collected, logs include the flag, learned gamma values,
effective gamma values, and delta norm.
