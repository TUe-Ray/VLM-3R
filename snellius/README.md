# Snellius SpatialFocus/VLM-3R smoke pipeline

This directory contains the Snellius-only landing paths, validators, and smoke
wrappers. Source `snellius/paths.env`; do not edit the Leonardo paths embedded
in shared training wrappers just to run on Snellius.

## Current validated path

Use these entry points in order:

1. `snellius/validate_target_bundle.sh`
2. `snellius/validate_migration.sh`
3. `snellius/smoke_environment_native.sbatch`
4. `snellius/smoke_spatialstack_train.sbatch`
5. `snellius/smoke_spatialstack_eval.sbatch`

The shared-node `smoke_spatialstack_train_1gpu.sbatch` and
`smoke_spatialstack_train_2gpu.sbatch` wrappers exercise the same four-record,
one-step training scope through gradient accumulation when a whole node is not
available.

The lower-footprint `smoke_spatialstack_eval_1gpu.sbatch` runs the same
evaluation path on one GPU. `smoke_spatialstack_eval_2gpu.sbatch` additionally
checks the distributed Accelerate path when a whole four-GPU node is slow to
start.

All active smoke jobs request at most 30 minutes and write separate stdout and
stderr files under `/scratch-shared/geusdd/VLM3R/logs/`. Account `tesei2748`
currently exposes only the `normal` QOS on Snellius, so the unavailable
`boost_qos_dbg` is not requested.

The training smoke performs one optimizer step on one four-GPU node. Its
single global batch deliberately covers all three annotation manifests:
ScanNet (two records), ScanNet++ (one), and route-plan (one). It validates the
saved config and all 18 trained SpatialStack branch tensors before atomically
publishing the checkpoint pointer.

The evaluation smoke consumes that pointer by default. Set
`SMOKE_PRETRAINED_LOCAL=/absolute/checkpoint/path` to select a checkpoint
explicitly. A four-rank smoke evaluates four records so every rank has work;
the one-GPU fallback evaluates one. Success requires parseable, non-empty
`results.json` and `vsibench.json` artifacts.

## Historical files

`smoke_environment.sbatch` and `smoke_training.template.sbatch` are retained
as pre-migration records and must not be submitted. They use obsolete
activation and logging behavior.

See `SNELLIUS_AGENT_HANDOVER.md` for completed job IDs, output locations,
known non-blocking environment warnings, and remaining operational risks.
