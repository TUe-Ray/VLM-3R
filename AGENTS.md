# Agent Instructions

## Repository Rules

- Check `git status --short` before edits.
- Do not modify user-changed files unless the task requires it.
- Do not edit files under `third_party/` unless explicitly requested.
- Prefer small, experiment-specific wrapper scripts over broad training-script changes.

## HPC / Slurm

- Do not run GPU-heavy training directly on the login node.
- For smoke tests or jobs under 30 minutes, use:
  `#SBATCH --qos=boost_qos_dbg`
  with `#SBATCH --time=00:30:00` or less.
- When submitting smoke-test training or eval jobs, make the Slurm job name start
  with `SMOKE`.
- When submitting debug jobs that use `#SBATCH --qos=boost_qos_dbg`, make the
  Slurm job name start with `DBG`, unless the job is a smoke test that already
  starts with `SMOKE`.
- Before submitting an official Slurm job for a run configuration that has not
  been run before, or whose changes are more than a simple parameter switch,
  submit a `DBG`/`SMOKE` job first. Monitor the debug/smoke job until it is safe
  enough to promote: the job starts, logs are written, there are no immediate
  import, config, path, data-loading, or shape errors, and it reaches the first
  expected progress signal or completes the intended smoke scope. Only then
  submit the official job.
- For multi-node training, make failures fail the whole allocation instead of
  leaving nodes idle. Prefer `srun --kill-on-bad-exit=1 --wait=30`, strict
  preflight checks, and a cleanup trap that self-cancels the current Slurm job
  on training failure.
- For official training or eval runs, remove debug/smoke prefixes such as
  `SMOKE` or `DBG` from the Slurm job name.
- After submitting an official training or eval job, monitor it until it is
  running stably: the job has started, logs are updating, data/file reads are
  succeeding, and the first expected training or evaluation progress appears.
- Training and evaluation jobs can fail because of transient HPC I/O or
  infrastructure issues, including failures to get, open, or read files. Do not
  assume every such failure is a code bug; check logs for infra symptoms and
  consider whether a retry or filesystem recovery is appropriate before changing
  code.
- Do not stop, cancel, kill, or otherwise interrupt any run unless its Slurm job
  name starts with `DBG` or `SMOKE`. For all non-`DBG`/`SMOKE` runs, ask the
  user for explicit permission before using commands such as `scancel`, `kill`,
  or `pkill`, or before changing scripts in a way that would stop the run.
- For normal runs, use:
  `#SBATCH --qos=normal`
- For jobs that do not require GPUs, use the budget-free serial partition:
  `#SBATCH --partition=lrd_all_serial`
  with `#SBATCH --qos=normal`, no GPU request, at most 4 physical cores
  (8 logical cores), and `#SBATCH --time=04:00:00` or less.
- For interactive GPU debugging, use `srun`.
- Before submitting long jobs, show the intended command or script and expected output path.
- Write Slurm logs under `logs/` with experiment-specific names.
- Keep Slurm stdout and stderr separated. Prefer `#SBATCH --output=...%x_%j.out`
  and `#SBATCH --error=...%x_%j.err`; do not merge training stderr into stdout
  with `2>&1 | tee` unless the user explicitly asks for a combined log.

## Environment

- Use the existing project environment unless instructed otherwise.
- Do not install or upgrade dependencies without asking.

## Preextracted Spatial Sidecars

- Extraction utilities live under `scripts/extraction/`. Use that path in new
  Slurm wrappers, for example:
  `python scripts/extraction/extract_cut3r_point_maps.py`.
- The provenance scripts are mainly under `logs/chore/` and
  `logs/chore/archived/`. Treat those as the record of what was extracted.

### CUT3R Decoder-Layer Features

- All CUT3R token sidecars share the same schema: `.pt` dict with
  `camera_tokens`, `patch_tokens`, and sometimes `metadata`.
  `camera_tokens` is frame-level CUT3R camera token data; `patch_tokens` is the
  729-token spatial patch grid. Feature dim is 768.
- Final-layer CUT3R token sidecars (the usual baseline features) use subdir
  `spatial_features`.
- Final-layer locations:
  `/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r/{scannet,scannetpp,arkitscenes}/spatial_features`
  for the FAST copy, and
  `/leonardo_work/EUHPC_D32_006/train_data/vlm3r/{scannet,scannetpp}/spatial_features`
  for the current WORK mirror fallback used by the main training wrappers.
  Older logs may still reference the legacy
  `/leonardo_work/EUHPC_D32_006/FAST/train_data/vlm3r` mirror.
- Intermediate decoder-layer sidecars live under
  `/leonardo_work/EUHPC_D32_006/VLM_3R_cut3r_min2N4_features/{scannet,scannetpp,arkitscenes}/`
  with subdirs:
  `spatial_features_dec_6` for decoder layer `6`,
  `spatial_features_dec_9` for decoder layer `9`,
  `spatial_features_dec_m2` for decoder layer `-2`,
  `spatial_features_dec_m4` for decoder layer `-4`.
- For CUT3R cross-attention or feature-alignment baselines, point
  `SPATIAL_FEATURES_ROOT` at the FAST root
  `/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r`
  or the training mirror
  `/leonardo_work/EUHPC_D32_006/train_data/vlm3r`,
  then set `SPATIAL_FEATURES_SUBDIR=spatial_features`,
  `MODEL_SPATIAL_TOWER=cut3r`,
  `MODEL_SPATIAL_TOWER_SELECT_FEATURE=all_tokens`,
  `MODEL_SPATIAL_FEATURE_DIM=768`.
- For SpatialStack or decoder-layer ablations, point
  `SPATIAL_FEATURES_ROOT=/leonardo_work/EUHPC_D32_006/VLM_3R_cut3r_min2N4_features`
  and choose the matching decoder-layer subdir, or use the multi-layer mapping
  form such as
  `6:spatial_features_dec_6;9:spatial_features_dec_9;12:/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r:spatial_features`.

### CUT3R Point Maps

- CUT3R point-map sidecars use subdir `spatial_features_points`.
- Schema: `.pt` dict with `point_maps_ref`, `point_maps_cam`, `camera_pose`,
  and `metadata`.
- `point_maps_ref` / `pts3d_in_other_view` means CUT3R reference/anchor-frame
  coordinates. `point_maps_cam` / `pts3d_in_self_view` means per-frame camera
  coordinates. Keep the selected coordinate source identical between training
  and evaluation for a checkpoint.
- Verified train/large root:
  `/leonardo_scratch/large/userexternal/shuang00/VLM_3R_cut3r_pointmaps/{scannet,scannetpp,arkitscenes}/spatial_features_points`.
- Verified fast/eval-style root:
  `/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r/{scannet,scannetpp,arkitscenes}/spatial_features_points`.
- Use for CUT3R Metric-Grounded Geometry Projection or GeoRoPE point-map
  geometry with:
  `GEOMETRY_SPATIAL_FEATURES_ROOT=<one of the roots above>`,
  `GEOMETRY_SPATIAL_FEATURES_SUBDIR=spatial_features_points`,
  `GEOMETRY_SPATIAL_TOWER_TYPE=cut3r`.

### Pi3X and VGGT Sidecars

- This file intentionally keeps only CUT3R sidecar details. Before modifying
  Pi3X or VGGT decoded features, point maps, schemas, or loader settings, read
  `docs/data-sidecars.md`.
- The archived Pi3X geometry wrapper is
  `scripts/archived/old_files/old_bash/train/rope/train_geo_rope_fusion_cut3r_pi3x_pos.sh`.

### Spatial Rank Head / P_geo

- `scripts/extraction/extract_spatial_rank_head.py` does not produce dataset
  sidecars. It extracts `spatial_rank_head.*` weights from a trained checkpoint
  into a small state dict, often called `p_geo.bin`.
- Use:
  `python scripts/extraction/extract_spatial_rank_head.py --checkpoint <ckpt> --output <p_geo.bin>`.

## Training / Evaluation Scripts

- The old `scripts/archived/old_files/old_bash/train/train_vsi*.sh` wrappers
  are legacy entry points.
- The current shared base wrappers are `train_cut3r_Baseline.sh` and
  `train_cut3r_spatialstack.sh`; avoid changing them unless the task actually
  requires it.
- Prefer creating or editing dedicated wrapper scripts for new experiments.
- Keep train/eval wrapper names descriptive, for example:
  `train_<feature>_<backbone>.sh`
  `eval_<feature>_<benchmark>.sh`

## Geometry / RoPE Design

- Before modifying geometry projection, GeoRoPE, or their training/evaluation
  configuration, read `docs/designs.md`.

## Verification

- For Python changes, run:
  `python -m py_compile <changed files>`
- For geometry projection changes, run:
  `conda run -n vlm3r python tests/test_metric_grounded_geometry_projection.py`
- For Slurm or shell wrapper changes, run:
  `bash -n <changed scripts>`
- If a dependency or environment is unavailable, report that clearly instead of silently skipping.

## Git

- Never revert user changes unless explicitly asked.
- When asked to commit, commit relevant files together and split unrelated changes into separate commits.
- Use Conventional Commits:
  `<type>[optional scope]: <description>`
