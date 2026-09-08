# Leonardo Runtime and Storage Instructions

Read this file on CINECA Leonardo login hosts and compute hosts such as
`lrdn*`. It supplies machine-specific runtime and storage instructions and
never relaxes the shared scientific contract in the root `AGENTS.md`.

## Slurm Runtime

- Do not run GPU-heavy training, evaluation, extraction, or model-forward
  diagnostics directly on a login node.
- Use `#SBATCH --qos=boost_qos_dbg` with `#SBATCH --time=00:30:00` or less for
  smoke tests and jobs under 30 minutes. Prefix smoke job names with `SMOKE`
  and other debug jobs with `DBG`.
- Before an official run with a new or materially changed configuration,
  submit the smallest meaningful `DBG`/`SMOKE` job. Monitor it until logs are
  written, inputs load, tensor shapes are valid, and the first expected
  progress signal appears or the intended smoke finishes.
- Use `#SBATCH --qos=normal` for normal runs. For CPU-only jobs, use
  `#SBATCH --partition=lrd_all_serial`, normal QoS, no GPU request, no more
  than four physical cores/eight logical cores, and at most four hours.
- For interactive GPU debugging, use `srun`.
- Multi-node launchers must fail the complete allocation when one worker
  fails. Prefer `srun --kill-on-bad-exit=1 --wait=30`, strict preflight checks,
  and a cleanup trap that self-cancels the current job after training failure.
- Do not stop, cancel, or kill a non-`DBG`/`SMOKE` run without explicit user
  permission. Diagnose transient HPC I/O and filesystem failures before
  treating them as code bugs.
- Before a long run, state the command or wrapper, requested resources, output
  root, and log paths. Keep Slurm stdout and stderr separate under `logs/`,
  preferably with `%x_%j.out` and `%x_%j.err`.

## Environment

- Use the existing project environment unless instructed otherwise. Do not
  install, remove, or upgrade dependencies without approval.
- Do not copy Leonardo GPU-count, memory, precision, or Slurm assumptions to
  mps or Snellius wrappers.

## Storage and Sidecars

- Extraction utilities live under `scripts/extraction/`. Provenance scripts
  under `logs/chore/` and `logs/chore/archived/` are historical records.
- Final CUT3R token sidecars use `spatial_features` under either:

      /leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r/{scannet,scannetpp,arkitscenes}/
      /leonardo_work/EUHPC_D32_006/train_data/vlm3r/{scannet,scannetpp}/

- Older records may reference
  `/leonardo_work/EUHPC_D32_006/FAST/train_data/vlm3r`; treat it as a legacy
  mirror, not an automatically interchangeable source.
- Intermediate CUT3R decoder features live under
  `/leonardo_work/EUHPC_D32_006/VLM_3R_cut3r_min2N4_features` with
  `spatial_features_dec_6`, `spatial_features_dec_9`,
  `spatial_features_dec_m2`, and `spatial_features_dec_m4`.
- CUT3R token `.pt` files contain `camera_tokens`, `patch_tokens`, and
  sometimes `metadata`; patch grids contain 729 tokens with feature dimension
  768.
- Cross-attention and feature-alignment runs use the final-layer root plus
  `SPATIAL_FEATURES_SUBDIR=spatial_features`, `MODEL_SPATIAL_TOWER=cut3r`,
  `MODEL_SPATIAL_TOWER_SELECT_FEATURE=all_tokens`, and
  `MODEL_SPATIAL_FEATURE_DIM=768`.
- SpatialStack layer mappings may use forms such as
  `6:spatial_features_dec_6;9:spatial_features_dec_9;12:/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r:spatial_features`.

## CUT3R Point Maps

- Point-map sidecars use `spatial_features_points` and contain
  `point_maps_ref`, `point_maps_cam`, `camera_pose`, and `metadata`.
- `point_maps_ref` / `pts3d_in_other_view` are reference-frame coordinates;
  `point_maps_cam` / `pts3d_in_self_view` are per-frame camera coordinates.
  Keep the coordinate source identical between training and evaluation.
- Verified roots are:

      /leonardo_scratch/large/userexternal/shuang00/VLM_3R_cut3r_pointmaps/{scannet,scannetpp,arkitscenes}/spatial_features_points
      /leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r/{scannet,scannetpp,arkitscenes}/spatial_features_points

- Geometry runs set `GEOMETRY_SPATIAL_FEATURES_ROOT` to the selected verified
  root, `GEOMETRY_SPATIAL_FEATURES_SUBDIR=spatial_features_points`, and
  `GEOMETRY_SPATIAL_TOWER_TYPE=cut3r`.
- Before modifying Pi3X or VGGT sidecars, point maps, schemas, or loaders, read
  `docs/data-sidecars.md`.

## Training and Evaluation

- The wrappers under `scripts/archived/old_files/old_bash/` are legacy entry
  points. Use current root or experiment-specific wrappers and inspect their
  checkpoint, cache, dataset, and sidecar defaults before submission.
- Shared base wrappers are `train_cut3r_Baseline.sh` and
  `train_cut3r_spatialstack.sh`. Prefer dedicated descriptive wrappers for new
  experiments instead of broad changes to them.
- The root `eval_spatialstack_vsibench.sh` is Leonardo site-specific and
  expects Slurm plus Leonardo storage unless its paths are explicitly
  overridden.
- Extract `spatial_rank_head.*` weights with:

      python scripts/extraction/extract_spatial_rank_head.py --checkpoint <ckpt> --output <p_geo.bin>

## Verification

- Run `python -m py_compile <changed files>` for Python changes and
  `bash -n <changed scripts>` for shell or Slurm wrappers.
- For geometry projection changes, run
  `conda run -n vlm3r python tests/test_metric_grounded_geometry_projection.py`
  when the environment and required assets are available.
