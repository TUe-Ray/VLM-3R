# Depth Training Handoff

Date: 2026-06-01

## Current Slurm Jobs

- Debug/smoke job: `43817006`
  - Job name: `SMOKE_depth20`
  - Script: `smoke_train_depth_20it.sh`
  - State when saved: `RUNNING`
  - Node: `lrdn0601`
  - Elapsed when saved: `00:02:23`
  - Log paths:
    - `logs/train/SMOKE_depth20_43817006.out`
    - `logs/train/SMOKE_depth20_43817006.err`

- Full training job: `43817021`
  - Job name: `cut3r_depth_loss`
  - Script: `train_depth_loss.sh`
  - Dependency: submitted with `afterok:43817006`
  - State when saved: `PENDING (Dependency)`
  - Log paths after it starts:
    - `logs/train/cut3r_depth_loss_43817021.out`
    - `logs/train/cut3r_depth_loss_43817021.err`

If `43817006` is cancelled or fails, `43817021` should not start because the dependency is `afterok`.

## Script State

- `train_BEV_loss.sh` is now BEV-only:
  - `MODEL_USE_BEV_SUPERVISION="True"`
  - `MODEL_USE_DEPTH_SUPERVISION="False"`

- `train_depth_loss.sh` is now depth-only and independent:
  - `MODEL_USE_BEV_SUPERVISION="False"`
  - `MODEL_USE_DEPTH_SUPERVISION="True"`
  - `MODEL_DEPTH_POINT_MAP_KEY="${MODEL_DEPTH_POINT_MAP_KEY:-point_maps_cam}"`
  - strict depth fallback flags default to `False`

- `train_BEV_depth_loss.sh` was added for BEV+depth:
  - `MODEL_USE_BEV_SUPERVISION="True"`
  - `MODEL_USE_DEPTH_SUPERVISION="True"`

- `smoke_train_depth_20it.sh` now calls `train_depth_loss.sh`, so it tests the independent depth-only script.

## Debug Job Preflight Observed

Before this handoff, `43817006` had already reached configuration printout using the new independent `train_depth_loss.sh`:

- `[BEV] ENABLE=False`
- `[DEPTH] ENABLE=True ... POINT_MAP=point_maps_cam`
- `use_bev_supervision: False`
- `use_depth_supervision: True`
- `depth_point_map_key: point_maps_cam`
- `torch_compile: True`

It had not yet reached first depth metrics at the time this status was saved.

## Resume Checklist

1. Check current Slurm state:
   ```bash
   squeue -j 43817006,43817021 -o '%i %.22j %.8T %.10M %.9l %.30R'
   sacct -j 43817006,43817021 --format=JobID,JobName%28,State,ExitCode,Elapsed,Timelimit,NodeList -P
   ```

2. If the debug job was cancelled or failed, cancel or ignore the dependent full job if it is still pending, then resubmit:
   ```bash
   sbatch smoke_train_depth_20it.sh
   sbatch --dependency=afterok:<new_debug_job_id> train_depth_loss.sh
   ```

3. If the debug job completed successfully, verify that full training started and check:
   ```text
   [BEV] ENABLE=False
   [DEPTH] ENABLE=True
   depth_point_map_key_used=point_maps_cam
   depth_target_space=camera
   ```

4. Important unrelated local changes existed before this handoff:
   - `scripts/probing/train_depth_probes.py`
   - several untracked semantic probing scripts under `scripts/probing/`
