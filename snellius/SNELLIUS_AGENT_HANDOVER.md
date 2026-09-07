# Snellius VLM-3R migration handover

## Scope and current conclusion

The accepted **training plus immediate VSI-Bench evaluation** bundle is fully
present on Snellius and passes both repository validators. Native CUDA/NCCL,
real SpatialStack training, checkpoint serialization, and real VSI-Bench
generation have passed. No official SFT or evaluation campaign has been
submitted.

The platform is ready for a user-approved official run. Both single-GPU and
two-rank Accelerate evaluation have passed. The redundant four-rank eval smoke
was cancelled while still pending because the same multi-GPU code path had
already passed and an entire node would only duplicate that proof.

## Authoritative source/provenance

* Leonardo repository: `/leonardo/home/userexternal/shuang00/VLM-3R`
* Snellius repository: `/home/geusdd/VLM-3R`
* branch: `feat/new_design`
* HEAD: `02ce5b541267369f7c9a61c5f0e6557a94d44b2d`
* CUT3R submodule: `third_party/CUT3R` at
  `51244364af3566d6473559f71a81b4accc75c424`
* Leonardo remains authoritative.  Do not delete the Leonardo source/assets.

The Snellius checkout includes the transferred authoritative working tree; do
not replace it with GitHub `main`.  Do not initialise Pi3, VGGT, EoMT, or copy
their payloads.

## Completed and validator-verified assets

Persistent campaign root (temporary Snellius shared scratch):

```
/scratch-shared/geusdd/VLM3R
```

The following were revalidated on 2026-09-07 with both
`validate_target_bundle.sh` and `validate_migration.sh`:

| Asset | Verified count | Absolute destination |
|---|---:|---|
| formal SFT JSON | 3 | `/scratch-shared/geusdd/VLM3R/data/vlm3r/VLM-3R-DATA/vsibench_train/` |
| formal SFT media | 2,405 | `/scratch-shared/geusdd/VLM3R/data/vlm3r/` |
| train CUT3R dec6 | 2,405 | `/scratch-shared/geusdd/VLM3R/spatial_features/cut3r/dec6/` |
| train CUT3R dec9 | 2,405 | `/scratch-shared/geusdd/VLM3R/spatial_features/cut3r/dec9/` |
| train CUT3R dec12 | 2,405 | `/scratch-shared/geusdd/VLM3R/spatial_features/cut3r/dec12/` |
| evaluation parquet metadata | 2 | `/scratch-shared/geusdd/VLM3R/data/vsibench/` |
| evaluation media | 288 | `/scratch-shared/geusdd/VLM3R/hf_cache/vsibench/` |
| eval CUT3R dec6/dec9/dec12 | 288 each | corresponding `spatial_features/cut3r/dec{6,9,12}/` roots |
| LLaVA-NeXT-Video-7B-Qwen2 | 37 files | `/scratch-shared/geusdd/VLM3R/models/LLaVA-NeXT-Video-7B-Qwen2/` |
| SigLIP | 8 files | `/scratch-shared/geusdd/VLM3R/models/siglip-so400m-patch14-384/` |
| CUT3R checkpoint | SHA-256 verified | `/home/geusdd/VLM-3R/third_party/CUT3R/src/cut3r_512_dpt_4_64.pth` |

Training and evaluation scene sets were verified disjoint.  The SHA-256 hashes
of all three formal JSON files and the CUT3R checkpoint pass.

The decoder roots contain 2,693 `.pt` files each because they contain the
2,405 training plus 288 evaluation sidecars.  This is expected.

Do not add dec-2, dec-4, point maps, Pi3/Pi3X, VGGT, EoMT, BEV assets, old
checkpoints, outputs, logs, or a full Hugging Face cache.

## Path configuration

Source before every smoke job:

```bash
source /home/geusdd/VLM-3R/snellius/paths.env
```

Important values:

```bash
REPO_DIR=/home/geusdd/VLM-3R
VLM3R_ROOT=/scratch-shared/geusdd/VLM3R
LOCAL_MODEL_BASE=$VLM3R_ROOT/models/LLaVA-NeXT-Video-7B-Qwen2
LOCAL_SIGLIP=$VLM3R_ROOT/models/siglip-so400m-patch14-384
DATA_ROOT=$VLM3R_ROOT/data/vlm3r
CUT3R_DEC6_ROOT=$VLM3R_ROOT/spatial_features/cut3r/dec6
CUT3R_DEC9_ROOT=$VLM3R_ROOT/spatial_features/cut3r/dec9
CUT3R_DEC12_ROOT=$VLM3R_ROOT/spatial_features/cut3r/dec12
TRAIN_SAVE_ROOT=$VLM3R_ROOT/checkpoints
```

`paths.env` also relocates `HF_HOME`, `XDG_CACHE_HOME`, `TRITON_CACHE_DIR`, and
`TORCH_EXTENSIONS_DIR` into scratch.  Do not use the exhausted Snellius project
space, `/scratch-local` for persistent data, or `/home` for large assets.

## Environment state

The environment exists at:

```
/home/geusdd/.conda/envs/vlm3r-snellius
```

Known installed compatibility baseline:

* Python 3.10.14
* PyTorch `2.1.1+cu121`
* `torch.version.cuda == 12.1`
* Transformers `4.40.0.dev0` (Leonardo source revision)
* DeepSpeed `0.14.4`
* editable `llava` and `lmms_eval`

Important activation rule: loading Snellius `Miniconda3/23.5.2-0` and running
`conda activate vlm3r-snellius` places the module base Python before the
environment Python.  **Do not use that activation sequence in smoke jobs.**
After loading `2023` and `CUDA/12.1.1`, use:

```bash
VLM3R_CONDA_PREFIX=/home/geusdd/.conda/envs/vlm3r-snellius
export CONDA_PREFIX="$VLM3R_CONDA_PREFIX"
export PATH="$VLM3R_CONDA_PREFIX/bin:$PATH"
```

The active smoke wrappers already implement this workaround.

Do not copy compiled Leonardo `.so` files. FlashAttention and CUROPE were built
from Snellius source and functionally validated on an allocated GPU node.

## Completed jobs and live distributed checks

Completed evidence:

* `26432129`: four-GPU environment/native smoke, `COMPLETED 0:0`.
  FlashAttention, CUT3R CUROPE, four-rank NCCL, and DeepSpeed ZeRO-2 passed.
* `26432343`: original four-GPU/one-step SpatialStack training smoke,
  `COMPLETED 0:0`.
* `26438561`: original real one-sample VSI-Bench generation smoke,
  `COMPLETED 0:0`.
* `26439184`: strengthened two-GPU/one-step training smoke,
  `COMPLETED 0:0` in 4m31s. Its four-record global batch covers ScanNet (2),
  ScanNet++ (1), and route-plan (1). Loss was 0.4964 and grad norm 7.4543.
  Config plus all 18 finite, updated SpatialStack tensors passed validation.
* `26439634`: strengthened one-GPU eval smoke, `COMPLETED 0:0` in 3m32s.
  It generated one real answer with nonzero SpatialStack injection and wrote
  validated `results.json` plus `vsibench.json`.
* `26439879`: two-GPU/two-rank distributed eval smoke, `COMPLETED 0:0` in
  5m13s. Both ranks generated a request with nonzero SpatialStack injection;
  aggregation produced exactly two effective and two logged samples.

Current checkpoint:

```
/scratch-shared/geusdd/VLM3R/checkpoints/SMOKE_spatialstack_dec6_9_12_26439184
```

Its path is atomically published in:

```
/scratch-shared/geusdd/VLM3R/migration/latest_spatialstack_smoke_checkpoint.txt
```

There are no required live jobs. Job `26439632`, the redundant four-rank eval,
was cancelled while pending after `26439879` passed. Do not cancel or alter
non-SMOKE runs without explicit user permission.

## Proactive pipeline hardening

The current wrappers now:

* propagate both child-process and `lmms_eval` caught exceptions as nonzero
  Slurm exits;
* explicitly select Accelerate `--multi_gpu` whenever world size is greater
  than one and reject GPU/rank mismatches before model load;
* require at least one eval request per rank;
* separate the Hugging Face datasets cache from VSI media paths, with both on
  shared scratch rather than the repository;
* fail unless non-empty, parseable `results.json` and `vsibench.json` exist;
  the effective and logged sample counts must also equal the requested limit;
* use a job-owned temporary directory and 30-minute smoke limits;
* derive a per-job distributed rendezvous port to avoid shared-node collisions;
* support one-/two-GPU shared-node backfill wrappers; and
* validate the saved training config and every trained SpatialStack tensor
  before updating the checkpoint pointer.

Account `tesei2748` exposes only `normal` QOS; `boost_qos_dbg` is unavailable.
Do not submit an official experiment campaign without the user's explicit
direction.

## Useful commands

Read-only validators:

```bash
/home/geusdd/VLM-3R/snellius/validate_target_bundle.sh
/home/geusdd/VLM-3R/snellius/validate_migration.sh
```

Dedicated wrappers:

```text
/home/geusdd/VLM-3R/snellius/smoke_environment_native.sbatch
/home/geusdd/VLM-3R/snellius/smoke_spatialstack_train.sbatch
/home/geusdd/VLM-3R/snellius/smoke_spatialstack_train_1gpu.sbatch
/home/geusdd/VLM-3R/snellius/smoke_spatialstack_train_2gpu.sbatch
/home/geusdd/VLM-3R/snellius/smoke_spatialstack_eval.sbatch
/home/geusdd/VLM-3R/snellius/smoke_spatialstack_eval_1gpu.sbatch
/home/geusdd/VLM-3R/snellius/smoke_spatialstack_eval_2gpu.sbatch
```

The formal SFT YAML rendered for Snellius is:

```text
/home/geusdd/VLM-3R/snellius/vsibench_data.snellius.yaml
```

## Explicitly not done

* No official full SFT/evaluation campaign has been submitted.
* The repository's CPU unit tests were not run because the existing project
  environment does not include `pytest`; no dependency was installed.
* Shared scratch is purgeable and is not a substitute for a resolved
  persistent project-space location for long official campaigns.
