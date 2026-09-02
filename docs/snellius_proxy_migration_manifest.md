# Snellius migration manifest: pre-SFT zero-cost proxy

Audit date: 2026-09-02. This was an audit only; no model, data, or proxy run was modified.

## A. Current proxy runtime

The authoritative formal path is `scripts/probing/evaluate_pre_sft_zero_cost_proxies.py`, launched by `scripts/probing/run_pre_sft_zero_cost_proxies_local.sh`. The wrapper names exactly the evaluator's five `CANDIDATES`: C1 VLM3R Baseline; C1 SpatialStack additive 0/1/2, 1/2/3, and 0/3/6; and C1 SpatialStack cross-attention 0/1/2. `pre_sft_zero_cost_proxies_v1/full_corrected` is the completed reference result.

`scripts/probing/run_pre_sft_baseline_trainable_proxy_smoke.py` is newer but distinct: it is a one-candidate Baseline sanity smoke that attaches fresh LoRA and scores LoRA + C1 fusion + the pretrained `mm_projector`. The chunked and gradient-coverage scripts validate that smoke; neither replaces the formal five-way evaluator.

| Item | Value |
|---|---|
| Branch | `machine/mps-edu-06` |
| Commit | `a4fac8767da588dc75284416316ead7994a12105` |
| Subject | `feat(probing): add chunked pre-sft proxy smoke` |
| Worktree before report creation | clean |

| Quantity | Current function/path | Definition |
|---|---|---|
| Params/trainable params | `structural_row`, `parameter_count`, `lora_parameter_count` | Materialized model plus historical rank-128 LoRA count. |
| FLOPs | `profile_forward_flops`, `ForwardFlopCounter` | Actual-forward dense linear/attention matmul count, two FLOPs/MAC. |
| GradNorm | `proxy_scores`, `grouped_proxy_scores` | `sum_p ||dL/dp||_2`. |
| SNIP | same | `sum |p * dL/dp|`. |
| Fisher | same | `sum (dL/dp)^2`, empirical diagonal Fisher. |
| Backward | `run_backward_scope`, `run_grouped_backward_scope` | Training-mode CE forward/backward; no optimizer exists or steps. |

`proxy_supervised_logits_only` uses the model's compact-label CE path to fit memory; FLOP accounting restores full-vocabulary lm-head cost analytically. Loss is ordinary `LlavaQwenForCausalLM` supervised next-token cross entropy.

Model/data construction is: `make_load_args` forces plain `pre_sft_fusion`, `model_base=None`, fp16, and `zero_spatial_features=False`; `load_model` loads the plain base, rejects a base containing adapter artifacts, replaces its vision tower from the supplied SigLIP directory, and builds a CUT3R-sidecar-only tower; `install_pre_sft_fusion` makes fresh candidate modules; `apply_c1_calibration_artifact` deterministically creates C1 matrices and scalar calibration; `install_forward_frame_loader` substitutes cached 32-frame RGB decoding only. The standard `LazySupervisedDataset`, collator, QA prompt, and loss remain active.

| Identifier | Architecture | Sidecar needed for the current smoke |
|---|---|---|
| `c1_vlm3r_native` | native C1 VLM3R/Baseline | final CUT3R token sidecar |
| `c1_ss_add_012` | additive LLM injections 0/1/2 | CUT3R decoder 6, 9, 12 sidecars |
| `c1_ss_add_123` | additive injections 1/2/3 | CUT3R decoder 6, 9, 12 sidecars |
| `c1_ss_add_036` | additive injections 0/3/6 | CUT3R decoder 6, 9, 12 sidecars |
| `c1_ss_cross_attn_012` | cross attention injections 0/1/2 | CUT3R decoder 6, 9, 12 sidecars |

There is no formal `CandidateSpec` for Base VLM/zero-spatial, selective fusion, SS+depth, Baseline+depth, Extra Object Token, GeoRoPE, or Visual 3D-RoPE. They must not be silently added to the formal study.

### Trainable-count gate

The formal evaluator scores fresh C1 fusion only and optionally the whole model. Its historical structural `sft_trainable_params` means LoRA + fusion and deliberately excludes the projector. The dedicated Baseline smoke's actual primary scope is fresh LoRA + C1 fusion + pretrained base projector. Its valid current result confirms:

| Group | Parameter elements |
|---|---:|
| LoRA | 322,961,408 |
| C1 VLM3R fusion block | 9,747,456 |
| `mm_projector` | 16,980,992 |
| Total | **349,689,856** |

It records rank 128, alpha 256, dropout 0.05, bias none, 196 LoRA targets, seed 42, and PEFT `0.4.0`. `init_lora_weights` is intentionally not passed, so PEFT's implementation controls initialization. This version is a migration gate: it can alter target selection, count, initialization, and initial SNIP.

## F. Calibration sample dependency manifest

The formal wrapper uses `--calibration-batches 1`; smoke mode enforces one candidate and one batch. `load_calibration_records` selects the first fixed-manifest video present in `by_video`. This is a small fixed calibration minibatch, not a QA subset sweep or the 2,400-video depth-probe data set.

| Field | Verified current value |
|---|---|
| Seed | 42 for the Baseline trainable smoke; formal C1 construction is deterministic |
| Video | `scannet/videos/scene0384_00.mp4` |
| Split | ScanNet train |
| RGB | 32 cached uint8 frames |
| Input token shape | `[1, 424]` |
| Supervised labels | 13 tokens |
| Actual QA item | first matching ScanNet annotation, id `7e8340c2-eb6e-4736-abfe-13ca82a9e62e`, answer `1.3` |
| Sample-index SHA-256 | `d478cb684958dfc25066821ec83d5216469577c9e282e33bdf87d3c88b200d8e` |

| Dependency | Exact source | Bytes | Runtime reason |
|---|---|---:|---|
| Fixed selection | `/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json` | 1,267,667 | Selects the video. |
| QA source | `/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1/manifests/merged_qa_scannet_train.json` | 32,018,875 | Contains the actual selected prompt/label. |
| YAML-required QA files | `merged_qa_scannetpp_train.json`; `merged_qa_route_plan_train.json` in that same `manifests` directory | 93,645,364; 3,511,046 | The unmodified YAML and `LazySupervisedDataset` open all three. |
| Frames | `/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1/frames/scannet/scene0384_00.pt` | 29,495,100 | Exact 32-frame forward input. |
| Baseline CUT3R | `/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features/scannet/spatial_features/scene0384_00.pt` | 35,882,486 | Native VLM3R camera/patch tokens. |
| SpatialStack extra CUT3R | `.../spatial_features_dec_6/scene0384_00.pt`; `.../spatial_features_dec_9/scene0384_00.pt` | 35,883,062 each | Required in addition to final layer for all SS candidates. |

The sample-index JSON contains old Leonardo annotation/point-map paths, but the current proxy only uses `video_path`; those strings are provenance. The compact target `probe_targets_2f_v1/targets/scannet/spatial_features_points/scene0384_00.pt` exists (4,482,272 bytes) but is never read. `--probe-targets-root` is only `Path.exists()` validated; create an empty destination directory and pass it rather than transferring the 11 GiB target tree.

## B. Mandatory transfer table

Measured with `du -sb`; GiB is 2^30. This is the smallest exact **five-candidate, one-minibatch** formal set if the source checkout is transferred rather than cloned.

| Category | Source path | Required by | Size | Why required | Destination suggestion |
|---|---|---|---:|---|---|
| Exact source | `/home/shaoruei/SpatialFocus` | all | 106,780,365 B (101.8 MiB) | Audited code imported directly. | `$SNELLIUS_PROXY_ROOT/SpatialFocus` |
| Base VLM | `/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2` | all | 16,074,699,554 B (14.97 GiB) | Plain pretrained Qwen/LLaVA, projector, tokenizer, config. | `$SNELLIUS_PROXY_ROOT/models/base/...` |
| SigLIP | `/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384` | all | 3,515,158,792 B (3.27 GiB) | Explicit vision tower/image processor. | `$SNELLIUS_PROXY_ROOT/models/base/...` |
| RGB | exact `scene0384_00.pt` frame file above | all | 29,495,100 B | Exact model input. | `$SNELLIUS_PROXY_ROOT/forward_frames/frames/scannet/` |
| CUT3R | final + dec6 + dec9 exact files above | formal roster | 107,648,610 B (102.7 MiB) | Sidecar-only spatial input; no CUT3R weights are loaded. | `$SNELLIUS_PROXY_ROOT/cut3r_features/scannet/` |
| QA | three listed JSON manifests | all | 129,175,285 B (123.2 MiB) | Required by unmodified dataset YAML. | `$SNELLIUS_PROXY_ROOT/qa_manifests/` |
| Selection | fixed sample-index JSON | all | 1,267,667 B | Fixed calibration selection. | `$SNELLIUS_PROXY_ROOT/provenance/` |
| C1 | five official directories listed below | all | 60,338 B | Deterministic architecture initialization. | `$SNELLIUS_PROXY_ROOT/c1/` |

**Mandatory formal total: 19,964,285,711 B (18.59 GiB).** Cloning exactly the audited commit can replace the 101.8 MiB source transfer; do not use another revision.

The full model directories are the safe offline unit. The base directory has all four safetensor shards plus index, config, tokenizer, processor, and generation assets; do not omit a shard. Its `_name_or_path` is an old `/mnt/bn/...` path, not a verified Hub id, so reacquisition is not established as safe without byte/hash comparison. SigLIP appears standard `google/siglip-so400m-patch14-384`, but transfer remains the exact offline choice; a verified re-download is only conditional.

## E. C1 dependency graph

`apply_c1_calibration_artifact` reads the selected top-level JSON only. It regenerates C1 matrices from repository code and scalar fields; it never opens a tensor/checkpoint or the referenced parent JSONs. The parent JSON references are nevertheless transferred for provenance and integrity.

```text
c1_ss_add_012 -> c1_additive_v1/official/spatialstack_add.json (29a540...80dc)
                -> base_r0.json (4c0c84...47cb)
                -> calibration_manifest_32.json (1f6085...7da)
c1_ss_add_036 -> c1_ss_add_036/official/spatialstack_add.json (f51409...42d9)
                -> same base_r0 + manifest
c1_ss_add_123 -> c1_ss_add_123/official/spatialstack_add.json (8ff199...a150)
                -> same base_r0 + manifest
c1_ss_cross_attn_012 -> c1_ss_cross_attn_v1/official/spatialstack_cross_attn_v1.json (d1080b...a520)
                         -> same base_r0 + manifest
c1_vlm3r_native -> c1_vlm3r_v1/official/vlm3r.json (edb6ab...8f9c)
                   -> same base_r0 + manifest
```

```text
/home/shaoruei/probe_outputs/c1_additive_v1/official/base_r0.json
/home/shaoruei/probe_outputs/c1_additive_v1/official/calibration_manifest_32.json
/home/shaoruei/probe_outputs/c1_additive_v1/official/spatialstack_add.json
/home/shaoruei/probe_outputs/c1_ss_add_036/official/spatialstack_add.json
/home/shaoruei/probe_outputs/c1_ss_add_123/official/spatialstack_add.json
/home/shaoruei/probe_outputs/c1_ss_cross_attn_v1/official/spatialstack_cross_attn_v1.json
/home/shaoruei/probe_outputs/c1_vlm3r_v1/official/vlm3r.json
```

All are `c1_calibration_v1`, `c1_structured_isometry_v1`, `no_training: true`, and 32-sample calibration evidence. They contain absolute `base_calibration`, `calibration_manifest`, and source-index provenance references, but no dense state artifact. Keep JSON byte-identical; use a filesystem mapping or code path configuration rather than editing a scientific artifact.

| Candidate | Canonical root/unit | Files in transferable unit | Measured size | Relocation/provenance |
|---|---|---|---:|---|
| SS additive 0/1/2 | `/home/shaoruei/probe_outputs/c1_additive_v1/official` | `spatialstack_add.json`, `base_r0.json`, `calibration_manifest_32.json` | 19,880 B | Owns the shared parent/manifest; JSON records their absolute paths and SHA-256. |
| SS additive 0/3/6 | `/home/shaoruei/probe_outputs/c1_ss_add_036/official` | `spatialstack_add.json` | 10,516 B | References the first unit's parent/manifest by path and hash. |
| SS additive 1/2/3 | `/home/shaoruei/probe_outputs/c1_ss_add_123/official` | `spatialstack_add.json` | 10,518 B | Same shared parent/manifest reference. |
| SS cross-attention | `/home/shaoruei/probe_outputs/c1_ss_cross_attn_v1/official` | `spatialstack_cross_attn_v1.json` | 11,592 B | Same shared parent/manifest reference. |
| VLM3R/Baseline | `/home/shaoruei/probe_outputs/c1_vlm3r_v1/official` | `vlm3r.json` | 7,832 B | Same shared parent/manifest reference. |

## Files regenerated on Snellius

| Item | Why it need not transfer |
|---|---|
| `runtime/<candidate>/.../config.json` patched runtime checkpoint | `patch_runtime_checkpoint` creates it from the transferred plain base and supplied SigLIP path. |
| Fresh C1 dense matrices/buffers | Recreated deterministically by `llava/model/c1_structured_isometry.py` from C1 JSON scalar/schema data. |
| Fresh Baseline LoRA parameters | `attach_intended_sft_lora` creates them with recorded seed/PEFT recipe; trained LoRA is prohibited. |
| Proxy `results.json`, CSV, Markdown, metadata, logs, temporary matplotlib/Triton caches | Outputs/caches are generated by the proxy launcher and must be isolated under a new Snellius output root. |
| Empty `--probe-targets-root` directory | It satisfies current existence validation; target files are not used. |

## C. Conditional assets and architecture status

| Candidate/asset | Current evidence | Conditional dependency | Migration status |
|---|---|---|---|
| Plain Base VLM / zero-spatial | `load_model` and depth code support related modes, but the formal proxy hardcodes `pre_sft_fusion`, `feature_preset="original"`, and no matching `CandidateSpec`. | Base + SigLIP + frames + QA; plain base needs no CUT3R. A zero-spatial initialization/scope must be specified first. | Not in the formal proxy. |
| Selective fusion | `diagnose_c1_eomt_selective_calibration.py` reuses C1 VLM3R but is forward-only, with no backward/proxy CandidateSpec. | Its fixed 32 samples are 943,843,200 B frames + 1,148,239,552 B final CUT3R + 109,875,328 B class logits + 597,305,216 B selective masks = 2,799,263,296 B (2.61 GiB). | Diagnostic only; do not migrate by default. Full EoMT cache is 31 GiB and is not needed for that fixed subset. |
| Extra Object Token | Present in post-SFT code/specs only; no C1 pre-SFT artifact/CandidateSpec. | Post-SFT EoMT object cache exists. | Unavailable for formal pre-SFT proxy. |
| SS+depth; Baseline+depth | Only post-SFT proxy paths exist. They require full 32-frame point maps and explicitly reject compact 2-frame targets. | The local migrated full point-map tree is not established; no truthful size is available. | Blocked/unavailable. |
| GeoRoPE fusion; Visual 3D-RoPE | Only post-SFT specs/paths exist and require geometry plus checkpoint-exact configuration. | Full `point_maps_ref` and an explicit pre-SFT initialization contract. | Unavailable for formal pre-SFT proxy. |

## D. Explicit exclusions

| Do not transfer | Evidence/reason |
|---|---|
| `adapter_model.bin`, `non_lora_trainables.bin`, trained fusion/projector state, trained LoRA, candidate SFT checkpoints | Formal base loading rejects adapter artifacts. Loading any trained candidate state violates the pre-SFT contract even with no optimizer update. |
| 2,400-video hidden-state/depth-probe caches and post-SFT probe features | No active proxy code reads feature-cache tensors or fits a depth probe. |
| Full 2,400 RGB cache and full CUT3R trees | Current one-minibatch path opens one RGB file and three CUT3R files only. Decoder -2/-4 are never named. |
| Entire 11 GiB compact target tree | Proxy validates only the root's existence; it never reads targets. |
| VSiBench test MP4s | Proxy reads cached ScanNet RGB, not VSiBench MP4s. VSI is a comparison label only. |
| Old Leonardo artifacts, Slurm wrappers, stale logs, optimizer/scheduler state | Not imported or executed by the active proxy. |
| Full point maps | Unsupported candidates might need them; supported candidates do not. |

## G. Environment manifest

Conda environment: `vlm3r` at `/home/shaoruei/miniconda3/envs/vlm3r`. `conda env export --from-history` reports Python 3.10, PyTorch 2.1.1, torchvision 0.16.1, and pytorch-cuda 12.1. Direct environment inspection reported Python 3.10.14, torch 2.1.1, transformers 4.40.0.dev0, PEFT 0.4.0, Accelerate 0.29.1, and torchvision 0.16.1.

```text
accelerate==0.29.1
bitsandbytes==0.41.0
deepspeed==0.14.4
einops==0.6.1
flash-attn==2.7.1.post1
numpy==1.26.4
peft==0.4.0
safetensors==0.7.0
sentencepiece==0.1.99
timm==1.0.25
tokenizers==0.15.2
torch==2.1.1
torchvision==0.16.1
transformers==4.40.0.dev0
```

Relevant runtime variables are `CUDA_VISIBLE_DEVICES=0,1`, `ENV_NAME=vlm3r`, `GPU=0`, `REPO_ROOT`, `OUTPUT_ROOT`/`OUTPUT_BASE`, `LOG_ROOT`, `MPLCONFIGDIR`, and `TRITON_CACHE_DIR`. The Baseline smoke uses fp16, auto device map, seed 42, GPU budgets `4GiB,6GiB`, and CPU offload `45GiB`; the formal wrapper defaults to `5GiB` per visible GPU and `45GiB` CPU. These are TITAN-V placement settings, not portable Snellius settings: run a new one-minibatch memory smoke first.

The installed `llava==1.7.0.dev0` editable declaration points to a stale Leonardo path. The scripts put the repository root on `sys.path` and therefore import checkout source directly. On Snellius run from the exact checkout or install that checkout editable; do not depend on the stale installed link. PEFT, Transformers, Torch, Accelerate, flash-attn/attention backend, dtype, and tokenizer versions may change LoRA module selection/count/init, loss numerics, attention, and fp16 behavior.

## H. Path remapping

| Current path | Use | Relocation |
|---|---|---|
| `/mnt/DATA_SSD/shaoruei/models/base/...` | evaluator defaults | pass `--base-model`, `--siglip-model` |
| `/mnt/DATA_SSD/.../forward_frames_32_v1` | frame loader default | pass `--forward-frames-root` |
| `/mnt/DATA_SSD/.../cut3r_features` | sidecar default | pass `--feature-root` |
| `/mnt/DATA_SSD/.../probe_targets_2f_v1` | existence check only | pass an empty existing `--probe-targets-root` |
| `scannet_depth_probe_local_data.yaml` absolute manifest paths | dataset construction | provide relocated YAML through `--data-yaml`, preserving its three-file order |
| `/home/shaoruei/probe_provenance/...sample_indices.json` | selection default | pass `--sample-indices` |
| `/home/shaoruei/probe_outputs/c1_*/official/*.json` | hardcoded `CANDIDATES` | current CLI has no C1-path override: preserve/map this layout or add a small source-level path configuration; never edit C1 JSON contents |
| `/home/shaoruei/probe_outputs/...` | output/log default | pass output/log variables or `--output-root` |
| post-SFT `complete/results.json` | Baseline smoke cost reference only | pass `--post-sft-artifact` when using that unmodified smoke |

## Output/reference artifacts (not model inputs)

| Artifact | Size | Purpose |
|---|---:|---|
| `VSI result.csv` | 864 B | Candidate-to-VSI mapping. |
| `post-sft-result-for-codex.xlsx` | 12,244 B | Supplementary reference. |
| `/home/shaoruei/probe_outputs/pre_sft_zero_cost_proxies_v1/final` | 109,523 B | Prior formal result. |
| `.../pre_sft_zero_cost_proxies_v1/full_corrected` | 109,312 B | Corrected formal result. |
| `.../pre_sft_zero_cost_proxies_v2/smoke_baseline_retry6_asymmetric_shards` | 50,432 B | Current Baseline smoke. |
| `.../validate_baseline_chunked_exact_v2_mask_replay` | 270,069 B | Exact-backward validation. |
| `.../runtime_lora_gradient_coverage` | 21,057 B | Gradient coverage diagnostic. |
| `.../post_sft_3d_zero_cost_proxies_v1/complete/results.json` | 39,708 B | Required only for the unmodified Baseline smoke's cost-reference report; it loads no model weights and must not be compared as a formal score. |

The reference selection is about 613 KiB. It is optional except for that 39,708-B JSON when running the dedicated smoke unchanged.

## I. Final summary

| Set | Total |
|---|---:|
| Mandatory formal five-candidate runtime | **19,964,285,711 B (18.59 GiB)** |
| Smallest exact Baseline GradNorm/SNIP/Fisher smoke | **about 18.53 GiB**: source/base/SigLIP/three QA manifests/selection/C1 unit, one RGB file, one final CUT3R file, and the 39,708-B cost-reference JSON |
| Reference/debug selection | ~613 KiB |
| Conditional selective 32-sample diagnostic | 2.61 GiB |
| Full EoMT cache (not needed) | 31 GiB |
| Conditional depth/GeoRoPE full point maps | unmeasured/unavailable locally |

## J. Transfer-ready list and K. example rsync

`docs/snellius_proxy_migration_files.txt` is a newline-separated mandatory formal list, relative to `/`, suitable for `rsync -aR --files-from=LIST / DEST/`. It deliberately excludes every 2,400-video tree and every post-SFT weight.

```bash
# Example only: do not execute until destination/layout is reviewed.
rsync -aR --info=progress2 --partial --append-verify \
  --files-from=/home/shaoruei/SpatialFocus/docs/snellius_proxy_migration_files.txt \
  / USER@snellius.example:/scratch/USER/spatialfocus-proxy/

# Optional references can move separately.
rsync -aR --info=progress2 --partial --append-verify \
  /home/shaoruei/SpatialFocus/VSI\ result.csv \
  /home/shaoruei/probe_outputs/pre_sft_zero_cost_proxies_v1/final \
  USER@snellius.example:/scratch/USER/spatialfocus-proxy-references/
```

Before a Snellius proxy run, verify commit, C1 JSON hashes, sample-index hash, absence of base adapter files, RGB/sidecar schemas, PEFT version and 349,689,856 Baseline primary count. Then perform one minibatch smoke before any full sweep.
