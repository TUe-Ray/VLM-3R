# Dual-path controlled experiment results

| Run | Seed | Spatial attention | Writeback scope | Writeback visibility | Evidence | Init std | VSI Avg | Object Count | Abs. Distance | Object Size | Room Size | Rel. Distance | Rel. Direction | Route Plan | Appearance Order |
|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2D base |  | disabled | — | — | — | — |  |  |  |  |  |  |  |  |  |
| SpatialStack 6/9/12 |  | — | — | — | legacy | — |  |  |  |  |  |  |  |  |  |
| Raw layer-12 |  | — | all_tokens | frame_local | final dense |  |  |  |  |  |  |  |  |  |  |
| A |  | global | all_tokens | frame_local | processed 6/9/12 |  |  |  |  |  |  |  |  |  |  |
| B |  | frame_local | all_tokens | frame_local | processed 6/9/12 |  |  |  |  |  |  |  |  |  |  |
| C |  | global | all_tokens | global | processed 6/9/12 |  |  |  |  |  |  |  |  |  |  |
| D |  | global | text_only | global | processed 6/9/12 |  |  |  |  |  |  |  |  |  |  |

Record the target-GPU preflight report and checkpoint source alongside every
row. The raw control changes both maturation and progressive feature-level
usage relative to the processed arm; do not interpret it as a perfectly
isolated maturation ablation.

## HPC smoke-test status — 2026-07-29

- Status: **in progress; no training authorization**. No checkpoint, parity, gradient, generation, save/reload, or full-size preflight result below should be inferred as passed.
- Checked-out revision: `f17ecc74bd9b1eecca3f2c155a115932bfe41101` (contains dual-path commit `83b5668`).
- Runtime selected: `vlm3r` (`/leonardo_work/EUHPC_D32_006/miniconda3/envs/vlm3r`), Python 3.10.14, PyTorch 2.1.1+cu121, Transformers 4.40.0.dev0; flash/memory-efficient/math SDPA flags enabled.
- Checkpoint mounts: canonical `LLaVA-NeXT-Video-7B-Qwen2` and donor `cut3r_spatialstack_45297963` are readable. The donor adapter metadata names the same canonical checkpoint as its base; Journey9ni is not a load source.
- Login host has no visible GPU. Slurm debug allocation is requested on an A100 node; GPU type/memory will be recorded only from a successful job.
- Static checks passed: `git diff --check`; `bash -n train_cut3r_dualpath.sh`; `python -m py_compile llava/model/cut3r_dual_path.py llava/model/language_model/llava_qwen.py llava/train/train.py`.
- Test runner limitation: `pytest` is not installed in any available project Conda environment. The five existing test functions were run directly after test-loader repair and passed. These existing tests cover hybrid masks, writeback visibility/text-only unchanged visual state, and cache repeat/reorder; they do not by themselves cover every requested integration case.
- Fixes during smoke setup: register the dynamically loaded module in `sys.modules` for Python 3.10 dataclass support; prevent PyTorch 2.1 SDPA all-masked writeback rows from producing NaNs by using a temporary allowed key and zeroing that query output exactly.
- GPU job `50861636` failed before model loading because `/tmp` is node-local and the probe was not staged to the compute node. Job `50861876` is the corrected queued `SMOKE_dualpath_ckpt` allocation; it has not yet produced a validation result.
- Gate: **NO-GO pending**. Do not launch training until the remaining required GPU phases pass.
- Corrected GPU job 50862838 used the verified local SigLIP mirror and reached canonical checkpoint loading on an A100 allocation, but failed before donor loading: LlavaQwenForCausalLM.generate(input_ids=...) reached embed_tokens(inputs) with inputs=None and raised TypeError: embedding indices must be a Tensor. This is an active generation API/preflight blocker, so checkpoint, parity, donor, gradient, cache, and memory claims remain unverified.
- The local-mirror job also emitted extensive PyTorch meta-parameter copy warnings while loading the vision tower. These must be assessed after the generation API blocker is fixed; they are not treated as a successful vision-tower load.
- Worktree caution: llava/model/language_model/llava_qwen.py became modified externally during this smoke run. It was not edited by this task after detecting the generation failure.
- Correction to the prior allocation-status line: job 50861876 was cancelled after its confirmed offline Hugging Face retry loop; it produced no model-validation result. Job 50862838 is the definitive GPU preflight failure recorded above.

## HPC smoke-test update — 2026-07-30

- Static verification after the implementation fixes passed: `git diff --check`, `bash -n train_cut3r_dualpath.sh`, and `python -m py_compile` for the changed dual-path Python modules. `pytest` remains unavailable; the five existing test functions pass when invoked directly.
- Checkpoint preflight job `50890109` passed on an NVIDIA A100-SXM-64GB allocation: direct canonical and donor loading used the required paths, the donor `0/1/2 -> 6/9/12` components were shape-compatible and storage-independent, and the disabled feature forward/greedy-generation parity was exact.
- Smoke jobs `50893072`, `50894402`, `50897443`, `50900613`, and `50989829` exposed and then verified small implementation fixes for feature-dimension initialization, exact 27x27 canonical pooling, Transformers 4.40 Qwen RoPE invocation, and sidecar-frame alignment. None produced a checkpoint.
- The current strict alignment smoke `50995541` intentionally stopped before the first forward/optimizer step. Sample `51bdbf173f` selected canonical source frames `[0, 10952]`, while its layer-6 sidecar `/leonardo_work/EUHPC_D32_006/VLM_3R_cut3r_min2N4_features/scannetpp/spatial_features_dec_6/51bdbf173f.pt` has shape `(32, 729, 768)` and no `frame_indices` field. Layers 9 and 12 have the same missing provenance condition.
- The historical layer-6/9 metadata records `num_frames=32`, `frames_upbound=32`, and `video_fps=1`, but not the actual sampled source-frame list. The current extractor does record this list; these legacy artifacts predate it. A 16-frame or two-frame canonical sample cannot be assumed to correspond to a subset of the historical 32-frame token sequence.
- A strict loader check was added so dual-path exact-index mode records canonical sampled indices, selects only provenance-verified sidecar frames, and rejects missing/incompatible metadata. It does not guess a mapping or modify sidecars.
- Status: **NO-GO**. No smoke checkpoint, evaluator run, initialization sweep, generation/cache validation, save/reload check, full-size memory preflight, or official training/evaluation job has been run. The required prerequisite is regenerated or otherwise externally verified 6/9/12 sidecars for the exact production sampler/frame count, with `frame_indices` metadata.

## HPC smoke-test update — 2026-07-30 (32-frame distributed run)

- All current smoke work uses the required **32 frames**, never the obsolete 16-frame preflight value. The exact-index sidecars under `dualpath_smoke_frame_provenance/frames32` were used with CUT3R decoder layers `6,9,12` mapped respectively to spatial blocks `0,1,2`.
- Job `51045013` (`SMOKE_dualpath_train32_z3h`) ran one-node four-rank ZeRO-3 on four NVIDIA A100-SXM-64GB GPUs. It successfully loaded the pure canonical checkpoint, loaded the donor-only SpatialStack checkpoint, copied the donor components into the independent dual branch (`[DUAL_PATH] enabled canonical=direct-base`), and loaded four copies of the same real, provenance-aligned 32-frame sample. The four-record fixture is only to provide one real batch per distributed rank; it does not modify the training dataset.
- The first training forward failed before a loss, gradient, optimizer step, or checkpoint: each A100 had only 0.5–0.7 GiB free and ZeRO-3 attempted one additional 888 MiB linear allocation. No configuration was silently changed in response. This is a failed 32-frame memory preflight, not an architecture pass.
- The previous one-record fixture was also unsuitable for four ranks: Accelerate delivered `None` batches with `dataloader_drop_last=True`. The dedicated four-record smoke fixture corrects only that distributed-test setup issue.
- Status remains **NO-GO**. There is no valid smoke checkpoint or evaluator result. Do not submit training until a permitted production-memory plan passes the 32-frame preflight; do not substitute 16 frames, compress tokens, or change global-attention semantics to make it fit.

## HPC memory-parity audit — 2026-07-30

- Controls and DualPath smoke runs use the same one-node, four-rank ZeRO-3 BF16 setup on four NVIDIA A100-SXM-64GB GPUs; they use the real, provenance-aligned 32-frame sample, 729 patches per frame, stride 1, `TARGET_GLOBAL_BATCH_SIZE=4`, one optimizer update, and the pure canonical checkpoint plus the required donor.  No Journey9ni adapter was loaded.
- The original SpatialStack `6/9/12` control (`51081158`) completed forward, backward, optimizer update, and checkpoint save (`loss=0.37562`, step runtime `64.88 s`).  This confirms that the current 32-frame environment and baseline wrapper fit.
- The donor is not retained by the DualPath implementation: per-rank allocated memory was about `6.32 GiB` immediately after donor load and `5.58 GiB` after extraction, deletion, collection, and cache release.  Constructing the final independent branch then used about `7.00 GiB`; this is canonical plus exactly the intended projectors, three cloned spatial blocks, and writeback.
- The dense writeback activation retention exposed by raw control `51091690` was corrected without changing writeback semantics.  Commit `9d0bc39` checkpoints the existing exact 512-query chunks, recomputing them in backward rather than retaining their dense attention intermediates.  The raw-layer-12 control `51093714` then completed (`loss=0.37869`, `72.36 s`) and its post-writeback allocation was `8.49–8.50 GiB` instead of the failed run's `38.36 GiB`.
- Processed Variant B (`51096750`, frame-local spatial, all-token/frame-local writeback) completed and saved (`loss=0.37982`, `81.80 s`).  Processed Variant D (`51098467`, global spatial, text-only/global writeback) also completed and saved (`loss=0.36726`, `77.58 s`).  Both log that custom projectors and all three spatial blocks are gradient-checkpointed, with exact tokenwise MLP chunks of 1024 and exact writeback-query chunks of 512.
- For raw/B/D, full-vocabulary logits and loss dominate the forward peak: `41.09–41.90 GiB` allocated and `55.40–57.03 GiB` reserved immediately after logits/loss.  Peak reserved memory through backward was `59.96 GiB` (raw), `61.09 GiB` (B), and `61.10 GiB` (D), out of `63.42 GiB` usable device memory.  These are successful functional smoke steps but **do not meet the required 10% peak-memory headroom** (`6.34 GiB`), and DeepSpeed reports allocator-cache flushes under pressure.
- Variant A (global spatial, all-token/frame-local writeback) has been submitted unchanged as `51101130` (`SMOKE_memparity_dualA_z3r1`).  Do not infer a result until the job reaches a terminal state and its per-rank audit is recorded.
- Status: **NO-GO pending Variant A and checkpoint-reload/evaluation checks.**  The measured controls show that donor duplication and uncheckpointed writeback were avoidable regressions and have been removed.  If the exact global/all-token Variant A also lacks the required headroom after completing its prescribed smoke chain, the next step is sequence/context parallelism or higher-memory hardware—not a reduction in frames, patches, blocks, or attention scope.
