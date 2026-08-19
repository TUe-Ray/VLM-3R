# C1 structured-isometric pre-SFT fusion probing

C1 is inference/calibration only. It loads the plain base VLM and SigLIP
weights, constructs a fresh native fusion topology, overwrites every
from-scratch fusion affine map deterministically, and freezes scalar values
from unlabeled forward activations. It never loads a LoRA, trained fusion
projector, depth target, point-map target, or probe result.

## Canonical matrices

For a supported dimension `d`, C1 builds normalized Sylvester Hadamard blocks
(`b` is the largest power of two no greater than 512 that divides `d`), applies
the fixed block perfect-shuffle, and uses:

`U_d = blockdiag(H_b/sqrt(b)) @ P_d @ blockdiag(H_b/sqrt(b))`.

Rectangular `nn.Linear(d_in, d_out)` weights are semi-isometric: expansion is
`U_out[:, :d_in] @ U_in.T`; contraction is the analogous row-isometry. All
matrices are built in CPU FP32 without reading RNG state, then copied to the
runtime module dtype/device. The official Q/K mode is `shared_canonical`.
`role_offset` is an opt-in fixed K-channel permutation for a future variant.

- SpatialStack additive: `S768->3584`, GELU, then `U3584.T`.
- SpatialStack CA V1: `Q=U3584`, `K=V=S768->3584`, `O=U3584.T`.
- VLM3R: outer `Q=U1152`, outer `K=V=S768->1152`, internal MHA Q/K/V and
  output maps are identities, and final `out_proj=U1152.T`.

Biases are zero and normalization affine defaults are scale one/bias zero.

## Calibration convention

The manifest selects fixed train videos but contains no labels or targets.
For each video, C1 uses the same cache-aware 32-frame preprocessing, image
placement, and real selected SFT user prompt as the existing probe dataset.
The assistant response is empty, so answer labels are not consumed.

`r0` is the median of all per-sample, per-layer (L0/L1/L2) base-model values
`RMS(H_after-H_before)/RMS(H_before)`, measured only at `visual_token_indices`.
It is a **per-injection-site** budget: each of SpatialStack's three native
injection sites targets `r0`; it is not divided by three.

Additive branches first use the calibration-set `s_pre=1/RMS(Z_pre)`. SS-CA
V1 calibrates `s_qk=1/sqrt(std(QK^T/sqrt(head_dim)))` and applies it to both Q
and K. Its sites calibrate sequentially L0, then L1 with L0 active, then L2
with L0/L1 active. For every final residual gain, C1 takes the median over
samples of `RMS(delta)/RMS(H)`, then sets `gain=r0/median_ratio`.

VLM3R applies the same QK convention, then solves positive `lambda` by
deterministic bracket-and-bisection against the median per-sample effective
projected ratio `RMS(mm_projector(visual+lambda*B)-E_base)/RMS(E_base)`.

## Commands

Set paths for the local server before running. The official manifest uses 32
samples; use `--max-samples 1` only for a runtime smoke.

```bash
BASE=/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2
SIGLIP=/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384
FRAMES=/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1
CUT3R=/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features
SAMPLES=/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json

conda run -n vlm3r python scripts/probing/make_c1_calibration_manifest.py \
  --source-sample-indices "$SAMPLES" --num-samples 32 \
  --output /home/shaoruei/probe_outputs/c1/calibration_manifest_32.json

CUDA_VISIBLE_DEVICES=0,1 conda run -n vlm3r python scripts/probing/c1_calibrate_fusion.py \
  --architecture base --model-path "$BASE" --siglip-path "$SIGLIP" \
  --forward-frames-root "$FRAMES" --output /home/shaoruei/probe_outputs/c1/base_r0.json \
  --calibration-manifest /home/shaoruei/probe_outputs/c1/calibration_manifest_32.json
```

Run the following once for each architecture (`spatialstack_add`,
`spatialstack_cross_attn_v1`, and `vlm3r`):

```bash
CUDA_VISIBLE_DEVICES=0,1 conda run -n vlm3r python scripts/probing/c1_calibrate_fusion.py \
  --architecture spatialstack_cross_attn_v1 --model-path "$BASE" --siglip-path "$SIGLIP" \
  --feature-root "$CUT3R" --spatial-features-subdir '6:spatial_features_dec_6;9:spatial_features_dec_9;12:spatial_features' \
  --forward-frames-root "$FRAMES" --base-calibration /home/shaoruei/probe_outputs/c1/base_r0.json \
  --calibration-manifest /home/shaoruei/probe_outputs/c1/calibration_manifest_32.json \
  --output /home/shaoruei/probe_outputs/c1/ss_ca_v1.json
```

Then extract ordinary features with the existing pipeline; only the opt-in
loader arguments are new:

```bash
CUDA_VISIBLE_DEVICES=0,1 SPATIALFOCUS_CPU_MERGE_LORA=1 conda run -n vlm3r python scripts/probing/extract_depth_probe_features.py \
  --model-label c1_ss_ca_v1 --model-loading-mode pre_sft_fusion \
  --pre-sft-fusion-variant c1_ss_cross_attn_v1 \
  --c1-calibration-json /home/shaoruei/probe_outputs/c1/ss_ca_v1.json \
  --model-path "$BASE" --siglip-path "$SIGLIP" --forward-frames-root "$FRAMES" \
  --sample-indices "$SAMPLES" --output-root /home/shaoruei/probe_cache/c1_ss_ca_v1 \
  --probe-targets-root /mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1 \
  --feature-root "$CUT3R" --spatial-features-subdir '6:spatial_features_dec_6;9:spatial_features_dec_9;12:spatial_features'
```

The resulting cache remains an ordinary probe cache; no probe training or
evaluation code is changed.

For the zero-spatial/base control, use the existing plain-base path (no C1
artifact and no spatial sidecars):

```bash
CUDA_VISIBLE_DEVICES=0,1 conda run -n vlm3r python scripts/probing/extract_depth_probe_features.py \
  --model-label pre_sft_base_vlm --model-loading-mode pre_sft_base_vlm \
  --model-path "$BASE" --siglip-path "$SIGLIP" --forward-frames-root "$FRAMES" \
  --sample-indices "$SAMPLES" --output-root /home/shaoruei/probe_cache/c1_base \
  --probe-targets-root /mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1
```
