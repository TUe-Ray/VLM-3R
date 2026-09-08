# C2 CCA-QK canonical probe initialization

C2 is an opt-in, forward-only extension of `c1_ss_cross_attn_v1`.  It starts
from C1's deterministic Q/K/V/O construction, uses the native V1 `q_proj` and
`k_proj` outputs as paired patch observations, replaces only Q/K by per-head
CCA affine maps, and keeps C1 V/O untouched.

The calibration command reuses the C1 manifest and normal cached 32-frame
probe preprocessing.  The C1 artifact and supplied manifest must have the
same SHA-256: this prevents a C2 run from silently inheriting a residual target
measured on different inputs.

```bash
conda run -n vlm3r python scripts/probing/c2_calibrate_spatialstack.py \
  --c1-calibration-json /home/shaoruei/probe_outputs/c1_ss_ca.json \
  --calibration-manifest /home/shaoruei/probe_outputs/c1_manifest.json \
  --output /home/shaoruei/probe_outputs/c2_ss_ca.pt \
  --model-path /mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2 \
  --siglip-path /mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384 \
  --feature-root /mnt/DATA_SSD/shaoruei/probing_data/cut3r_features \
  --spatial-features-subdir '6:spatial_features_dec_6;9:spatial_features_dec_9;12:spatial_features' \
  --forward-frames-root /mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1 \
  --device cuda:0 --device-map auto --dtype float16
```

For the smallest real-input smoke check, add `--max-samples 2 --max-layers 1`
and use a C1 artifact built from that same two-video prefix manifest.  Such an
artifact is intentionally marked incomplete and cannot be used for extraction.

After a complete calibration, normal feature extraction loads it without
refitting CCA:

```bash
conda run -n vlm3r python scripts/probing/extract_depth_probe_features.py \
  --model-label c2_ss_ca --model-loading-mode pre_sft_fusion \
  --pre-sft-fusion-variant c1_ss_cross_attn_v1 \
  --c2-calibration-path /home/shaoruei/probe_outputs/c2_ss_ca.pt \
  --model-path /mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2 \
  --siglip-path /mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384 \
  --feature-root /mnt/DATA_SSD/shaoruei/probing_data/cut3r_features \
  --spatial-features-subdir '6:spatial_features_dec_6;9:spatial_features_dec_9;12:spatial_features' \
  --forward-frames-root /mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1 \
  --probe-targets-root /mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1 \
  --device cuda:0 --device-map auto --dtype float16
```

For a CCA head, the saved affine maps implement
`(Q_C1 - mu_q) A` and `(K_C1 - mu_k) B`, where
`A = Sigma_XX_reg^-1/2 U` and `B = Sigma_YY_reg^-1/2 V`.  Statistics are
float64.  The saved scalar uses C1's declared target:
`s_qk = sqrt(sigma_target_C1 / sigma_raw_C2)`, because V1 multiplies both Q
and K by `s_qk`.  Residual alpha is recomputed with C1's stated aggregation:
median over samples of `RMS(delta)/RMS(pre-injection visual hidden)`.

V1 has no camera tokens under C1 (C1 rejects that configuration), so all
native geometry tokens are aligned CUT3R patch tokens.  C2 does not define a
V2/camera-token variant.
