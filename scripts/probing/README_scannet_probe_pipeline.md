# Generic ScanNet Probe Pipeline

This wrapper runs the fixed ScanNet semantic/depth probe pipeline for any saved
VLM-3R checkpoint path.

## Original-like model

```bash
bash scripts/probing/submit_scannet_probe_pipeline.sh \
  --model-label my_model_12345678 \
  --model-path /leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/my_model_12345678 \
  --arch-preset original
```

Default feature levels:

```text
siglip_output,projected_features,layer_0,layer_1,layer_2,layer_3,layer_6,layer_9,layer_12,layer_15,layer_18,layer_21,layer_24,layer_27
```

## SpatialStack-like model

```bash
bash scripts/probing/submit_scannet_probe_pipeline.sh \
  --model-label my_spatialstack_12345678 \
  --model-path /leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/my_spatialstack_12345678 \
  --arch-preset spatialstack
```

Default feature levels:

```text
siglip_output,projected_features,layer_0,layer_1,layer_2,layer_3,layer_6,layer_9,layer_12,layer_15,layer_18,layer_21,layer_24,layer_27
```

## Wait for a training job

```bash
bash scripts/probing/submit_scannet_probe_pipeline.sh \
  --model-label my_model_12345678 \
  --model-path /leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/my_model_12345678 \
  --arch-preset original \
  --train-dependency auto
```

With `auto`, the script tries to infer the trailing numeric job id from the
model label or checkpoint directory name. You can also pass it explicitly:

```bash
  --training-job-id 12345678 --train-dependency auto
```

## Custom feature levels

The default is the complete new-model policy above. Pass `--feature-levels`
only when intentionally completing missing layers; a subset is allowed for
that workflow and does not trigger a full rerun.

```bash
bash scripts/probing/submit_scannet_probe_pipeline.sh \
  --model-label my_model_12345678 \
  --model-path /leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/my_model_12345678 \
  --arch-preset llm_only \
  --feature-levels layer_1,layer_2,layer_3,layer_6
```

## Dry run

```bash
bash scripts/probing/submit_scannet_probe_pipeline.sh \
  --model-label my_model_12345678 \
  --model-path /leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/my_model_12345678 \
  --arch-preset original \
  --dry-run
```

The dry run writes the task TSV and prints the expected output paths, but does
not submit Slurm jobs.

## Outputs

Per-layer probe metrics are written under:

```text
outputs/scannet_semantic_full/depth_probes_scannet/<model_label>/<feature>/metrics.json
outputs/scannet_semantic_full/semantic_probes_scannet/<model_label>/<feature>/metrics.json
```

Aggregated outputs:

```text
outputs/scannet_semantic_full/depth_probe_scannet_<model_label>_results.csv
outputs/scannet_semantic_full/semantic_probe_scannet_<model_label>_results.csv
```
