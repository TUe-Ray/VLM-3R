# SpatialFocus 🔭

SpatialFocus is an ongoing research codebase for spatial reasoning from monocular video, built on top of [VLM-3R](https://github.com/VITA-Group/VLM-3R).

The project explores query-guided visual representations that emphasize spatially relevant information for vision-language reasoning. It extends the VLM-3R framework with components for training, evaluation, offline deployment, and spatial feature extraction in GPU cluster environments.

> This repository is under active development. Some components may change as the research evolves.

## Prerequisites ✅

- Linux environment with GPU support
- Conda (Miniconda or Mambaforge recommended)
- CUDA-compatible GPUs
- Fast local or scratch storage for model caches and datasets

## Project Structure 🗂️

```text
SpatialFocus/
|-- llava/                    # Core LLaVA-NeXT / VLM-3R model code
|-- third_party/
|   |-- CUT3R/                # Geometry encoder submodule
|   |-- EoMT/                 # EoMT submodule
|   |-- Pi3/                  # Pi3 submodule
|   `-- VGGT/                 # VGGT spatial encoder submodule
|-- thinking-in-space/        # VSiBench / VSTiBench evaluation framework
|-- scripts/                  # Training, inference, and utility scripts
|-- vlm_3r_data_process/      # Data processing pipeline
|-- playground/demo/          # Demo videos and images
|-- eval_vsi_snellius.sh      # Example cluster evaluation job
`-- README.md
```

External dependencies are included as git submodules where needed.

## Installation 🛠️

### 1. Clone the repository

```bash
git clone <your-repo-url> SpatialFocus
cd SpatialFocus
git submodule update --init --recursive
```

### 2. Create the training and inference environment

```bash
conda create -n vlm3r python=3.10 -y
conda activate vlm3r

pip install --upgrade pip
conda install pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

pip install -e ".[train]"
pip install flash-attn==2.7.1.post1 --no-build-isolation
pip install decord openai accelerate==0.29.1
```

Tested package versions in this environment include:

- PyTorch 2.1.1 (CUDA 12.1)
- FlashAttention 2.7.1.post1
- DeepSpeed 0.14.4
- Transformers 4.40.0.dev0
- PEFT 0.4.0
- Accelerate 0.29.1

### 3. Create the evaluation environment

```bash
conda create -n vsibench python=3.10 -y
conda activate vsibench

conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.vsibench.txt

cd thinking-in-space
pip install -e .
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
cd ..
```

Tested package versions in this environment include:

- PyTorch 2.1.1 (CUDA 12.1)
- FlashAttention 2.7.3
- Transformers 4.40.0
- PEFT 0.10.0
- s2wrapper 0.1

### 4. Build CUT3R

Run the following from the repository root with the vlm3r environment activated:

```bash
conda activate vlm3r

cd third_party/CUT3R
pip install -r requirements.txt

cd src/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../..
```

To download the CUT3R checkpoint:

```bash
cd third_party/CUT3R/src
pip install gdown
gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link
cd ../../..
```

### 5. Set up VGGT

VGGT is included as a submodule under `third_party/VGGT`. The spatial encoder adapter can load weights from Hugging Face by default:

```bash
--spatial_tower vggt \
--spatial_tower_select_feature all_tokens \
--spatial_feature_dim 2048 \
--vggt_weights_path facebook/VGGT-1B
```

For offline cluster jobs, pre-cache the VGGT checkpoint and pass the local path with `--vggt_weights_path /path/to/VGGT-1B`.

## Offline Cluster Setup 🧊

This repository supports offline GPU cluster workflows, where compute nodes do not have internet access.

Before submitting jobs to offline nodes, all required models and datasets should be cached in advance.

### Step 1. Pre-cache assets on a node with internet access

If you use the legacy helper script:

```bash
bash prepare_offline_cache.sh
```

Expected cached assets include:

- Journey9ni/vlm-3r-llava-qwen2-lora (LoRA weights)
- lmms-lab/LLaVA-NeXT-Video-7B-Qwen2 (base model)
- google/siglip-so400m-patch14-384 (vision encoder)
- nyu-visionx/VSI-Bench (evaluation dataset)

If any repository requires authentication, set HF_TOKEN in your environment before downloading. 🔐

### Step 2. Submit an evaluation job

```bash
sbatch eval_vsi_snellius.sh
```

The script is configured for offline execution with HF_*_OFFLINE=1 enabled.

Useful environment variables include:

| Variable | Default | Description |
|---|---|---|
| FAST_ROOT | /path/to/scratch | Fast scratch storage root |
| HF_HOME | $FAST_ROOT/hf_cache | Hugging Face cache directory |
| MODEL_ROOT | $FAST_ROOT/hf_models/VLM3R | Local model directory |
| NUM_PROCESSES | 4 | Number of GPUs |
| MAX_FRAMES_NUM | 32 | Maximum video frames per sample |

## Evaluation 📏

### VSiBench (thinking-in-space native script)

Run from the thinking-in-space directory with the vsibench environment activated:

```bash
conda activate vsibench
cd thinking-in-space
bash eval_vlm_3r_vsibench.sh
```

### VSTiBench (thinking-in-space native script)

```bash
conda activate vsibench
cd thinking-in-space
bash eval_vlm_3r_vstibench.sh
```

### VSiBench (cluster parity script at repository root)

```bash
conda activate vsibench
sbatch eval_vsi_snellius.sh
```

## Geometry Retention vs Correctness Diagnostic

Use `scripts/analysis/analyze_geometry_retention_vs_correctness.py` to test whether VSI-Bench samples with better hidden-state retention of CUT3R token similarity are more likely to be answered correctly. The prediction file must already contain `correctness` as `1` or `0`; the script intentionally does not guess answer matching.

Recommended first run:

```bash
python scripts/analysis/analyze_geometry_retention_vs_correctness.py \
    --model_path /path/to/original_vlm3r_checkpoint \
  --data_json /path/to/vsi_val.json \
  --prediction_json /path/to/original_predictions_with_correctness.json \
  --spatial_feature_dir /path/to/cut3r_features \
  --output_dir outputs/geometry_retention_original \
  --layers 1,4 \
  --anchors_per_frame 128 \
  --positive_top_percent 10 \
  --negative_bottom_percent 30 \
  --negative_mode bottom \
  --seed 42
```

For LoRA checkpoints that need a separate base model, add `--model_base /path/to/base_model`. If media paths in the JSON are relative, set `--image_folder` and `--video_folder`. Use `--sample_start` with `--num_samples` for smoke tests that should begin at a known covered dataset index. The script uses VLM-3R visual metadata to exclude text, answer, padding, newline, special, camera/prefix, and alignment-only tokens.

The CUT3R sidecar feature directory must cover the same scenes as the eval JSON. If only a subset of datasets has sidecars, filter the eval and prediction JSONs to that subset before running the diagnostic; otherwise the skip summary will report missing feature files.

Main outputs:

- `geometry_retention_per_sample.csv`: one row per sample, layer, and representation.
- `category_correct_vs_wrong.csv`: correct-vs-wrong geometry gap, rank accuracy, margin loss, and bootstrap confidence intervals by category.
- `overall_correlations.csv`: point-biserial and Spearman associations between correctness and geometry metrics.
- `geometry_bins_accuracy.csv` and `plots/*.png`: quintile-bin accuracy and category plots.
- `geometry_retention_triplets.csv`: sampled triplets when `--save_per_triplet true` is set.

Interpretation: if correct samples have higher `geometry_gap_mean` or `geometry_rank_acc`, CUT3R relational topology retained in LLM visual states is aligned with spatial reasoning success. If the effect appears mainly in `Abs Dist` or `Rel Dist`, the latent ranking is most useful for local metric reasoning. Weak trends for `Room Size`, `Rel Dir`, or `Route Plan` suggest that global layout, viewpoint, or navigation may need richer targets such as physical 3D coordinates, object-level relations, semihard negatives, H4 supervision, or layout-aware supervision.

## Pre-extracting Spatial Features ⚡

Use the extraction pipeline to precompute spatial features before training:

```bash
python scripts/extract_spatial_features.py \
  --input-dir /path/to/video/dataset \
  --output-dir /path/to/save/extracted_features \
  --cut3r-weights-path third_party/CUT3R/src/cut3r_512_dpt_4_64.pth \
  --processor-config-path processor_config.json \
  --gpu-ids 0,1,2,3
```

## Notes 📝

- This repository is still under active development.
- Interfaces, scripts, and directory structures may change as the project evolves.
- Several workflows are currently optimized for offline SLURM-based GPU clusters.
- Depending on your environment, you may need to adjust paths, cache locations, and job settings.

## License 📄

This repository is released under the Apache License 2.0. See [LICENSE](LICENSE).

Some third-party components may be distributed under different licenses. In particular:

- CUT3R is licensed under CC BY-NC-SA 4.0.
- VGGT is distributed under its own license in `third_party/VGGT/LICENSE.txt`; note that VGGT checkpoint licenses differ between the original and commercial-use checkpoints.
- This may impose restrictions on commercial use.

Review all dependency licenses before production or commercial use.

## Acknowledgements 🙏

This project builds on several excellent open-source projects:

- [VLM-3R](https://github.com/VITA-Group/VLM-3R)
- [CUT3R](https://github.com/CUT3R/CUT3R)
- [VGGT](https://github.com/facebookresearch/vggt)
- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [thinking-in-space](https://github.com/vision-x-nyu/thinking-in-space)
