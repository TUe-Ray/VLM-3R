# 🔭 SpatialFocus

> **Spatial understanding from monocular video — built on VLM-3R.**

This project extends [VLM-3R](https://github.com/VITA-Group/VLM-3R) with 3D reconstructive instruction tuning for training and evaluating Vision-Language Models on spatial reasoning tasks. Designed to run on **offline GPU clusters** where compute nodes have no internet access.

**Author:** Ruei · s.huang5@student.tue.nl

---

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Offline Setup](#offline-setup-no-internet-compute-nodes)
- [Evaluation](#evaluation)
- [Inference (Demo)](#inference-demo)
- [Pre-extracting Spatial Features](#pre-extracting-spatial-features-optional)
- [Security Notes](#security-notes)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## ✅ Prerequisites

- HPC cluster with GPU nodes
- Conda (Miniconda or Mambaforge)
- Fast scratch storage for model cache (models are large, ~20GB+)

---

## 🗂️ Project Structure

```
SpatialFocus/
├── llava/                    # Core LLaVA-NeXT model code
├── CUT3R/                    # Geometry encoder (submodule)
├── thinking-in-space/        # VSiBench / VSTiBench eval framework (submodule)
├── third_party/eomt/         # EOMT (submodule)
├── scripts/                  # Training, inference, and utility scripts
├── vlm_3r_data_process/      # Data generation pipeline
├── playground/demo/          # Demo videos/images
├── prepare_offline_cache.sh  # 📥 Pre-cache assets for offline nodes
├── eval_offline.sh           # 🚀 Offline eval SBATCH job
└── README.md
```

---

## 🛠️ Installation

### 1. Clone and Initialize Submodules

```bash
git clone <your-repo-url> SpatialFocus
cd SpatialFocus
git submodule update --init --recursive
```

---

### 2. 🧠 Environment: `vlm3r` — Training & Inference

```bash
conda create -n vlm3r python=3.10 -y
conda activate vlm3r

pip install --upgrade pip
conda install pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

pip install -e ".[train]"
pip install flash-attn==2.7.1.post1 --no-build-isolation
pip install decord openai accelerate==0.29.1
```

<details>
<summary>📦 Key packages</summary>

| Package | Version |
|---|---|
| PyTorch | 2.1.1 (CUDA 12.1) |
| FlashAttention | 2.7.1.post1 |
| DeepSpeed | 0.14.4 |
| Transformers | 4.40.0.dev0 |
| PEFT | 0.4.0 |
| Accelerate | 0.29.1 |

</details>

---

### 3. 📊 Environment: `vsibench` — Evaluation

```bash
conda create -n vsibench python=3.10 -y
conda activate vsibench

conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

cd thinking-in-space
pip install -e .
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
pip install flash-attn==2.7.3 --no-build-isolation
pip install transformers==4.40.0 peft==0.10.0 huggingface_hub[hf_xet]
cd ..
```

<details>
<summary>📦 Key packages</summary>

| Package | Version |
|---|---|
| PyTorch | 2.1.1 (CUDA 12.1) |
| FlashAttention | 2.7.3 |
| Transformers | 4.40.0 |
| PEFT | 0.10.0 |
| s2wrapper | 0.1 |

</details>

---

### 4. ⚙️ Build CUT3R (Geometry Encoder)

```bash
conda activate vlm3r

cd CUT3R
pip install -r requirements.txt

# Build the RoPE CUDA extension
cd src/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../..

# Download the CUT3R checkpoint
cd src
pip install gdown
gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link
cd ../..
```

---

## 📦 Offline Setup (No-Internet Compute Nodes)

Compute nodes without internet need all models and datasets pre-cached **before** submitting a job.

### Step 1 — Pre-cache on a Node with Internet

```bash
bash prepare_offline_cache.sh
```

This downloads and symlinks into `$MODEL_ROOT`:

| Asset | Type |
|---|---|
| `Journey9ni/vlm-3r-llava-qwen2-lora` | LoRA weights |
| `lmms-lab/LLaVA-NeXT-Video-7B-Qwen2` | Base model |
| `google/siglip-so400m-patch14-384` | Vision encoder |
| `nyu-visionx/VSI-Bench` | Evaluation dataset |

> 💡 Set `HF_TOKEN` in your environment if any repo requires authentication.

### Step 2 — Submit the Evaluation Job

```bash
sbatch eval_offline.sh
```

Runs on 1 node / 4 GPUs with `HF_*_OFFLINE=1` enforced. Configure via environment variables:

| Variable | Default | Description |
|---|---|---|
| `FAST_ROOT` | `/path/to/scratch` | Fast scratch storage root |
| `HF_HOME` | `$FAST_ROOT/hf_cache` | HuggingFace cache directory |
| `MODEL_ROOT` | `$FAST_ROOT/hf_models/VLM3R` | Local model directory |
| `NUM_PROCESSES` | `4` | Number of GPUs |
| `MAX_FRAMES_NUM` | `32` | Max video frames per sample |

---

## 📏 Evaluation

### VSiBench

```bash
conda activate vsibench
cd thinking-in-space
bash eval_vlm_3r_vsibench.sh
```

### VSTiBench

```bash
conda activate vsibench
cd thinking-in-space
bash eval_vlm_3r_vstibench.sh
```

---

## 🎬 Inference (Demo)

**From video:**
```bash
conda activate vlm3r
CUDA_VISIBLE_DEVICES=0 bash scripts/video/demo/video_demo.sh \
    Journey9ni/vlm-3r-llava-qwen2-lora \
    qwen_1_5 32 2 average grid True \
    playground/demo/47334096.mp4 \
    lmms-lab/LLaVA-NeXT-Video-7B-Qwen2
```

**From images:**
```bash
conda activate vlm3r
bash scripts/image/demo/image_demo.sh \
    Journey9ni/vlm-3r-llava-qwen2-lora \
    qwen_1_5 2 average grid True \
    playground/demo/scene_47334096_imgs \
    lmms-lab/LLaVA-NeXT-Video-7B-Qwen2
```

---

## ⚡ Pre-extracting Spatial Features (Optional)

Pre-compute CUT3R features to speed up training — the training script detects and loads them automatically.

```bash
python scripts/extract_spatial_features.py \
    --input-dir /path/to/video/dataset \
    --output-dir /path/to/save/extracted_features \
    --cut3r-weights-path CUT3R/src/CUT3R_weights.pth \
    --processor-config-path processor_config.json \
    --gpu-ids 0,1,2,3
```

Expected data layout:
```
your_data_folder/
├── videos/
│   └── scene0191_00.mp4
└── spatial_features/
    └── scene0191_00.pt
```

---

## 🔐 Security Notes

- All credentials (`HF_TOKEN`, `OPENAI_API_KEY`, `WANDB_API_KEY`) must be passed via environment variables — never hardcode them.
- ⚠️ The `scripts/archived/` directory contains legacy scripts from the upstream project with hardcoded credentials. **Do not push it to any public repository.**

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

> ⚠️ The CUT3R geometry encoder is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), which **restricts commercial use**. Ensure you comply with all dependency licenses before use.

---

## 🙏 Acknowledgements

This project is based on [VLM-3R](https://github.com/VITA-Group/VLM-3R) by Zhiwen Fan et al. (CVPR 2026).

Built on top of:
- [CUT3R](https://github.com/CUT3R/CUT3R) — Spatial feature encoder
- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) — Base VLM framework
- [thinking-in-space](https://github.com/vision-x-nyu/thinking-in-space) — Evaluation framework

If you use VLM-3R in your work, please cite the original paper:

```bibtex
@misc{fan2025vlm3rvisionlanguagemodelsaugmented,
      title={VLM-3R: Vision-Language Models Augmented with Instruction-Aligned 3D Reconstruction}, 
      author={Zhiwen Fan and Jian Zhang and Renjie Li and Junge Zhang and Runjin Chen and Hezhen Hu and Kevin Wang and Huaizhi Qu and Shijie Zhou and Dilin Wang and Zhicheng Yan and Hongyu Xu and Justin Theiss and Tianlong Chen and Jiachen Li and Zhengzhong Tu and Zhangyang Wang and Rakesh Ranjan},
      year={2025},
      eprint={2505.20279},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.20279}, 
}
```
