#!/bin/bash
#SBATCH -A EUHPC_D32_006
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH -J vlm3r_pi3x_scannet_extract
#SBATCH -o /leonardo/home/userexternal/shuang00/VLM-3R/logs/%x-%j.out
#SBATCH -e /leonardo/home/userexternal/shuang00/VLM-3R/logs/%x-%j.err

module purge
module load profile/deeplrn
module load cuda/12.1

source ~/.bashrc
conda activate vlm3r

export CUDA_HOME=/leonardo/prod/opt/compilers/cuda/12.1/none
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

REPO=/leonardo/home/userexternal/shuang00/VLM-3R

DATA_ROOT=/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r

python $REPO/scripts/extract_spatial_features_pi3x.py \
  --input-dir $DATA_ROOT/scannet/videos \
  --output-dir $DATA_ROOT/scannet/spatial_features_pi3x \
  --pi3x-weights-path /leonardo_scratch/fast/EUHPC_D32_006/hf_models/Pi3X \
  --processor-config-path $REPO/processor_config.json \
  --filter-by-jsons \
    $DATA_ROOT/VLM-3R-DATA/vsibench_train/merged_qa_scannet_train.json \
    $DATA_ROOT/VLM-3R-DATA/vsibench_train/merged_qa_route_plan_train.json \
  --gpu-ids 0 \
  --batch-size 1 \
  --frames-upbound 32 \
  --precision bf16
