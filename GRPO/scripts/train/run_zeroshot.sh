#!/bin/bash

# Zero-shot Recaller Selection Script

# Default values
CUDA_DEVICE=${1:-0}
DATASET=${2:-ml-1m}

echo "Running Zero-shot Recaller Selection"
echo "CUDA_VISIBLE_DEVICES: $CUDA_DEVICE"
echo "Dataset: $DATASET"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
PYTHONPATH=~/AmazonReviews2023 \
WANDB_MODE=disabled \
python GRPO/models/main_zeroshot.py \
    --dataset "$DATASET" \
    --data_path dataset \
    --checkpoint_dir ./checkpoints \
    --model_name Qwen/Qwen3-4B-Instruct-2507 \
    --recbole_models itemknn lightgcn pop \
    --eval_k 50 \
    --profile_cutoff 20 \
    --prompt_top_k 3 \
    --max_new_tokens 32 \
    --seed 42 \
    --bf16 \
    --max_samples 100
