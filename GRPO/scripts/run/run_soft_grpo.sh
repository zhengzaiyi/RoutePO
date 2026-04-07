#!/bin/bash

# SofT-GRPO Quick Test Script
# Usage: ./run_soft_grpo.sh [dataset] [stage] [model_size]
# Example: ./run_soft_grpo.sh ml-1m full small

set -e

DATASET=${1:-"ml-1m"}
STAGE=${2:-"full"}  # Options: data, sft, grpo, test, full
MODEL_SIZE=${3:-"small"}  # Options: small, large

# Model selection
if [ "$MODEL_SIZE" = "small" ]; then
    MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
    BATCH_SIZE=4
    GRAD_ACCUM=4
elif [ "$MODEL_SIZE" = "large" ]; then
    MODEL_NAME="Llama-3.2-1B-Instruct" 
    BATCH_SIZE=2
    GRAD_ACCUM=8
else
    echo "Unknown model size: $MODEL_SIZE"
    exit 1
fi

# Dataset-specific recallers
if [ "$DATASET" = "ml-1m" ]; then
    RECALLERS="BPR SASRec"
elif [ "$DATASET" = "steam" ]; then
    RECALLERS="BPR SASRec LightGCN"
elif [ "$DATASET" = "Amazon_All_Beauty" ]; then
    RECALLERS="BPR SASRec Pop"
else
    RECALLERS="BPR SASRec"
fi

echo "======================================"
echo "SofT-GRPO Training Pipeline"
echo "Dataset: $DATASET"
echo "Stage: $STAGE"
echo "Model: $MODEL_NAME"
echo "Recallers: $RECALLERS"
echo "======================================"

# Common arguments
COMMON_ARGS="--dataset $DATASET --data_path ./dataset --recbole_models $RECALLERS --seed 42"
TRAIN_ARGS="--model_name $MODEL_NAME --per_device_train_batch_size $BATCH_SIZE --gradient_accumulation_steps $GRAD_ACCUM"

# SofT-GRPO specific args
GRPO_ARGS="--tau_gumbel 1.0 --top_p 0.9 --epsilon 0.2 --beta 0.01 --num_generations 4 --grpo_lr 1e-6 --grpo_epochs 1"

cd "$(dirname "$0")/.."

case $STAGE in
    "data")
        echo "Generating SFT data..."
        python GRPO/models/main_pure.py $COMMON_ARGS --gen_sft_data
        ;;
    "sft")
        echo "Running SFT training..."
        python GRPO/models/main_pure.py $COMMON_ARGS $TRAIN_ARGS --do_sft --learning_rate 5e-5 --num_train_epochs 3
        ;;
    "grpo")
        echo "Running SofT-GRPO training..."
        python GRPO/models/main_pure.py $COMMON_ARGS $TRAIN_ARGS $GRPO_ARGS --do_grpo
        ;;
    "test")
        echo "Testing model..."
        python GRPO/models/main_pure.py $COMMON_ARGS --do_test
        ;;
    "full")
        echo "Running full pipeline: Data -> SFT -> GRPO -> Test"
        
        echo "Step 1: Generating data..."
        python GRPO/models/main_pure.py $COMMON_ARGS --gen_sft_data
        
        echo "Step 2: SFT training..."
        python GRPO/models/main_pure.py $COMMON_ARGS $TRAIN_ARGS --do_sft --learning_rate 5e-5 --num_train_epochs 3
        
        echo "Step 3: SofT-GRPO training..."
        python GRPO/models/main_pure.py $COMMON_ARGS $TRAIN_ARGS $GRPO_ARGS --do_grpo
        
        echo "Step 4: Testing..."
        python GRPO/models/main_pure.py $COMMON_ARGS --do_test
        ;;
    "standalone")
        echo "Running standalone SofT-GRPO..."
        
        echo "Generating GRPO data..."
        python GRPO/models/soft_grpo.py $COMMON_ARGS --gen_data
        
        echo "Training SofT-GRPO..."
        python GRPO/models/soft_grpo.py $COMMON_ARGS $TRAIN_ARGS $GRPO_ARGS --do_train
        
        echo "Evaluating..."
        python GRPO/models/soft_grpo.py $COMMON_ARGS --do_eval
        ;;
    "debug")
        echo "Running debug mode with small steps..."
        python GRPO/models/main_pure.py $COMMON_ARGS $TRAIN_ARGS $GRPO_ARGS \
            --gen_sft_data --do_sft --do_grpo --do_test \
            --learning_rate 1e-4 --num_train_epochs 1 \
            --max_steps 50 --logging_steps 5 --save_steps 25 \
            --per_device_train_batch_size 1 --gradient_accumulation_steps 2
        ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Available stages: data, sft, grpo, test, full, standalone, debug"
        exit 1
        ;;
esac

echo "======================================"
echo "SofT-GRPO $STAGE completed successfully!"
echo "======================================"
