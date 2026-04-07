#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# DATASET=Musical_Instruments
# export MASTER_PORT=12346
# export CUDA_VISIBLE_DEVICES=2

# DATASET=Sports_and_Outdoors
# export MASTER_PORT=12344
# export CUDA_VISIBLE_DEVICES=0

export MASTER_PORT=12366
# export CUDA_VISIBLE_DEVICES=4

# DATASET=Gift_Cards
# export MASTER_PORT=12347
# export CUDA_VISIBLE_DEVICES=3

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export WANDB_PROJECT="pure-sft"
# export WANDB_MODE=disabled

# Usage: ./pure_sft.sh <dataset_name> <gpu_ids> [models]
# Example: ./pure_sft.sh Amazon_All_Beauty 0           (single GPU)
# Example: ./pure_sft.sh Amazon_All_Beauty 0,1,2,3     (multi GPU)
# Example: ./pure_sft.sh Amazon_All_Beauty 0,1 "SASRec ItemKNN LightGCN"

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <dataset_name> <gpu_ids> [models]"
    echo "Example: $0 Amazon_All_Beauty 0           (single GPU)"
    echo "Example: $0 Amazon_All_Beauty 0,1,2,3     (multi GPU)"
    echo "Example: $0 Amazon_All_Beauty 0,1 \"SASRec ItemKNN LightGCN\""
    exit 1
fi

# 运行命令
cd "$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=$2

# Count GPUs for multi-GPU training
IFS=',' read -ra GPU_ARRAY <<< "$2"
NUM_GPUS=${#GPU_ARRAY[@]}
if [ $NUM_GPUS -gt 1 ]; then
    SFT_LAUNCH="torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT"
else
    SFT_LAUNCH="python"
fi
profile_cutoff=20
model_name=meta-llama/Llama-3.2-1B-Instruct
train_k=50
eval_k=50
# model_name=microsoft/deberta-v3-base
# model_name=mistralai/Ministral-3-3B-Base-2512

models="${3:-SASRec ItemKNN LightGCN}"

echo "================================================"
echo "Dataset: $1"
echo "GPU: $2"
echo "Model: $model_name"
echo "================================================"

echo "================================================"
echo "Generating pure SFT data..."
echo "================================================"
python GRPO/models/main_pure.py \
    --dataset $1 \
    --data_path dataset \
    --checkpoint_dir ./checkpoints \
    --output_dir GRPO/data/pure_models \
    --model_name $model_name \
    --recbole_models $models\
    --gen_sft_test \
    --train_k $train_k \
    --eval_k $eval_k \
    --seed 42 \
    --padding_side left \
    --random_history_selection \
    --profile_cutoff $profile_cutoff \
    --gen_sft_train \
    --gen_sft_eval \
    --autoregressive \

echo "================================================"
echo "Training pure SFT model... (GPUs: $2, num_processes: $NUM_GPUS)"
echo "================================================"
$SFT_LAUNCH GRPO/models/main_pure.py \
    --dataset $1 \
    --data_path dataset \
    --checkpoint_dir ./checkpoints \
    --output_dir GRPO/data/pure_models \
    --model_name $model_name \
    --recbole_models $models\
    --do_sft \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --warmup_steps 100 \
    --logging_steps 20 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --max_length 1536 \
    --train_k $train_k \
    --eval_k $eval_k \
    --seed 42 \
    --bf16 \
    --gradient_checkpointing \
    --padding_side left \
    --random_history_selection \
    --profile_cutoff $profile_cutoff \
    --autoregressive \

echo "================================================"
echo "Training pure SFT model (not autoregressive)... (GPUs: $2, num_processes: $NUM_GPUS)"
echo "================================================"
$SFT_LAUNCH GRPO/models/main_pure.py \
    --dataset $1 \
    --data_path dataset \
    --checkpoint_dir ./checkpoints \
    --output_dir GRPO/data/pure_models \
    --model_name $model_name \
    --recbole_models $models\
    --do_sft \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --warmup_steps 100 \
    --logging_steps 20 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --max_length 1536 \
    --train_k $train_k \
    --eval_k $eval_k \
    --seed 42 \
    --bf16 \
    --gradient_checkpointing \
    --padding_side left \
    --random_history_selection \
    --profile_cutoff $profile_cutoff \

echo "================================================" 
echo "Testing pure SFT model..."
echo "================================================"
python GRPO/models/main_pure.py \
    --dataset $1 \
    --data_path dataset \
    --checkpoint_dir ./checkpoints \
    --output_dir GRPO/data/pure_models \
    --model_name $model_name \
    --recbole_models $models\
    --do_test_sft \
    --train_k $train_k \
    --eval_k $eval_k \
    --seed 42 \
    --padding_side left \
    --random_history_selection \
    --profile_cutoff $profile_cutoff \
    --merge_method top_k \


echo "================================================"
echo "Pure SFT training completed!"
echo "================================================"
