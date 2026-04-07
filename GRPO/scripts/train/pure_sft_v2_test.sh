#!/bin/bash

# =============================================================================
# main_pure_v2.py V3 测试脚本 (无数据泄露版本)
# =============================================================================
# 
# 用法: ./pure_sft_v2_test.sh <dataset> <gpu_id>
#
# 示例:
#   ./pure_sft_v2_test.sh Amazon_All_Beauty 0
#   ./pure_sft_v2_test.sh Amazon_Video_Games 1
#
# =============================================================================

export PYTHONPATH=<path_to_routepo>
export MASTER_PORT=12367
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export WANDB_MODE=disabled

# 检查参数
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <dataset> <gpu_id>"
    echo ""
    echo "Examples:"
    echo "  $0 Amazon_All_Beauty 0"
    echo "  $0 Amazon_Video_Games 1"
    exit 1
fi

DATASET=$1
GPU_ID=$2

cd <path_to_routepo>
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 基础参数
profile_cutoff=20
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
RECBOLE_MODELS="BPR SASRec FPMC SimpleX LightGCN ItemKNN"

echo "================================================"
echo "Main Pure V3 Test (No Data Leakage)"
echo "================================================"
echo "Dataset: $DATASET"
echo "GPU: $GPU_ID"
echo "Model: $MODEL_NAME"
echo "================================================"

# Step 1: 生成数据 (V3: 一次生成 train/eval/test)
echo ""
echo "================================================"
echo "[Step 1/3] Generating SFT data (V3)..."
echo "================================================"
python GRPO/models/main_pure_v2.py \
    --dataset $DATASET \
    --data_path dataset \
    --checkpoint_dir ./checkpoints \
    --output_dir GRPO/data/pure_models \
    --model_name $MODEL_NAME \
    --recbole_models $RECBOLE_MODELS \
    --gen_sft_data \
    --final_k 50 \
    --seed 42 \
    --padding_side left \
    --profile_cutoff $profile_cutoff

if [ $? -ne 0 ]; then
    echo "Error: Data generation failed!"
    exit 1
fi

Step 2: 训练模型
echo ""
echo "================================================"
echo "[Step 2/3] Training SFT model (V3)..."
echo "================================================"
python GRPO/models/main_pure_v2.py \
    --dataset $DATASET \
    --data_path dataset \
    --checkpoint_dir ./checkpoints \
    --output_dir GRPO/data/pure_models \
    --model_name $MODEL_NAME \
    --recbole_models $RECBOLE_MODELS \
    --do_sft \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --warmup_steps 100 \
    --logging_steps 20 \
    --save_steps 500 \
    --eval_steps 500 \
    --max_length 1536 \
    --final_k 50 \
    --seed 42 \
    --gradient_checkpointing \
    --padding_side left \
    --profile_cutoff $profile_cutoff

if [ $? -ne 0 ]; then
    echo "Error: Training failed!"
    exit 1
fi

# Step 3: 测试模型
echo ""
echo "================================================"
echo "[Step 3/3] Testing SFT model (V3)..."
echo "================================================"
python GRPO/models/main_pure_v2.py \
    --dataset $DATASET \
    --data_path dataset \
    --checkpoint_dir ./checkpoints \
    --output_dir GRPO/data/pure_models \
    --model_name $MODEL_NAME \
    --recbole_models $RECBOLE_MODELS \
    --do_test_sft \
    --final_k 50 \
    --seed 42 \
    --padding_side left \
    --profile_cutoff $profile_cutoff

if [ $? -ne 0 ]; then
    echo "Error: Testing failed!"
    exit 1
fi

echo ""
echo "================================================"
echo "V3 Test completed successfully!"
echo "Dataset: $DATASET"
echo "================================================"
