#!/bin/bash

# =============================================================================
# test_recaller_combos_v2.sh - 测试不同 recaller 组合 (V2 版本)
# =============================================================================
# 
# 用法: ./test_recaller_combos_v2.sh <dataset> <gpu_id>
#
# 示例:
#   ./test_recaller_combos_v2.sh Amazon_All_Beauty 0
#   ./test_recaller_combos_v2.sh ml-1m 1
#
# =============================================================================

export PYTHONPATH=<path_to_routepo>
export MASTER_PORT=12368
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export WANDB_MODE=disabled

# 检查参数
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <dataset> <gpu_id>"
    echo ""
    echo "Examples:"
    echo "  $0 Amazon_All_Beauty 0"
    echo "  $0 ml-1m 1"
    exit 1
fi

DATASET=$1
GPU_ID=$2

cd <path_to_routepo>
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 基础参数
profile_cutoff=20
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"

# 定义要测试的 recaller 组合
RECALLER_COMBOS=(
    # Two-model combinations
    "BPR SASRec"
    "BPR ItemKNN"
    "BPR LightGCN"
    "SASRec ItemKNN"
    "SASRec LightGCN"
    "ItemKNN LightGCN"
    # Three-model combinations
    "BPR SASRec ItemKNN"
    "BPR SASRec LightGCN"
    "BPR ItemKNN LightGCN"
    "SASRec ItemKNN LightGCN"
    # Four-model combination
    "BPR SASRec ItemKNN LightGCN"
)

echo "================================================"
echo "Testing Recaller Combinations (V2)"
echo "================================================"
echo "Dataset: $DATASET"
echo "GPU: $GPU_ID"
echo "Model: $MODEL_NAME"
echo "Number of combinations: ${#RECALLER_COMBOS[@]}"
echo "================================================"

# 创建结果目录
mkdir -p results

# 汇总文件
SUMMARY_FILE="results/summary_v2_${DATASET}_$(date +%Y%m%d_%H%M%S).txt"
echo "Testing recaller combinations (V2) for $DATASET" > $SUMMARY_FILE
echo "Started at: $(date)" >> $SUMMARY_FILE
echo "================================================" >> $SUMMARY_FILE

# 测试每个组合
for combo in "${RECALLER_COMBOS[@]}"; do
    echo ""
    echo "================================================"
    echo "Testing combination: $combo"
    echo "================================================"
    
    # 转换空格分隔的字符串为数组
    read -ra MODELS <<< "$combo"
    combo_name=$(echo $combo | tr ' ' '_')
    
    echo "[Step 1/3] Generating SFT data for $combo_name..."
    python GRPO/models/main_pure_v2.py \
        --dataset $DATASET \
        --data_path dataset \
        --checkpoint_dir ./checkpoints \
        --output_dir GRPO/data/pure_models \
        --model_name $MODEL_NAME \
        --recbole_models ${MODELS[@]} \
        --gen_sft_data \
        --final_k 50 \
        --seed 42 \
        --padding_side left \
        --profile_cutoff $profile_cutoff
    
    if [ $? -ne 0 ]; then
        echo "Error generating data for $combo_name, skipping..."
        echo "FAILED: $combo_name (data generation)" >> $SUMMARY_FILE
        continue
    fi
    
    echo "[Step 2/3] Training SFT model for $combo_name..."
    python GRPO/models/main_pure_v2.py \
        --dataset $DATASET \
        --data_path dataset \
        --checkpoint_dir ./checkpoints \
        --output_dir GRPO/data/pure_models \
        --model_name $MODEL_NAME \
        --recbole_models ${MODELS[@]} \
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
        echo "Error training model for $combo_name, skipping..."
        echo "FAILED: $combo_name (training)" >> $SUMMARY_FILE
        continue
    fi
    
    echo "[Step 3/3] Testing SFT model for $combo_name..."
    python GRPO/models/main_pure_v2.py \
        --dataset $DATASET \
        --data_path dataset \
        --checkpoint_dir ./checkpoints \
        --output_dir GRPO/data/pure_models \
        --model_name $MODEL_NAME \
        --recbole_models ${MODELS[@]} \
        --do_test_sft \
        --final_k 50 \
        --seed 42 \
        --padding_side left \
        --profile_cutoff $profile_cutoff
    
    if [ $? -eq 0 ]; then
        echo "SUCCESS: $combo_name" >> $SUMMARY_FILE
        # 从 JSON 结果文件提取关键指标
        result_file="results/pure_v2_results_${DATASET}_${combo_name}.json"
        if [ -f "$result_file" ]; then
            echo "  Results file: $result_file" >> $SUMMARY_FILE
            python -c "
import json
with open('$result_file') as f:
    r = json.load(f)
print(f\"  Accuracy: {r.get('accuracy', 'N/A'):.4f}\")
print(f\"  F1 Macro: {r.get('f1_macro', 'N/A'):.4f}\")
print(f\"  Predicted NDCG: {r.get('avg_predicted_ndcg', 'N/A'):.4f}\")
print(f\"  True Best NDCG: {r.get('avg_true_ndcg', 'N/A'):.4f}\")
if 'base_model_results' in r:
    for name, res in r['base_model_results'].items():
        print(f\"  {name} NDCG: {res['avg_ndcg']:.4f}\")
" >> $SUMMARY_FILE 2>/dev/null
        fi
    else
        echo "FAILED: $combo_name (testing)" >> $SUMMARY_FILE
    fi
    
    echo ""
done

echo "================================================"
echo "All combinations tested!"
echo "Summary saved to: $SUMMARY_FILE"
echo "Individual results saved in: results/pure_v2_results_${DATASET}_*.json"
echo "================================================"

echo "" >> $SUMMARY_FILE
echo "Completed at: $(date)" >> $SUMMARY_FILE

# 打印汇总
cat $SUMMARY_FILE










