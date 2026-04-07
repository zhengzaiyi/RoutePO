#!/bin/bash

# =============================================================================
# test_baseline_pairs.sh - 测试所有recallers组合的baseline性能
# =============================================================================
# 
# 用法: ./test_baseline_pairs.sh <dataset> <gpu_id> [combo_size]
#
# 示例:
#   ./test_baseline_pairs.sh ml-1m 0          # 默认3个model的组合
#   ./test_baseline_pairs.sh ml-1m 0 3        # 3个model的组合
#   ./test_baseline_pairs.sh ml-1m 0 2        # 2个model的组合(pair)
#   ./test_baseline_pairs.sh Amazon_All_Beauty 1 4  # 4个model的组合
#
# =============================================================================

export PYTHONPATH=<path_to_routepo>
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export WANDB_MODE=disabled

# 检查参数
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <dataset> <gpu_id> [combo_size]"
    echo ""
    echo "Examples:"
    echo "  $0 ml-1m 0          # 默认3个model的组合"
    echo "  $0 ml-1m 0 3        # 3个model的组合"
    echo "  $0 ml-1m 0 2        # 2个model的组合(pair)"
    echo "  $0 Amazon_All_Beauty 1 4  # 4个model的组合"
    exit 1
fi

DATASET=$1
GPU_ID=$2
COMBO_SIZE=${3:-3}  # 默认组合大小为3
SCORE_NORM=${4:-minmax}

cd <path_to_routepo>
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 基础参数
profile_cutoff=20
final_k=50
seed=42
score_norm=$SCORE_NORM
# 定义所有可用的recallers
ALL_RECALLERS=("BPR" "SASRec" "ItemKNN" "LightGCN" "Pop" "SimpleX")
TOTAL_RECALLERS=${#ALL_RECALLERS[@]}

# 验证组合大小
if [ $COMBO_SIZE -lt 1 ] || [ $COMBO_SIZE -gt $TOTAL_RECALLERS ]; then
    echo "Error: combo_size must be between 1 and $TOTAL_RECALLERS"
    exit 1
fi

echo "================================================"
echo "Testing All Recaller Combinations Baseline Performance"
echo "================================================"
echo "Dataset: $DATASET"
echo "GPU: $GPU_ID"
echo "Combo size: $COMBO_SIZE"
echo "Available recallers: ${ALL_RECALLERS[@]}"
echo "Total recallers: $TOTAL_RECALLERS"
echo "================================================"

# 创建结果目录
mkdir -p results

# 递归函数生成所有组合 C(n, k)
# 参数: start_idx, remaining_k, current_combo_str
generate_combinations() {
    local start_idx=$1
    local remaining_k=$2
    local current_combo_str="$3"
    local i
    local max_idx
    local new_combo_str
    
    if [ $remaining_k -eq 0 ]; then
        # 组合完成，添加到全局数组
        combos+=("$current_combo_str")
        return
    fi
    
    max_idx=$((TOTAL_RECALLERS - remaining_k))
    for ((i=$start_idx; i<=$max_idx; i++)); do
        if [ -z "$current_combo_str" ]; then
            new_combo_str="${ALL_RECALLERS[$i]}"
        else
            new_combo_str="$current_combo_str ${ALL_RECALLERS[$i]}"
        fi
        generate_combinations $((i+1)) $((remaining_k-1)) "$new_combo_str"
    done
}

# 生成所有组合
combos=()
if [ $COMBO_SIZE -eq 1 ]; then
    # 特殊情况：单个model
    for recaller in "${ALL_RECALLERS[@]}"; do
        combos+=("$recaller")
    done
elif [ $COMBO_SIZE -eq 2 ]; then
    # 特殊情况：两两组合（保持原逻辑）
    for i in "${!ALL_RECALLERS[@]}"; do
        for j in "${!ALL_RECALLERS[@]}"; do
            if [ $i -lt $j ]; then
                combos+=("${ALL_RECALLERS[$i]} ${ALL_RECALLERS[$j]}")
            fi
        done
    done
elif [ $COMBO_SIZE -eq 3 ]; then
    # 特殊情况：三个组合（三层循环，性能更好）
    for i in "${!ALL_RECALLERS[@]}"; do
        for j in "${!ALL_RECALLERS[@]}"; do
            for k in "${!ALL_RECALLERS[@]}"; do
                if [ $i -lt $j ] && [ $j -lt $k ]; then
                    combos+=("${ALL_RECALLERS[$i]} ${ALL_RECALLERS[$j]} ${ALL_RECALLERS[$k]}")
                fi
            done
        done
    done
else
    # 通用情况：使用递归函数（支持任意大小的组合）
    generate_combinations 0 $COMBO_SIZE ""
fi

echo "Total combinations to test: ${#combos[@]}"
echo ""

# 汇总文件
SUMMARY_FILE="results/baseline_combos_summary_${DATASET}_size${COMBO_SIZE}_$(date +%Y%m%d_%H%M%S).txt"
echo "Baseline Performance for Recaller Combinations (size=$COMBO_SIZE) - $DATASET" > $SUMMARY_FILE
echo "Started at: $(date)" >> $SUMMARY_FILE
echo "Total combinations: ${#combos[@]}" >> $SUMMARY_FILE
echo "Combo size: $COMBO_SIZE" >> $SUMMARY_FILE
echo "================================================" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# 测试每个组合
combo_num=0
for combo in "${combos[@]}"; do
    combo_num=$((combo_num + 1))
    combo_name=$(echo "$combo" | tr ' ' '_')
    
    echo ""
    echo "================================================"
    echo "[$combo_num/${#combos[@]}] Testing: $combo"
    echo "================================================"
    
    # 运行baseline测试
    python GRPO/models/main_pure.py \
        --dataset $DATASET \
        --data_path dataset \
        --checkpoint_dir ./checkpoints \
        --output_dir GRPO/data/pure_models \
        --model_name meta-llama/Llama-3.2-1B-Instruct \
        --recbole_models $combo \
        --test_baseline \
        --final_k $final_k \
        --seed $seed \
        --padding_side left \
        --random_history_selection \
        --profile_cutoff $profile_cutoff \
        --merge_method top_k \
        --score_norm $score_norm
    
    # 检查结果文件是否存在
    result_file="results/baseline_results_${DATASET}_${combo_name}.json"
    if [ -f "$result_file" ]; then
        echo "✅ Results saved to: $result_file"
        
        # 提取关键指标并写入汇总文件
        if command -v jq &> /dev/null; then
            # 如果有jq，提取NDCG@50和Recall@50
            ndcg50=$(jq -r '.avg_score_weight."ndcg@50" // "N/A"' $result_file 2>/dev/null)
            recall50=$(jq -r '.avg_score_weight."recall@50" // "N/A"' $result_file 2>/dev/null)
            echo "$combo_name: NDCG@50=$ndcg50, Recall@50=$recall50" >> $SUMMARY_FILE
        else
            echo "$combo_name: Results saved (use jq to extract metrics)" >> $SUMMARY_FILE
        fi
    else
        echo "❌ Warning: Result file not found: $result_file"
        echo "$combo_name: FAILED" >> $SUMMARY_FILE
    fi
done

echo ""
echo "================================================"
echo "All combinations tested!"
echo "================================================"
echo "Summary saved to: $SUMMARY_FILE"
echo ""
echo "Individual results saved to:"
echo "  results/baseline_results_${DATASET}_*.json"
echo ""
