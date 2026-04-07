#!/bin/bash

# =============================================================================
# run_all_sft_combos.sh - 对所有recaller三模型组合，分别在tmux窗口中运行pure_sft.sh
# =============================================================================
#
# 用法: ./run_all_sft_combos.sh <dataset> <gpu_ids> [combo_size] [session_name]
#
# 参数:
#   dataset       - 数据集名称 (如 ml-1m, Amazon_All_Beauty)
#   gpu_ids       - 可用GPU列表，逗号分隔 (如 0,1,2,3)
#   combo_size    - 组合大小，默认3
#   session_name  - tmux session名称，默认 sft_combos
#
# 示例:
#   ./run_all_sft_combos.sh ml-1m 0,1,2,3           # 用GPU 0-3轮流跑所有3模型组合
#   ./run_all_sft_combos.sh ml-1m 0,1 2              # 用GPU 0,1轮流跑所有2模型组合
#   ./run_all_sft_combos.sh ml-1m 4,5,6,7 3 my_sess  # 自定义session名称
#
# GPU分配策略: 轮流分配 (round-robin)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PURE_SFT_SCRIPT="$SCRIPT_DIR/pure_sft.sh"

# 检查参数
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <dataset> <gpu_ids> [combo_size] [session_name]"
    echo ""
    echo "Arguments:"
    echo "  dataset       - Dataset name (e.g., ml-1m, Amazon_All_Beauty)"
    echo "  gpu_ids       - Comma-separated GPU IDs (e.g., 0,1,2,3)"
    echo "  combo_size    - Combination size (default: 3)"
    echo "  session_name  - Tmux session name (default: sft_combos)"
    echo ""
    echo "Examples:"
    echo "  $0 ml-1m 0,1,2,3              # 3-model combos on GPUs 0-3"
    echo "  $0 ml-1m 0,1 2                # 2-model combos on GPUs 0,1"
    echo "  $0 ml-1m 4,5,6,7 3 my_sess    # Custom session name"
    exit 1
fi

DATASET=$1
GPU_IDS_STR=$2
COMBO_SIZE=${3:-3}
SESSION_NAME=${4:-sft_combos}

# 检查pure_sft.sh是否存在
if [ ! -f "$PURE_SFT_SCRIPT" ]; then
    echo "Error: pure_sft.sh not found at: $PURE_SFT_SCRIPT"
    exit 1
fi

# 解析GPU ID列表
IFS=',' read -ra GPU_IDS <<< "$GPU_IDS_STR"
NUM_GPUS=${#GPU_IDS[@]}

if [ $NUM_GPUS -eq 0 ]; then
    echo "Error: No GPU IDs provided"
    exit 1
fi

# 定义所有可用的recallers
ALL_RECALLERS=("BPR" "SASRec" "ItemKNN" "LightGCN" "Pop" "SASRec")
TOTAL_RECALLERS=${#ALL_RECALLERS[@]}

# 验证组合大小
if [ $COMBO_SIZE -lt 1 ] || [ $COMBO_SIZE -gt $TOTAL_RECALLERS ]; then
    echo "Error: combo_size must be between 1 and $TOTAL_RECALLERS"
    exit 1
fi

# -------------------------------------------------------
# 递归函数生成所有组合 C(n, k)
# 参数: start_idx, remaining_k, current_combo_str
# -------------------------------------------------------
generate_combinations() {
    local start_idx=$1
    local remaining_k=$2
    local current_combo_str="$3"
    local i
    local max_idx
    local new_combo_str
    
    if [ $remaining_k -eq 0 ]; then
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

# -------------------------------------------------------
# 生成所有组合
# -------------------------------------------------------
combos=()
generate_combinations 0 $COMBO_SIZE ""

TOTAL_COMBOS=${#combos[@]}

echo "================================================"
echo "Run Pure SFT for All Recaller Combinations"
echo "================================================"
echo "Dataset:         $DATASET"
echo "Available GPUs:  ${GPU_IDS[@]} ($NUM_GPUS GPUs)"
echo "Combo size:      $COMBO_SIZE"
echo "All recallers:   ${ALL_RECALLERS[@]}"
echo "Total combos:    $TOTAL_COMBOS"
echo "Tmux session:    $SESSION_NAME"
echo "SFT script:      $PURE_SFT_SCRIPT"
echo "================================================"
echo ""

# -------------------------------------------------------
# 自动kill已有的同名tmux session，避免冲突
# -------------------------------------------------------
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Killing existing tmux session: $SESSION_NAME"
    tmux kill-session -t "$SESSION_NAME"
fi

# -------------------------------------------------------
# 创建tmux session
# -------------------------------------------------------
echo "Creating new tmux session: $SESSION_NAME"
tmux new-session -d -s "$SESSION_NAME" -n "control"
# 在control窗口显示摘要信息
tmux send-keys -t "$SESSION_NAME:control" \
    "echo '=== SFT Combos Control Window ===' && echo 'Dataset: $DATASET' && echo 'Total combos: $TOTAL_COMBOS' && echo 'GPUs: ${GPU_IDS[@]}' && echo 'Use: tmux list-windows -t $SESSION_NAME to see all windows'" \
    Enter

# -------------------------------------------------------
# 为每个组合创建tmux窗口并运行
# -------------------------------------------------------
gpu_idx=0

for ((c=0; c<$TOTAL_COMBOS; c++)); do
    combo="${combos[$c]}"
    # 生成窗口名：用下划线连接模型名，如 BPR_SASRec_ItemKNN
    window_name=$(echo "$combo" | tr ' ' '_')
    
    # 轮流分配GPU (round-robin)
    gpu_id=${GPU_IDS[$((gpu_idx % NUM_GPUS))]}
    gpu_idx=$((gpu_idx + 1))
    
    echo "[$((c+1))/$TOTAL_COMBOS] Window: $window_name | GPU: $gpu_id | Models: $combo"
    
    # 在tmux中新建窗口并运行pure_sft.sh
    tmux new-window -t "$SESSION_NAME" -n "$window_name"
    tmux send-keys -t "$SESSION_NAME:$window_name" \
        "echo '=== Combo $((c+1))/$TOTAL_COMBOS: $combo (GPU $gpu_id) ===' && p bash $PURE_SFT_SCRIPT $DATASET $gpu_id \"$combo\"" \
        Enter
done

echo ""
echo "================================================"
echo "All $TOTAL_COMBOS combinations launched!"
echo "================================================"
echo ""
echo "Tmux session: $SESSION_NAME"
echo "GPU assignment (round-robin over ${GPU_IDS[@]}):"
echo ""

# 打印GPU分配汇总表
gpu_idx=0
for ((c=0; c<$TOTAL_COMBOS; c++)); do
    combo="${combos[$c]}"
    window_name=$(echo "$combo" | tr ' ' '_')
    gpu_id=${GPU_IDS[$((gpu_idx % NUM_GPUS))]}
    gpu_idx=$((gpu_idx + 1))
    printf "  [%2d] GPU %s : %s\n" $((c+1)) "$gpu_id" "$combo"
done

echo ""
echo "Useful commands:"
echo "  tmux attach -t $SESSION_NAME            # 进入session"
echo "  tmux list-windows -t $SESSION_NAME       # 列出所有窗口"
echo "  tmux select-window -t $SESSION_NAME:N    # 切换到第N个窗口"
echo "  tmux kill-session -t $SESSION_NAME       # 终止整个session"
echo ""
