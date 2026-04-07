#!/bin/bash

# =============================================================================
# test_score_norms_tmux.sh - 在tmux session中为每个score_norm创建窗口并测试
# =============================================================================
# 
# 用法: ./test_score_norms_tmux.sh <dataset> <gpu_id> [combo_size] [session_name]
#
# 示例:
#   ./test_score_norms_tmux.sh Steam 0          # 默认3个model的组合
#   ./test_score_norms_tmux.sh Steam 0 3        # 3个model的组合
#   ./test_score_norms_tmux.sh Steam 0 2 score_test  # 自定义session名称
#
# =============================================================================

# 检查参数
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <dataset> <gpu_id> [combo_size] [session_name]"
    echo ""
    echo "Examples:"
    echo "  $0 Steam 0                    # 默认3个model的组合"
    echo "  $0 Steam 0 3                  # 3个model的组合"
    echo "  $0 Steam 0 2 score_test       # 自定义session名称"
    exit 1
fi

DATASET=$1
GPU_ID=$2
COMBO_SIZE=${3:-3}  # 默认组合大小为3
SESSION_NAME=${4:-"score_norm_test_${DATASET}"}

# 定义所有要测试的score_norm方法
SCORE_NORMS=(
    "none"
    "minmax"
    "zscore"
    "softmax"
    "percentile"
    "rank_reciprocal"
    "rank_exp"
    # "platt"  # platt需要额外参数，可以单独测试
)

# 脚本路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/test_baseline_pairs.sh"

# 检查测试脚本是否存在
if [ ! -f "$TEST_SCRIPT" ]; then
    echo "Error: Test script not found: $TEST_SCRIPT"
    exit 1
fi

echo "================================================"
echo "Setting up tmux session for score_norm testing"
echo "================================================"
echo "Dataset: $DATASET"
echo "GPU: $GPU_ID"
echo "Combo size: $COMBO_SIZE"
echo "Session name: $SESSION_NAME"
echo "Number of score_norms: ${#SCORE_NORMS[@]}"
echo "Score norms: ${SCORE_NORMS[@]}"
echo "================================================"

# 创建或连接到tmux session
if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Creating new tmux session: $SESSION_NAME"
    tmux new-session -d -s "$SESSION_NAME" -n "main"
    # 在第一个窗口显示session信息
    tmux send-keys -t "$SESSION_NAME:main" "echo 'Session: $SESSION_NAME'; echo 'Dataset: $DATASET'; echo 'Testing ${#SCORE_NORMS[@]} score_norm methods'; echo ''; echo 'Windows created:'" C-m
    sleep 1
else
    echo "Using existing tmux session: $SESSION_NAME"
fi

# 为每个score_norm创建窗口并运行测试
for score_norm in "${SCORE_NORMS[@]}"; do
    window_name="norm_${score_norm}"
    
    echo "Creating window: $window_name for score_norm=$score_norm"
    
    # 创建新窗口
    tmux new-window -t "$SESSION_NAME" -n "$window_name"
    
    # 在新窗口中运行测试命令
    # 注意：需要先切换到正确的目录
    tmux send-keys -t "$SESSION_NAME:$window_name" "cd <path_to_routepo>" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo '=============================================='" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo 'Testing score_norm: $score_norm'" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo 'Dataset: $DATASET'" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo 'Combo size: $COMBO_SIZE'" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo \"Started at: \$(date)\"" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo '=============================================='" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo ''" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "$TEST_SCRIPT $DATASET $GPU_ID $COMBO_SIZE $score_norm" C-m
    
    # 短暂延迟，确保窗口创建完成
    sleep 0.5
done

# 列出所有窗口
echo ""
echo "================================================"
echo "All windows created!"
echo "================================================"
echo "Session name: $SESSION_NAME"
echo "Total windows: $(( ${#SCORE_NORMS[@]} + 1 ))"  # +1 for main window
echo ""
echo "Windows:"
tmux list-windows -t "$SESSION_NAME" -F "  #{window_index}: #{window_name}"
echo ""
echo "To attach to the session, run:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "To detach from tmux, press: Ctrl+b, then d"
echo "To switch windows: Ctrl+b, then window number (0-9)"
echo "To list windows: Ctrl+b, then w"
echo ""

# 可选：自动连接到session（注释掉如果需要手动连接）
# echo "Attaching to session..."
# tmux attach -t "$SESSION_NAME"
