#!/bin/bash
# =============================================================================
# run_train_pure_grid.sh - 枚举 datasets × combinations × model_names，
# 每种配置在一个 tmux 窗口中运行 train_pure.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_pure.sh"
SESSION_NAME="Food_re"

# =============================================================================
# 在这里手动配置三个列表
# =============================================================================
datasets=(
    # "ml-1m"
    # "steam"
    # "Amazon_Books"
    "Food"
)

combinations=(
    "ItemKNN LightGCN Pop"
    # "EASE RecVAE SLIMElastic NeuMF"
    # "BPR SASRec LightGCN"
    # "SASRec ItemKNN LightGCN"
)

model_names=(
    "meta-llama/Llama-3.2-1B-Instruct"
    # "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen3-4B-Instruct-2507"
)

get_max_length() {
    case "$1" in
        Food) echo 11024 ;;
        *)    echo 1536 ;;
    esac
}

# =============================================================================
# 启动
# =============================================================================
total=$(( ${#datasets[@]} * ${#combinations[@]} * ${#model_names[@]} ))

echo "================================================"
echo "Train Pure Grid Launch"
echo "================================================"
echo "Datasets:      ${datasets[*]}"
echo "Combinations:  ${#combinations[@]}"
echo "Models:        ${#model_names[@]}"
echo "Total runs:    $total"
echo "Tmux session:  $SESSION_NAME"
echo "================================================"

# 清理已有 session
tmux kill-session -t "$SESSION_NAME" 2>/dev/null
tmux new-session -d -s "$SESSION_NAME" -n "control"
tmux send-keys -t "$SESSION_NAME:control" \
    "echo 'Train Pure Grid: $total runs. Use tmux list-windows -t $SESSION_NAME'" Enter

counter=0
base_port=12400

# DATA_EVAL_GPU cycles over GPUs 0-7 per window (for data gen and evaluation)
num_data_eval_gpus=8
for dataset in "${datasets[@]}"; do
    for combo in "${combinations[@]}"; do
        for model in "${model_names[@]}"; do
            port=$((base_port + counter))
            data_eval_gpu=$((counter % num_data_eval_gpus))
            model_short=$(basename "$model")
            combo_short=$(echo "$combo" | tr ' ' '_')
            window_name="${dataset}__${combo_short}__${model_short}"
            window_name="${window_name//[-.]/_}"
            window_name="${window_name:0:60}"

            ml=$(get_max_length "$dataset")
            echo "[$((counter+1))/$total] $window_name (port $port, DATA_EVAL_GPU=$data_eval_gpu, MAX_LENGTH=$ml)"

            tmux new-window -t "$SESSION_NAME" -n "$window_name"
            tmux send-keys -t "$SESSION_NAME:$window_name" \
                "export TRAIN_MODELS='$combo' TRAIN_MODEL_NAME='$model' MASTER_PORT=$port DATA_EVAL_GPU=$data_eval_gpu MAX_LENGTH=$ml && source ~/.bashrc && conda activate pp && p bash $TRAIN_SCRIPT $dataset" \
                Enter

            counter=$((counter + 1))
        done
    done
done

echo ""
echo "Launched $counter runs in tmux session: $SESSION_NAME"
echo "  tmux attach -t $SESSION_NAME"
echo "  tmux list-windows -t $SESSION_NAME"
echo "  tmux kill-session -t $SESSION_NAME"
