#!/bin/bash
# =============================================================================
# run_ablation_grid.sh - Launch ablation study across datasets × combos × models
# Each (dataset, combo, model) gets 3 ablation variants in separate tmux windows
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/../train/train_pure_ablation.sh"
SESSION_NAME="ablation_food_rl"

# =============================================================================
# Configuration
# =============================================================================
datasets=(
    "Food"
    # "steam"
    # "ml-1m"
)

combinations=(
    "ItemKNN LightGCN Pop"
)

model_names=(
    "meta-llama/Llama-3.2-1B-Instruct"
    # "Qwen/Qwen3-4B-Instruct-2507"
)

ablation_modes=(
    # "no_sft"
    "no_rl"
    # "no_snack"
)

get_max_length() {
    case "$1" in
        Food) echo 11024 ;;
        *)    echo 1536 ;;
    esac
}

# =============================================================================
# Launch
# =============================================================================
total=$(( ${#datasets[@]} * ${#combinations[@]} * ${#model_names[@]} * ${#ablation_modes[@]} ))

echo "================================================"
echo "Ablation Study Grid Launch"
echo "================================================"
echo "Datasets:       ${datasets[*]}"
echo "Combinations:   ${#combinations[@]}"
echo "Models:         ${#model_names[@]}"
echo "Ablation modes: ${ablation_modes[*]}"
echo "Total runs:     $total"
echo "Tmux session:   $SESSION_NAME"
echo "================================================"

tmux kill-session -t "$SESSION_NAME" 2>/dev/null
tmux new-session -d -s "$SESSION_NAME" -n "control"
tmux send-keys -t "$SESSION_NAME:control" \
    "echo 'Ablation Grid: $total runs. Use tmux list-windows -t $SESSION_NAME'" Enter

counter=0
base_port=12500
num_data_eval_gpus=8

for dataset in "${datasets[@]}"; do
    for combo in "${combinations[@]}"; do
        for model in "${model_names[@]}"; do
            for ablation in "${ablation_modes[@]}"; do
                port=$((base_port + counter))
                data_eval_gpu=$((counter % num_data_eval_gpus))
                model_short=$(basename "$model")
                combo_short=$(echo "$combo" | tr ' ' '_')
                window_name="${dataset}__${combo_short}__${model_short}__${ablation}"
                window_name="${window_name//[-.]/_}"
                window_name="${window_name:0:60}"

                ml=$(get_max_length "$dataset")
                echo "[$((counter+1))/$total] $window_name (port $port, GPU=$data_eval_gpu)"

                tmux new-window -t "$SESSION_NAME" -n "$window_name"
                tmux send-keys -t "$SESSION_NAME:$window_name" \
                    "export TRAIN_MODELS='$combo' TRAIN_MODEL_NAME='$model' MASTER_PORT=$port DATA_EVAL_GPU=$data_eval_gpu MAX_LENGTH=$ml ABLATION_MODE=$ablation && source ~/.bashrc && conda activate pp && bash $TRAIN_SCRIPT $dataset" \
                    Enter

                counter=$((counter + 1))
            done
        done
    done
done

echo ""
echo "Launched $counter runs in tmux session: $SESSION_NAME"
echo "  tmux attach -t $SESSION_NAME"
echo "  tmux list-windows -t $SESSION_NAME"
echo "  tmux kill-session -t $SESSION_NAME"
