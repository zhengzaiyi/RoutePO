#!/bin/bash
# =============================================================================
# run_param_study_grid.sh - Parameter study: history length & recall item number
#
# Sweeps over:
#   PARAM_TYPE=history_length  -> profile_cutoff values
#   PARAM_TYPE=prompt_top_k    -> prompt_top_k values
#
# Each (dataset, combo, model, param_value) runs the full train_pure.sh pipeline
# in a separate tmux window.
# =============================================================================
export CUDA_VISIBLE_DEVICES=4,5,6,7
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/../train/train_pure.sh"
SESSION_NAME="param_ptk_ml"

# =============================================================================
# Configuration
# =============================================================================
datasets=(
    # "Food"
    # "steam"
    "ml-1m"
)

combinations=(
    "ItemKNN LightGCN Pop"
)

model_names=(
    "meta-llama/Llama-3.2-1B-Instruct"
)

# Parameter study values
history_lengths=(
    # 0
    # 5 
    # 10 
    # 15
    20 
    # 25 
    # 50
    # 500000
)
prompt_top_ks=(
    # 0 
    1 
    # 3 
    5 
    10
)

# Which study to run: "history_length", "prompt_top_k", or "both"
PARAM_TYPE="${PARAM_TYPE:-prompt_top_k}"

get_max_length() {
    case "$1" in
        Food) echo 11024 ;;
        *)    echo 1536 ;;
    esac
}

# =============================================================================
# Build param list
# =============================================================================
declare -a param_configs=()

if [ "$PARAM_TYPE" = "history_length" ] || [ "$PARAM_TYPE" = "both" ]; then
    for hl in "${history_lengths[@]}"; do
        param_configs+=("hl_${hl}:${hl}:3")
    done
fi

if [ "$PARAM_TYPE" = "prompt_top_k" ] || [ "$PARAM_TYPE" = "both" ]; then
    for ptk in "${prompt_top_ks[@]}"; do
        param_configs+=("ptk_${ptk}:20:${ptk}")
    done
fi

total=$(( ${#datasets[@]} * ${#combinations[@]} * ${#model_names[@]} * ${#param_configs[@]} ))

echo "================================================"
echo "Parameter Study Grid Launch"
echo "================================================"
echo "Datasets:      ${datasets[*]}"
echo "Combinations:  ${#combinations[@]}"
echo "Models:        ${#model_names[@]}"
echo "PARAM_TYPE:    $PARAM_TYPE"
echo "Param configs: ${#param_configs[@]}"
echo "Total runs:    $total"
echo "Tmux session:  $SESSION_NAME"
echo "================================================"
for pc in "${param_configs[@]}"; do
    IFS=':' read -r name cutoff ptk <<< "$pc"
    echo "  $name -> profile_cutoff=$cutoff, prompt_top_k=$ptk"
done
echo "================================================"

tmux kill-session -t "$SESSION_NAME" 2>/dev/null
tmux new-session -d -s "$SESSION_NAME" -n "control"
tmux send-keys -t "$SESSION_NAME:control" \
    "echo 'Param Study: $total runs. Use tmux list-windows -t $SESSION_NAME'" Enter

counter=0
base_port=12600
num_data_eval_gpus=8

for dataset in "${datasets[@]}"; do
    for combo in "${combinations[@]}"; do
        for model in "${model_names[@]}"; do
            for pc in "${param_configs[@]}"; do
                IFS=':' read -r param_name cutoff ptk <<< "$pc"
                sleep_num=$((counter * 28800 + 28800))
                port=$((base_port + counter))
                data_eval_gpu=$((counter % num_data_eval_gpus))
                model_short=$(basename "$model")
                combo_short=$(echo "$combo" | tr ' ' '_')
                window_name="${dataset}__${combo_short}__${param_name}"
                window_name="${window_name//[-.]/_}"
                window_name="${window_name:0:60}"

                ml=$(get_max_length "$dataset")
                echo "[$((counter+1))/$total] $window_name (port $port, GPU=$data_eval_gpu, cutoff=$cutoff, ptk=$ptk)"

                tmux new-window -t "$SESSION_NAME" -n "$window_name"
                tmux send-keys -t "$SESSION_NAME:$window_name" \
                    "sleep 1 && export TRAIN_MODELS='$combo' TRAIN_MODEL_NAME='$model' MASTER_PORT=$port DATA_EVAL_GPU=$data_eval_gpu MAX_LENGTH=$ml PARAM_PROFILE_CUTOFF=$cutoff PARAM_PROMPT_TOP_K=$ptk && source ~/.bashrc && conda activate pp && bash $TRAIN_SCRIPT $dataset" \
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
