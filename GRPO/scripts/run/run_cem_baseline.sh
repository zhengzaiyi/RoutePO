#!/bin/bash
# =============================================================================
# CEM Baseline Experiments
# Run CEM-based fusion optimization across different datasets and recaller combos
# =============================================================================

set -e  # Exit on error
export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
# Configuration
DATA_PATH="./dataset"
CHECKPOINT_DIR="./checkpoints"
OUTPUT_DIR="results/cem"
SEED=42
FINAL_K=50
MIN_THRES=0.001
TRAIN_K=20

# main_pure test dataset config (set PURE_MODEL_NAME and PURE_PROFILE_CUTOFF to match train_pure.sh)
PURE_OUTPUT_DIR="GRPO/data/pure_models"
PURE_MODEL_NAME="${PURE_MODEL_NAME:-meta-llama/Llama-3.2-1B-Instruct}"
PURE_PROFILE_CUTOFF="${PURE_PROFILE_CUTOFF:-500000}"

# CEM parameters
CEM_ITERS=20
CEM_POPULATION=256
CEM_ELITE_FRAC=0.1

# User limits (set to empty string for all users)
NUM_TRAIN_USERS=""
NUM_TEST_USERS=""

# Datasets to run
DATASETS=("steam" "Food")

# Recaller combinations to test
# Each combination is a space-separated list of models
declare -a RECALLER_COMBOS=(
    "LightGCN ItemKNN Pop"
)

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="$OUTPUT_DIR/cem_experiments_$(date +%Y%m%d_%H%M%S).log"
echo "CEM Baseline Experiments" | tee "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# Counter for experiments
TOTAL_EXPERIMENTS=$((${#DATASETS[@]} * ${#RECALLER_COMBOS[@]}))
CURRENT=0

# Run experiments
for DATASET in "${DATASETS[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "============================================" | tee -a "$LOG_FILE"
    echo "Dataset: $DATASET" | tee -a "$LOG_FILE"
    echo "============================================" | tee -a "$LOG_FILE"
    
    for COMBO in "${RECALLER_COMBOS[@]}"; do
        CURRENT=$((CURRENT + 1))
        
        # Convert combo string to array
        read -ra MODELS <<< "$COMBO"
        COMBO_NAME=$(IFS="_"; echo "${MODELS[*]}")
        
        # Build test dataset path (matching main_pure.py get_paths convention)
        PURE_MODEL_SHORT=$(basename "$PURE_MODEL_NAME")
        SORTED_MODELS=($(echo "${MODELS[@]}" | tr ' ' '\n' | sort | tr '\n' ' '))
        SORTED_COMBO=$(IFS="_"; echo "${SORTED_MODELS[*]}")
        TEST_DATASET_PATH="${PURE_OUTPUT_DIR}/${DATASET}/${PURE_MODEL_SHORT}_pure_sft_data_${SORTED_COMBO}_${PURE_PROFILE_CUTOFF}/test"
        
        echo "" | tee -a "$LOG_FILE"
        echo "[$CURRENT/$TOTAL_EXPERIMENTS] Running: $DATASET with ${MODELS[*]}" | tee -a "$LOG_FILE"
        echo "Time: $(date)" | tee -a "$LOG_FILE"
        echo "-------------------------------------------" | tee -a "$LOG_FILE"
        
        # Build command
        CMD="python GRPO/baselines/baseline_cem.py \
            --dataset $DATASET \
            --data_path $DATA_PATH \
            --checkpoint_dir $CHECKPOINT_DIR \
            --output_dir $OUTPUT_DIR \
            --recbole_models ${MODELS[*]} \
            --final_k $FINAL_K \
            --cem_iters $CEM_ITERS \
            --cem_population $CEM_POPULATION \
            --cem_elite_frac $CEM_ELITE_FRAC \
            --seed $SEED \
            --min_thres $MIN_THRES \
            --train_k $TRAIN_K"
        
        # Add user limits if specified
        if [ -n "$NUM_TRAIN_USERS" ]; then
            CMD="$CMD --num_train_users $NUM_TRAIN_USERS"
        fi
        if [ -n "$NUM_TEST_USERS" ]; then
            CMD="$CMD --num_test_users $NUM_TEST_USERS"
        fi
        if [ -d "$TEST_DATASET_PATH" ]; then
            CMD="$CMD --test_dataset_path $TEST_DATASET_PATH"
            echo "Using pre-generated test dataset: $TEST_DATASET_PATH" | tee -a "$LOG_FILE"
        else
            echo "Warning: test dataset not found at $TEST_DATASET_PATH, using inter_dataset" | tee -a "$LOG_FILE"
        fi
        
        echo "Command: $CMD" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        
        # Run the experiment
        if $CMD 2>&1 | tee -a "$LOG_FILE"; then
            echo "✅ Success: $DATASET - $COMBO_NAME" | tee -a "$LOG_FILE"
        else
            echo "❌ Failed: $DATASET - $COMBO_NAME" | tee -a "$LOG_FILE"
        fi
        
        echo "" | tee -a "$LOG_FILE"
    done
done

echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "All experiments completed at: $(date)" | tee -a "$LOG_FILE"
echo "Results saved to: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

# Summary of results
echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "RESULTS SUMMARY" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

for result_file in "$OUTPUT_DIR"/cem_results_*.json; do
    if [ -f "$result_file" ]; then
        echo "" | tee -a "$LOG_FILE"
        echo "File: $(basename $result_file)" | tee -a "$LOG_FILE"
        # Extract key metrics using python
        python -c "
import json
with open('$result_file') as f:
    data = json.load(f)
    config = data.get('config', {})
    cem = data.get('cem_fusion', {}).get('cem_optimized', {})
    weights = data.get('cem_fusion', {}).get('optimized_weights', {})
    print(f\"  Dataset: {config.get('dataset', 'N/A')}\")
    print(f\"  Recallers: {config.get('recaller_combo', 'N/A')}\")
    print(f\"  NDCG@50: {cem.get('ndcg@50', 0):.4f}\")
    print(f\"  Recall@50: {cem.get('recall@50', 0):.4f}\")
    print(f\"  Weights: {weights}\")
" 2>/dev/null || echo "  (Could not parse results)" | tee -a "$LOG_FILE"
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "Done!" | tee -a "$LOG_FILE"

