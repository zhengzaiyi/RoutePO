#!/bin/bash
# =============================================================================
# Inference Latency Benchmark
# Measure per-sample recall-channel latency and extra overhead for
# GRPO, CEM, and PG across datasets and recaller combinations.
# =============================================================================

set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda activate pp
export CUDA_VISIBLE_DEVICES=2,3,6,7
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"

# ---- Configuration ---------------------------------------------------------
DATA_PATH="./dataset"
CHECKPOINT_DIR="./checkpoints"
OUTPUT_DIR="results/latency"
SEED=42
EVAL_K=50

# GRPO / pure model config
PURE_OUTPUT_DIR="GRPO/data/pure_models"
PURE_MODEL_NAME="${PURE_MODEL_NAME:-meta-llama/Llama-3.2-1B-Instruct}"
PURE_PROFILE_CUTOFF="${PURE_PROFILE_CUTOFF:-500000}"
MAX_LENGTH="${MAX_LENGTH:-1536}"

# Optional: limit test samples (0 = use all)
MAX_SAMPLES="${MAX_SAMPLES:-0}"
WARMUP_SAMPLES="${WARMUP_SAMPLES:-10}"

# ---- Datasets & recaller combos to benchmark ------------------------------
DATASETS=("ml-1m" "steam" "Food")

declare -a RECALLER_COMBOS=(
    "LightGCN ItemKNN Pop"
)

# ---- Setup -----------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/latency_benchmark_$(date +%Y%m%d_%H%M%S).log"
echo "Inference Latency Benchmark" | tee "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

TOTAL_EXPERIMENTS=$((${#DATASETS[@]} * ${#RECALLER_COMBOS[@]}))
CURRENT=0

# ---- Run -------------------------------------------------------------------
for DATASET in "${DATASETS[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "============================================" | tee -a "$LOG_FILE"
    echo "Dataset: $DATASET" | tee -a "$LOG_FILE"
    echo "============================================" | tee -a "$LOG_FILE"

    # Dataset-specific max_length
    if [ "$DATASET" = "Food" ]; then
        ds_max_length="${MAX_LENGTH:-11024}"
    else
        ds_max_length="${MAX_LENGTH:-1536}"
    fi

    for COMBO in "${RECALLER_COMBOS[@]}"; do
        CURRENT=$((CURRENT + 1))

        read -ra MODELS <<< "$COMBO"
        COMBO_NAME=$(IFS="_"; echo "${MODELS[*]}")

        PURE_MODEL_SHORT=$(basename "$PURE_MODEL_NAME")
        SORTED_MODELS=($(echo "${MODELS[@]}" | tr ' ' '\n' | sort | tr '\n' ' '))
        SORTED_COMBO=$(IFS="_"; echo "${SORTED_MODELS[*]}")

        TEST_DATASET_PATH="${PURE_OUTPUT_DIR}/${DATASET}/${PURE_MODEL_SHORT}_pure_sft_data_${SORTED_COMBO}_${PURE_PROFILE_CUTOFF}/test"

        # Try to find GRPO checkpoint
        GRPO_MODEL_PATH="${PURE_OUTPUT_DIR}/${DATASET}/${PURE_MODEL_SHORT}_pure_grpo_${SORTED_COMBO}_pc${PURE_PROFILE_CUTOFF}"
        if [ ! -d "$GRPO_MODEL_PATH" ]; then
            GRPO_MODEL_PATH="${PURE_OUTPUT_DIR}/${DATASET}/${PURE_MODEL_SHORT}_pure_grpo_${SORTED_COMBO}"
        fi

        echo "" | tee -a "$LOG_FILE"
        echo "[$CURRENT/$TOTAL_EXPERIMENTS] $DATASET | ${MODELS[*]}" | tee -a "$LOG_FILE"
        echo "  Test dataset : $TEST_DATASET_PATH" | tee -a "$LOG_FILE"
        echo "  GRPO model   : $GRPO_MODEL_PATH" | tee -a "$LOG_FILE"
        echo "-------------------------------------------" | tee -a "$LOG_FILE"

        CMD="python GRPO/scripts/benchmark_latency.py \
            --dataset $DATASET \
            --data_path $DATA_PATH \
            --checkpoint_dir $CHECKPOINT_DIR \
            --recbole_models ${MODELS[*]} \
            --eval_k $EVAL_K \
            --seed $SEED \
            --warmup_samples $WARMUP_SAMPLES \
            --output_dir $OUTPUT_DIR \
            --model_name $PURE_MODEL_NAME \
            --max_length $ds_max_length \
            --bf16"

        if [ -n "$MAX_SAMPLES" ] && [ "$MAX_SAMPLES" -gt 0 ] 2>/dev/null; then
            CMD="$CMD --max_samples $MAX_SAMPLES"
        fi

        if [ -d "$TEST_DATASET_PATH" ]; then
            CMD="$CMD --test_dataset_path $TEST_DATASET_PATH"
        else
            echo "  WARNING: test dataset not found at $TEST_DATASET_PATH" | tee -a "$LOG_FILE"
            echo "  Skipping this combination." | tee -a "$LOG_FILE"
            continue
        fi

        if [ -d "$GRPO_MODEL_PATH" ]; then
            CMD="$CMD --grpo_model_path $GRPO_MODEL_PATH"
        else
            echo "  NOTE: GRPO model not found, will skip GRPO timing." | tee -a "$LOG_FILE"
        fi

        echo "  Command: $CMD" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"

        if $CMD 2>&1 | tee -a "$LOG_FILE"; then
            echo "  Success: $DATASET - $COMBO_NAME" | tee -a "$LOG_FILE"
        else
            echo "  Failed: $DATASET - $COMBO_NAME" | tee -a "$LOG_FILE"
        fi
    done
done

echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "All benchmarks completed at: $(date)" | tee -a "$LOG_FILE"
echo "Results saved to: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

# ---- Summary ---------------------------------------------------------------
echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "RESULTS SUMMARY" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

for result_file in "$OUTPUT_DIR"/latency_*.json; do
    if [ -f "$result_file" ]; then
        echo "" | tee -a "$LOG_FILE"
        echo "File: $(basename "$result_file")" | tee -a "$LOG_FILE"
        python3 -c "
import json, sys
with open('$result_file') as f:
    d = json.load(f)
cfg = d['config']
rc  = d['recall_channel']['total_per_sample']
cem = d['cem_extra']['total_extra']
pg  = d['pg_extra']['total_extra']
print(f\"  Dataset       : {cfg['dataset']}\")
print(f\"  Recallers     : {cfg['recbole_models']}\")
print(f\"  Recall Channel: {rc['mean_ms']:.3f} ms  (std {rc['std_ms']:.3f})\")
grpo = d.get('grpo_extra')
if grpo:
    ge = grpo['total_extra']
    print(f\"  GRPO Extra    : {ge['mean_ms']:.3f} ms  (std {ge['std_ms']:.3f})\")
else:
    print(f\"  GRPO Extra    : N/A (no checkpoint)\")
print(f\"  CEM Extra     : {cem['mean_ms']:.3f} ms  (std {cem['std_ms']:.3f})\")
print(f\"  PG Extra      : {pg['mean_ms']:.3f} ms  (std {pg['std_ms']:.3f})\")
" 2>/dev/null || echo "  (Could not parse results)" | tee -a "$LOG_FILE"
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "Done!" | tee -a "$LOG_FILE"
