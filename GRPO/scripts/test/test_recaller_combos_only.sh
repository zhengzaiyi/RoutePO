#!/bin/bash

# Quick testing script - only runs testing on already trained models
# Use this when you've already trained models and just want to re-run tests

export PYTHONPATH=<path_to_routepo>
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export WANDB_MODE=disabled

# Usage: ./test_recaller_combos_only.sh <dataset_name> <gpu_id>
# Example: ./test_recaller_combos_only.sh Amazon_All_Beauty 0

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <dataset_name> <gpu_id>"
    echo "Example: $0 Amazon_All_Beauty 0"
    exit 1
fi

cd <path_to_routepo>
export CUDA_VISIBLE_DEVICES=$2

DATASET=$1
profile_cutoff=20
model_name=meta-llama/Llama-3.2-1B-Instruct

# Define recaller combinations to test
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
    # Four-model combinations
    "BPR SASRec ItemKNN LightGCN"
)

echo "================================================"
echo "Quick Testing Recaller Combinations (Test Only)"
echo "Dataset: $DATASET"
echo "GPU: $2"
echo "Number of combinations: ${#RECALLER_COMBOS[@]}"
echo "================================================"

mkdir -p results

SUMMARY_FILE="results/summary_test_only_${DATASET}_$(date +%Y%m%d_%H%M%S).txt"
echo "Quick testing recaller combinations for $DATASET" > $SUMMARY_FILE
echo "Started at: $(date)" >> $SUMMARY_FILE
echo "================================================" >> $SUMMARY_FILE

for combo in "${RECALLER_COMBOS[@]}"; do
    echo ""
    echo "Testing: $combo"
    
    read -ra MODELS <<< "$combo"
    combo_name=$(echo $combo | tr ' ' '_')
    
    python GRPO/models/main_pure.py \
        --dataset $DATASET \
        --data_path dataset \
        --checkpoint_dir ./checkpoints \
        --output_dir GRPO/data/pure_models \
        --model_name $model_name \
        --recbole_models ${MODELS[@]} \
        --do_test_sft \
        --final_k 50 \
        --seed 42 \
        --padding_side left \
        --profile_cutoff $profile_cutoff
    
    if [ $? -eq 0 ]; then
        echo "SUCCESS: $combo_name" >> $SUMMARY_FILE
        result_file="results/pure_results_${DATASET}_${combo_name}.json"
        if [ -f "$result_file" ]; then
            python -c "
import json
with open('$result_file') as f:
    r = json.load(f)
print(f\"  Accuracy: {r.get('accuracy', 'N/A'):.4f}\")
print(f\"  F1 Macro: {r.get('f1_macro', 'N/A'):.4f}\")
print(f\"  Predicted NDCG: {r.get('avg_predicted_ndcg', 'N/A'):.4f}\")
" >> $SUMMARY_FILE 2>/dev/null
        fi
    else
        echo "FAILED: $combo_name" >> $SUMMARY_FILE
    fi
done

echo "" >> $SUMMARY_FILE
echo "Completed at: $(date)" >> $SUMMARY_FILE
echo ""
echo "================================================"
echo "Summary:"
cat $SUMMARY_FILE



