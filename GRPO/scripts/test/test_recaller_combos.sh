#!/bin/bash

export PYTHONPATH=<path_to_routepo>
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export WANDB_MODE=disabled

# Usage: ./test_recaller_combos.sh <dataset_name> <gpu_id>
# Example: ./test_recaller_combos.sh Amazon_All_Beauty 0

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

# Define all available recallers
ALL_RECALLERS=("BPR" "SASRec" "ItemKNN" "LightGCN" "GRU4Rec")

# Define recaller combinations to test
# Format: "model1 model2 model3"
RECALLER_COMBOS=(
    # Two-model combinations
    # "BPR SASRec"
    # "BPR ItemKNN"
    # "BPR LightGCN"
    "SASRec ItemKNN"
    "SASRec LightGCN"
    "ItemKNN LightGCN"
    "FPMC SASRec"
    "FPMC ItemKNN"
    "FPMC LightGCN"
    # Three-model combinations
    "BPR SASRec ItemKNN"
    "BPR SASRec LightGCN"
    "BPR ItemKNN LightGCN"
    "SASRec ItemKNN LightGCN"
    # Four-model combinations
    "BPR SASRec ItemKNN LightGCN"
    # Five-model combination (all)
    "BPR SASRec ItemKNN LightGCN GRU4Rec"
)

echo "================================================"
echo "Testing Recaller Combinations"
echo "Dataset: $DATASET"
echo "GPU: $2"
echo "Model: $model_name"
echo "Number of combinations: ${#RECALLER_COMBOS[@]}"
echo "================================================"

# Create results directory
mkdir -p results

# Summary file
SUMMARY_FILE="results/summary_${DATASET}_$(date +%Y%m%d_%H%M%S).txt"
echo "Testing recaller combinations for $DATASET" > $SUMMARY_FILE
echo "Started at: $(date)" >> $SUMMARY_FILE
echo "================================================" >> $SUMMARY_FILE

# Test each combination
for combo in "${RECALLER_COMBOS[@]}"; do
    echo ""
    echo "================================================"
    echo "Testing combination: $combo"
    echo "================================================"
    
    # Convert space-separated string to array for recbole_models
    read -ra MODELS <<< "$combo"
    combo_name=$(echo $combo | tr ' ' '_')
    
    echo "Step 1: Generating SFT data for $combo_name..."
    python GRPO/models/main_pure.py \
        --dataset $DATASET \
        --data_path dataset \
        --checkpoint_dir ./checkpoints \
        --output_dir GRPO/data/pure_models \
        --model_name $model_name \
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
    
    echo "Step 2: Training SFT model for $combo_name..."
    python GRPO/models/main_pure.py \
        --dataset $DATASET \
        --data_path dataset \
        --checkpoint_dir ./checkpoints \
        --output_dir GRPO/data/pure_models \
        --model_name $model_name \
        --recbole_models ${MODELS[@]} \
        --do_sft \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --learning_rate 1e-3 \
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
    
    echo "Step 3: Testing SFT model for $combo_name..."
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
        # Extract key metrics from JSON result file
        result_file="results/pure_results_${DATASET}_${combo_name}.json"
        if [ -f "$result_file" ]; then
            echo "  Results file: $result_file" >> $SUMMARY_FILE
            # Use python to extract metrics
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
echo "Individual results saved in: results/pure_results_${DATASET}_*.json"
echo "================================================"

echo "" >> $SUMMARY_FILE
echo "Completed at: $(date)" >> $SUMMARY_FILE

# Print summary
cat $SUMMARY_FILE



