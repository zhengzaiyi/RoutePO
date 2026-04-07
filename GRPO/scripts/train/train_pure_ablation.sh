#!/bin/bash
# =============================================================================
# train_pure_ablation.sh - Ablation study variants of train_pure.sh
#
# Controlled by ABLATION_MODE environment variable:
#   no_sft   - Skip SFT, GRPO from base model directly
#   no_rl    - Skip GRPO, SFT only
#   no_snack - Full pipeline but test with score merge (--merge_method average)
# =============================================================================

export PYTHONPATH=~/AmazonReviews2023
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export WANDB_PROJECT="pure-grpo-ablation"

ABLATION_MODE="${ABLATION_MODE:-no_rl}"

if [ "$1" = "Food" ]; then
    max_length="${MAX_LENGTH:-11024}"
else
    max_length="${MAX_LENGTH:-1536}"
fi

PARALLEL_SIZE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
DATA_EVAL_GPU="${DATA_EVAL_GPU:-4}"
models="${TRAIN_MODELS:-LightGCN ItemKNN Pop}"
train_k=20
eval_k=50
model_name="${TRAIN_MODEL_NAME:-meta-llama/Llama-3.2-1B-Instruct}"
profile_cutoff=500000

echo "================================================"
echo "Ablation Study: ABLATION_MODE=${ABLATION_MODE}"
echo "Dataset: $1 | Model: $model_name"
echo "Recallers: $models"
echo "================================================"

# Step 1: Generate data (always needed)
echo "================================================"
echo "Generating SFT data..."
echo "================================================"
CUDA_VISIBLE_DEVICES=$DATA_EVAL_GPU python GRPO/models/main_pure.py \
    --dataset $1 \
    --data_path dataset \
    --checkpoint_dir ./checkpoints \
    --output_dir GRPO/data/pure_models \
    --model_name $model_name \
    --recbole_models $models \
    --gen_sft_test \
    --train_k $train_k \
    --eval_k $eval_k \
    --seed 42 \
    --padding_side left \
    --random_history_selection \
    --profile_cutoff $profile_cutoff \
    --gen_sft_train \
    --gen_sft_eval \
    --autoregressive \

# ---------- Ablation: w/o SFT ----------
if [ "$ABLATION_MODE" = "no_sft" ]; then
    echo "================================================"
    echo "[Ablation: w/o SFT] Skipping SFT, running GRPO from base model"
    echo "================================================"

    accelerate launch --config_file GRPO/configs/soft_acc.yaml \
        GRPO/models/main_pure.py \
        --do_grpo \
        --skip_sft_init \
        --dataset $1 \
        --data_path dataset \
        --model_name $model_name \
        --output_dir GRPO/data/pure_models \
        --recbole_models $models \
        --train_k $train_k \
        --eval_k $eval_k \
        --logging_steps 10 \
        --save_steps 500 \
        --tau_gumbel 1.0 \
        --top_p 0.9 \
        --noise_scale 0.1 \
        --epsilon 0.2 \
        --beta 0.1 \
        --sync_ref_model \
        --merge_method top_k \
        --ref_model_sync_steps 500 \
        --max_length $max_length \
        --num_generations 8 \
        --grpo_lr 1e-6 \
        --grpo_epochs 1 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --bf16 \
        --seed 42 \
        --profile_cutoff $profile_cutoff

    echo "================================================"
    echo "[Ablation: w/o SFT] Testing GRPO model..."
    echo "================================================"
    CUDA_VISIBLE_DEVICES=$DATA_EVAL_GPU python GRPO/models/main_pure.py \
        --dataset $1 \
        --data_path dataset \
        --checkpoint_dir ./checkpoints \
        --output_dir GRPO/data/pure_models \
        --model_name $model_name \
        --recbole_models $models \
        --do_test_grpo \
        --train_k $train_k \
        --eval_k $eval_k \
        --seed 42 \
        --padding_side left \
        --random_history_selection \
        --profile_cutoff $profile_cutoff \
        --merge_method top_k \
        --max_length $max_length

# ---------- Ablation: w/o RL ----------
elif [ "$ABLATION_MODE" = "no_rl" ]; then
    echo "================================================"
    echo "[Ablation: w/o RL] Training SFT only (autoregressive)..."
    echo "================================================"
    accelerate launch --config_file GRPO/configs/soft_acc.yaml \
        GRPO/models/main_pure.py \
        --dataset $1 \
        --data_path dataset \
        --checkpoint_dir ./checkpoints \
        --output_dir GRPO/data/pure_models \
        --model_name $model_name \
        --recbole_models $models \
        --do_sft \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --learning_rate 1e-5 \
        --num_train_epochs 3 \
        --warmup_steps 100 \
        --logging_steps 20 \
        --save_steps 1000 \
        --eval_steps 1000 \
        --max_length $max_length \
        --train_k $train_k \
        --eval_k $eval_k \
        --seed 42 \
        --bf16 \
        --gradient_checkpointing \
        --padding_side left \
        --random_history_selection \
        --profile_cutoff $profile_cutoff \
        --autoregressive

    echo "================================================"
    echo "[Ablation: w/o RL] Training SFT (classification)..."
    echo "================================================"
    accelerate launch --config_file GRPO/configs/soft_acc.yaml \
        GRPO/models/main_pure.py \
        --dataset $1 \
        --data_path dataset \
        --checkpoint_dir ./checkpoints \
        --output_dir GRPO/data/pure_models \
        --model_name $model_name \
        --recbole_models $models \
        --do_sft \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --learning_rate 1e-5 \
        --num_train_epochs 3 \
        --warmup_steps 100 \
        --logging_steps 20 \
        --save_steps 1000 \
        --eval_steps 1000 \
        --max_length $max_length \
        --train_k $train_k \
        --eval_k $eval_k \
        --seed 42 \
        --bf16 \
        --gradient_checkpointing \
        --padding_side left \
        --random_history_selection \
        --profile_cutoff $profile_cutoff

    echo "================================================"
    echo "[Ablation: w/o RL] Testing SFT model only..."
    echo "================================================"
    CUDA_VISIBLE_DEVICES=$DATA_EVAL_GPU python GRPO/models/main_pure.py \
        --dataset $1 \
        --data_path dataset \
        --checkpoint_dir ./checkpoints \
        --output_dir GRPO/data/pure_models \
        --model_name $model_name \
        --recbole_models $models \
        --do_test_sft \
        --train_k $train_k \
        --eval_k $eval_k \
        --seed 42 \
        --padding_side left \
        --random_history_selection \
        --profile_cutoff $profile_cutoff \
        --merge_method top_k \
        --max_length $max_length

# ---------- Ablation: w/o SNACK (score merge) ----------
elif [ "$ABLATION_MODE" = "no_snack" ]; then
    echo "================================================"
    echo "[Ablation: w/o SNACK] Full pipeline with score merge at test"
    echo "================================================"

    # AR SFT
    accelerate launch --config_file GRPO/configs/soft_acc.yaml \
        GRPO/models/main_pure.py \
        --dataset $1 \
        --data_path dataset \
        --checkpoint_dir ./checkpoints \
        --output_dir GRPO/data/pure_models \
        --model_name $model_name \
        --recbole_models $models \
        --do_sft \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --learning_rate 1e-5 \
        --num_train_epochs 3 \
        --warmup_steps 100 \
        --logging_steps 20 \
        --save_steps 1000 \
        --eval_steps 1000 \
        --max_length $max_length \
        --train_k $train_k \
        --eval_k $eval_k \
        --seed 42 \
        --bf16 \
        --gradient_checkpointing \
        --padding_side left \
        --random_history_selection \
        --profile_cutoff $profile_cutoff \
        --autoregressive

    # Classification SFT
    accelerate launch --config_file GRPO/configs/soft_acc.yaml \
        GRPO/models/main_pure.py \
        --dataset $1 \
        --data_path dataset \
        --checkpoint_dir ./checkpoints \
        --output_dir GRPO/data/pure_models \
        --model_name $model_name \
        --recbole_models $models \
        --do_sft \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --learning_rate 1e-5 \
        --num_train_epochs 3 \
        --warmup_steps 100 \
        --logging_steps 20 \
        --save_steps 1000 \
        --eval_steps 1000 \
        --max_length $max_length \
        --train_k $train_k \
        --eval_k $eval_k \
        --seed 42 \
        --bf16 \
        --gradient_checkpointing \
        --padding_side left \
        --random_history_selection \
        --profile_cutoff $profile_cutoff

    # GRPO
    accelerate launch --config_file GRPO/configs/soft_acc.yaml \
        GRPO/models/main_pure.py \
        --do_grpo \
        --dataset $1 \
        --data_path dataset \
        --model_name $model_name \
        --output_dir GRPO/data/pure_models \
        --recbole_models $models \
        --train_k $train_k \
        --eval_k $eval_k \
        --logging_steps 10 \
        --save_steps 500 \
        --tau_gumbel 1.0 \
        --top_p 0.9 \
        --noise_scale 0.1 \
        --epsilon 0.2 \
        --beta 0.1 \
        --sync_ref_model \
        --merge_method top_k \
        --ref_model_sync_steps 500 \
        --max_length $max_length \
        --num_generations 8 \
        --grpo_lr 1e-6 \
        --grpo_epochs 1 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --bf16 \
        --seed 42 \
        --profile_cutoff $profile_cutoff

    # Test with score merge (average) instead of top_k
    echo "================================================"
    echo "[Ablation: w/o SNACK] Testing with --merge_method average..."
    echo "================================================"
    CUDA_VISIBLE_DEVICES=$DATA_EVAL_GPU python GRPO/models/main_pure.py \
        --dataset $1 \
        --data_path dataset \
        --checkpoint_dir ./checkpoints \
        --output_dir GRPO/data/pure_models \
        --model_name $model_name \
        --recbole_models $models \
        --do_test_sft \
        --do_test_grpo \
        --train_k $train_k \
        --eval_k $eval_k \
        --seed 42 \
        --padding_side left \
        --random_history_selection \
        --profile_cutoff $profile_cutoff \
        --merge_method average \
        --max_length $max_length

else
    echo "ERROR: Unknown ABLATION_MODE='${ABLATION_MODE}'"
    echo "Valid options: no_sft, no_rl, no_snack"
    exit 1
fi

echo "================================================"
echo "Ablation '${ABLATION_MODE}' completed for $1"
echo "================================================"
