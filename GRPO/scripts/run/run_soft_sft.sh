#!/bin/bash

dataset=$1

# Step 1: Generate SFT data
python -m GRPO.main_soft \
    --gen_sft_data \
    --dataset $dataset \
    --data_path "./dataset" \
    --model_name "meta-llama/Llama-3.2-1B-Instruct" \
    --output_dir "GRPO/soft_models" \
    --final_k 50 \
    --norm_type "static" \
    --num_train_samples 1000 \
    --seed 42

# Step 2: Train with SFT
python -m GRPO.main_soft \
    --do_sft \
    --dataset $dataset \
    --model_name "meta-llama/Llama-3.2-1B-Instruct" \
    --output_dir "GRPO/soft_models" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-5 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 100 \
    --max_length 1536 \
    --final_k 50 \
    --seed 42

# Step 3: Test the model (optional)
# python -m GRPO.main_soft \
#     --do_test \
#     --do_sft \
#     --dataset $dataset \
#     --model_name "meta-llama/Llama-3.2-1B-Instruct" \
#     --output_dir "GRPO/soft_models" \
#     --final_k 50

# Training with LoRA (more memory efficient)
# python -m GRPO.main_soft \
#     --do_sft \
#     --use_lora \
#     --dataset $dataset \
#     --model_name "meta-llama/Llama-3.2-1B-Instruct" \
#     --output_dir "GRPO/soft_models" \
#     --lora_r 16 \
#     --lora_alpha 32 \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 16 \
#     --gradient_accumulation_steps 2 \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --gradient_checkpointing
