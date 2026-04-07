#!/bin/bash

export PYTHONPATH=<path_to_routepo>

# DATASET=Musical_Instruments
# export MASTER_PORT=12346
# export CUDA_VISIBLE_DEVICES=2

# DATASET=Sports_and_Outdoors
# export MASTER_PORT=12344
# export CUDA_VISIBLE_DEVICES=0

export MASTER_PORT=12366
# export CUDA_VISIBLE_DEVICES=4

# DATASET=Gift_Cards
# export MASTER_PORT=12347
# export CUDA_VISIBLE_DEVICES=3

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
# export WANDB_MODE=disabled


# 运行命令
cd <path_to_routepo>

PARALLEL_SIZE=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=4,5,6,7

# echo "================================================"
# echo "Running Soft Token Training: Generate SFT Data"
# echo "================================================"
# python GRPO/models/main_soft.py \
#     --gen_sft_data \
#     --dataset $1 \
#     --data_path dataset \
#     --model_name meta-llama/Llama-3.2-1B-Instruct \
#     --output_dir GRPO/soft_models \
#     --final_k 50 \
#     --norm_type static \
#     --num_train_samples 1000 \
#     --seed 42

# echo "================================================"
# echo "Running Soft Token Training: SFT"
# echo "================================================"
# accelerate launch --config_file GRPO/configs/soft_acc.yaml \
#     GRPO/models/main_soft.py \
#     --do_sft \
#     --dataset $1 \
#     --data_path dataset \
#     --model_name meta-llama/Llama-3.2-1B-Instruct \
#     --output_dir GRPO/soft_models \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 8 \
#     --learning_rate 5e-5 \
#     --warmup_steps 100 \
#     --logging_steps 10 \
#     --save_steps 500 \
#     --eval_steps 100 \
#     --max_length 1536 \
#     --final_k 50 \
#     --seed 42

# echo "================================================"
# echo "Running Soft Token Training: SFT + RL"
# echo "================================================"
# accelerate launch --config_file GRPO/configs/soft_acc.yaml \
#     GRPO/models/main_soft.py \
#     --do_sft \
#     --do_rl \
#     --dataset $1 \
#     --data_path dataset \
#     --model_name meta-llama/Llama-3.2-1B-Instruct \
#     --output_dir GRPO/soft_models \
#     --final_k 50 \
#     --logging_steps 10 \
#     --eval_steps 100 \
#     --save_steps 500 \
#     --grpo_learning_rate 1e-5 \
#     --grpo_batch_size 2 \
#     --grpo_num_train_epochs 3 \
#     --kl_coef 0.1 \
#     --gradient_checkpointing \
#     --seed 42

echo "================================================"
echo "Running Soft Token Training: RL Only"
echo "================================================"
accelerate launch --config_file GRPO/configs/soft_acc.yaml \
    GRPO/models/main_soft.py \
    --do_rl \
    --dataset $1 \
    --data_path dataset \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --output_dir GRPO/soft_models \
    --final_k 50 \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 500 \
    --grpo_learning_rate 1e-5 \
    --grpo_batch_size 2 \
    --grpo_num_train_epochs 3 \
    --seed 42

