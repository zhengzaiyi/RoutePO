
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
# export CUDA_VISIBLE_DEVICES=$2
# python GRPO/models/main_trl.py \
#     --use_hf_local \
#     --dataset $1 \
#     --data_path dataset \
#     --do_train \  
#     --use_vllm \
#     --hf_model meta-llama/Llama-3.2-1B-Instruct

PARALLEL_SIZE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# echo "================================================"
# echo "Running with SFT and RL"
# echo "================================================"
# accelerate launch --config_file GRPO/configs/acc.yaml\
#     GRPO/models/main_trl.py \
#     --use_hf_local \
#     --dataset $1 \
#     --data_path dataset \
#     --parallel_size 2 \
#     --do_sft \
#     --do_rl \
#     --use_vllm \
#     --hf_model meta-llama/Llama-3.2-1B-Instruct

echo "================================================"
echo "Running ONLY with RL"
echo "================================================"
accelerate launch --config_file GRPO/configs/acc.yaml \
    GRPO/models/main_trl.py \
    --use_hf_local \
    --dataset $1 \
    --data_path dataset \
    --parallel_size 2 \
    --do_rl \
    --use_vllm \
    --hf_model meta-llama/Llama-3.2-1B-Instruct