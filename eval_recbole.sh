#!/bin/bash
MODELS=(
    "BPR"
    "LightGCN"
    "SimpleX"
    "SASRec"
)
EPOCHS=(500)
DATASETS=(
    "ml-1m"
    # "ml-10m"
    "steam"
)
LRS=(1e-3)
# Create a new tmux session
SESSION_NAME="recaller_all"
# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null || true
tmux new-session -d -s $SESSION_NAME

# 初始化计数器
window_id=0

# Run commands in tmux windows
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for EPOCH in "${EPOCHS[@]}"; do
            for LR in "${LRS[@]}"; do
                DEVICE_ID=$((window_id % 8))
                
                if [ $window_id -eq 0 ]; then
                    # 第一个命令在初始窗口执行
                    tmux send-keys -t $SESSION_NAME:0 "CUDA_VISIBLE_DEVICES=$DEVICE_ID python evaluate_recbole.py --dataset $DATASET --models_to_train ${MODEL} --epochs ${EPOCH} --learning_rate ${LR}" Enter
                else
                    # 创建新窗口并执行命令
                    tmux new-window -t $SESSION_NAME -n "window_${window_id}"
                    tmux send-keys -t $SESSION_NAME:${window_id} "CUDA_VISIBLE_DEVICES=$DEVICE_ID python evaluate_recbole.py --dataset $DATASET --models_to_train ${MODEL} --epochs ${EPOCH} --learning_rate ${LR}" Enter
                fi    
                ((window_id++))
            done
        done
    done
done

# Attach to the tmux session
tmux attach-session -t $SESSION_NAME