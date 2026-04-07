#!/bin/bash
export CUDA_VISIBLE_DEVICES=$2
cd <path_to_routepo> && python -m GRPO.generate_recaller_metrics_data \
    --dataset $1 \
    --data_path ./dataset \
    --checkpoint_dir ./checkpoints \
    --recbole_models BPR SASRec LightGCN Pop ItemKNN GRU4Rec SimpleX \
    --output_dir ./GRPO/recaller_metrics_data \
    --final_k 50 \
    --min_history_len $3 \
    --ks 5 10 20 50 \
    --seed 42