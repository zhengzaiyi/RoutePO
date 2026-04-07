export PYTHONPATH=<path_to_routepo>
export CUDA_VISIBLE_DEVICES=$2
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

# 运行测试命令
cd <path_to_routepo>
python GRPO/models/main_trl.py \
    --use_hf_local \
    --dataset $1 \
    --checkpoint_dir checkpoints_test \
    --data_path dataset \
    --recbole_models BPR ItemKNN FPMC Pop SASRec LightGCN SimpleX\
    --do_test_recaller \
    --use_vllm