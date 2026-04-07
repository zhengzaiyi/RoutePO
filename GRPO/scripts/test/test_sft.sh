export PYTHONPATH=<path_to_routepo>
export CUDA_VISIBLE_DEVICES=$2
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

# 运行测试命令
cd <path_to_routepo>
python GRPO/models/main_trl.py \
    --use_hf_local \
    --dataset $1 \
    --data_path dataset \
    --recbole_models BPR ItemKNN FPMC Pop SASRec\
    --do_test \
    --do_test_sft\
    --use_vllm