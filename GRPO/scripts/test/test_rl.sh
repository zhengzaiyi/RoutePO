export PYTHONPATH=<path_to_routepo>
export CUDA_VISIBLE_DEVICES=$2
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

# 运行测试命令

echo "================================================" 
echo "Testing pure GRPO model..."
echo "================================================"
python GRPO/models/main_pure.py \
    --dataset $1 \
    --data_path dataset \
    --checkpoint_dir ./checkpoints \
    --output_dir GRPO/data/pure_models \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --recbole_models BPR SASRec LightGCN \
    --do_test_grpo \
    --final_k 50 \
    --seed 42 \
    --padding_side left \
    --profile_cutoff 20