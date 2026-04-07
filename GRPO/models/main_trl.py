import argparse
import json
import os
import random
from functools import partial
from typing import List
import numpy as np

import outlines
import torch
from datasets import Dataset
from huggingface_hub import InferenceClient
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, SFTTrainer, SFTConfig
from vllm.sampling_params import GuidedDecodingParams

from GRPO.core.agents import UserProfileAgent, LLMRouterAgent, create_model_config
from GRPO.core.data import load_dataset
from GRPO.models.main import initialize_recallers
from GRPO.core.recallers import RecBoleRecaller
from GRPO.trainers.trl_trainer import GRPOTrainer
from GRPO.core.utils import set_seed, build_prompt, multi_channel_recall, ndcg_at_k, evaluate, evaluate_recallers, recall_at_k
from accelerate import Accelerator
from collections import defaultdict


def create_rl_dataset(
    profile_agent: UserProfileAgent, 
    user_ids: List[int],
    histories: List[List[int]],
    target_items: List[int],
    recaller_names: List[str]
):
    """Create TRL dataset containing information needed for reward calculation"""
    dataset = []
    for i, uid in enumerate(user_ids):
        hist = histories[i]
        if 0 in hist:
            hist = hist[:hist.index(0)]
        prof_json = '' if profile_agent is None else profile_agent.forward(uid, hist)
        content = build_prompt(prof_json, recaller_names) if profile_agent is not None else str(recaller_names)
        dataset.append({
            "prompt": [{"role": "user", "content": content}],
            "uid": uid,                           # User ID
            "histories": hist,                 # User history
            "target_items": [target_items[i]] if type(target_items[i]) == int else target_items[i]  # Test items
        })
    return Dataset.from_list(dataset)

def create_sft_dataset(
    profile_agent: UserProfileAgent, 
    user_ids: List[int],
    histories: List[List[int]],
    target_items: List[int],
    recallers: List[RecBoleRecaller],
    final_k: int,
    norm_type: str = 'static', # oracle, static
    profile_cutoff: int = 20,
    min_history_for_augmentation: int = 30,
    augmentation_step: int = 10,
):
    dataset = []
    metrics = {recaller: defaultdict(list) for recaller in recallers.keys()}
    
    for i, uid in enumerate(user_ids):
        hist = histories[i]
        if 0 in hist:
            hist = hist[:hist.index(0)]
        
        # Determine history lengths for augmentation (from main_pure.py)
        history_lengths = [len(hist)]
        if len(hist) >= min_history_for_augmentation:
            history_lengths = list(range(profile_cutoff, len(hist), augmentation_step)) + [len(hist)]
        
        for hist_len in history_lengths:
            current_hist = hist[:hist_len]
            
            # Prepare ground-truth (20% of history + target) - from main_pure.py
            n_gt = max(1, min(5, int(len(current_hist) * 0.2)))
            if len(current_hist) > 1:
                gt_items = current_hist[-n_gt:] + [target_items[i]]
                eval_hist = current_hist[:-n_gt]
        else:
            gt_items = [target_items[i]]
            eval_hist = current_hist
        
            # Skip if evaluation history too short
        if len(eval_hist) < 5:
            continue
        
            # Find best recaller using eval_hist and gt_items
        best_ndcg, best_recaller = -1, None
        for recaller_name in recallers.keys():
            items = recallers[recaller_name].recall(uid, int(final_k), eval_hist)
            item_ids = [item[0] for item in items] if items else []
            ndcg = ndcg_at_k(item_ids, gt_items, k=final_k)
            metrics[recaller_name]["ndcg"].append(ndcg)
            metrics[recaller_name]["recall@10"].append(recall_at_k(item_ids, gt_items, k=10))
                
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_recaller = recaller_name
                    
        assert best_recaller is not None
            
            # Generate prompt using eval_hist with profile_cutoff
        prompt = build_prompt(profile_agent.forward(uid, eval_hist, cut_off=profile_cutoff), available_models=recallers.keys())
        if norm_type == 'oracle':
            ground_truth_output = {
                recaller_name: {
                        "top-k": final_k if recaller_name == best_recaller else 0,
                    "score-weight": 1.0 if recaller_name == best_recaller else 0.0
                } for recaller_name in recallers.keys()
            }
        elif norm_type == 'static':
            recaller_scores = {name: ndcg_at_k([item[0] for item in items], gt_items, k=final_k) 
                            if (items := recallers[name].recall(uid, final_k, eval_hist)) else 0
                            for name in recallers.keys()}
            
            scores = np.array(list(recaller_scores.values()))
            weights = np.exp(scores * 2) / np.exp(scores * 2).sum()  # softmax with temperature=0.5
            weights = weights * 0.9 + np.random.dirichlet(np.ones(len(weights))) * 0.1  # 10% noise

            total_k = int(final_k * 1.5)
            k_values = np.maximum(5, (weights * total_k).astype(int))
            k_values = (k_values * min(1, total_k / k_values.sum())).astype(int)
                
            ground_truth_output = {
                    name: {"top-k": int(k), "score-weight": float(w)}
                    for name, k, w in zip(recaller_scores.keys(), k_values, weights/weights.sum())
            }
        else:
            raise ValueError(f"Invalid norm type: {norm_type}")
        ground_truth_output = json.dumps(ground_truth_output, indent=2)
        dataset.append({
            "prompt": prompt + '\n\n',
            "completion": ground_truth_output,
                "user_id": uid,
                "history_len_used": len(eval_hist),
            })
    
    # Print statistics (from main_pure.py)
    print(f"\nDataset created: {len(dataset)} samples from {len(set(d['user_id'] for d in dataset))} users")
    for recaller_name in recallers.keys():
        if metrics[recaller_name]["ndcg"]:
            avg_ndcg = np.mean(metrics[recaller_name]["ndcg"])
            print(f"{recaller_name}: avg NDCG = {avg_ndcg:.4f}")
    
    return Dataset.from_list(dataset)

def parse_args():
    ap = argparse.ArgumentParser(description='GRPO Training (TRL) for Recommendation Systems')
    ap.add_argument('--dataset', type=str, default='ml-1m')
    ap.add_argument('--data_path', type=str, default='./data')
    ap.add_argument('--final_k', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--recbole_models', type=str, nargs='+', default=['BPR', 'SASRec', 'FPMC', 'Pop', 'ItemKNN'])
    ap.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    ap.add_argument('--use_hf_local', action='store_true')
    ap.add_argument('--hf_model', type=str, default='meta-llama/Llama-3.2-1B-Instruct')
    ap.add_argument('--output_dir', type=str, default='GRPO/grpo_models')
    ap.add_argument('--do_rl', action='store_true')
    ap.add_argument('--do_eval', action='store_true')
    ap.add_argument('--do_test', action='store_true')
    ap.add_argument('--do_test_sft', action='store_true')
    ap.add_argument('--do_test_rl', action='store_true')
    ap.add_argument('--do_test_recaller', action='store_true')
    ap.add_argument('--use_vllm', action='store_true')
    ap.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    ap.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    ap.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    ap.add_argument('--use_peft', action='store_true', help='Whether to use PEFT')
    ap.add_argument('--parallel_size', type=int, default=1, help='Parallel size')
    ap.add_argument('--gen_sft_data', action='store_true')
    ap.add_argument('-df', '--do_sft', action='store_true')
    args = ap.parse_args()
    return args

def main():
    args = parse_args()
    if not args.use_hf_local:
        raise RuntimeError("TRL entrypoint requires --use_hf_local to provide a local HF model.")
    
    # accelerator = Acceleratosr()
    set_seed(args.seed)
    model_save_dir = f"{args.output_dir}/{args.dataset}/{args.hf_model.split('/')[-1]}"
    os.makedirs(model_save_dir, exist_ok=True)
    
    inter_dataset = load_dataset(args.dataset, args.data_path, seed=args.seed) 
    profile_agent = UserProfileAgent(inter_dataset, args.dataset)
    cut_off_users = min(len(inter_dataset.train_user_ids), 100000)
    model_config = create_model_config(
        available_models=[model_name.lower() for model_name in args.recbole_models],
        default_configs=None,
        class_name="ModelConfigs"
    )
    grpo_config = GRPOConfig(
        output_dir=model_save_dir,
        save_steps=500,
        save_total_limit=10,   
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_model_impl="vllm",
        vllm_tensor_parallel_size=args.parallel_size,
        # model_init_kwargs={
        #     "load_in_8bit": True,
        # },
        adam_beta1=0.9,
        adam_beta2=0.99,
        beta=0.001,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=3,
        num_train_epochs=3,
        learning_rate=1e-5,
        optim="paged_adamw_8bit",
        lr_scheduler_type="constant",
        vllm_gpu_memory_utilization=0.3,
        seed=3407,
        max_prompt_length=2048,
        max_completion_length=1024,
        generation_batch_size=16,
        num_generations=4,
        gradient_checkpointing=True,
        run_name="vanilla_" + f"_lr{1e-5}_kl{1e-3}",
        report_to = "wandb"
    )
    grpo_config.vllm_guided_decoding_json=model_config.model_json_schema()
    if args.gen_sft_data or args.do_rl or args.do_eval or args.do_test or args.do_test_rl or args.do_test_recaller:
        recallers = initialize_recallers(
            model_names=args.recbole_models,
            dataset_name=args.dataset,
            checkpoint_dir=args.checkpoint_dir,
            data_path=args.data_path,
            seed=args.seed,
            use_latest_checkpoint=True,
            num_items=inter_dataset.ds.item_num
        )
    # Initialize HF models
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # accelerator.wait_for_everyone()
    # Run training, evaluation, and testing
    model_name = args.hf_model
    if args.gen_sft_data:
        # separatelt generate sft data to avoid gpu id conflict
        assert not args.do_sft and not args.do_rl and not args.do_eval and not args.do_test and not args.do_test_rl and not args.do_test_recaller
        sft_dataset = create_sft_dataset(
            profile_agent=profile_agent,
            user_ids=inter_dataset.train_user_ids[:cut_off_users],
            histories=inter_dataset.train_histories[:cut_off_users],
            target_items=inter_dataset.train_target_items[:cut_off_users],
            recallers=recallers,
            final_k=args.final_k
        )
        if os.path.exists(f'{model_save_dir}_sft_data'):
            os.rmtree(f'{model_save_dir}_sft_data')
        sft_dataset.save_to_disk(f'{model_save_dir}_sft_data')
        return
    if args.do_sft:
        # if len(os.listdir(f'{model_save_dir}_sft')) == 0:
        model_save_dir = f"{model_save_dir}_sft"
        if not os.path.exists(model_save_dir) or len(os.listdir(model_save_dir)) == 0:
            sft_config = SFTConfig(
                output_dir=model_save_dir,
                save_total_limit=2,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                num_train_epochs=10,
                save_strategy="steps",
                save_steps=200,
                logging_steps=10,
                bf16=True,
                gradient_accumulation_steps=3,
                learning_rate=1e-5,
                # optim="paged_adamw_8bit",
                # lr_scheduler_type="constant",
                seed=3407,
                max_length=2048,
                gradient_checkpointing=True,
                run_name="sft_" + f"_lr{1e-5}",
                report_to="wandb",
                # completion_only_loss=True
            )
            sft_dataset = Dataset.load_from_disk(f'{model_save_dir}_data')
            trainer = SFTTrainer(
                model=model_name, 
                train_dataset=sft_dataset, 
                processing_class=tokenizer, 
                args=sft_config,
                # load_in_8bit=True,
                # tokenizer=tokenizer,
                # formatting_func=lambda e: [e["prompt"] + e["completion"]],
            )
            trainer.train()
        model_name = get_last_checkpoint(model_save_dir)          
    if args.do_rl:
        trl_train_dataset = create_rl_dataset(
            profile_agent=profile_agent,
            user_ids=inter_dataset.train_user_ids[:cut_off_users],
            histories=inter_dataset.train_histories[:cut_off_users],
            target_items=inter_dataset.train_target_items[:cut_off_users],
            recaller_names=[model_name.lower() for model_name in args.recbole_models]
        )
        model_save_dir = f"{model_save_dir}_rl"
        grpo_config.output_dir = model_save_dir
        def reward_func(completions, uid, histories, target_items, **kwargs):
            rewards = []
            recall_results = multi_channel_recall(
                completions=completions, 
                uid=uid, 
                histories=histories, 
                recallers=recallers, 
                total_item=args.final_k
            )
            for i, recall_result in enumerate(recall_results):
                ndcg = ndcg_at_k(recall_result, target_items[i], k=args.final_k)
                rewards.append(ndcg)
            return rewards
        trainer = GRPOTrainer(
            model=model_name,
            reward_funcs=reward_func,
            args=grpo_config,
            train_dataset=trl_train_dataset,
            processing_class=tokenizer,
            eval_dataset=create_rl_dataset(
                profile_agent=profile_agent,
                user_ids=inter_dataset.eval_user_ids,
                histories=inter_dataset.eval_histories,
                target_items=inter_dataset.eval_target_items,
                recaller_names=[model_name.lower() for model_name in args.recbole_models]
            ) if args.do_eval else None,
            # peft_config=peft_config if args.use_peft else None,
        )
        trainer.train()
        # accelerator.wait_for_everyone() 
    if args.do_test:
        trl_test_dataset = create_rl_dataset(
            profile_agent=profile_agent,
            user_ids=inter_dataset.test_user_ids[:cut_off_users],
            histories=inter_dataset.test_histories[:cut_off_users],
            target_items=inter_dataset.test_target_items[:cut_off_users],
            recaller_names=[model_name.lower() for model_name in args.recbole_models]
        )
        if args.do_test_sft or args.do_test_rl:
            model_save_dir = f"{model_save_dir}_sft" if args.do_test_sft else f"{model_save_dir}"
            model_save_dir = f"{model_save_dir}_rl" if args.do_test_rl else model_save_dir
            model_name = get_last_checkpoint(model_save_dir)
        model = outlines.from_transformers(
            AutoModelForCausalLM.from_pretrained(
                model_name, 
                trust_remote_code=True, 
                device_map="auto", 
                torch_dtype=torch.float16
                # load_in_8bit=True
            ),
            tokenizer
        )
        completions = []
        for i in tqdm(range(len(trl_test_dataset))):
            prompts = trl_test_dataset[i]['prompt'][0]['content']
            completions.append(model(prompts, model_config, max_new_tokens=200))
        import pickle
        with open(f"completions/completions_{args.dataset}_"
          f"{'_'.join(k for k, v in vars(args).items() if isinstance(v, bool) and v)}"
          f".pkl", "wb") as f:
            pickle.dump(completions, f)
        metrics = evaluate(
            completions=completions,
            uid=trl_test_dataset["uid"],
            histories=trl_test_dataset["histories"],
            recallers=recallers,
            final_k=args.final_k,
            target_items=trl_test_dataset["target_items"],
            ks=[10, 50],
        )
        print(json.dumps({"multi_channel_metrics": metrics}, indent=4))
    if args.do_test_recaller:
        trl_test_dataset = create_rl_dataset(
            profile_agent=None,
            user_ids=inter_dataset.test_user_ids[:cut_off_users],
            histories=inter_dataset.test_histories[:cut_off_users],
            target_items=inter_dataset.test_target_items[:cut_off_users],
            recaller_names=[model_name.lower() for model_name in args.recbole_models]
        )
        recaller_metrics = evaluate_recallers(
            recallers=recallers,
            uid=trl_test_dataset["uid"],
            histories=trl_test_dataset["histories"],
            target_items=trl_test_dataset["target_items"],
            final_k=args.final_k,
            ks=[10, 50],
        )
        print(json.dumps(recaller_metrics, indent=4))

    
if __name__ == "__main__":
    main()