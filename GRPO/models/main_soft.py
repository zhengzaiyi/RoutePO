import argparse
import json
import os
import random
from functools import partial
from typing import List, Dict
import numpy as np

import outlines
import torch
from datasets import Dataset
from huggingface_hub import InferenceClient
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
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
from GRPO.models.soft_model import get_model_and_tokenizer, get_mixed_model_and_tokenizer
from GRPO.trainers.soft_sft_trainer import SoftSFTTrainer
from GRPO.models.soft_utils import build_soft_template, generate_soft_completions

# Dataset generation configuration (from main_pure.py)
MIN_HISTORY_FOR_AUGMENTATION = 30
AUGMENTATION_STEP = 10

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
                        "top-k": 1.0 if recaller_name == best_recaller else 0,
                        "score-weight": 1.0 if recaller_name == best_recaller else 0.0
                    } for recaller_name in recallers.keys()
                }
            elif norm_type == 'static':
                recaller_scores = {name: ndcg_at_k([item[0] for item in items], gt_items, k=final_k) 
                                if (items := recallers[name].recall(uid, final_k, eval_hist)) else 0
                                for name in recallers.keys()}
                
                scores = np.array(list(recaller_scores.values()))
                weights = np.exp(scores) / np.exp(scores).sum()  # softmax with temperature=0.5

                k_values = weights / weights.sum()
                
                # k_values is already normalized by final_k, so it should be in a reasonable range
                # We keep the normalized k_values (not converting back to int) for Beta distribution
                ground_truth_output = {
                    name: {"top-k": float(k), "score-weight": float(w)}
                    for name, k, w in zip(recaller_scores.keys(), k_values, weights/weights.sum())
                }
            else:
                raise ValueError(f"Invalid norm type: {norm_type}")
            
            completion_lines = ["{"]
            value_targets = []
            for recaller_name in recallers.keys():
                completion_lines.append(f"  {recaller_name}: {{")
                inner_lines = []
                hparams = ['top-k', 'score-weight']
                for hp_idx, hp in enumerate(hparams):
                    suffix = "," if hp_idx < len(hparams) - 1 else ""
                    inner_lines.append(f"    {hp}: [num][soft_token]{suffix}")
                    value_targets.append(float(ground_truth_output[recaller_name][hp]))
                completion_lines.extend(inner_lines)
                completion_lines.append("  },")
            completion_lines[-1] = completion_lines[-1].rstrip(",")
            completion_lines.append("}")
            completion = "\n".join(completion_lines)
            dataset.append({
                "prompt": prompt + "\n\n",
                "completion": completion,
                "value_targets": value_targets,
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
        content = build_prompt(prof_json, recaller_names) if profile_agent is not None else recaller_names
        dataset.append({
            "prompt": content,
            "uid": uid,
            "histories": histories[i],
            "target_items": [target_items[i]]
        })
    return Dataset.from_list(dataset)

def get_soft_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens = {"additional_special_tokens": ["[num]", "[soft_token]"]}
    num_added = tokenizer.add_special_tokens(special_tokens)
    return tokenizer



class SoftDataCollator:
    """Custom data collator that supports mixed token-level supervision.
    
    For each feature we expect:
    - prompt: base prompt text
    - completion: completion text containing placeholders like [num][soft_token]
    - value_targets: list of floats aligned with the occurrences of [soft_token] in completion
    """
    
    def __init__(self, tokenizer, max_length=1536):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_index = -100  # Standard PyTorch ignore index for cross-entropy
        self.soft_token_id = tokenizer.convert_tokens_to_ids("[soft_token]")
    
    def __call__(self, features):
        batch_size = len(features)
        
        prompts = [f["prompt"] for f in features]
        completions = [f["completion"] for f in features]
        value_targets_list = [f.get("value_targets", []) for f in features]
        
        prompt_encodings = []
        for prompt in prompts:
            encoding = self.tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=True,
                # max_length=self.max_length,
                return_tensors=None,
            )
            prompt_encodings.append(encoding)
        
        full_texts = [
            prompt + completion for prompt, completion in zip(prompts, completions)
        ]
        
        model_inputs = self.tokenizer(
            full_texts,
            # max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        
        input_ids = model_inputs["input_ids"]
        labels = input_ids.clone()
        value_labels = torch.zeros_like(input_ids, dtype=torch.float32)
        value_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        for idx in range(batch_size):
            prompt_length = len(prompt_encodings[idx]["input_ids"])
            labels[idx, :prompt_length] = self.ignore_index
            
            soft_positions = (input_ids[idx] == self.soft_token_id).nonzero(as_tuple=True)[0]
            targets = value_targets_list[idx]
            if len(targets) != len(soft_positions):
                raise ValueError(
                    f"Mismatch between value_targets ({len(targets)}) and [soft_token] occurrences ({len(soft_positions)})"
                )
            if len(soft_positions) > 0:
                value_mask[idx, soft_positions] = True
                value_labels[idx, soft_positions] = torch.tensor(targets, dtype=torch.float32)
                labels[idx, soft_positions] = self.ignore_index
        
        labels[labels == self.tokenizer.pad_token_id] = self.ignore_index
        
        model_inputs["labels"] = labels
        model_inputs["value_labels"] = value_labels
        model_inputs["value_mask"] = value_mask
        
        return model_inputs
    
    def __call__alternative(self, features):
        """
        Alternative implementation using token search instead of length calculation.
        This might be more accurate for some tokenizers.
        """
        batch_size = len(features)
        
        # Extract data
        prompts = [f['prompt'] for f in features]
        completions = [f['completion'] for f in features]
        value_labels = [f['value_labels'] for f in features]
        
        # Get special token ids
        num_token_id = self.tokenizer.convert_tokens_to_ids('[num]')
        soft_token_id = self.tokenizer.convert_tokens_to_ids('[soft_token]')
        
        # Tokenize full sequences
        full_texts = [p + c for p, c in zip(prompts, completions)]
        model_inputs = self.tokenizer(
            full_texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Create labels
        labels = model_inputs["input_ids"].clone()
        
        # Find completion tokens and mask everything before them
        for idx in range(batch_size):
            input_ids = model_inputs["input_ids"][idx]
            
            # Find the position of [num] or [soft_token]
            num_pos = (input_ids == num_token_id).nonzero(as_tuple=True)[0]
            soft_pos = (input_ids == soft_token_id).nonzero(as_tuple=True)[0]
            
            if len(num_pos) > 0:
                # Mask everything before [num]
                completion_start = num_pos[0].item()
                labels[idx, :completion_start] = self.ignore_index
            elif len(soft_pos) > 0:
                # Mask everything before [soft_token]
                completion_start = soft_pos[0].item()
                labels[idx, :completion_start] = self.ignore_index
            else:
                # If neither token found, mask everything (shouldn't happen)
                print(f"Warning: No completion token found in example {idx}")
                labels[idx, :] = self.ignore_index
        
        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = self.ignore_index
        
        model_inputs["labels"] = labels
        model_inputs["value_labels"] = torch.tensor(value_labels, dtype=torch.float32)
        
        return model_inputs


def parse_args():
    ap = argparse.ArgumentParser(description='Soft Token SFT Training for Recommendation Systems')
    ap.add_argument('--dataset', type=str, default='Amazon_All_Beauty')
    ap.add_argument('--data_path', type=str, default='./dataset')
    ap.add_argument('--final_k', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--recbole_models', type=str, nargs='+', default=['BPR', 'SASRec'])
    ap.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    ap.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    ap.add_argument('--output_dir', type=str, default='GRPO/soft_models')
    ap.add_argument('--gen_sft_data', action='store_true')
    ap.add_argument('--do_sft', action='store_true')
    ap.add_argument('--do_rl', action='store_true', help='Run GRPO reinforcement learning')
    ap.add_argument('--do_test', action='store_true')
    ap.add_argument('--do_test_sft', action='store_true')
    ap.add_argument('--do_test_rl', action='store_true', help='Test RL model')
    ap.add_argument('--do_test_sft_rl', action='store_true', help='Test SFT+RL model')
    ap.add_argument('--use_lora', action='store_true')
    ap.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    ap.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    ap.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    ap.add_argument('--norm_type', type=str, default='static', choices=['oracle', 'static'])
    ap.add_argument('--num_train_samples', type=int, default=100000, help='Number of training samples')
    ap.add_argument('--per_device_train_batch_size', type=int, default=2)
    ap.add_argument('--gradient_accumulation_steps', type=int, default=1)
    ap.add_argument('--learning_rate', type=float, default=5e-5)
    ap.add_argument('--num_train_epochs', type=int, default=3)
    ap.add_argument('--warmup_steps', type=int, default=100)
    ap.add_argument('--logging_steps', type=int, default=10)
    ap.add_argument('--save_steps', type=int, default=500)
    ap.add_argument('--eval_steps', type=int, default=100)
    ap.add_argument('--max_length', type=int, default=1536)
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--gradient_checkpointing', action='store_true')
    # GRPO specific arguments
    ap.add_argument('--kl_coef', type=float, default=0.1, help='KL coefficient for GRPO')
    ap.add_argument('--grpo_num_samples', type=int, default=10000, help='Number of samples for GRPO training')
    ap.add_argument('--grpo_batch_size', type=int, default=2, help='Batch size for GRPO training')
    ap.add_argument('--grpo_learning_rate', type=float, default=1e-5, help='Learning rate for GRPO')
    ap.add_argument('--grpo_num_train_epochs', type=int, default=1, help='Number of epochs for GRPO')
    # Parameters from main_pure.py for dataset creation
    ap.add_argument('--profile_cutoff', type=int, default=20,
                   help='Cutoff length for user profiles and minimum history length for augmentation.')
    ap.add_argument('--min_history_for_augmentation', type=int, default=30,
                   help='Minimum history length for data augmentation')
    ap.add_argument('--augmentation_step', type=int, default=10,
                   help='Step size for history length augmentation')
    args = ap.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    
    model_save_dir = f"{args.output_dir}/{args.dataset}/{args.model_name.split('/')[-1]}"
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Load dataset
    
    model_save_dir_rl = f"{args.output_dir}/{args.dataset}/{args.model_name.split('/')[-1]}_rl"
    grpo_config = GRPOConfig(
            output_dir=model_save_dir_rl,
            generation_batch_size=16,
            learning_rate=args.grpo_learning_rate,
            per_device_train_batch_size=args.grpo_batch_size,
            per_device_eval_batch_size=args.grpo_batch_size,
            num_train_epochs=args.grpo_num_train_epochs,
            do_eval=False,
            logging_steps=args.logging_steps,
            save_strategy="epoch",
            beta=0.0,  # In GRPOConfig, 'beta' is the KL penalty coefficient
            max_prompt_length=2048,
            max_completion_length=300,
            seed=args.seed,
            bf16=True,
            report_to="none",
            gradient_checkpointing=args.gradient_checkpointing
        )
    
    # Generate or load SFT data - only initialize recallers when generating data, testing, or RL training
    if args.gen_sft_data or args.do_test_sft or args.do_test_rl or args.do_test_sft_rl or args.do_rl or args.do_test:
        inter_dataset = load_dataset(args.dataset, args.data_path, seed=args.seed)
        profile_agent = UserProfileAgent(inter_dataset, args.dataset)
        cut_off_users = min(len(inter_dataset.train_user_ids), args.num_train_samples)
        recallers = initialize_recallers(
            model_names=args.recbole_models,
            dataset_name=args.dataset,
            checkpoint_dir=args.checkpoint_dir,
            data_path=args.data_path,
            seed=args.seed,
            use_latest_checkpoint=True,
            num_items=inter_dataset.ds.item_num
        )
    
    if args.gen_sft_data:
        # Generate SFT data separately to avoid GPU conflicts
        assert not args.do_sft and not args.do_test_sft and not args.do_rl and not args.do_test_rl and not args.do_test_sft_rl and not args.do_test
        
        # Get training data
        user_ids = inter_dataset.train_user_ids[:cut_off_users]
        histories = inter_dataset.train_histories[:cut_off_users]
        target_items = inter_dataset.train_target_items[:cut_off_users]
        
        # Create train dataset
        train_dataset = create_sft_dataset(
            profile_agent=profile_agent,
            user_ids=user_ids,
            histories=histories,
            target_items=target_items,
            recallers=recallers,
            final_k=args.final_k,
            norm_type=args.norm_type,
            profile_cutoff=args.profile_cutoff,
            min_history_for_augmentation=args.min_history_for_augmentation,
            augmentation_step=args.augmentation_step
        )
        
        # Create eval dataset from test set (like main_pure.py)
        eval_dataset = create_sft_dataset(
            profile_agent=profile_agent,
            user_ids=inter_dataset.test_user_ids[:5000],
            histories=inter_dataset.test_histories[:5000],
            target_items=inter_dataset.test_target_items[:5000],
            recallers=recallers,
            final_k=args.final_k,
            norm_type=args.norm_type,
            profile_cutoff=args.profile_cutoff,
            min_history_for_augmentation=args.min_history_for_augmentation,
            augmentation_step=args.augmentation_step
        )
        
        # Save datasets
        if os.path.exists(f'{model_save_dir}_sft_data'):
            import shutil
            shutil.rmtree(f'{model_save_dir}_sft_data')
        os.makedirs(f'{model_save_dir}_sft_data', exist_ok=True)
        train_dataset.save_to_disk(f'{model_save_dir}_sft_data/train')
        eval_dataset.save_to_disk(f'{model_save_dir}_sft_data/eval')
        print(f"SFT datasets saved to {model_save_dir}_sft_data")
        print(f"Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")
        return
    
    if args.do_sft:
        model_save_dir = f"{model_save_dir}_sft"
        if not os.path.exists(model_save_dir) or len(os.listdir(model_save_dir)) == 0:
            # Load model and tokenizer
            print("Loading model and tokenizer...")
            model, tokenizer = get_mixed_model_and_tokenizer(args.model_name)
            
            # Add padding token if needed
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Enable gradient checkpointing if requested
            if args.gradient_checkpointing:
                model.gradient_checkpointing_enable()
            
            # Apply LoRA if requested
            if args.use_lora:
                print("Applying LoRA...")
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    modules_to_save=["value_head"]  # Also train the value head
                )
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
            
            # Load datasets
            train_dataset = Dataset.load_from_disk(f'{model_save_dir[:-4]}_sft_data/train')
            eval_dataset = Dataset.load_from_disk(f'{model_save_dir[:-4]}_sft_data/eval')
            
            print(f"Total training samples: {len(train_dataset)}")
            print(f"Total eval samples: {len(eval_dataset)}")
            print(f"Sample data point: {train_dataset[0]}")
            
            # Set up training arguments
            sft_config = TrainingArguments(
                output_dir=model_save_dir,
                save_total_limit=2,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                num_train_epochs=args.num_train_epochs,
                save_strategy="steps",
                save_steps=args.save_steps,
                logging_steps=args.logging_steps,
                evaluation_strategy="steps",  # Enable evaluation!
                eval_steps=args.eval_steps,
                bf16=True,
                learning_rate=args.learning_rate,
                warmup_steps=args.warmup_steps,
                seed=args.seed,
                gradient_checkpointing=args.gradient_checkpointing,
                run_name=f"soft_sft_{args.dataset}_lr{args.learning_rate}",
                report_to="none",
                metric_for_best_model="eval_loss",  # Use eval_loss instead of loss
                greater_is_better=False,
                load_best_model_at_end=True,  # Load best model at the end
                remove_unused_columns=False,  # Important for custom columns
                label_names=["labels", "value_labels", "value_mask"],  # Explicitly declare our label fields
                fp16=args.fp16 and not args.bf16,
                dataloader_drop_last=True,
            )
            
            # Create data collator
            data_collator = SoftDataCollator(tokenizer=tokenizer, max_length=args.max_length)
            
            # Create trainer
            trainer = SoftSFTTrainer(
                model=model,
                args=sft_config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )
            
            # Train
            print("Starting training...")
            trainer.train()
            
            # Save final model
            print("Saving final model...")
            trainer.save_model()
            tokenizer.save_pretrained(model_save_dir)
        
        model_name = get_last_checkpoint(model_save_dir)
        print(f"Using checkpoint: {model_name}")
    
    if args.do_rl:
        # GRPO training
        if args.do_sft:
            # Use SFT model as starting point
            model_save_dir_base = f"{args.output_dir}/{args.dataset}/{args.model_name.split('/')[-1]}_sft"
            model_name = get_last_checkpoint(model_save_dir_base)
        else:
            model_name = args.model_name
            
        print(f"Starting GRPO training with model: {model_name}")
        
        # Create RL dataset
        trl_train_dataset = create_rl_dataset(
            profile_agent=profile_agent,
            user_ids=inter_dataset.train_user_ids[:cut_off_users],
            histories=inter_dataset.train_histories[:cut_off_users],
            target_items=inter_dataset.train_target_items[:cut_off_users],
            recaller_names=[m.lower() for m in args.recbole_models]
        )
        
        # Load tokenizer
        tokenizer = get_soft_tokenizer(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Define reward function for soft tokens
        def soft_reward_func(completions, uid, histories, target_items, **kwargs):
            """Calculate rewards for soft token predictions"""
            rewards = []
            recall_results = multi_channel_recall(
                completions=completions, 
                uid=uid, 
                histories=histories, 
                recallers=recallers, 
                final_k=args.final_k
            )
            for i, recall_result in enumerate(recall_results):
                ndcg = ndcg_at_k(recall_result, target_items[i], k=args.final_k)
                rewards.append(ndcg)
            return rewards
        
        # Create trainer
        # Use mixed model with value head so that beta sampling can work
        mixed_model, processing_tokenizer = get_mixed_model_and_tokenizer(model_name)
        trainer = GRPOTrainer(
            model=mixed_model,
            reward_funcs=soft_reward_func,
            args=grpo_config,
            train_dataset=trl_train_dataset,
            processing_class=processing_tokenizer,
        )
        # Enable beta-sampling generation path and pass model names to template
        trainer.use_beta_sampling = True
        trainer.beta_template_models = [m.lower() for m in args.recbole_models]
        
        # Train
        print("Starting GRPO training...")
        trainer.train()
        
        # Update model name for testing
        model_name = get_last_checkpoint(model_save_dir_rl)
        print(f"GRPO training completed. Model saved to: {model_name}")
    
    if args.do_test_sft or args.do_test_rl or args.do_test_sft_rl or args.do_test:
        # Determine model path
        if args.do_test_rl:
            model_name = get_last_checkpoint(f"{args.output_dir}/{args.dataset}/{args.model_name.split('/')[-1]}_rl")
        elif args.do_test_sft:
            model_name = get_last_checkpoint(f"{args.output_dir}/{args.dataset}/{args.model_name.split('/')[-1]}_sft")
        elif args.do_test_sft_rl:
            model_name = get_last_checkpoint(f"{args.output_dir}/{args.dataset}/{args.model_name.split('/')[-1]}_sft_rl")
        elif args.do_test:
            model_name = args.model_name
        
        print(f"Testing model: {model_name}")
        
        # Create test dataset
        cut_off_test_users = min(len(inter_dataset.test_user_ids), 100000)
        trl_test_dataset = create_rl_dataset(
            profile_agent=profile_agent,
            user_ids=inter_dataset.test_user_ids[:cut_off_test_users],
            histories=inter_dataset.test_histories[:cut_off_test_users],
            target_items=inter_dataset.test_target_items[:cut_off_test_users],
            recaller_names=[m.lower() for m in args.recbole_models]
        )
        
        # Load model and generate completions
        model, tokenizer = get_mixed_model_and_tokenizer(model_name)
        model.eval()
        completions = generate_soft_completions(
            model=model,
            tokenizer=tokenizer,
            test_dataset=trl_test_dataset,
            model_names=[m.lower() for m in args.recbole_models],
            max_length=args.max_length
        )
        
        # Save and evaluate
        os.makedirs("completions", exist_ok=True)
        import pickle
        completion_file = f"completions/soft_completions_{args.dataset}_{'rl' if args.do_test_rl else 'sft'}.pkl"
        with open(completion_file, "wb") as f:
            pickle.dump(completions, f)
        print(f"Saved {len(completions)} completions to {completion_file}")
        
        # Evaluate
        from GRPO.core.utils import evaluate
        metrics = evaluate(
            completions=completions,
            uid=trl_test_dataset["uid"],
            histories=trl_test_dataset["histories"],
            recallers=recallers,
            final_k=args.final_k,
            target_items=trl_test_dataset["target_items"],
            ks=[10, 50],
        )
        
        print("Evaluation Results:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()