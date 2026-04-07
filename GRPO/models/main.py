import argparse
import os
import random
import json
import numpy as np
from typing import List

from GRPO.core.utils import set_seed
from GRPO.core.data import load_dataset
from GRPO.core.recallers import RecBoleRecaller
from GRPO.core.agents import UserProfileAgent, LLMRouterAgent


def create_recaller(model_name: str, dataset_name: str, checkpoint_dir: str, data_path: str, seed: int, use_latest_checkpoint: bool = True, num_items: int = 0) -> RecBoleRecaller:
    """Create a recaller with proper configuration for the specified model"""
    
    # Find checkpoint if requested
    checkpoint_path = f"{checkpoint_dir}/{dataset_name}"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    
    # Model-specific configurations (epochs: 100 for trainable models; main_pure uses this via config_dict.update)
    model_configs = {
        'BPR': {
            'seed': seed,
            'epochs': 100,
            'learning_rate': 0.001,
            'embedding_size': 64,
            'train_batch_size': 2048,
        },
        'SASRec': {
            'seed': seed,
            'epochs': 100,
            'learning_rate': 0.001,
            'hidden_size': 64,
            'max_seq_length': 50,
            'train_batch_size': 256,
        },
        'Pop': {
            'seed': seed,
            'epochs': 1,  # Pop doesn't need training
        },
        'ItemKNN': {
            'seed': seed,
            'k': 50,
            'shrink': 0.0,
        },
        'FPMC': {
            'seed': seed,
            'epochs': 100,
            'learning_rate': 0.001,
            'embedding_size': 64,
        },
        'GRU4Rec': {
            'seed': seed,
            'epochs': 100,
            'learning_rate': 0.001,
            'embedding_size': 64,
            'hidden_size': 128,
        },
        'LightGCN': {
            'seed': seed,
            'epochs': 100,
        },
        'SimpleX': {
            'seed': seed,
            'epochs': 100,
        },
        'RaCT': {
            'seed': seed,
            'epochs': 100,
            'latent_dimension': 64,
            'learning_rate': 0.001,
        },
        'EASE': {
            'seed': seed,
            'reg_weight': 250.0,
        },
        'RecVAE': {
            'seed': seed,
            'epochs': 100,
            'latent_dimension': 64,
            'learning_rate': 0.001,
        },
        'SLIMElastic': {
            'seed': seed,
            'alpha': 0.2,
            'l1_ratio': 0.02,
        },
        'NeuMF': {
            'seed': seed,
            'epochs': 100,
            'mf_embedding_size': 64,
            'mlp_embedding_size': 64,
            'learning_rate': 0.001,
        },
    }
    
    # Get configuration for the specific model
    config_dict = model_configs.get(model_name, {'seed': seed})
    
    # Create the recaller
    recaller = RecBoleRecaller(
        model_name=model_name,
        dataset_name=dataset_name,
        checkpoint_path=checkpoint_path,
        data_path=data_path,
        config_dict=config_dict,
        num_items=num_items
    )
    
    return recaller


def initialize_recallers(
    model_names: List[str], dataset_name: str, checkpoint_dir: str, data_path: str, seed: int, use_latest_checkpoint: bool = True, num_items: int = 0) -> dict:
    """Initialize all recallers with proper error handling"""
    
    recallers = {}
    failed_models = []
    
    # Supported models in RecBole (with correct casing)
    supported_models = ['BPR', 'SASRec', 'Pop', 'ItemKNN', 'FPMC', 'GRU4Rec', 'LightGCN', 'SimpleX', 'RaCT', 'EASE', 'RecVAE', 'SLIMElastic', 'NeuMF']
    # Case-insensitive mapping to correct model names
    model_name_map = {m.lower(): m for m in supported_models}
    
    for model_name in model_names:
        # Normalize model name (case-insensitive lookup)
        normalized_name = model_name_map.get(model_name.lower(), model_name)
        
        # Handle legacy model name mappings
        if model_name.lower() == 'itemcf':
            normalized_name = 'ItemKNN'
        
        print(f"Initializing {normalized_name} recaller...")
        
        if normalized_name not in supported_models:
            print(f"Warning: Model {model_name} not supported, skipping...")
            failed_models.append(model_name)
            continue
        
        recaller = create_recaller(
            model_name=normalized_name,
            dataset_name=dataset_name,
            checkpoint_dir=checkpoint_dir,
            data_path=data_path,
            seed=seed,
            use_latest_checkpoint=use_latest_checkpoint,
            num_items=num_items
        )
        
        recallers[model_name.lower()] = recaller
        print(f"✅ Successfully initialized {normalized_name} recaller")
    
    # Ensure we have at least one recaller
    if not recallers:
        print("No recallers successfully initialized, creating Pop model as fallback...")
        fallback_recaller = create_recaller(
            model_name='Pop',
            dataset_name=dataset_name,
            checkpoint_dir=checkpoint_dir,
            data_path=data_path,
            seed=seed,
            use_latest_checkpoint=False,
            num_items=num_items
        )
        recallers['pop'] = fallback_recaller
        print("✅ Fallback Pop recaller initialized")
    
    if failed_models:
        print(f"⚠️  Failed to initialize models: {failed_models}")
    
    print(f"📊 Total recallers initialized: {len(recallers)} ({list(recallers.keys())})")
    return recallers


def main(argv: List[str] = None):
    ap = argparse.ArgumentParser(description='GRPO Training for Recommendation Systems')
    ap.add_argument('--dataset', type=str, default='ml-100k')
    ap.add_argument('--data_path', type=str, default='./data')
    ap.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    ap.add_argument('--group_size', type=int, default=4)
    ap.add_argument('--final_k', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--recbole_models', type=str, nargs='+', default=['SASRec', 'BPR'])
    ap.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    ap.add_argument('--use_latest_checkpoint', action='store_true')
    ap.add_argument('--router_only', action='store_true', help='Run in router-only mode')
    ap.add_argument('--eval_only', action='store_true', help='Only evaluate router without training (overrides --train_router)')
    ap.add_argument('--router_batch_size', type=int, default=32, help='Batch size for router training')
    ap.add_argument('--use_hf_local', action='store_true')
    ap.add_argument('--hf_model', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    # Advanced/legacy args removed: users_per_batch, train_router, router_strategy,
    # save_router_json, hf_dtype, hf_device, use_trl_grpo, api_base, api_key, api_model
    args = ap.parse_args(argv)

    set_seed(args.seed)
    inter = load_dataset(args.dataset, args.data_path, seed=args.seed)

    # Initialize recallers using the new simplified approach
    recallers = initialize_recallers(
        model_names=args.recbole_models,
        dataset_name=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        data_path=args.data_path,
        seed=args.seed,
        use_latest_checkpoint=args.use_latest_checkpoint,
        num_items=inter.num_items
    )
    data_map_path = os.path.join(args.data_path, f'{args.dataset}', f'{args.dataset}.data_maps')
    reviews_path = os.path.join(args.data_path, f'{args.dataset}', f'{args.dataset}.reviews')
    with open(data_map_path, 'r') as f:
        data_maps = json.load(f)
    with open(reviews_path, 'r') as f:
        reviews = json.load(f) 
    profile_agent = UserProfileAgent(inter.num_items, data_maps, reviews)
    # Initialize router
    router = LLMRouterAgent(
        n_per_user=args.group_size,
        available_models=list(recallers.keys())
    )
    
    # Initialize HF models if needed
    model = ref_model = tokenizer = None
    if args.use_hf_local:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        model = AutoModelForCausalLM.from_pretrained(
            args.hf_model, 
            trust_remote_code=True,
            device_map="auto"
        )
        
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.hf_model, 
            trust_remote_code=True,
            device_map="auto"
        )
        for p in ref_model.parameters(): 
            p.requires_grad = False
        ref_model.eval()

    if args.router_only:
        users = list(range(inter.num_users))
        
        # Determine if we should train or just evaluate (default: train unless eval_only)
        should_train_router = not args.eval_only
        
        if should_train_router:
            print("Router training is temporarily disabled (no trainer implemented)")
            # TODO: Implement GRPO training logic or use TRL
            return
        
        # Simplified router evaluation mode
        print("Evaluating router performance...")
        cfg = GRPOConfig(beta=1.0, group_size=args.group_size)
        
        total_recalls = []
        for uid in users:
            hist = inter.user2item_list_train.get(uid, [])
            prof_json = profile_agent.forward(uid, hist).profile_json
            
            # Use router to generate routes directly
            result = router.forward(prof_json, n_candidates=cfg.group_size, temperature=cfg.temperature)
            routes = result.get("routes", [])
            
            if not routes:
                continue
                
            # Select first route for evaluation
            chosen_route = routes[0]
            
            # Calculate recommendation results
            from collections import defaultdict
            candidates = defaultdict(int)
            
            for model_usage in chosen_route:
                model_name = model_usage.get("name", "")
                k = model_usage.get("k", 10)
                weight = model_usage.get("weight", 1.0)
                
                if model_name in recallers:
                    items = recallers[model_name].recall(uid, int(k), hist)
                    for item in items:
                        candidates[item[0]] += item[1] * weight
            
            # Sort by score and take top-k
            final_candidates = sorted(candidates.keys(), key=lambda x: candidates[x], reverse=True)[:args.final_k]
            
            # Calculate recall
            from .utils import recall_at_k
            recall = recall_at_k(final_candidates, inter.user2items_test.get(uid, []), args.final_k)
            total_recalls.append(recall)
        
        avg_recall = sum(total_recalls) / len(total_recalls) if total_recalls else 0.0
        print(f"Done. Router evaluation avg Recall@{args.final_k} = {avg_recall:.4f}")
        return

    print("Note: Selector training has been deprecated. Use --router_only for router training.")


if __name__ == '__main__':
    main()
