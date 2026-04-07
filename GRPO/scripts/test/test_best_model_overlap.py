#!/usr/bin/env python3
"""Test script to analyze best model overlap between training and test sets"""

import argparse
from collections import defaultdict

import torch.distributed as dist
import os
import sys
sys.path.append('<path_to_routepo>')
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
if not dist.is_initialized():
    dist.init_process_group(backend='gloo', rank=0, world_size=1)

from GRPO.core.data import load_dataset
from GRPO.models.main import initialize_recallers
from GRPO.core.utils import ndcg_at_k, set_seed

def analyze_best_model_overlap(args):
    set_seed(args.seed)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    inter_data = load_dataset(args.dataset, args.data_path, seed=args.seed)
    
    # Initialize recallers
    print(f"Initializing recallers: {args.recbole_models}")
    recallers = initialize_recallers(
        model_names=args.recbole_models,
        dataset_name=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        data_path=args.data_path,
        seed=args.seed,
        use_latest_checkpoint=True,
        num_items=inter_data.ds.item_num
    )
    
    # Analyze training set
    print("\nAnalyzing training set...")
    train_best_models = {}  # Now stores list of best models
    train_scores = defaultdict(lambda: defaultdict(float))
    
    for i, uid in enumerate(inter_data.train_user_ids[:args.max_users]):
        hist = inter_data.train_histories[i]
        target = [inter_data.train_target_items[i]] if isinstance(inter_data.train_target_items[i], int) else inter_data.train_target_items[i]
        
        best_score = -1
        best_models = []  # Can have multiple best models
        
        for model_name, recaller in recallers.items():
            items = recaller.recall(uid, args.k, hist)
            item_ids = [item[0] for item in items] if items else []
            score = ndcg_at_k(item_ids, target, k=args.k)
            train_scores[model_name][uid] = score
            
            if score > best_score:
                best_score = score
                best_models = [model_name]
            elif score == best_score and score > 0:  # Add to best models if tied
                best_models.append(model_name)
        
        train_best_models[uid] = best_models
    
    # Analyze test set
    print("\nAnalyzing test set...")
    test_best_models = {}  # Now stores list of best models
    test_scores = defaultdict(lambda: defaultdict(float))
    
    for i, uid in enumerate(inter_data.test_user_ids[:args.max_users]):
        hist = inter_data.test_histories[i]
        target = [inter_data.test_target_items[i]] if isinstance(inter_data.test_target_items[i], int) else inter_data.test_target_items[i]
        
        best_score = -1
        best_models = []  # Can have multiple best models
        
        for model_name, recaller in recallers.items():
            items = recaller.recall(uid, args.k, hist)
            item_ids = [item[0] for item in items] if items else []
            score = ndcg_at_k(item_ids, target, k=args.k)
            test_scores[model_name][uid] = score
            
            if score > best_score:
                best_score = score
                best_models = [model_name]
            elif score == best_score and score > 0:  # Add to best models if tied
                best_models.append(model_name)
        
        test_best_models[uid] = best_models
    
    # Calculate overlap
    common_users = set(train_best_models.keys()) & set(test_best_models.keys())
    # Count overlap if there's any intersection between best models
    overlap_count = sum(1 for uid in common_users 
                       if set(train_best_models[uid]) & set(test_best_models[uid]))
    exact_overlap_count = sum(1 for uid in common_users 
                             if set(train_best_models[uid]) == set(test_best_models[uid]))
    no_overlap_count = sum(1 for uid in common_users 
                          if not (set(train_best_models[uid]) & set(test_best_models[uid])))
    overlap_rate = overlap_count / len(common_users) if common_users else 0
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    # Model performance summary
    print("\nModel Performance Summary (NDCG@{:d}):".format(args.k))
    print("-"*40)
    for model_name in args.recbole_models:
        train_avg = sum(train_scores[model_name].values()) / len(train_scores[model_name]) if train_scores[model_name] else 0
        test_avg = sum(test_scores[model_name].values()) / len(test_scores[model_name]) if test_scores[model_name] else 0
        print(f"{model_name:15s} Train: {train_avg:.4f}  Test: {test_avg:.4f}")
    
    # Best model distribution
    print("\nBest Model Distribution:")
    print("-"*40)
    train_dist = defaultdict(int)
    test_dist = defaultdict(int)
    train_ties = 0  # Count users with tied best models
    test_ties = 0
    
    for models in train_best_models.values():
        if len(models) > 1:
            train_ties += 1
        for model in models:  # Count each model in the list
            train_dist[model] += 1
    
    for models in test_best_models.values():
        if len(models) > 1:
            test_ties += 1
        for model in models:  # Count each model in the list
            test_dist[model] += 1
    
    print("Train set:")
    for model, count in sorted(train_dist.items(), key=lambda x: -x[1]):
        print(f"  {model:15s} {count:4d} times as best ({count/len(train_best_models)*100:5.1f}% of users)")
    print(f"  Users with tied best models: {train_ties} ({train_ties/len(train_best_models)*100:5.1f}%)")
    
    print("\nTest set:")
    for model, count in sorted(test_dist.items(), key=lambda x: -x[1]):
        print(f"  {model:15s} {count:4d} times as best ({count/len(test_best_models)*100:5.1f}% of users)")
    print(f"  Users with tied best models: {test_ties} ({test_ties/len(test_best_models)*100:5.1f}%)")
    
    # Overlap analysis
    print("\nBest Model Overlap Analysis:")
    print("-"*40)
    print(f"Common users: {len(common_users)}")
    if len(common_users) > 0:
        print(f"  - Exact same best models: {exact_overlap_count} ({exact_overlap_count/len(common_users)*100:.1f}%)")
        print(f"  - Partial overlap: {overlap_count - exact_overlap_count} ({(overlap_count - exact_overlap_count)/len(common_users)*100:.1f}%)")
        print(f"  - No overlap: {no_overlap_count} ({no_overlap_count/len(common_users)*100:.1f}%)")
        print(f"Total overlap rate: {overlap_rate:.2%} (any intersection)")
    else:
        print("  No common users found!")
    
    # Transition matrix
    print("\nTransition Matrix (Train -> Test):")
    print("(Note: Values are fractional for users with tied best models)")
    print("-"*40)
    transition = defaultdict(lambda: defaultdict(int))
    # For users with multiple best models, count all transitions
    for uid in common_users:
        for train_model in train_best_models[uid]:
            for test_model in test_best_models[uid]:
                transition[train_model][test_model] += 1/len(train_best_models[uid])  # Fractional count
    
    # Print header
    print("{:15s}".format('Train\\Test'), end="")
    for model in args.recbole_models:
        print(f"{model:>10s}", end="")
    print()
    
    # Print matrix
    for train_model in args.recbole_models:
        print(f"{train_model:15s}", end="")
        for test_model in args.recbole_models:
            count = transition[train_model][test_model]
            print(f"{count:>10.1f}", end="")  # Format as float with 1 decimal
        print()

def main():
    parser = argparse.ArgumentParser(description='Analyze best model overlap between training and test sets')
    parser.add_argument('--dataset', type=str, default='ml-1m', help='Dataset name')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--recbole_models', type=str, nargs='+', 
                        default=['BPR', 'SASRec', 'FPMC', 'Pop', 'ItemKNN'],
                        help='RecBole models to evaluate')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--k', type=int, default=10, help='Rank cutoff for NDCG@k')
    parser.add_argument('--max_users', type=int, default=1000, help='Maximum number of users to analyze')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    analyze_best_model_overlap(args)

if __name__ == '__main__':
    main()
