#!/usr/bin/env python3
"""
CEM-based fusion optimization baseline.

This script uses Cross-Entropy Method (CEM) to optimize fusion weights
for combining multiple recommendation channels without any LLM.
"""

import argparse
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from datetime import datetime
import numpy as np
from datasets import Dataset
from typing import List, Dict

from GRPO.core.data import load_dataset
from GRPO.models.main import initialize_recallers
from GRPO.core.recallers import RecBoleRecaller
from GRPO.core.utils import set_seed, ndcg_at_k, recall_at_k
from GRPO.baselines.cem_utils import (
    cem_optimize_fusion_weights,
    build_user_candidates_from_recalls,
    fuse_by_quota,
    recall_at_L,
)


def optimize_cem_weights(
    train_dataset,
    recallers: Dict[str, RecBoleRecaller],
    recaller_names: List[str],
    final_k: int = 50,
    cem_iters: int = 20,
    cem_population: int = 256,
    cem_elite_frac: float = 0.1,
    device: str = "cuda",
    min_thres: float = 0.001,
    train_k: int = 20,
):
    """
    Optimize fusion weights using CEM on training/validation set.
    
    Args:
        train_dataset: Training/validation dataset with user_id, history, and ground truth
        recallers: Dictionary of recaller name -> RecBoleRecaller
        recaller_names: Ordered list of recaller names  
        final_k: Number of items to recommend (L in the paper)
        cem_iters: Number of CEM iterations
        cem_population: Population size for CEM
        cem_elite_frac: Elite fraction for CEM
        device: Device to run CEM on
        
    Returns:
        Optimized weights tensor (K,) and optimization history
    """
    print("\n" + "="*60)
    print("CEM Weight Optimization (Training Phase)")
    print("="*60)
    
    K = len(recaller_names)
    M = final_k * 3  # Get more candidates per channel for better fusion
    
    # Extract user data from training dataset
    user_ids = []
    eval_hists = []
    gt_items_list = []
    full_hists = []
    
    for example in train_dataset:
        user_ids.append(example["user_id"])
        
        # Get evaluation history
        if "history" in example:
            hist = example["history"]
            if isinstance(hist, list) and 0 in hist:
                hist = hist[:hist.index(0)]
            eval_hists.append(hist)
        else:
            eval_hists.append([])
        
        # Get ground truth items
        if "target_items" in example:
            gt = example["target_items"]
            if isinstance(gt, int):
                gt = [gt]
            gt_items_list.append(set(gt))
        else:
            gt_items_list.append(set())
        
        # Get full history if available
        if "full_hist" in example:
            full_hists.append(example["full_hist"])
        else:
            full_hists.append(None)
    
    print(f"Processing {len(user_ids)} users with {K} recallers")
    
    # Build user_candidates: List[List[List[int]]]
    def recaller_wrapper(recaller_name):
        """Wrapper to match the signature expected by build_user_candidates_from_recalls"""
        def recall_fn(uids: List[int], topk: int) -> List[List[int]]:
            results = []
            for i, uid in enumerate(uids):
                hist = eval_hists[i]
                full_hist = full_hists[i] if full_hists[i] else None
                gt = list(gt_items_list[i]) if gt_items_list[i] else None
                
                items = recallers[recaller_name.lower()].recall(
                    uid, topk, hist, 
                    full_hist=full_hist, 
                    gt_items=gt
                )
                item_ids = [item[0] for item in items]
                results.append(item_ids)
            return results
        return recall_fn
    
    # Create recaller wrappers
    recaller_fns = [recaller_wrapper(name) for name in recaller_names]
    
    print(f"Building candidate lists (M={M} per channel)...")
    user_candidates = build_user_candidates_from_recalls(
        user_ids=user_ids,
        recallers=recaller_fns,
        M=M
    )
    
    if min_thres > 0:
        valid_indices = []
        for u in range(len(user_ids)):
            best_ndcg = max(
                ndcg_at_k(user_candidates[u][k][:train_k], list(gt_items_list[u]), train_k)
                for k in range(K)
            )
            if best_ndcg >= min_thres:
                valid_indices.append(u)
        n_before = len(user_ids)
        user_ids = [user_ids[i] for i in valid_indices]
        eval_hists = [eval_hists[i] for i in valid_indices]
        gt_items_list = [gt_items_list[i] for i in valid_indices]
        full_hists = [full_hists[i] for i in valid_indices]
        user_candidates = [user_candidates[i] for i in valid_indices]
        print(f"Filtered to {len(user_ids)}/{n_before} users (min_thres={min_thres}, train_k={train_k})")
    
    # Evaluate individual recallers on training set
    print("\n" + "="*60)
    print("Individual Recaller Performance (Training Set)")
    print("="*60)
    for k, name in enumerate(recaller_names):
        # Get top-k recommendations from this recaller
        recaller_lists = []
        for u in range(len(user_candidates)):
            recaller_lists.append(user_candidates[u][k][:final_k])
        
        # Compute metrics for this recaller
        ndcg_scores_k = []
        recall_scores_k = []
        for rec_list, gt in zip(recaller_lists, gt_items_list):
            if len(gt) > 0:
                ndcg_scores_k.append(ndcg_at_k(rec_list, list(gt), final_k))
                recall_scores_k.append(recall_at_k(rec_list, list(gt), final_k))
        
        avg_ndcg_k = float(np.mean(ndcg_scores_k)) if ndcg_scores_k else 0.0
        avg_recall_k = float(np.mean(recall_scores_k)) if recall_scores_k else 0.0
        
        print(f"{name}: NDCG@{final_k}={avg_ndcg_k:.4f}, Recall@{final_k}={avg_recall_k:.4f}")
    
    print(f"\nRunning CEM optimization (iters={cem_iters}, pop={cem_population})...")
    best_w, best_score, hist = cem_optimize_fusion_weights(
        user_candidates=user_candidates,
        ground_truth=gt_items_list,
        L=final_k,
        K=K,
        device=device,
        iters=cem_iters,
        population=cem_population,
        elite_frac=cem_elite_frac,
        seed=42
    )
    
    print("\n" + "="*60)
    print("CEM Optimization Results (Training)")
    print("="*60)
    print(f"Best Recall@{final_k}: {best_score:.4f}")
    print(f"\nOptimized Weights:")
    for i, name in enumerate(recaller_names):
        print(f"  {name}: {best_w[i].item():.4f}")
    
    return best_w, hist


def evaluate_cem_fusion(
    test_dataset,
    recallers: Dict[str, RecBoleRecaller],
    recaller_names: List[str],
    weights: torch.Tensor,
    final_k: int = 50,
    device: str = "cuda",
    min_thres: float = 0.001,
    train_k: int = 20,
):
    """
    Evaluate CEM-optimized fusion weights on test set.
    
    Args:
        test_dataset: Test dataset with user_id, history, and ground truth
        recallers: Dictionary of recaller name -> RecBoleRecaller
        recaller_names: Ordered list of recaller names  
        weights: Optimized weights tensor (K,) from training phase
        final_k: Number of items to recommend (L in the paper)
        device: Device to run on
        
    Returns:
        Dictionary with evaluation results
    """
    print("\n" + "="*60)
    print("CEM-Based Fusion Evaluation (Test Phase)")
    print("="*60)
    
    K = len(recaller_names)
    M = final_k * 3  # Get more candidates per channel for better fusion
    
    # Extract user data from test dataset
    user_ids = []
    eval_hists = []
    gt_items_list = []
    full_hists = []
    
    for example in test_dataset:
        # Get evaluation history
        if "history" in example:
            hist = example["history"]
            if isinstance(hist, list) and 0 in hist:
                hist = hist[:hist.index(0)]
        else:
            hist = []
        
        if len(hist) < 5:
            continue
        
        user_ids.append(example["user_id"])
        eval_hists.append(hist)
        
        # Get ground truth items
        if "target_items" in example:
            gt = example["target_items"]
            if isinstance(gt, int):
                gt = [gt]
            gt_items_list.append(set(gt))
        else:
            gt_items_list.append(set())
        
        # Get full history if available
        if "full_hist" in example:
            full_hists.append(example["full_hist"])
        else:
            full_hists.append(None)
    
    print(f"Processing {len(user_ids)} users with {K} recallers (filtered history < 5)")
    
    # Build user_candidates: List[List[List[int]]]
    # user_candidates[u][k] = list of M items for user u from recaller k
    def recaller_wrapper(recaller_name):
        """Wrapper to match the signature expected by build_user_candidates_from_recalls"""
        def recall_fn(uids: List[int], topk: int) -> List[List[int]]:
            results = []
            for i, uid in enumerate(uids):
                hist = eval_hists[i]
                full_hist = full_hists[i] if full_hists[i] else None
                gt = list(gt_items_list[i]) if gt_items_list[i] else None
                
                items = recallers[recaller_name.lower()].recall(
                    uid, topk, hist, 
                    full_hist=full_hist, 
                    gt_items=gt
                )
                item_ids = [item[0] for item in items]
                results.append(item_ids)
            return results
        return recall_fn
    
    # Create recaller wrappers
    recaller_fns = [recaller_wrapper(name) for name in recaller_names]
    
    print(f"Building candidate lists (M={M} per channel)...")
    user_candidates = build_user_candidates_from_recalls(
        user_ids=user_ids,
        recallers=recaller_fns,
        M=M
    )
    
    if min_thres > 0:
        valid_indices = []
        for u in range(len(user_ids)):
            best_ndcg = max(
                ndcg_at_k(user_candidates[u][k][:train_k], list(gt_items_list[u]), train_k)
                for k in range(K)
            )
            if best_ndcg >= min_thres:
                valid_indices.append(u)
        n_before = len(user_ids)
        user_ids = [user_ids[i] for i in valid_indices]
        eval_hists = [eval_hists[i] for i in valid_indices]
        gt_items_list = [gt_items_list[i] for i in valid_indices]
        full_hists = [full_hists[i] for i in valid_indices]
        user_candidates = [user_candidates[i] for i in valid_indices]
        print(f"Filtered to {len(user_ids)}/{n_before} users (min_thres={min_thres}, train_k={train_k})")
    
    print(f"Using optimized weights from training phase...")
    print(f"Weights:")
    for i, name in enumerate(recaller_names):
        print(f"  {name}: {weights[i].item():.4f}")
    
    # Evaluate individual recallers
    print("\n" + "="*60)
    print("Individual Recaller Performance")
    print("="*60)
    individual_results = {}
    
    for k, name in enumerate(recaller_names):
        # Get top-k recommendations from this recaller
        recaller_lists = []
        for u in range(len(user_candidates)):
            recaller_lists.append(user_candidates[u][k][:final_k])
        
        # Compute metrics for this recaller
        ndcg_scores_k = []
        recall_scores_k = []
        for rec_list, gt in zip(recaller_lists, gt_items_list):
            if len(gt) > 0:
                ndcg_scores_k.append(ndcg_at_k(rec_list, list(gt), final_k))
                recall_scores_k.append(recall_at_k(rec_list, list(gt), final_k))
        
        avg_ndcg_k = float(np.mean(ndcg_scores_k)) if ndcg_scores_k else 0.0
        avg_recall_k = float(np.mean(recall_scores_k)) if recall_scores_k else 0.0
        
        # Evaluate at different k values
        metrics_k = {
            f"ndcg@{final_k}": avg_ndcg_k,
            f"recall@{final_k}": avg_recall_k,
        }
        
        for eval_k in [10, 20, 50]:
            recaller_lists_k = []
            for u in range(len(user_candidates)):
                recaller_lists_k.append(user_candidates[u][k][:eval_k])
            
            ndcg_k = []
            recall_k = []
            for rec_list, gt in zip(recaller_lists_k, gt_items_list):
                if len(gt) > 0:
                    ndcg_k.append(ndcg_at_k(rec_list, list(gt), eval_k))
                    recall_k.append(recall_at_k(rec_list, list(gt), eval_k))
            
            metrics_k[f"ndcg@{eval_k}"] = float(np.mean(ndcg_k)) if ndcg_k else 0.0
            metrics_k[f"recall@{eval_k}"] = float(np.mean(recall_k)) if recall_k else 0.0
        
        individual_results[name] = metrics_k
        
        print(f"\n{name}:")
        print(f"  NDCG@{final_k}: {avg_ndcg_k:.4f}, Recall@{final_k}: {avg_recall_k:.4f}")
        for eval_k in [10, 20, 50]:
            print(f"  NDCG@{eval_k}: {metrics_k[f'ndcg@{eval_k}']:.4f}, "
                  f"Recall@{eval_k}: {metrics_k[f'recall@{eval_k}']:.4f}")
    
    # Compute NDCG with optimized weights
    fused_lists = fuse_by_quota(user_candidates, weights, final_k)
    
    ndcg_scores = []
    recall_scores = []
    for fused, gt in zip(fused_lists, gt_items_list):
        if len(gt) > 0:
            ndcg = ndcg_at_k(fused, list(gt), final_k)
            recall = recall_at_k(fused, list(gt), final_k)
            ndcg_scores.append(ndcg)
            recall_scores.append(recall)
    
    avg_ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
    avg_recall = float(np.mean(recall_scores)) if recall_scores else 0.0
    
    print(f"\nOptimized Fusion Performance:")
    print(f"  NDCG@{final_k}: {avg_ndcg:.4f}")
    print(f"  Recall@{final_k}: {avg_recall:.4f}")
    
    # Also evaluate at k=10, 20, 50
    results = {
        "cem_optimized": {
            f"recall@{final_k}": avg_recall,
            f"ndcg@{final_k}": avg_ndcg,
        },
        "optimized_weights": {name: weights[i].item() for i, name in enumerate(recaller_names)},
        "individual_recallers": individual_results,
    }
    
    # Evaluate at different k values
    for k in [10, 20, 50]:
        fused_k = fuse_by_quota(user_candidates, weights, k)
        ndcg_k = []
        recall_k = []
        for fused, gt in zip(fused_k, gt_items_list):
            if len(gt) > 0:
                ndcg_k.append(ndcg_at_k(fused, list(gt), k))
                recall_k.append(recall_at_k(fused, list(gt), k))
        
        results["cem_optimized"][f"ndcg@{k}"] = float(np.mean(ndcg_k)) if ndcg_k else 0.0
        results["cem_optimized"][f"recall@{k}"] = float(np.mean(recall_k)) if recall_k else 0.0
    
    print(f"\nCEM-Optimized Performance at Different k:")
    for k in [10, 20, 50]:
        print(f"  k={k}: NDCG={results['cem_optimized'][f'ndcg@{k}']:.4f}, "
              f"Recall={results['cem_optimized'][f'recall@{k}']:.4f}")
    
    return results


def create_dataset_from_inter_dataset(inter_dataset, split='test', num_users=None):
    """
    Create a dataset from inter_dataset for CEM optimization/evaluation.
    Merges multiple examples for the same user:
    - history: selects the shortest one
    - target_items: union of all target_items
    
    Args:
        inter_dataset: Dataset loaded from load_dataset
        split: 'train', 'eval', or 'test'
        num_users: Number of users to use (None for all)
        
    Returns:
        List of dicts with user_id, history, target_items (one per user)
    """
    if split == 'train':
        user_ids = inter_dataset.train_user_ids
        histories = inter_dataset.train_histories
        target_items = inter_dataset.train_target_items
    elif split == 'eval':
        user_ids = inter_dataset.eval_user_ids
        histories = inter_dataset.eval_histories
        target_items = inter_dataset.eval_target_items
    else:  # test
        user_ids = inter_dataset.test_user_ids
        histories = inter_dataset.test_histories
        target_items = inter_dataset.test_target_items
    
    # Group examples by user_id
    user_examples = {}
    for uid, hist, target in zip(user_ids, histories, target_items):
        # Normalize target to list
        if isinstance(target, int):
            target_list = [target]
        else:
            target_list = target if isinstance(target, list) else [target]
        
        if uid not in user_examples:
            user_examples[uid] = {
                "histories": [],
                "target_items": set()
            }
        
        # Store history
        user_examples[uid]["histories"].append(hist)
        # Add target items to set (for union)
        user_examples[uid]["target_items"].update(target_list)
    
    # Merge examples for each user
    dataset = []
    for uid, examples in user_examples.items():
        # Select shortest history
        histories_list = examples["histories"]
        shortest_hist = min(histories_list, key=lambda h: len(h) if isinstance(h, list) else 0)
        
        # Get union of target_items
        merged_target_items = sorted(list(examples["target_items"]))
        
        dataset.append({
            "user_id": uid,
            "history": shortest_hist,
            "target_items": merged_target_items,
            "full_hist": shortest_hist  # Use same history as full_hist for simplicity
        })
    
    # Sort by user_id for consistency
    dataset.sort(key=lambda x: x["user_id"])
    
    # Apply num_users cutoff after merging (to ensure we get exactly num_users unique users)
    if num_users is not None:
        dataset = dataset[:num_users]
    
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(description='CEM-based Fusion Optimization Baseline')
    # Data
    parser.add_argument('--dataset', type=str, default='Amazon_All_Beauty')
    parser.add_argument('--data_path', type=str, default='./dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--output_dir', type=str, default='results')
    # Model
    parser.add_argument('--recbole_models', type=str, nargs='+', default=['BPR', 'SASRec', 'LightGCN'])
    # CEM parameters
    parser.add_argument('--final_k', type=int, default=50, help='Number of items to recommend')
    parser.add_argument('--cem_iters', type=int, default=20, help='Number of CEM iterations')
    parser.add_argument('--cem_population', type=int, default=256, help='CEM population size')
    parser.add_argument('--cem_elite_frac', type=float, default=0.1, help='CEM elite fraction')
    parser.add_argument('--use_eval_for_training', action='store_true', 
                       help='Use eval set for weight optimization (default: use train set)')
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_users', type=int, default=None, help='Number of training users (None for all)')
    parser.add_argument('--num_test_users', type=int, default=None, help='Number of test users (None for all)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run CEM on')
    parser.add_argument('--min_thres', type=float, default=0.001,
                       help='Minimum best-recaller NDCG threshold for user filtering (matching main_pure.py).')
    parser.add_argument('--train_k', type=int, default=20,
                       help='Top-k for computing NDCG during filtering (matching main_pure.py train_k).')
    parser.add_argument('--test_dataset_path', type=str, default=None,
                       help='Path to main_pure pre-generated test dataset (HuggingFace Dataset). '
                            'When provided, uses the exact same filtered users/history/gt as main_pure.')
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    print("="*60)
    print("CEM-Based Fusion Optimization Baseline")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Recallers: {args.recbole_models}")
    print(f"Final K: {args.final_k}")
    print(f"CEM Iterations: {args.cem_iters}")
    print(f"CEM Population: {args.cem_population}")
    print("="*60)
    
    # Load dataset
    print("\n1. Loading dataset...")
    inter_dataset = load_dataset(args.dataset, args.data_path, seed=args.seed)
    print(f"   Train users: {len(inter_dataset.train_user_ids)}")
    print(f"   Eval users: {len(inter_dataset.eval_user_ids)}")
    print(f"   Test users: {len(inter_dataset.test_user_ids)}")
    
    # Initialize recallers
    print("\n2. Initializing recallers...")
    recallers = initialize_recallers(
        model_names=args.recbole_models,
        dataset_name=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        data_path=args.data_path,
        seed=args.seed,
        use_latest_checkpoint=True,
        num_items=inter_dataset.ds.item_num
    )
    print(f"   Initialized {len(recallers)} recallers: {list(recallers.keys())}")
    
    # Create training dataset for weight optimization
    print("\n3. Creating training dataset for weight optimization...")
    train_split = 'eval' if args.use_eval_for_training else 'train'
    train_dataset = create_dataset_from_inter_dataset(
        inter_dataset, 
        split=train_split,
        num_users=args.num_train_users
    )
    eval_dataset = create_dataset_from_inter_dataset(
        inter_dataset, 
        split='eval',
        num_users=args.num_train_users
    )
    print(f"   Created {train_split} dataset with {len(train_dataset)} users")
    
    # Optimize weights on training set
    print("\n4. Optimizing weights on training set...")
    recaller_names = sorted(args.recbole_models)
    optimized_weights, opt_hist = optimize_cem_weights(
        train_dataset=eval_dataset,
        recallers=recallers,
        recaller_names=recaller_names,
        final_k=args.final_k,
        cem_iters=args.cem_iters,
        cem_population=args.cem_population,
        cem_elite_frac=args.cem_elite_frac,
        device=args.device if torch.cuda.is_available() else "cpu",
        min_thres=args.min_thres,
        train_k=args.train_k,
    )
    
    # Create test dataset
    print("\n5. Creating test dataset...")
    if args.test_dataset_path:
        from datasets import Dataset as HFDataset
        hf_test = HFDataset.load_from_disk(args.test_dataset_path)
        test_dataset = []
        for ex in hf_test:
            hist = ex["history"]
            if isinstance(hist, list) and 0 in hist:
                hist = hist[:hist.index(0)]
            gt = ex["target_items"]
            if isinstance(gt, int):
                gt = [gt]
            test_dataset.append({
                "user_id": ex["user_id"],
                "history": hist,
                "target_items": gt,
                "full_hist": ex.get("full_hist", hist),
            })
        print(f"   Loaded pre-generated test dataset from {args.test_dataset_path}: {len(test_dataset)} users")
    else:
        test_dataset = create_dataset_from_inter_dataset(
            inter_dataset, 
            split='test',
            num_users=args.num_test_users
        )
        print(f"   Created test dataset with {len(test_dataset)} users")
    
    # Evaluate on test set with optimized weights
    print("\n6. Evaluating on test set with optimized weights...")
    cem_results = evaluate_cem_fusion(
        test_dataset=test_dataset,
        recallers=recallers,
        recaller_names=recaller_names,
        weights=optimized_weights,
        final_k=args.final_k,
        device=args.device if torch.cuda.is_available() else "cpu",
        min_thres=args.min_thres,
        train_k=args.train_k,
    )
    
    # Save results
    print("\n7. Saving results...")
    os.makedirs(args.output_dir, exist_ok=True)
    recaller_combo = "_".join(sorted(args.recbole_models))
    result_filename = f"{args.output_dir}/cem_results_{args.dataset}_{recaller_combo}.json"
    
    results = {
        "cem_fusion": cem_results,
        "training_history": {
            "best_score": [float(s) for s in opt_hist["best_score"]],
            "mean_score": [float(s) for s in opt_hist["mean_score"]],
        },
        "config": {
            "dataset": args.dataset,
            "recbole_models": args.recbole_models,
            "recaller_combo": recaller_combo,
            "final_k": args.final_k,
            "cem_iters": args.cem_iters,
            "cem_population": args.cem_population,
            "cem_elite_frac": args.cem_elite_frac,
            "train_split": train_split,
            "train_samples": len(train_dataset),
            "test_samples": len(test_dataset),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    }
    
    with open(result_filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"   Results saved to: {result_filename}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Recaller Combo: {recaller_combo}")
    print(f"Training Split: {train_split}")
    print(f"Train Users: {len(train_dataset)}")
    print(f"Test Users: {len(test_dataset)}")
    
    cem_ndcg = cem_results['cem_optimized'].get(f'ndcg@{args.final_k}', 0.0)
    cem_recall = cem_results['cem_optimized'].get(f'recall@{args.final_k}', 0.0)
    print(f"\nCEM-Optimized Fusion Performance:")
    print(f"  NDCG@{args.final_k}: {cem_ndcg:.4f}")
    print(f"  Recall@{args.final_k}: {cem_recall:.4f}")
    print(f"\nOptimized Weights:")
    for name, weight in cem_results['optimized_weights'].items():
        print(f"  {name}: {weight:.4f}")
    
    # Print individual recaller comparison
    if 'individual_recallers' in cem_results:
        print(f"\n" + "="*60)
        print("Individual Recaller Comparison")
        print("="*60)
        for name, metrics in cem_results['individual_recallers'].items():
            print(f"{name}: NDCG@{args.final_k}={metrics[f'ndcg@{args.final_k}']:.4f}, "
                  f"Recall@{args.final_k}={metrics[f'recall@{args.final_k}']:.4f}")
    
    print("="*60)


if __name__ == "__main__":
    main()
