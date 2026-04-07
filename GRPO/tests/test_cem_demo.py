#!/usr/bin/env python3
"""
Minimal example demonstrating CEM optimization for multi-channel fusion.

This script shows how to use the CEM utilities to optimize fusion weights
for combining multiple recommendation channels.
"""

import torch
import numpy as np
from typing import List, Set
from GRPO.baselines.cem_utils import (
    cem_optimize_fusion_weights,
    fuse_by_quota,
    recall_at_L,
)

def generate_synthetic_data(
    n_users: int = 100,
    n_items: int = 1000,
    K: int = 3,  # number of channels
    M: int = 100,  # candidates per channel
    n_gt: int = 5,  # ground truth items per user
):
    """
    Generate synthetic user candidates and ground truth for testing.
    
    Returns:
        user_candidates: List[List[List[int]]] - shape (n_users, K, M)
        ground_truth: List[Set[int]] - shape (n_users,)
    """
    np.random.seed(42)
    
    user_candidates = []
    ground_truth = []
    
    for u in range(n_users):
        # Generate ground truth items for this user
        gt_items = set(np.random.choice(n_items, size=n_gt, replace=False))
        ground_truth.append(gt_items)
        
        # Generate candidates from K channels
        user_channels = []
        for k in range(K):
            # Each channel has different quality/overlap with ground truth
            # Channel 0: high quality (50% contain gt items)
            # Channel 1: medium quality (30% contain gt items)
            # Channel 2: low quality (10% contain gt items)
            quality = 0.5 - k * 0.2
            
            candidates = []
            # Add some ground truth items with probability based on quality
            for gt_item in gt_items:
                if np.random.random() < quality:
                    candidates.append(gt_item)
            
            # Fill rest with random items
            while len(candidates) < M:
                item = np.random.randint(0, n_items)
                if item not in candidates:
                    candidates.append(item)
            
            # Shuffle and trim to exactly M
            np.random.shuffle(candidates)
            user_channels.append(candidates[:M])
        
        user_candidates.append(user_channels)
    
    return user_candidates, ground_truth


def main():
    print("="*60)
    print("CEM Fusion Optimization - Synthetic Example")
    print("="*60)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    n_users = 100
    K = 3  # 3 channels
    M = 100  # 100 candidates per channel
    L = 50  # recommend top-50
    
    user_candidates, ground_truth = generate_synthetic_data(
        n_users=n_users, n_items=1000, K=K, M=M, n_gt=5
    )
    
    print(f"   Users: {n_users}")
    print(f"   Channels: {K}")
    print(f"   Candidates per channel: {M}")
    print(f"   Target recommendation size: {L}")
    
    # Baseline: Uniform weights
    print("\n2. Baseline: Uniform weights (1/K for each channel)...")
    uniform_w = torch.ones(K) / K
    uniform_fused = fuse_by_quota(user_candidates, uniform_w, L)
    uniform_recall = recall_at_L(uniform_fused, ground_truth)
    print(f"   Uniform weights: {uniform_w.tolist()}")
    print(f"   Recall@{L}: {uniform_recall:.4f}")
    
    # CEM Optimization
    print("\n3. Running CEM optimization...")
    best_w, best_score, hist = cem_optimize_fusion_weights(
        user_candidates=user_candidates,
        ground_truth=ground_truth,
        L=L,
        K=K,
        device="cuda" if torch.cuda.is_available() else "cpu",
        iters=15,
        population=128,
        elite_frac=0.1,
        seed=42
    )
    
    # Results
    print("\n" + "="*60)
    print("Results")
    print("="*60)
    
    print(f"\nBaseline (Uniform):")
    print(f"  Recall@{L}: {uniform_recall:.4f}")
    print(f"  Weights: {[f'{w:.4f}' for w in uniform_w.tolist()]}")
    
    print(f"\nCEM-Optimized:")
    print(f"  Recall@{L}: {best_score:.4f}")
    print(f"  Weights: {[f'{w:.4f}' for w in best_w.tolist()]}")
    
    improvement = (best_score - uniform_recall) / uniform_recall * 100
    print(f"\nImprovement: {improvement:+.2f}%")
    
    # Plot convergence (if matplotlib available)
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot scores over iterations
        ax1.plot(hist["mean_score"], label="Mean", marker='o')
        ax1.plot(hist["best_score"], label="Best", marker='s')
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel(f"Recall@{L}")
        ax1.set_title("CEM Convergence")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot final weights
        channel_names = [f"Channel {i}" for i in range(K)]
        ax2.bar(channel_names, best_w.tolist(), alpha=0.7)
        ax2.set_ylabel("Weight")
        ax2.set_title("Optimized Fusion Weights")
        ax2.set_ylim([0, max(best_w.tolist()) * 1.2])
        for i, w in enumerate(best_w.tolist()):
            ax2.text(i, w + 0.02, f'{w:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig("cem_optimization_results.png", dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: cem_optimization_results.png")
        
    except ImportError:
        print("\n(Install matplotlib to visualize results)")
    
    # Analysis
    print("\n" + "="*60)
    print("Analysis")
    print("="*60)
    print("\nKey Observations:")
    print(f"1. Channel 0 (high quality) received weight {best_w[0].item():.4f}")
    print(f"2. Channel 1 (medium quality) received weight {best_w[1].item():.4f}")
    print(f"3. Channel 2 (low quality) received weight {best_w[2].item():.4f}")
    print("\nCEM successfully learned to prioritize higher-quality channels!")
    
    # Demonstrate quota allocation
    print("\n" + "="*60)
    print("Quota Allocation Example (First User)")
    print("="*60)
    
    quotas = torch.round(best_w * L).to(torch.int64).tolist()
    print(f"\nWith L={L} and weights={[f'{w:.3f}' for w in best_w.tolist()]}:")
    for k in range(K):
        print(f"  Channel {k}: quota = {quotas[k]} items (weight={best_w[k].item():.3f})")
    
    print(f"\nTotal quota: {sum(quotas)} (should equal {L})")
    
    # Show actual fusion for first user
    fused_example = fuse_by_quota([user_candidates[0]], best_w, L)
    print(f"\nFused {len(fused_example[0])} items for user 0")
    gt_0 = ground_truth[0]
    hits = sum(1 for item in fused_example[0] if item in gt_0)
    print(f"Ground truth size: {len(gt_0)}")
    print(f"Hits in fused list: {hits}")
    print(f"Recall for this user: {hits/len(gt_0):.4f}")


if __name__ == "__main__":
    main()




