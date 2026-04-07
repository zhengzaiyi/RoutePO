#!/usr/bin/env python3
"""
Policy Gradient-based personalized fusion optimization baseline.

This script uses Policy Gradient (REINFORCE) to optimize personalized fusion weights
for combining multiple recommendation channels.

Based on: Huang et al. 2024 - "Unleashing the Potential of Multi-Channel Fusion 
in Retrieval for Personalized Recommendations"

Key differences from CEM baseline:
- CEM: globally unified weights (same for all users)
- PG: personalized weights (different for each user based on their profile)
"""

import argparse
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from datetime import datetime
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from GRPO.core.data import load_dataset
from GRPO.models.main import initialize_recallers
from GRPO.core.recallers import RecBoleRecaller
from GRPO.core.utils import set_seed, ndcg_at_k, recall_at_k
from GRPO.baselines.cem_utils import (
    build_user_candidates_from_recalls,
    fuse_by_quota,
    recall_at_L,
)


def extract_pretrained_embeddings(
    recaller: RecBoleRecaller,
    num_users: int,
    num_items: int,
    embedding_dim: int,
    device: str = "cuda"
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Extract pre-trained user and item embeddings from RecBoleRecaller.
    
    Args:
        recaller: RecBoleRecaller instance with loaded model
        num_users: Number of users
        num_items: Number of items
        embedding_dim: Embedding dimension
        device: Device to place embeddings on
        
    Returns:
        Tuple of (user_embeddings, item_embeddings) tensors or (None, None) if not available
    """
    model = recaller.model
    user_emb = None
    item_emb = None
    
    # Try to extract embeddings from different RecBole model types
    # Most RecBole models have user_embedding and item_embedding attributes
    if hasattr(model, 'user_emb'):
        try:
            user_emb = model.user_emb.weight.data.clone()  # (num_users+1, embedding_dim)
            # Ensure it matches expected size
            if user_emb.size(0) < num_users + 1:
                # Pad with zeros if needed
                padding = torch.zeros(num_users + 1 - user_emb.size(0), embedding_dim, device=device)
                user_emb = torch.cat([user_emb, padding], dim=0)
            elif user_emb.size(0) > num_users + 1:
                # Truncate if needed
                user_emb = user_emb[:num_users + 1]
            user_emb = user_emb.to(device)
            print(f"  Extracted user embeddings: {user_emb.shape}")
        except Exception as e:
            print(f"  Warning: Could not extract user embeddings: {e}")
    
    if hasattr(model, 'item_emb'):
        try:
            item_emb = model.item_emb.weight.data.clone()  # (num_items+1, embedding_dim)
            # Ensure it matches expected size
            if item_emb.size(0) < num_items + 1:
                # Pad with zeros if needed
                padding = torch.zeros(num_items + 1 - item_emb.size(0), embedding_dim, device=device)
                item_emb = torch.cat([item_emb, padding], dim=0)
            elif item_emb.size(0) > num_items + 1:
                # Truncate if needed
                item_emb = item_emb[:num_items + 1]
            item_emb = item_emb.to(device)
            print(f"  Extracted item embeddings: {item_emb.shape}")
        except Exception as e:
            print(f"  Warning: Could not extract item embeddings: {e}")
    
    # For some models, embeddings might be in different attributes
    # Try alternative names
    if user_emb is None and hasattr(model, 'user_emb'):
        try:
            user_emb = model.user_emb.weight.data.clone()
            if user_emb.size(0) < num_users + 1:
                padding = torch.zeros(num_users + 1 - user_emb.size(0), embedding_dim, device=device)
                user_emb = torch.cat([user_emb, padding], dim=0)
            elif user_emb.size(0) > num_users + 1:
                user_emb = user_emb[:num_users + 1]
            user_emb = user_emb.to(device)
            print(f"  Extracted user embeddings (alt): {user_emb.shape}")
        except Exception:
            pass
    
    if item_emb is None and hasattr(model, 'item_emb'):
        try:
            item_emb = model.item_emb.weight.data.clone()
            if item_emb.size(0) < num_items + 1:
                padding = torch.zeros(num_items + 1 - item_emb.size(0), embedding_dim, device=device)
                item_emb = torch.cat([item_emb, padding], dim=0)
            elif item_emb.size(0) > num_items + 1:
                item_emb = item_emb[:num_items + 1]
            item_emb = item_emb.to(device)
            print(f"  Extracted item embeddings (alt): {item_emb.shape}")
        except Exception:
            pass
    
    return user_emb, item_emb


class ChannelRepresentationModule(nn.Module):
    """
    Module to compute channel representations from retrieved items.
    
    As per paper Section C.3:
    - "The top 10 items retrieved by each channel are pooled to represent the channel"
    """
    def __init__(
        self, 
        num_items: int, 
        embedding_dim: int, 
        top_k_items: int = 10,
        pretrained_item_emb: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.top_k_items = top_k_items
        
        # Item embeddings (initialized from pre-trained model if provided)
        self.item_embeddings = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        if pretrained_item_emb is not None:
            with torch.no_grad():
                self.item_embeddings.weight.data.copy_(pretrained_item_emb)
                print(f"  Initialized channel item embeddings from pre-trained model")
        
    def forward(self, channel_items: torch.Tensor) -> torch.Tensor:
        """
        Compute channel representations by pooling item embeddings.
        
        Args:
            channel_items: (batch_size, num_channels, top_k_items) - item IDs per channel
            
        Returns:
            channel_repr: (batch_size, num_channels, embedding_dim)
        """
        batch_size, num_channels, top_k = channel_items.shape
        
        # Get item embeddings: (batch_size, num_channels, top_k, embedding_dim)
        item_embeds = self.item_embeddings(channel_items)
        
        # Mean pooling over items in each channel: (batch_size, num_channels, embedding_dim)
        channel_repr = item_embeds.mean(dim=2)
        
        return channel_repr


class UserRepresentationModule(nn.Module):
    """
    Module to compute user representations from their history.
    
    In the paper, pre-trained representations from SimpleX are used.
    Can be initialized with pre-trained embeddings from RecBoleRecaller.
    """
    def __init__(
        self, 
        num_users: int, 
        num_items: int, 
        embedding_dim: int, 
        max_hist_len: int = 50,
        pretrained_user_emb: Optional[torch.Tensor] = None,
        pretrained_item_emb: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.max_hist_len = max_hist_len
        
        # User embeddings (initialized from pre-trained model if provided)
        self.user_embeddings = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        if pretrained_user_emb is not None:
            with torch.no_grad():
                self.user_embeddings.weight.data.copy_(pretrained_user_emb)
                print(f"  Initialized user embeddings from pre-trained model")
        
        # Item embeddings for history encoding (initialized from pre-trained model if provided)
        self.item_embeddings = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        if pretrained_item_emb is not None:
            with torch.no_grad():
                self.item_embeddings.weight.data.copy_(pretrained_item_emb)
                print(f"  Initialized item embeddings from pre-trained model")
        
    def forward(self, user_ids: torch.Tensor, history: torch.Tensor = None) -> torch.Tensor:
        """
        Compute user representations.
        
        Args:
            user_ids: (batch_size,) - user IDs
            history: (batch_size, max_hist_len) - user history item IDs (optional)
            
        Returns:
            user_repr: (batch_size, embedding_dim)
        """
        # Get user embeddings
        user_embed = self.user_embeddings(user_ids)  # (batch_size, embedding_dim)
        
        if history is not None:
            # Encode history by mean pooling
            hist_embed = self.item_embeddings(history)  # (batch_size, max_hist_len, embedding_dim)
            # Mask out padding (0s)
            mask = (history > 0).float().unsqueeze(-1)  # (batch_size, max_hist_len, 1)
            hist_embed = (hist_embed * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            
            # Combine user embedding and history
            user_repr = user_embed + hist_embed
        else:
            user_repr = user_embed
            
        return user_repr


class PolicyNetwork(nn.Module):
    """
    Policy network for personalized weight assignment.
    
    Takes user representation and channel representations as input,
    outputs Dirichlet distribution parameters for weight sampling.
    
    Based on paper Equations 17-18:
    - Uses softplus activation to ensure positive Dirichlet parameters
    - Applies clipping with delta_max and epsilon for stability
    """
    def __init__(
        self, 
        num_channels: int, 
        embedding_dim: int,
        hidden_dim: int = 128,
        delta_max: float = 10.0,
        epsilon: float = 1e-6
    ):
        super().__init__()
        self.num_channels = num_channels
        self.embedding_dim = embedding_dim
        self.delta_max = delta_max
        self.epsilon = epsilon
        
        # Input: user repr + channel repr (concatenated)
        input_dim = embedding_dim + num_channels * embedding_dim
        
        # MLP to output Dirichlet parameters
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_channels)
        )
        
    def forward(
        self, 
        user_repr: torch.Tensor, 
        channel_repr: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dirichlet parameters for weight distribution.
        
        Args:
            user_repr: (batch_size, embedding_dim)
            channel_repr: (batch_size, num_channels, embedding_dim)
            
        Returns:
            alpha: (batch_size, num_channels) - Dirichlet parameters
        """
        batch_size = user_repr.size(0)
        
        # Flatten channel representations
        channel_flat = channel_repr.view(batch_size, -1)  # (batch_size, num_channels * embedding_dim)
        
        # Concatenate user and channel representations
        combined = torch.cat([user_repr, channel_flat], dim=1)  # (batch_size, input_dim)
        
        # Get raw Dirichlet parameters
        raw_alpha = self.network(combined)  # (batch_size, num_channels)
        
        # Apply softplus and clipping as per Equation 18
        # alpha = softplus(raw) clipped to [epsilon, delta_max]
        alpha = F.softplus(raw_alpha)
        alpha = torch.clamp(alpha, min=self.epsilon, max=self.delta_max)
        
        return alpha
    
    def sample_weights(
        self, 
        alpha: torch.Tensor, 
        num_samples: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample weights from Dirichlet distribution.
        
        Args:
            alpha: (batch_size, num_channels) - Dirichlet parameters
            num_samples: number of weight samples per user
            
        Returns:
            weights: (batch_size, num_samples, num_channels)
            log_probs: (batch_size, num_samples)
        """
        batch_size, num_channels = alpha.shape
        
        # Create Dirichlet distribution
        dist = Dirichlet(alpha)
        
        # Sample weights
        weights = dist.sample((num_samples,))  # (num_samples, batch_size, num_channels)
        weights = weights.permute(1, 0, 2)  # (batch_size, num_samples, num_channels)
        
        # Compute log probabilities
        log_probs = dist.log_prob(weights.permute(1, 0, 2))  # (num_samples, batch_size)
        log_probs = log_probs.permute(1, 0)  # (batch_size, num_samples)
        
        return weights, log_probs


class PersonalizedFusionPG:
    """
    Personalized fusion using Policy Gradient (REINFORCE).
    
    Based on paper Section 4.2 and Algorithm description.
    """
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_channels: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        learning_rate: float = 1e-4,
        reg_weight: float = 1.0,
        delta_max: float = 10.0,
        epsilon: float = 1e-6,
        num_samples: int = 1,
        top_k_channel_items: int = 10,
        device: str = "cuda",
        pretrained_user_emb: Optional[torch.Tensor] = None,
        pretrained_item_emb: Optional[torch.Tensor] = None
    ):
        self.num_users = num_users
        self.num_items = num_items
        self.num_channels = num_channels
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.reg_weight = reg_weight
        self.num_samples = num_samples
        self.top_k_channel_items = top_k_channel_items
        self.device = device
        
        # Initialize modules with pre-trained embeddings if provided
        self.user_module = UserRepresentationModule(
            num_users, num_items, embedding_dim,
            pretrained_user_emb=pretrained_user_emb,
            pretrained_item_emb=pretrained_item_emb
        ).to(device)
        
        self.channel_module = ChannelRepresentationModule(
            num_items, embedding_dim, top_k_channel_items,
            pretrained_item_emb=pretrained_item_emb
        ).to(device)
        
        self.policy = PolicyNetwork(
            num_channels, embedding_dim, hidden_dim, delta_max, epsilon
        ).to(device)
        
        # Optimizer
        params = list(self.user_module.parameters()) + \
                 list(self.channel_module.parameters()) + \
                 list(self.policy.parameters())
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)
        
    def get_channel_items_tensor(
        self, 
        user_candidates: List[List[List[int]]], 
        user_indices: List[int]
    ) -> torch.Tensor:
        """
        Extract top-k items per channel for the given users.
        
        Args:
            user_candidates: List[u][k] = list of items for user u from channel k
            user_indices: indices of users to process
            
        Returns:
            channel_items: (len(user_indices), num_channels, top_k_channel_items)
        """
        batch_size = len(user_indices)
        num_channels = len(user_candidates[0])
        
        channel_items = torch.zeros(
            batch_size, num_channels, self.top_k_channel_items, 
            dtype=torch.long, device=self.device
        )
        
        for i, u_idx in enumerate(user_indices):
            for k in range(num_channels):
                items = user_candidates[u_idx][k][:self.top_k_channel_items]
                for j, item in enumerate(items):
                    if j < self.top_k_channel_items:
                        channel_items[i, k, j] = item
                        
        return channel_items
    
    def compute_reward(
        self, 
        user_candidates: List[List[List[int]]],
        weights: torch.Tensor,
        ground_truth: List[set],
        user_indices: List[int],
        L: int
    ) -> torch.Tensor:
        """
        Compute Recall@L reward for each user with given weights.
        
        Args:
            user_candidates: candidate items per user per channel
            weights: (batch_size, num_channels) normalized weights
            ground_truth: ground truth items per user
            user_indices: indices of users in current batch
            L: number of items to recommend
            
        Returns:
            rewards: (batch_size,)
        """
        batch_size = len(user_indices)
        rewards = []
        
        for i, u_idx in enumerate(user_indices):
            # Get candidates for this user
            user_cand = [user_candidates[u_idx]]
            w = weights[i]
            
            # Fuse candidates using quota-based method
            fused = fuse_by_quota(user_cand, w, L)[0]
            
            # Compute recall
            gt = ground_truth[u_idx]
            if len(gt) > 0:
                hit = sum(1 for item in fused if item in gt)
                recall = hit / len(gt)
            else:
                recall = 0.0
            rewards.append(recall)
            
        return torch.tensor(rewards, device=self.device)
    
    def train_step(
        self,
        user_ids: List[int],
        histories: List[List[int]],
        user_candidates: List[List[List[int]]],
        ground_truth: List[set],
        L: int,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Perform one training step using REINFORCE.
        
        Args:
            user_ids: list of user IDs
            histories: list of user histories
            user_candidates: candidate items per user per channel
            ground_truth: ground truth items per user
            L: number of items to recommend
            batch_size: batch size for training
            
        Returns:
            dict with training metrics
        """
        self.user_module.train()
        self.channel_module.train()
        self.policy.train()
        
        num_users = len(user_ids)
        indices = list(range(num_users))
        np.random.shuffle(indices)
        
        total_loss = 0.0
        total_reward = 0.0
        num_batches = 0
        
        for start in range(0, num_users, batch_size):
            end = min(start + batch_size, num_users)
            batch_indices = indices[start:end]
            batch_size_actual = len(batch_indices)
            
            # Prepare batch data
            batch_user_ids = torch.tensor(
                [user_ids[i] for i in batch_indices], 
                dtype=torch.long, device=self.device
            )
            
            # Prepare history tensor
            max_hist_len = max(len(histories[i]) for i in batch_indices) if histories else 1
            max_hist_len = min(max_hist_len, 50)
            batch_history = torch.zeros(
                batch_size_actual, max_hist_len, 
                dtype=torch.long, device=self.device
            )
            for i, idx in enumerate(batch_indices):
                hist = histories[idx][:max_hist_len] if histories[idx] else []
                for j, item in enumerate(hist):
                    batch_history[i, j] = item
            
            # Get channel items tensor
            channel_items = self.get_channel_items_tensor(user_candidates, batch_indices)
            
            # Forward pass
            user_repr = self.user_module(batch_user_ids, batch_history)
            channel_repr = self.channel_module(channel_items)
            alpha = self.policy(user_repr, channel_repr)
            
            # Sample weights (S=1 as per paper)
            weights, log_probs = self.policy.sample_weights(alpha, num_samples=self.num_samples)
            
            # Compute rewards for each sample
            batch_rewards = []
            for s in range(self.num_samples):
                w = weights[:, s, :]  # (batch_size, num_channels)
                reward = self.compute_reward(
                    user_candidates, w, ground_truth, batch_indices, L
                )
                batch_rewards.append(reward)
            
            # Stack rewards: (batch_size, num_samples)
            rewards = torch.stack(batch_rewards, dim=1)
            
            # Compute baseline (batch-level mean reward for variance reduction)
            # When num_samples=1, use batch mean instead of per-sample mean
            if self.num_samples == 1:
                # Use batch mean as baseline (all samples share the same baseline)
                baseline = rewards.mean()  # scalar
            else:
                # Use per-sample mean across multiple samples
                baseline = rewards.mean(dim=1, keepdim=True)
            
            # Policy gradient loss: -E[log_prob * (reward - baseline)]
            advantage = rewards - baseline
            pg_loss = -(log_probs * advantage).mean()
            
            # Regularization: entropy bonus for exploration
            # Adding entropy encourages exploration and prevents collapse to deterministic policy
            # But reg_weight should be small enough to not dominate the PG loss
            entropy = Dirichlet(alpha).entropy().mean()
            # Entropy bonus: -reg_weight * (-entropy) = reg_weight * entropy added to loss
            # We want to MAXIMIZE entropy (minimize -entropy), so:
            # loss = pg_loss - reg_weight * entropy
            # This encourages higher entropy (more exploration)
            # Use a small reg_weight (e.g., 0.01-0.1) to not dominate PG loss
            reg_loss = -self.reg_weight * entropy
            
            # Total loss
            loss = pg_loss + reg_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_reward += rewards.mean().item()
            num_batches += 1
        
        return {
            "loss": total_loss / num_batches,
            "avg_reward": total_reward / num_batches,
        }
    
    def predict_weights(
        self,
        user_ids: List[int],
        histories: List[List[int]],
        user_candidates: List[List[List[int]]]
    ) -> torch.Tensor:
        """
        Predict personalized weights for users (using mean of Dirichlet).
        
        Args:
            user_ids: list of user IDs
            histories: list of user histories
            user_candidates: candidate items per user per channel
            
        Returns:
            weights: (num_users, num_channels)
        """
        self.user_module.eval()
        self.channel_module.eval()
        self.policy.eval()
        
        num_users = len(user_ids)
        all_weights = []
        
        with torch.no_grad():
            batch_size = 64
            for start in range(0, num_users, batch_size):
                end = min(start + batch_size, num_users)
                batch_indices = list(range(start, end))
                batch_size_actual = len(batch_indices)
                
                # Prepare batch data
                batch_user_ids = torch.tensor(
                    [user_ids[i] for i in batch_indices], 
                    dtype=torch.long, device=self.device
                )
                
                # Prepare history tensor
                max_hist_len = max(len(histories[i]) for i in batch_indices) if histories else 1
                max_hist_len = min(max_hist_len, 50)
                batch_history = torch.zeros(
                    batch_size_actual, max_hist_len, 
                    dtype=torch.long, device=self.device
                )
                for i, idx in enumerate(batch_indices):
                    hist = histories[idx][:max_hist_len] if histories[idx] else []
                    for j, item in enumerate(hist):
                        batch_history[i, j] = item
                
                # Get channel items tensor
                channel_items = self.get_channel_items_tensor(user_candidates, batch_indices)
                
                # Forward pass
                user_repr = self.user_module(batch_user_ids, batch_history)
                channel_repr = self.channel_module(channel_items)
                alpha = self.policy(user_repr, channel_repr)
                
                # Use mean of Dirichlet as weights: E[w] = alpha / sum(alpha)
                weights = alpha / alpha.sum(dim=1, keepdim=True)
                all_weights.append(weights)
        
        return torch.cat(all_weights, dim=0)


def train_pg_model(
    train_dataset,
    recallers: Dict[str, RecBoleRecaller],
    recaller_names: List[str],
    num_users: int,
    num_items: int,
    final_k: int = 50,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    reg_weight: float = 1.0,
    embedding_dim: int = 64,
    device: str = "cuda",
    min_thres: float = 0.001,
    train_k: int = 20,
):
    """
    Train Policy Gradient model for personalized fusion.
    
    Args:
        train_dataset: Training dataset with user_id, history, and ground truth
        recallers: Dictionary of recaller name -> RecBoleRecaller
        recaller_names: Ordered list of recaller names
        num_users: Total number of users
        num_items: Total number of items
        final_k: Number of items to recommend (L in the paper)
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate (η₂ in paper)
        reg_weight: Regularization weight (λ in paper)
        embedding_dim: Embedding dimension
        device: Device to run on
        
    Returns:
        Trained PersonalizedFusionPG model and training history
    """
    print("\n" + "="*60)
    print("Policy Gradient Training (Personalized Weights)")
    print("="*60)
    
    K = len(recaller_names)
    M = final_k * 3  # Get more candidates per channel for better fusion
    
    # Extract user data from training dataset
    user_ids = []
    eval_hists = []
    gt_items_list = []
    
    for example in train_dataset:
        user_ids.append(example["user_id"])
        
        if "history" in example:
            hist = example["history"]
            if isinstance(hist, list) and 0 in hist:
                hist = hist[:hist.index(0)]
            eval_hists.append(hist)
        else:
            eval_hists.append([])
        
        if "target_items" in example:
            gt = example["target_items"]
            if isinstance(gt, int):
                gt = [gt]
            gt_items_list.append(set(gt))
        else:
            gt_items_list.append(set())
    
    print(f"Processing {len(user_ids)} users with {K} recallers")
    print(f"Learning rate: {learning_rate}, Reg weight: {reg_weight}")
    
    # Build user_candidates
    def recaller_wrapper(recaller_name):
        def recall_fn(uids: List[int], topk: int) -> List[List[int]]:
            results = []
            for i, uid in enumerate(uids):
                hist = eval_hists[i]
                gt = list(gt_items_list[i]) if gt_items_list[i] else None
                
                items = recallers[recaller_name.lower()].recall(
                    uid, topk, hist, 
                    full_hist=hist, 
                    gt_items=gt
                )
                item_ids = [item[0] for item in items]
                results.append(item_ids)
            return results
        return recall_fn
    
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
        user_candidates = [user_candidates[i] for i in valid_indices]
        print(f"Filtered to {len(user_ids)}/{n_before} users (min_thres={min_thres}, train_k={train_k})")
    
    # Extract pre-trained embeddings from recallers (prefer SimpleX as per paper)
    print(f"\nExtracting pre-trained embeddings from recallers...")
    pretrained_user_emb = None
    pretrained_item_emb = None
    
    # Try to get embeddings from SimpleX first (as per paper Section C.3)
    if 'SimpleX' in recaller_names or 'simplex' in [k.lower() for k in recallers.keys()]:
        simplex_key = 'simplex' if 'simplex' in recallers else 'SimpleX'
        if simplex_key in recallers:
            print(f"  Extracting embeddings from SimpleX recaller...")
            user_emb, item_emb = extract_pretrained_embeddings(
                recallers[simplex_key], num_users, num_items, embedding_dim, device
            )
            if user_emb is not None:
                pretrained_user_emb = user_emb
            if item_emb is not None:
                pretrained_item_emb = item_emb
    
    # If SimpleX not available, try other models
    if pretrained_user_emb is None or pretrained_item_emb is None:
        for name in recaller_names:
            recaller_key = name.lower()
            if recaller_key in recallers and recaller_key != 'simplex':
                print(f"  Trying to extract embeddings from {name} recaller...")
                user_emb, item_emb = extract_pretrained_embeddings(
                    recallers[recaller_key], num_users, num_items, embedding_dim, device
                )
                if pretrained_user_emb is None and user_emb is not None:
                    pretrained_user_emb = user_emb
                    print(f"    Using user embeddings from {name}")
                if pretrained_item_emb is None and item_emb is not None:
                    pretrained_item_emb = item_emb
                    print(f"    Using item embeddings from {name}")
    
    if pretrained_user_emb is None and pretrained_item_emb is None:
        print("  Warning: No pre-trained embeddings found. Using random initialization.")
    elif pretrained_user_emb is None:
        print("  Warning: No pre-trained user embeddings found. Using random initialization.")
    elif pretrained_item_emb is None:
        print("  Warning: No pre-trained item embeddings found. Using random initialization.")
    else:
        print("  Successfully loaded pre-trained embeddings!")
    
    # Initialize PG model with pre-trained embeddings
    print(f"\nInitializing Policy Gradient model...")
    pg_model = PersonalizedFusionPG(
        num_users=num_users,
        num_items=num_items,
        num_channels=K,
        embedding_dim=embedding_dim,
        hidden_dim=128,
        learning_rate=learning_rate,
        reg_weight=reg_weight,
        delta_max=10.0,  # As per paper
        epsilon=1e-6,     # As per paper
        num_samples=1,    # S=1 as per paper
        top_k_channel_items=10,  # As per paper
        device=device,
        pretrained_user_emb=pretrained_user_emb,
        pretrained_item_emb=pretrained_item_emb
    )
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    history = {"loss": [], "avg_reward": []}
    best_reward = -float('inf')
    best_state = None
    
    for epoch in range(num_epochs):
        metrics = pg_model.train_step(
            user_ids=user_ids,
            histories=eval_hists,
            user_candidates=user_candidates,
            ground_truth=gt_items_list,
            L=final_k,
            batch_size=batch_size
        )
        
        history["loss"].append(metrics["loss"])
        history["avg_reward"].append(metrics["avg_reward"])
        
        if metrics["avg_reward"] > best_reward:
            best_reward = metrics["avg_reward"]
            best_state = {
                'user_module': pg_model.user_module.state_dict(),
                'channel_module': pg_model.channel_module.state_dict(),
                'policy': pg_model.policy.state_dict(),
            }
        
        print(f"[PG] Epoch {epoch+1:02d}/{num_epochs}: "
              f"Loss={metrics['loss']:.4f}, Avg Recall@{final_k}={metrics['avg_reward']:.4f}")
    
    # Load best model
    if best_state is not None:
        pg_model.user_module.load_state_dict(best_state['user_module'])
        pg_model.channel_module.load_state_dict(best_state['channel_module'])
        pg_model.policy.load_state_dict(best_state['policy'])
    
    print(f"\nBest training Recall@{final_k}: {best_reward:.4f}")
    
    return pg_model, history, user_candidates


def evaluate_pg_fusion(
    test_dataset,
    recallers: Dict[str, RecBoleRecaller],
    recaller_names: List[str],
    pg_model: PersonalizedFusionPG,
    final_k: int = 50,
    device: str = "cuda",
    min_thres: float = 0.001,
    train_k: int = 20,
):
    """
    Evaluate PG-optimized fusion on test set.
    
    Args:
        test_dataset: Test dataset with user_id, history, and ground truth
        recallers: Dictionary of recaller name -> RecBoleRecaller
        recaller_names: Ordered list of recaller names
        pg_model: Trained PersonalizedFusionPG model
        final_k: Number of items to recommend
        device: Device to run on
        
    Returns:
        Dictionary with evaluation results
    """
    print("\n" + "="*60)
    print("Policy Gradient Fusion Evaluation (Test Phase)")
    print("="*60)
    
    K = len(recaller_names)
    M = final_k * 3
    
    # Extract user data from test dataset
    user_ids = []
    eval_hists = []
    gt_items_list = []
    
    for example in test_dataset:
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
        
        if "target_items" in example:
            gt = example["target_items"]
            if isinstance(gt, int):
                gt = [gt]
            gt_items_list.append(set(gt))
        else:
            gt_items_list.append(set())
    
    print(f"Processing {len(user_ids)} users with {K} recallers (filtered history < 5)")
    
    # Build user_candidates
    def recaller_wrapper(recaller_name):
        def recall_fn(uids: List[int], topk: int) -> List[List[int]]:
            results = []
            for i, uid in enumerate(uids):
                hist = eval_hists[i]
                gt = list(gt_items_list[i]) if gt_items_list[i] else None
                
                items = recallers[recaller_name.lower()].recall(
                    uid, topk, hist, 
                    full_hist=hist, 
                    gt_items=gt
                )
                item_ids = [item[0] for item in items]
                results.append(item_ids)
            return results
        return recall_fn
    
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
        user_candidates = [user_candidates[i] for i in valid_indices]
        print(f"Filtered to {len(user_ids)}/{n_before} users (min_thres={min_thres}, train_k={train_k})")
    
    # Evaluate individual recallers
    print("\n" + "="*60)
    print("Individual Recaller Performance")
    print("="*60)
    individual_results = {}
    
    for k, name in enumerate(recaller_names):
        recaller_lists = []
        for u in range(len(user_candidates)):
            recaller_lists.append(user_candidates[u][k][:final_k])
        
        ndcg_scores_k = []
        recall_scores_k = []
        for rec_list, gt in zip(recaller_lists, gt_items_list):
            if len(gt) > 0:
                ndcg_scores_k.append(ndcg_at_k(rec_list, list(gt), final_k))
                recall_scores_k.append(recall_at_k(rec_list, list(gt), final_k))
        
        avg_ndcg_k = float(np.mean(ndcg_scores_k)) if ndcg_scores_k else 0.0
        avg_recall_k = float(np.mean(recall_scores_k)) if recall_scores_k else 0.0
        
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
    
    # Predict personalized weights
    print("\nPredicting personalized weights...")
    personalized_weights = pg_model.predict_weights(
        user_ids=user_ids,
        histories=eval_hists,
        user_candidates=user_candidates
    )
    
    # Compute metrics with personalized fusion
    ndcg_scores = []
    recall_scores = []
    all_weights = []
    
    for u in range(len(user_candidates)):
        w = personalized_weights[u]
        all_weights.append(w.cpu().numpy())
        
        # Fuse candidates
        fused = fuse_by_quota([user_candidates[u]], w, final_k)[0]
        gt = gt_items_list[u]
        
        if len(gt) > 0:
            ndcg = ndcg_at_k(fused, list(gt), final_k)
            recall = recall_at_k(fused, list(gt), final_k)
            ndcg_scores.append(ndcg)
            recall_scores.append(recall)
    
    avg_ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
    avg_recall = float(np.mean(recall_scores)) if recall_scores else 0.0
    
    print(f"\nPersonalized Fusion Performance:")
    print(f"  NDCG@{final_k}: {avg_ndcg:.4f}")
    print(f"  Recall@{final_k}: {avg_recall:.4f}")
    
    # Compute average weights
    # avg_weights = np.mean(all_weights, axis=0)
    avg_weights = personalized_weights.mean(dim=0)
    print(f"\nAverage Personalized Weights:")
    for i, name in enumerate(recaller_names):
        print(f"  {name}: {avg_weights[i]:.4f}")
    
    # Evaluate at different k values
    results = {
        "pg_optimized": {
            f"recall@{final_k}": avg_recall,
            f"ndcg@{final_k}": avg_ndcg,
        },
        "avg_weights": {name: float(avg_weights[i]) for i, name in enumerate(recaller_names)},
        "per_user_weights": {
            int(user_ids[u]): {name: float(personalized_weights[u][i]) for i, name in enumerate(recaller_names)}
            for u in range(len(user_ids))
        },
        "individual_recallers": individual_results,
    }
    
    for k in [10, 20, 50]:
        ndcg_k = []
        recall_k = []
        for u in range(len(user_candidates)):
            w = personalized_weights[u]
            fused = fuse_by_quota([user_candidates[u]], w, k)[0]
            gt = gt_items_list[u]
            if len(gt) > 0:
                ndcg_k.append(ndcg_at_k(fused, list(gt), k))
                recall_k.append(recall_at_k(fused, list(gt), k))
        
        results["pg_optimized"][f"ndcg@{k}"] = float(np.mean(ndcg_k)) if ndcg_k else 0.0
        results["pg_optimized"][f"recall@{k}"] = float(np.mean(recall_k)) if recall_k else 0.0
    
    print(f"\nPG-Optimized Performance at Different k:")
    for k in [10, 20, 50]:
        print(f"  k={k}: NDCG={results['pg_optimized'][f'ndcg@{k}']:.4f}, "
              f"Recall={results['pg_optimized'][f'recall@{k}']:.4f}")
    
    return results


def create_dataset_from_inter_dataset(inter_dataset, split='test', num_users=None):
    """
    Create a dataset from inter_dataset for PG optimization/evaluation.
    
    For train split: returns all examples without merging (preserves multiple samples per user)
    For eval/test splits: merges multiple examples for the same user:
    - history: selects the shortest one
    - target_items: union of all target_items
    """
    if split == 'train':
        user_ids = inter_dataset.train_user_ids
        histories = inter_dataset.train_histories
        target_items = inter_dataset.train_target_items
        
        # For train split, return all examples without merging
        dataset = []
        for uid, hist, target in zip(user_ids, histories, target_items):
            if isinstance(target, int):
                target_list = [target]
            else:
                target_list = target if isinstance(target, list) else [target]
            
            dataset.append({
                "user_id": uid,
                "history": hist if isinstance(hist, list) else [],
                "target_items": target_list,
                "full_hist": hist if isinstance(hist, list) else []
            })
        
        # Apply num_users limit if specified (limit by unique users)
        if num_users is not None:
            unique_users = sorted(set(user_ids))
            if len(unique_users) > num_users:
                selected_users = set(unique_users[:num_users])
                dataset = [ex for ex in dataset if ex["user_id"] in selected_users]
        
        return dataset
    
    elif split == 'eval':
        user_ids = inter_dataset.eval_user_ids
        histories = inter_dataset.eval_histories
        target_items = inter_dataset.eval_target_items
    else:  # test
        user_ids = inter_dataset.test_user_ids
        histories = inter_dataset.test_histories
        target_items = inter_dataset.test_target_items
    
    # For eval/test splits, merge multiple examples for the same user
    user_examples = {}
    for uid, hist, target in zip(user_ids, histories, target_items):
        if isinstance(target, int):
            target_list = [target]
        else:
            target_list = target if isinstance(target, list) else [target]
        
        if uid not in user_examples:
            user_examples[uid] = {
                "histories": [],
                "target_items": set()
            }
        
        user_examples[uid]["histories"].append(hist)
        user_examples[uid]["target_items"].update(target_list)
    
    # Merge examples for each user
    dataset = []
    for uid, examples in user_examples.items():
        histories_list = examples["histories"]
        shortest_hist = min(histories_list, key=lambda h: len(h) if isinstance(h, list) else 0)
        merged_target_items = sorted(list(examples["target_items"]))
        
        dataset.append({
            "user_id": uid,
            "history": shortest_hist,
            "target_items": merged_target_items,
            "full_hist": shortest_hist
        })
    
    dataset.sort(key=lambda x: x["user_id"])
    
    if num_users is not None:
        dataset = dataset[:num_users]
    
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Policy Gradient-based Personalized Fusion Optimization')
    
    # Data
    parser.add_argument('--dataset', type=str, default='Amazon_All_Beauty')
    parser.add_argument('--data_path', type=str, default='./dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--output_dir', type=str, default='results')
    
    # Model
    parser.add_argument('--recbole_models', type=str, nargs='+', default=['BPR', 'SASRec', 'LightGCN'])
    parser.add_argument('--embedding_dim', type=int, default=64, help='Embedding dimension')
    
    # PG parameters (from paper Section C.3)
    parser.add_argument('--final_k', type=int, default=50, help='Number of items to recommend (L)')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                       help='Learning rate η₂ (paper: {1e-5, 5e-5, 1e-4})')
    parser.add_argument('--reg_weight', type=float, default=1.0,
                       help='Regularization weight λ (paper: {0.5, 1, 5})')
    parser.add_argument('--delta_max', type=float, default=10.0,
                       help='Max Dirichlet parameter (paper: 10.0)')
    parser.add_argument('--epsilon', type=float, default=1e-6,
                       help='Min Dirichlet parameter epsilon (paper: 1e-6)')
    parser.add_argument('--num_samples', type=int, default=1,
                       help='Number of weight samples S (paper: 1)')
    parser.add_argument('--top_k_channel_items', type=int, default=10,
                       help='Top-k items to represent each channel (paper: 10)')
    
    parser.add_argument('--use_eval_for_training', action='store_true', 
                       help='Use eval set for weight optimization (default: use train set)')
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_users', type=int, default=None, help='Number of training users')
    parser.add_argument('--num_test_users', type=int, default=None, help='Number of test users')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
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
    print("Policy Gradient-Based Personalized Fusion Optimization")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Recallers: {args.recbole_models}")
    print(f"Final K: {args.final_k}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Regularization Weight: {args.reg_weight}")
    print(f"Epochs: {args.num_epochs}")
    print("="*60)
    
    # Set device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load dataset
    print("\n1. Loading dataset...")
    inter_dataset = load_dataset(args.dataset, args.data_path, seed=args.seed)
    print(f"   Train users: {len(inter_dataset.train_user_ids)}")
    print(f"   Eval users: {len(inter_dataset.eval_user_ids)}")
    print(f"   Test users: {len(inter_dataset.test_user_ids)}")
    print(f"   Num items: {inter_dataset.ds.item_num}")
    
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
    
    # Create training dataset
    print("\n3. Creating training dataset...")
    train_split = 'eval' if args.use_eval_for_training else 'train'
    # train_dataset = create_dataset_from_inter_dataset(
    #     inter_dataset, 
    #     split=train_split,
    #     num_users=args.num_train_users
    # )
    train_dataset = create_dataset_from_inter_dataset(
        inter_dataset, 
        split='eval',
        num_users=args.num_train_users
    )
    print(f"   Created {train_split} dataset with {len(train_dataset)} users")
    
    # Train PG model
    print("\n4. Training Policy Gradient model...")
    recaller_names = sorted(args.recbole_models)
    pg_model, train_history, _ = train_pg_model(
        train_dataset=train_dataset,
        recallers=recallers,
        recaller_names=recaller_names,
        num_users=inter_dataset.ds.user_num,
        num_items=inter_dataset.ds.item_num,
        final_k=args.final_k,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        reg_weight=args.reg_weight,
        embedding_dim=args.embedding_dim,
        device=device,
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
    
    # Evaluate on test set
    print("\n6. Evaluating on test set...")
    pg_results = evaluate_pg_fusion(
        test_dataset=test_dataset,
        recallers=recallers,
        recaller_names=recaller_names,
        pg_model=pg_model,
        final_k=args.final_k,
        device=device,
        min_thres=args.min_thres,
        train_k=args.train_k,
    )
    
    # Save results
    print("\n7. Saving results...")
    os.makedirs(args.output_dir, exist_ok=True)
    recaller_combo = "_".join(sorted(args.recbole_models))
    result_filename = f"{args.output_dir}/pg_results_{args.dataset}_{recaller_combo}.json"
    
    results = {
        "pg_fusion": pg_results,
        "training_history": {
            "loss": train_history["loss"],
            "avg_reward": train_history["avg_reward"],
        },
        "config": {
            "dataset": args.dataset,
            "recbole_models": args.recbole_models,
            "recaller_combo": recaller_combo,
            "final_k": args.final_k,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "reg_weight": args.reg_weight,
            "delta_max": args.delta_max,
            "epsilon": args.epsilon,
            "num_samples": args.num_samples,
            "top_k_channel_items": args.top_k_channel_items,
            "embedding_dim": args.embedding_dim,
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
    
    pg_ndcg = pg_results['pg_optimized'].get(f'ndcg@{args.final_k}', 0.0)
    pg_recall = pg_results['pg_optimized'].get(f'recall@{args.final_k}', 0.0)
    print(f"\nPG-Optimized Personalized Fusion Performance:")
    print(f"  NDCG@{args.final_k}: {pg_ndcg:.4f}")
    print(f"  Recall@{args.final_k}: {pg_recall:.4f}")
    print(f"\nAverage Personalized Weights:")
    for name, weight in pg_results['avg_weights'].items():
        print(f"  {name}: {weight:.4f}")
    
    # Print individual recaller comparison
    if 'individual_recallers' in pg_results:
        print(f"\n" + "="*60)
        print("Individual Recaller Comparison")
        print("="*60)
        for name, metrics in pg_results['individual_recallers'].items():
            print(f"{name}: NDCG@{args.final_k}={metrics[f'ndcg@{args.final_k}']:.4f}, "
                  f"Recall@{args.final_k}={metrics[f'recall@{args.final_k}']:.4f}")
    
    print("="*60)


if __name__ == "__main__":
    main()

