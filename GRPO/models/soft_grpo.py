"""
SofT-GRPO: Soft Token Group Relative Policy Optimization

Core algorithm implementing:
- Gumbel-Softmax soft sampling for distributional actions
- Noise-based importance ratio computation
- Multi-channel recall with softmax weight aggregation
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np


@dataclass
class SoftSampleBuffer:
    """Buffer for storing soft sampling information during rollout."""
    support_set: torch.Tensor  # (T, K) indices of top-p tokens at each step
    q_prime: torch.Tensor  # (T, K) log_p + epsilon values
    log_p_old_noise: torch.Tensor  # (T,) log probability of noise under old policy
    y_soft: torch.Tensor  # (T, K) soft token probabilities (Gumbel-Softmax output)
    

def sample_gumbel(shape: Tuple, device: torch.device, eps: float = 1e-20) -> torch.Tensor:
    """Sample from Gumbel(0, 1) distribution."""
    u = torch.rand(shape, device=device).clamp(eps, 1 - eps)
    return -torch.log(-torch.log(u))


def gumbel_log_pdf(epsilon: torch.Tensor) -> torch.Tensor:
    """Compute log pdf of Gumbel(0,1): log f_G(ε) = -ε - e^{-ε}"""
    return -epsilon - torch.exp(-epsilon)


def top_p_filter(logits: torch.Tensor, p: float = 0.9) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply nucleus (top-p) sampling to get support set and renormalized probabilities.
    
    Args:
        logits: Raw logits (batch_size, vocab_size) or (vocab_size,)
        p: Cumulative probability threshold
        
    Returns:
        support_indices: Indices of tokens in support set
        renorm_probs: Renormalized probabilities over support set
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find cutoff index where cumsum exceeds p
    cutoff_mask = cumsum_probs <= p
    # Always include at least one token
    cutoff_mask[:, 0] = True
    
    # Get support set
    support_mask = torch.zeros_like(probs, dtype=torch.bool)
    support_mask.scatter_(1, sorted_indices, cutoff_mask)
    
    # Get indices and renormalized probs
    support_indices = support_mask.nonzero(as_tuple=False)
    
    # Renormalize probabilities
    renorm_probs = probs.clone()
    renorm_probs[~support_mask] = 0
    renorm_probs = renorm_probs / renorm_probs.sum(dim=-1, keepdim=True)
    
    return support_mask, renorm_probs


def soft_sample_step(
    logits: torch.Tensor,
    tau_g: float = 1.0,
    top_p: float = 0.9,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform one soft sampling step using Gumbel-Softmax with top-p filtering.
    
    Args:
        logits: Raw logits from model (vocab_size,) or (batch, vocab_size)
        tau_g: Gumbel temperature
        top_p: Nucleus sampling threshold
        device: Device for computation
        
    Returns:
        y_soft: Soft token distribution (Gumbel-Softmax output)
        support_mask: Boolean mask for support set
        q_prime: log_p + epsilon values for cached replay
        log_p_noise: Log probability of noise (for importance ratio)
    """
    if device is None:
        device = logits.device
    
    # Get support set via top-p
    support_mask, renorm_probs = top_p_filter(logits, top_p)
    
    # Sample Gumbel noise for support set
    epsilon = sample_gumbel(renorm_probs.shape, device)
    
    # Compute q' = log(p) + epsilon (only for support)
    log_renorm_probs = torch.log(renorm_probs + 1e-10)
    q_prime = log_renorm_probs + epsilon
    
    # Mask non-support tokens
    q_prime = torch.where(support_mask, q_prime, torch.tensor(float('-inf'), device=device))
    
    # Apply softmax with temperature to get soft distribution
    y_soft = F.softmax(q_prime / tau_g, dim=-1)
    
    # Compute log probability of noise (sum over support)
    # log p_noise = sum_i (-epsilon_i - exp(-epsilon_i))
    log_p_noise = gumbel_log_pdf(epsilon)
    log_p_noise = (log_p_noise * support_mask.float()).sum(dim=-1)
    
    return y_soft, support_mask, q_prime, log_p_noise


def compute_soft_embedding(
    y_soft: torch.Tensor,
    embeddings: torch.Tensor,
    support_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute soft embedding as weighted sum of token embeddings.
    
    Args:
        y_soft: Soft token probabilities (batch, vocab_size) or (vocab_size,)
        embeddings: Token embedding matrix (vocab_size, hidden_dim)
        support_mask: Optional mask to restrict to support set
        
    Returns:
        soft_emb: Weighted embedding (batch, hidden_dim) or (hidden_dim,)
    """
    if y_soft.dim() == 1:
        y_soft = y_soft.unsqueeze(0)
    
    if support_mask is not None:
        y_soft = y_soft * support_mask.float()
    
    # (batch, vocab) @ (vocab, hidden) -> (batch, hidden)
    soft_emb = torch.matmul(y_soft, embeddings)
    
    return soft_emb.squeeze(0) if soft_emb.size(0) == 1 else soft_emb


def compute_soft_importance_ratio(
    q_prime: torch.Tensor,
    new_logits: torch.Tensor,
    support_mask: torch.Tensor,
    log_p_old_noise: torch.Tensor,
    tau_g: float = 1.0
) -> torch.Tensor:
    """
    Compute importance ratio for soft step based on noise density ratio.
    
    Args:
        q_prime: Cached log_p_old + epsilon values
        new_logits: New policy logits
        support_mask: Boolean mask for support set
        log_p_old_noise: Cached log noise probability under old policy
        tau_g: Gumbel temperature
        
    Returns:
        r_soft: Soft step importance ratio
    """
    # Get new policy probabilities restricted to support set
    new_probs = F.softmax(new_logits, dim=-1)
    
    # Renormalize over support set
    new_probs_support = new_probs * support_mask.float()
    new_probs_support = new_probs_support / (new_probs_support.sum(dim=-1, keepdim=True) + 1e-10)
    
    # Compute new epsilon values: epsilon_new = q' - log(p_new)
    log_new_probs = torch.log(new_probs_support + 1e-10)
    epsilon_new = q_prime - log_new_probs
    
    # Compute log probability of new noise
    log_p_new_noise = gumbel_log_pdf(epsilon_new)
    log_p_new_noise = (log_p_new_noise * support_mask.float()).sum(dim=-1)
    
    # Importance ratio
    r_soft = torch.exp(log_p_new_noise - log_p_old_noise)
    
    return r_soft


def replay_soft_context(
    q_prime: torch.Tensor,
    support_mask: torch.Tensor,
    tau_g: float = 1.0
) -> torch.Tensor:
    """
    Replay soft token distribution from cached q' values.
    
    Args:
        q_prime: Cached log_p + epsilon values
        support_mask: Boolean mask for support set
        tau_g: Gumbel temperature
        
    Returns:
        y_soft: Reconstructed soft token distribution
    """
    # Mask non-support
    q_prime_masked = torch.where(
        support_mask,
        q_prime,
        torch.tensor(float('-inf'), device=q_prime.device)
    )
    
    # Apply temperature and softmax
    y_soft = F.softmax(q_prime_masked / tau_g, dim=-1)
    
    return y_soft


# ============== Multi-Channel Recall Functions ==============

def multi_channel_recall_with_softmax_weights(
    softmax_weights: torch.Tensor,
    recallers: Dict,
    recaller_names: List[str],
    user_id: int,
    history: List[int],
    total_k: int
) -> List[Tuple[int, float]]:
    """
    Perform multi-channel recall using softmax weights from classification.
    
    Args:
        softmax_weights: Softmax probability vector over recallers (num_recallers,)
        recallers: Dictionary of recaller objects
        recaller_names: List of recaller names matching softmax order
        user_id: User ID for recall
        history: User interaction history
        total_k: Total number of items to return
        
    Returns:
        List of (item_id, weighted_score) tuples sorted by score
    """
    candidates = defaultdict(float)
    
    for i, name in enumerate(recaller_names):
        weight = softmax_weights[i].item() if torch.is_tensor(softmax_weights[i]) else softmax_weights[i]
        
        if name.lower() in recallers:
            recaller = recallers[name.lower()]
            items = recaller.recall(user_id, total_k, history)
            
            for item_id, score in items:
                candidates[item_id] += score * weight
    
    # Sort by weighted score
    sorted_items = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:total_k]


def compute_ndcg_reward(
    candidates: List[Tuple[int, float]],
    ground_truth: List[int],
    k: int
) -> float:
    """
    Compute NDCG@k reward for candidate list.
    
    Args:
        candidates: List of (item_id, score) tuples
        ground_truth: List of ground truth item IDs
        k: Cutoff for NDCG computation
        
    Returns:
        NDCG@k score
    """
    import math
    
    if not ground_truth:
        return 0.0
    
    gt_set = set(ground_truth)
    rec_list = [item_id for item_id, _ in candidates[:k]]
    
    # DCG@k
    dcg = 0.0
    for i, item_id in enumerate(rec_list):
        if item_id in gt_set:
            dcg += 1.0 / math.log2(i + 2)
    
    # IDCG@k
    num_relevant = min(len(gt_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(num_relevant))
    
    return dcg / idcg if idcg > 0 else 0.0


def soft_grpo_reward_function(
    softmax_outputs: List[torch.Tensor],
    user_ids: List[int],
    histories: List[List[int]],
    ground_truths: List[List[int]],
    recallers: Dict,
    recaller_names: List[str],
    final_k: int = 50
) -> List[float]:
    """
    Compute NDCG rewards for batch of soft samples using multi-channel recall.
    
    This function directly uses classification softmax vectors as recaller weights.
    
    Args:
        softmax_outputs: List of softmax vectors from classification model
        user_ids: Batch of user IDs
        histories: Batch of user histories
        ground_truths: Batch of ground truth items
        recallers: Dictionary of recaller objects
        recaller_names: List of recaller names matching softmax order
        final_k: Top-k for NDCG computation
        
    Returns:
        List of NDCG rewards
    """
    rewards = []
    
    for i, softmax_weights in enumerate(softmax_outputs):
        try:
            # Multi-channel recall with softmax weights
            candidates = multi_channel_recall_with_softmax_weights(
                softmax_weights=softmax_weights,
                recallers=recallers,
                recaller_names=recaller_names,
                user_id=user_ids[i],
                history=histories[i],
                total_k=final_k
            )
            
            # Compute NDCG reward
            reward = compute_ndcg_reward(candidates, ground_truths[i], final_k)
            rewards.append(reward)
            
        except Exception as e:
            print(f"Error computing reward for sample {i}: {e}")
            rewards.append(0.0)
    
    return rewards


# ============== Loss Functions ==============

def compute_soft_grpo_loss(
    per_token_logps: torch.Tensor,
    old_per_token_logps: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    epsilon: float = 0.2,
    soft_ratios: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute clipped PPO-style loss for SofT-GRPO.
    
    For soft steps, uses noise-based importance ratios.
    For discrete steps, uses standard policy ratios.
    
    Args:
        per_token_logps: Current policy log probs (batch, seq_len)
        old_per_token_logps: Old policy log probs (batch, seq_len)
        advantages: Group-normalized advantages (batch,)
        completion_mask: Mask for completion tokens (batch, seq_len)
        epsilon: Clipping coefficient
        soft_ratios: Optional pre-computed soft importance ratios
        
    Returns:
        Clipped PPO loss (scalar)
    """
    # Discrete step ratios
    log_ratio = per_token_logps - old_per_token_logps
    ratio = torch.exp(log_ratio)
    
    # If soft_ratios provided, use them for corresponding positions
    if soft_ratios is not None:
        ratio = soft_ratios
    
    # Expand advantages for token-level
    advantages = advantages.unsqueeze(-1)
    
    # Clipped objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    
    # Take minimum (pessimistic bound)
    loss_per_token = -torch.min(surr1, surr2)
    
    # Average over valid tokens
    loss = (loss_per_token * completion_mask).sum() / (completion_mask.sum() + 1e-8)
    
    return loss


def compute_kl_penalty(
    per_token_logps: torch.Tensor,
    ref_per_token_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    beta: float = 0.01
) -> torch.Tensor:
    """
    Compute KL divergence penalty from reference policy.
    
    Args:
        per_token_logps: Current policy log probs
        ref_per_token_logps: Reference policy log probs
        completion_mask: Mask for completion tokens
        beta: KL penalty weight
        
    Returns:
        Weighted KL penalty (scalar)
    """
    # Approximate KL: exp(log_ref - log_pi) - (log_ref - log_pi) - 1
    per_token_kl = (
        torch.exp(ref_per_token_logps - per_token_logps)
        - (ref_per_token_logps - per_token_logps)
        - 1
    )
    
    kl = (per_token_kl * completion_mask).sum() / (completion_mask.sum() + 1e-8)
    
    return beta * kl


def normalize_advantages(
    rewards: torch.Tensor,
    num_generations: int,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute group-normalized advantages (GRPO style).
    
    Args:
        rewards: Reward tensor (batch_size,)
        num_generations: Number of generations per prompt (G)
        eps: Small constant for numerical stability
        
    Returns:
        Normalized advantages
    """
    # Reshape to (num_prompts, num_generations)
    rewards_grouped = rewards.view(-1, num_generations)
    
    # Group mean and std
    mean_rewards = rewards_grouped.mean(dim=1, keepdim=True)
    std_rewards = rewards_grouped.std(dim=1, keepdim=True)
    
    # Normalize
    advantages = (rewards_grouped - mean_rewards) / (std_rewards + eps)
    
    # Flatten back
    return advantages.view(-1)


