"""
Utilities for soft token generation and beta sampling.
Shared between main_soft.py and trl_trainer.py to avoid code duplication.
Extended with Gumbel-Softmax utilities for SofT-GRPO.
"""

import torch
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Optional, Callable, Union
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass
from GRPO.models.normalization import normalize_scores


# ============== Gumbel-Softmax Utilities for SofT-GRPO ==============

@dataclass
class SoftStepCache:
    """Cache for soft sampling step to enable replay during policy update."""
    support_indices: torch.Tensor  # Indices in vocabulary
    q_prime: torch.Tensor  # log_p + epsilon values
    log_p_old_noise: float  # Log probability of noise
    y_soft: torch.Tensor  # Soft distribution output


def sample_gumbel_noise(shape: Tuple, device: torch.device, eps: float = 1e-20) -> torch.Tensor:
    """Sample from Gumbel(0, 1) distribution."""
    u = torch.rand(shape, device=device).clamp(eps, 1 - eps)
    return -torch.log(-torch.log(u))


def gumbel_softmax_sample(
    logits: torch.Tensor,
    tau: float = 1.0,
    top_p: float = 0.9,
    hard: bool = False,
    noise_scale: float = 1.0
) -> Tuple[torch.Tensor, SoftStepCache]:
    """
    Gumbel-Softmax sampling with top-p filtering for SofT-GRPO.
    
    Args:
        logits: Logits from classification model (num_classes,)
        tau: Temperature for Gumbel-Softmax
        top_p: Nucleus sampling probability threshold
        hard: If True, return hard one-hot (straight-through estimator)
        noise_scale: Scale factor for Gumbel noise (0.0 = no noise, 1.0 = standard)
        
    Returns:
        y_soft: Soft probability distribution
        cache: SoftStepCache for importance ratio computation
    """
    device = logits.device
    
    # Apply top-p filtering
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    
    # Find cutoff
    mask = cumsum <= top_p
    mask[0] = True  # Always include top-1
    
    # Create support set
    support_indices = sorted_indices[mask]
    support_probs = probs[support_indices]
    
    # Renormalize
    support_probs = support_probs / support_probs.sum()
    log_support_probs = torch.log(support_probs + 1e-10)
    
    # Sample Gumbel noise (scaled to control exploration vs exploitation)
    epsilon = sample_gumbel_noise(support_probs.shape, device) * noise_scale
    
    # Compute q' = log(p) + epsilon
    # When noise_scale=0, q_prime = log(p), output = softmax(log(p)/tau) = p^(1/tau) (normalized)
    # When noise_scale=1, standard Gumbel-Softmax
    q_prime = log_support_probs + epsilon
    
    # Gumbel-Softmax
    y_soft_support = F.softmax(q_prime / tau, dim=-1)
    
    # Expand to full vocabulary
    y_soft = torch.zeros_like(logits)
    y_soft[support_indices] = y_soft_support.to(logits.dtype)
    
    # Compute log noise probability
    log_p_noise = (-epsilon - torch.exp(-epsilon)).sum().item()
    
    # Cache for replay
    cache = SoftStepCache(
        support_indices=support_indices,
        q_prime=q_prime,
        log_p_old_noise=log_p_noise,
        y_soft=y_soft
    )
    
    if hard:
        # Straight-through estimator
        idx = y_soft.argmax(dim=-1)
        y_hard = F.one_hot(idx, num_classes=logits.size(-1)).float()
        y_soft = (y_hard - y_soft).detach() + y_soft
    
    return y_soft, cache


def compute_soft_ratio(
    cache: SoftStepCache,
    new_logits: torch.Tensor,
    tau: float = 1.0
) -> float:
    """
    Compute importance ratio for soft step using cached noise.
    
    Args:
        cache: SoftStepCache from old policy sampling
        new_logits: Logits from new policy
        tau: Temperature
        
    Returns:
        Importance ratio r_soft
    """
    device = new_logits.device
    
    # Get new probs for support set
    new_probs = F.softmax(new_logits, dim=-1)
    new_support_probs = new_probs[cache.support_indices]
    
    # Renormalize
    new_support_probs = new_support_probs / (new_support_probs.sum() + 1e-10)
    log_new_support_probs = torch.log(new_support_probs + 1e-10)
    
    # Compute epsilon_new = q' - log(p_new)
    epsilon_new = cache.q_prime - log_new_support_probs
    
    # Compute log probability under new noise
    log_p_new_noise = (-epsilon_new - torch.exp(-epsilon_new)).sum().item()
    
    # Importance ratio
    r_soft = torch.exp(torch.tensor(log_p_new_noise - cache.log_p_old_noise))
    
    return r_soft.item()


def replay_soft_sample(cache: SoftStepCache, tau: float = 1.0) -> torch.Tensor:
    """
    Replay soft sample from cached q' values.
    
    Args:
        cache: SoftStepCache from original sampling
        tau: Temperature
        
    Returns:
        Reconstructed y_soft
    """
    y_soft_support = F.softmax(cache.q_prime / tau, dim=-1)
    y_soft = cache.y_soft.clone()
    y_soft[cache.support_indices] = y_soft_support
    return y_soft


# ============== Multi-Channel Recall ==============

ItemScoreList = List[Tuple[int, float]]


# -----------------------------
# Rescoring / normalization fns
# -----------------------------

def _to_tensor_scores(items: ItemScoreList, device=None, dtype=torch.float32):
    ids = [int(i) for i, _ in items]
    scores = torch.tensor([float(s) for _, s in items], device=device, dtype=dtype)
    return ids, scores


def rescore_none(items: ItemScoreList, **kwargs) -> ItemScoreList:
    return items


def rescore_minmax(items: ItemScoreList, eps: float = 1e-12, **kwargs) -> ItemScoreList:
    if not items:
        return items
    ids, s = _to_tensor_scores(items)
    s_min, s_max = torch.min(s), torch.max(s)
    denom = (s_max - s_min).clamp_min(eps)
    s2 = (s - s_min) / denom
    return list(zip(ids, s2.tolist()))


def rescore_zscore(items: ItemScoreList, eps: float = 1e-12, **kwargs) -> ItemScoreList:
    if not items:
        return items
    ids, s = _to_tensor_scores(items)
    mu = torch.mean(s)
    sigma = torch.std(s, unbiased=False).clamp_min(eps)
    s2 = (s - mu) / sigma
    return list(zip(ids, s2.tolist()))


def rescore_softmax(items: ItemScoreList, temperature: float = 1.0, **kwargs) -> ItemScoreList:
    """
    Converts channel scores to a probability distribution over the returned items.
    Note: this is per-channel and per-query.
    """
    if not items:
        return items
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    ids, s = _to_tensor_scores(items)
    s2 = torch.softmax(s / temperature, dim=0)
    return list(zip(ids, s2.tolist()))


def rescore_rank_reciprocal(
    items: ItemScoreList,
    offset: float = 1.0,
    **kwargs
) -> ItemScoreList:
    """
    Uses rank only. Highest-ranked gets 1/(offset), next 1/(offset+1), ...
    Assumes input `items` is already sorted descending by the channel.
    """
    if not items:
        return items
    out = []
    for r, (item_id, _) in enumerate(items, start=0):
        out.append((int(item_id), float(1.0 / (offset + r))))
    return out


def rescore_rank_exp(
    items: ItemScoreList,
    alpha: float = 1.0,
    **kwargs
) -> ItemScoreList:
    """
    Uses rank only: exp(-alpha * rank). Higher alpha => faster decay.
    Assumes input `items` is already sorted descending by the channel.
    """
    if not items:
        return items
    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    out = []
    for r, (item_id, _) in enumerate(items, start=0):
        out.append((int(item_id), float(math.exp(-alpha * r))))
    return out


def rescore_percentile(
    items: ItemScoreList,
    eps: float = 1e-12,
    **kwargs
) -> ItemScoreList:
    """
    Percentile / quantile normalization using empirical CDF estimated from the
    returned scores (typically top-K). Produces values in [0, 1].

    Interpretation: how many scores in this channel list are <= current score.
    This is a practical approximation when you don't have a global per-channel CDF.
    """
    if not items:
        return items
    ids, s = _to_tensor_scores(items)
    n = s.numel()
    if n == 1:
        return [(ids[0], 1.0)]

    # Compute ranks of scores among this list (ties handled by average rank).
    # Approach: sort scores, assign average rank to ties, then normalize to [0,1].
    s_sorted, idx_sorted = torch.sort(s, descending=False)  # ascending for CDF
    ranks = torch.empty_like(s_sorted, dtype=torch.float32)

    # Assign average rank for ties
    start = 0
    while start < n:
        end = start + 1
        while end < n and torch.isclose(s_sorted[end], s_sorted[start], rtol=0, atol=eps):
            end += 1
        # average rank in [0, n-1]
        avg_rank = 0.5 * (start + (end - 1))
        ranks[start:end] = avg_rank
        start = end

    # Convert rank to percentile in [0,1]
    pct_sorted = ranks / float(n - 1)
    # Map back to original order
    pct = torch.empty_like(pct_sorted)
    pct[idx_sorted] = pct_sorted
    return list(zip(ids, pct.tolist()))


def rescore_platt(
    items: ItemScoreList,
    a: float,
    b: float,
    **kwargs
) -> ItemScoreList:
    """
    Platt scaling: sigmoid(a*s + b). Requires learned (a, b) per channel.
    """
    if not items:
        return items
    ids, s = _to_tensor_scores(items)
    s2 = torch.sigmoid(a * s + b)
    return list(zip(ids, s2.tolist()))


# Registry for convenience
RESCORERS: Dict[str, Callable[..., ItemScoreList]] = {
    "none": rescore_none,
    "minmax": rescore_minmax,
    "zscore": rescore_zscore,
    "softmax": rescore_softmax,
    "rank_reciprocal": rescore_rank_reciprocal,
    "rank_exp": rescore_rank_exp,
    "percentile": rescore_percentile,
    "platt": rescore_platt,
}


def apply_rescore(
    items: ItemScoreList,
    method: Optional[str],
    *,
    method_kwargs: Optional[dict] = None
) -> ItemScoreList:
    if method is None or method == "none":
        return items
    method = method.lower()
    if method not in RESCORERS:
        raise ValueError(f"Unknown rescore method: {method}. Available: {sorted(RESCORERS.keys())}")
    kw = method_kwargs or {}
    return RESCORERS[method](items, **kw)


# -----------------------------------
# Your function, updated (merge之前)
# -----------------------------------

def multi_channel_recall_score(
    softmax_weights: torch.Tensor,
    recallers: Dict,
    recaller_names: List[str],
    user_id: int,
    history: List[int],
    total_k: int,
    full_hist: List[int] = None,
    gt_items: List[int] = None,
    score_norm: Optional[str] = "none",
    score_norm_kwargs: Optional[dict] = None,
    normalize_scores: Optional[str] = None,  # Deprecated: use score_norm instead
) -> List[Tuple[int, float]]:
    """
    Multi-channel recall using softmax weights from classification, with
    per-channel rescoring BEFORE merging to ensure cross-channel comparability.

    Args:
        softmax_weights: Probability distribution over recallers (len == len(recaller_names))
        recallers: Dict of recaller name -> RecBoleRecaller (keyed by lowercased name)
        recaller_names: Ordered list of recaller names
        user_id: User ID
        history: User history
        total_k: Number of items to return
        full_hist: All interacted items (optional). If provided along with gt_items,
                  items in full_hist but not in gt_items will be excluded
        gt_items: Ground truth items to keep as valid candidates (optional)
        score_norm: Rescoring method applied per channel *before* merging.
            Options:
              - None / "none": no change
              - "minmax": per-channel min-max to [0,1]
              - "zscore": per-channel z-score (mean 0 std 1)
              - "softmax": per-channel softmax distribution
              - "percentile": per-channel empirical CDF to [0,1] (robust)
              - "rank_reciprocal": 1/(offset+rank) using rank only
              - "rank_exp": exp(-alpha*rank) using rank only
              - "platt": sigmoid(a*s+b) (requires score_norm_kwargs={"a":..., "b":...})
        score_norm_kwargs: optional kwargs for the chosen normalization.
        normalize_scores: Deprecated parameter, use score_norm instead. For backward compatibility,
                         if score_norm is None or "none" and normalize_scores is provided, it will be used.

    Returns:
        List of (item_id, score) sorted by weighted merged score
    """
    # Backward compatibility: use normalize_scores if score_norm is not set
    if (score_norm is None or score_norm == "none") and normalize_scores is not None:
        score_norm = normalize_scores
    
    candidates = defaultdict(float)

    for i, name in enumerate(recaller_names):
        w_i = softmax_weights[i]
        weight = w_i.item() if torch.is_tensor(w_i) else float(w_i)

        name_lower = name.lower()
        if name_lower not in recallers:
            continue

        items: ItemScoreList = recallers[name_lower].recall(
            user_id, total_k, history, full_hist=full_hist, gt_items=gt_items
        )

        if not items:
            continue

        # Rescore / normalize per channel BEFORE merging
        rescored_items = apply_rescore(items, score_norm, method_kwargs=score_norm_kwargs)

        # Merge rescored scores with weights
        for item_id, s in rescored_items:
            candidates[int(item_id)] += float(s) * weight

    sorted_items = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:total_k]


def multi_channel_merge_cached(
    softmax_weights: torch.Tensor,
    cached_outputs: Dict[str, List[Tuple[int, float]]],
    recaller_names: List[str],
    total_k: int,
    score_norm: Optional[str] = "none",
    score_norm_kwargs: Optional[dict] = None,
) -> List[Tuple[int, float]]:
    """
    Merge cached recaller outputs using softmax weights.
    
    Same as multi_channel_recall_score but uses pre-cached recaller outputs
    instead of calling recallers directly. Useful for avoiding redundant recall calls.
    
    Args:
        softmax_weights: Probability distribution over recallers
        cached_outputs: Dict of recaller_name (lowercase) -> [(item_id, score), ...]
        recaller_names: Ordered list of recaller names (matches softmax_weights order)
        total_k: Number of items to return
        score_norm: Rescoring method (same options as multi_channel_recall_score)
        score_norm_kwargs: Optional kwargs for normalization
    
    Returns:
        List of (item_id, score) sorted by weighted merged score
    """
    candidates = defaultdict(float)
    
    for i, name in enumerate(recaller_names):
        w_i = softmax_weights[i]
        weight = w_i.item() if torch.is_tensor(w_i) else float(w_i)
        
        name_lower = name.lower()
        if name_lower not in cached_outputs:
            continue
        
        items = cached_outputs[name_lower]
        if not items:
            continue
        
        # Rescore / normalize per channel BEFORE merging
        if score_norm and score_norm != "none":
            items = apply_rescore(items, score_norm, method_kwargs=score_norm_kwargs)
        
        for item_id, s in items:
            candidates[int(item_id)] += float(s) * weight
    
    sorted_items = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:total_k]


def multi_channel_recall_top_k(
    recallers: Dict,
    recaller_names: List[str],
    user_id: int,
    history: List[int],
    total_k: int,
    full_hist: List[int] = None,
    gt_items: List[int] = None,
    weights: Optional[Union[Dict[str, float], List[float]]] = None,
    quota_mode: str = "round",     # {"floor", "round", "ceil"}
    tie_break: str = "next_score",      # {"fixed", "next_score"}
) -> List[Tuple[int, float]]:
    """
    Multi-channel recall merge using prefix-quota (deficit) scheduling.

    For each prefix length t, channel k should contribute about w_k * t items.
    At step t (1-indexed), we compute quota_k(t) and pick the channel with
    largest deficit: deficit_k = quota_k(t) - selected_k.

    Assumption: no channel exhaustion (each channel provides enough items).
    Dedup is handled by skipping seen items without increasing selected_k.
    """
    num_recallers = len(recaller_names)
    if num_recallers == 0 or total_k <= 0:
        return []

    names_lower = [n.lower() for n in recaller_names]

    # ----- weights normalize -----
    if weights is None:
        w = {n: 1.0 / num_recallers for n in names_lower}
    elif isinstance(weights, torch.Tensor):
        weights_list = weights.detach().cpu().tolist()
        assert len(weights_list) == num_recallers, "weights tensor must align with recaller_names"
        w = {names_lower[i]: float(weights_list[i]) for i in range(num_recallers)}
    elif isinstance(weights, list):
        assert len(weights) == num_recallers, "weights list must align with recaller_names"
        w = {names_lower[i]: float(weights[i]) for i in range(num_recallers)}
    elif isinstance(weights, dict):
        w = {}
        for i, n in enumerate(names_lower):
            orig = recaller_names[i]
            w[n] = float(weights.get(n, weights.get(orig, 0.0)))
    else:
        raise TypeError("weights must be None, list, dict, or torch.Tensor")

    for k_, v in w.items():
        if v < 0:
            raise ValueError(f"Negative weight for channel {k_}: {v}")
    s = sum(w.values())
    if s <= 0:
        w = {n: 1.0 / num_recallers for n in names_lower}
    else:
        w = {k_: v / s for k_, v in w.items()}

    # ----- get recall lists -----
    recaller_items: Dict[str, List[Tuple[int, float]]] = {}
    assert all(name.lower() in recallers for name in recaller_names), \
        f"Recaller names {recaller_names} are not in recallers {recallers.keys()}"

    for name in recaller_names:
        name_lower = name.lower()
        items = recallers[name_lower].recall(
            user_id, total_k, history, full_hist=full_hist, gt_items=gt_items
        )
        recaller_items[name_lower] = items if items else []

    return _merge_top_k(recaller_items, names_lower, w, total_k, quota_mode, tie_break)


def multi_channel_topk_cached(
    softmax_weights: torch.Tensor,
    cached_outputs: Dict[str, List[Tuple[int, float]]],
    recaller_names: List[str],
    total_k: int,
    quota_mode: str = "round",
    tie_break: str = "next_score",
) -> List[Tuple[int, float]]:
    """
    Merge cached recaller outputs using prefix-quota (top-k) scheduling.
    
    Same as multi_channel_recall_top_k but uses pre-cached recaller outputs.
    """
    num_recallers = len(recaller_names)
    if num_recallers == 0 or total_k <= 0:
        return []

    names_lower = [n.lower() for n in recaller_names]

    # Normalize weights
    w = {}
    for i, n in enumerate(names_lower):
        w_i = softmax_weights[i]
        w[n] = w_i.item() if torch.is_tensor(w_i) else float(w_i)
    
    s = sum(w.values())
    if s <= 0:
        w = {n: 1.0 / num_recallers for n in names_lower}
    else:
        w = {k_: v / s for k_, v in w.items()}

    # Use cached outputs directly
    recaller_items = {n: cached_outputs.get(n, []) for n in names_lower}
    
    return _merge_top_k(recaller_items, names_lower, w, total_k, quota_mode, tie_break)


def _merge_top_k(
    recaller_items: Dict[str, List[Tuple[int, float]]],
    names_lower: List[str],
    w: Dict[str, float],
    total_k: int,
    quota_mode: str,
    tie_break: str,
) -> List[Tuple[int, float]]:
    """Internal function for prefix-quota merge logic."""
    
    def quota_fn(weight: float, t: int) -> int:
        x = weight * t
        if quota_mode == "floor":
            return int(math.floor(x))
        if quota_mode == "ceil":
            return int(math.ceil(x))
        if quota_mode == "round":
            return int(round(x))
        raise ValueError(f"Unknown quota_mode: {quota_mode}")

    merged_items: List[Tuple[int, float]] = []
    seen_items = set()
    ptr = {n: 0 for n in names_lower}
    selected = {n: 0 for n in names_lower}

    if all(len(recaller_items.get(n, [])) == 0 for n in names_lower):
        return []

    hard_cap = total_k * 50

    while len(merged_items) < total_k and hard_cap > 0:
        hard_cap -= 1
        t_next = len(merged_items) + 1

        deficits = {n: quota_fn(w[n], t_next) - selected[n] for n in names_lower}

        best = None
        best_def = None
        for n in names_lower:
            d = deficits[n]
            if best is None or d > best_def:
                best, best_def = n, d
            elif d == best_def and tie_break == "next_score":
                i_best = ptr[best]
                i_n = ptr[n]
                s_best = recaller_items[best][i_best][1] if i_best < len(recaller_items[best]) else float("-inf")
                s_n = recaller_items[n][i_n][1] if i_n < len(recaller_items[n]) else float("-inf")
                if s_n > s_best:
                    best = n
                    best_def = d

        items = recaller_items[best]
        while True:
            if ptr[best] >= len(items):
                return merged_items[:total_k]

            item_id, score = items[ptr[best]]
            ptr[best] += 1
            if item_id in seen_items:
                continue

            merged_items.append((item_id, score))
            seen_items.add(item_id)
            selected[best] += 1
            break

    return merged_items[:total_k]


def compute_ndcg_at_k(rec_list: List[int], gt_items: List[int], k: int) -> float:
    """Compute NDCG@k."""
    import math
    if not gt_items:
        return 0.0
    
    gt = set(gt_items) if isinstance(gt_items, list) else {gt_items}
    dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(rec_list[:k]) if item in gt)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt), k)))
    
    return dcg / idcg if idcg > 0 else 0.0


def soft_grpo_reward(
    softmax_outputs: List[torch.Tensor],
    user_ids: List[int],
    histories: List[List[int]],
    ground_truths: List[List[int]],
    recallers: Dict,
    recaller_names: List[str],
    final_k: int = 50
) -> List[float]:
    """
    Compute NDCG rewards using multi-channel recall with softmax weights.
    
    Args:
        softmax_outputs: Softmax vectors from classification
        user_ids: User IDs
        histories: User histories
        ground_truths: Ground truth items
        recallers: Recaller dictionary
        recaller_names: Ordered recaller names
        final_k: Top-k for NDCG
        
    Returns:
        List of NDCG rewards
    """
    rewards = []
    
    for i, weights in enumerate(softmax_outputs):
        try:
            candidates = multi_channel_recall_score(
                weights, recallers, recaller_names,
                user_ids[i], histories[i], final_k
            )
            rec_list = [item_id for item_id, _ in candidates]
            reward = compute_ndcg_at_k(rec_list, ground_truths[i], final_k)
            rewards.append(reward)
        except Exception as e:
            print(f"Reward computation error: {e}")
            rewards.append(0.0)
    
    return rewards


# ============== Original Beta Sampling Functions ==============

def build_soft_template(model_names: List[str], use_top_k: bool = False) -> str:
    """Build JSON template with [num][soft_token] placeholders.
    
    Args:
        model_names: List of model names to include in template
        
    Returns:
        JSON string template with placeholders
    """
    lines = ["{"]
    for i, name in enumerate(model_names):
        lines.append(f'  "{name}": {{')
        if use_top_k:
            lines.append('    "top-k": [num][soft_token],')
        lines.append('    "score-weight": [num][soft_token]')
        lines.append("  }" + ("," if i < len(model_names) - 1 else ""))
    lines.append("}")
    return "\n".join(lines)


def sample_from_logits(logit, clamp_range=(1e-6, 1.0 - 1e-6), sample=True):
    """Sample from logits using sigmoid activation.
    
    Args:
        logit: Raw logit value (unbounded real number)
        clamp_range: Range to clamp sampled values
        sample: Whether to sample from Bernoulli or return sigmoid probability
        
    Returns:
        Sampled value in [0, 1] range
    """
    # Convert logit to probability
    prob = torch.sigmoid(logit).clamp(*clamp_range)
    
    if sample:
        # Sample from Bernoulli distribution
        dist = torch.distributions.Bernoulli(prob)
        return dist.sample().item()
    else:
        # Return probability directly
        return prob.item()


def sample_from_beta_params(alpha, beta, clamp_range=(1e-6, 1.0 - 1e-6), sample=True, apply_transform=True):
    """Sample from Beta distribution with given parameters.
    
    Args:
        alpha: Alpha parameter
        beta: Beta parameter  
        clamp_range: Range to clamp sampled values
        sample: Whether to sample or return mean/mode
        apply_transform: Whether to apply softplus + 1.0 transform to alpha/beta
        
    Returns:
        Sampled value from Beta distribution
    """
    # Process parameters if needed
    if apply_transform:
        alpha_processed = torch.nn.functional.softplus(alpha) + 1.0
        beta_processed = torch.nn.functional.softplus(beta) + 1.0
    else:
        alpha_processed = alpha
        beta_processed = beta
    
    # Sample and clamp
    if sample:
        dist = torch.distributions.Beta(alpha_processed, beta_processed)
        return dist.sample().clamp(*clamp_range).item()
    else:
        # return mean value
        # return alpha_processed / (alpha_processed + beta_processed)
        return (alpha_processed - 1) / (alpha_processed + beta_processed - 2)
        # TODO: try (alpha_processed - 1) / (alpha_processed + beta_processed - 2)


def replace_placeholders_with_values(template: str, values: List[float], precision: int = 6) -> str:
    """Replace [num][soft_token] placeholders with numeric values.
    
    Args:
        template: Template string containing placeholders
        values: List of numeric values to substitute
        precision: Number of decimal places for formatting
        
    Returns:
        Template with placeholders replaced by values
    """
    result = template
    for val in values:
        result = result.replace("[num][soft_token]", f"{float(val):.{precision}f}", 1)
    
    # Fill remaining placeholders with default value
    while "[num][soft_token]" in result:
        result = result.replace("[num][soft_token]", f"{0.0:.{precision}f}", 1)
    
    return result


def generate_soft_completions(model, tokenizer, test_dataset, model_names: List[str], max_length: int = 1536) -> List[str]:
    """Generate completions using soft token beta sampling.
    
    Args:
        model: Model with value_head for beta parameter prediction
        tokenizer: Tokenizer with [num] and [soft_token] special tokens
        test_dataset: Dataset containing prompts
        model_names: List of model names for template
        max_length: Maximum sequence length
        
    Returns:
        List of generated completion strings
    """
    completions = []
    device = next(model.parameters()).device
    model.eval()
    model.to(device)
    soft_token_id = tokenizer.convert_tokens_to_ids("[soft_token]")
    num_token_id = tokenizer.convert_tokens_to_ids("[num]")
    
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="Generating completions"):
            # Extract prompt
            prompt = test_dataset[i]['prompt']
            prompt_text = prompt[0]['content'] if isinstance(prompt, list) and isinstance(prompt[0], dict) else str(prompt)
            
            # Build full text with template
            template = build_soft_template(model_names)
            full_text = prompt_text + "\n\n" + template
            
            # Tokenize and forward pass
            inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            # Need to explicitly request hidden states
            outputs = model(**inputs, output_hidden_states=True)
            
            # Sample values from logits (following new BCE logic)
            sampled_values = []
            if hasattr(model, 'value_head') and hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                input_ids = inputs['input_ids'][0]
                num_positions = (input_ids == num_token_id).nonzero(as_tuple=True)[0]
                soft_positions = (input_ids == soft_token_id).nonzero(as_tuple=True)[0]
                
                if len(num_positions) > 0 and len(soft_positions) > 0:
                    # Get hidden states at [num] positions
                    hidden_states = outputs.hidden_states[-1][0, num_positions, :]  # (K, H)
                    value_preds = model.value_head(hidden_states)  # (K, 2)
                    
                    # Sample from logits (only use first output as logit)
                    for pred in value_preds:
                        logit = pred[0]  # Only use the first output (mu)
                        sampled_values.append(sample_from_logits(logit))
            
            # Use default values if no sampling occurred
            if not sampled_values:
                num_soft_tokens = template.count("[num][soft_token]")
                sampled_values = [0.5] * num_soft_tokens
            
            # Replace placeholders with sampled values
            completion_text = replace_placeholders_with_values(template, sampled_values)
            completions.append(completion_text)
    
    return completions
