import math
import torch
from typing import Dict, List, Tuple, Callable, Optional

# -----------------------------
# Utilities
# -----------------------------

def _ensure_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, device=device)

def dirichlet_sample(alpha: torch.Tensor, n: int) -> torch.Tensor:
    """
    Sample n weight vectors from Dirichlet(alpha).
    alpha: (K,)
    returns: (n, K), each row sums to 1.
    """
    dist = torch.distributions.Dirichlet(alpha)
    return dist.sample((n,))

def dirichlet_mle_alpha(elite_w: torch.Tensor, iters: int = 200, lr: float = 0.05, eps: float = 1e-8) -> torch.Tensor:
    """
    Approximate MLE of Dirichlet parameters alpha given samples elite_w.
    elite_w: (E, K) on simplex.
    Returns alpha_hat: (K,)
    Notes:
      - We optimize alpha directly with gradient descent on negative log-likelihood.
      - For stability, we parameterize alpha = softplus(raw) + eps.
    """
    device = elite_w.device
    E, K = elite_w.shape

    # clamp weights to avoid log(0)
    w = elite_w.clamp_min(eps)

    raw = torch.zeros(K, device=device, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=lr)

    for _ in range(iters):
        alpha = torch.nn.functional.softplus(raw) + eps  # (K,)
        alpha0 = alpha.sum()

        # log-likelihood for Dirichlet:
        # L = E[ log Gamma(alpha0) - sum log Gamma(alpha_k) + sum (alpha_k-1) log w_k ]
        # We maximize L; so minimize -L.
        loglik = (
            torch.lgamma(alpha0)
            - torch.lgamma(alpha).sum()
            + ((alpha - 1.0) * w.log()).sum(dim=1).mean()
        )

        loss = -loglik
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        alpha_hat = torch.nn.functional.softplus(raw) + eps
    return alpha_hat

def fuse_by_quota(
    user_candidates: List[List[List[int]]],
    w: torch.Tensor,
    L: int,
) -> List[List[int]]:
    """
    Quota-based fusion (list-level union):
      - For each user u and channel k, take top q_{u,k} items, where q_{u,k} = round(w_k * L)
      - Union them in a stable way (priority by channel order, then rank within channel).
    user_candidates: list over users, each is list over channels, each is list of item ids length M_k
    w: (K,) sums to 1
    L: desired total candidate budget
    Returns fused list per user, up to length L (deduped).
    """
    K = len(user_candidates[0])
    quotas = torch.round(w * L).to(torch.int64).tolist()

    # Fix rounding so total == L
    s = sum(quotas)
    if s != L:
        # distribute the difference by largest fractional parts
        # compute fractional parts from w*L
        frac = (w * L) - torch.floor(w * L)
        order = torch.argsort(frac, descending=(s < L)).tolist()
        diff = abs(L - s)
        for i in range(diff):
            k = order[i % K]
            quotas[k] += 1 if s < L else -1
        # safety
        quotas = [max(0, q) for q in quotas]

    fused = []
    for u in range(len(user_candidates)):
        seen = set()
        out = []
        for k in range(K):
            take = quotas[k]
            for item in user_candidates[u][k][:take]:
                if item not in seen:
                    seen.add(item)
                    out.append(item)
                if len(out) >= L:
                    break
            if len(out) >= L:
                break
        fused.append(out)
    return fused

def recall_at_L(
    fused: List[List[int]],
    ground_truth: List[set],
) -> float:
    """
    Compute average Recall@L across users:
      recall_u = |pred ∩ gt| / |gt|
    fused: list of predicted item lists
    ground_truth: list of gt item sets
    """
    assert len(fused) == len(ground_truth)
    recs = []
    for pred, gt in zip(fused, ground_truth):
        if len(gt) == 0:
            continue
        hit = 0
        for it in pred:
            if it in gt:
                hit += 1
        recs.append(hit / len(gt))
    return float(sum(recs) / max(1, len(recs)))

# -----------------------------
# Main: CEM optimizer
# -----------------------------

def cem_optimize_fusion_weights(
    user_candidates: List[List[List[int]]],
    ground_truth: List[set],
    L: int,
    K: int,
    device: str = "cuda",
    iters: int = 20,
    population: int = 256,
    elite_frac: float = 0.1,
    alpha_init: float = 1.0,
    alpha_smooth: float = 0.7,   # EMA smoothing on alpha
    mle_iters: int = 200,
    mle_lr: float = 0.05,
    seed: int = 42,
) -> Tuple[torch.Tensor, float, Dict]:
    """
    Returns:
      w_best: (K,)
      best_score: recall@L
      info: dict with history
    """
    torch.manual_seed(seed)
    device = torch.device(device)

    alpha = torch.full((K,), float(alpha_init), device=device)  # Dirichlet parameters
    best_w = None
    best_score = -1e9

    hist = {"best_score": [], "mean_score": [], "alpha": [], "best_w": []}

    for t in range(iters):
        with torch.no_grad():
            W = dirichlet_sample(alpha, population)  # (P, K)

            scores = []
            for i in range(population):
                w = W[i]
                fused = fuse_by_quota(user_candidates, w, L)
                s = recall_at_L(fused, ground_truth)
                scores.append(s)
            scores_t = _ensure_tensor(scores, device=device)  # (P,)

            mean_s = float(scores_t.mean().item())
            topk = max(1, int(math.ceil(population * elite_frac)))
            elite_idx = torch.topk(scores_t, k=topk, largest=True).indices
            elite_w = W[elite_idx]  # (E, K)

            # Track best
            elite_best_score = float(scores_t[elite_idx[0]].item())
            elite_best_w = W[elite_idx[0]]
            if elite_best_score > best_score:
                best_score = elite_best_score
                best_w = elite_best_w.clone()

        # MLE update alpha from elite (need gradients for optimization)
        alpha_hat = dirichlet_mle_alpha(elite_w, iters=mle_iters, lr=mle_lr)
        # Smooth update (EMA)
        with torch.no_grad():
            alpha = alpha_smooth * alpha + (1.0 - alpha_smooth) * alpha_hat

        hist["best_score"].append(best_score)
        hist["mean_score"].append(mean_s)
        hist["alpha"].append(alpha.detach().cpu())
        hist["best_w"].append(best_w.detach().cpu())

        print(f"[CEM] iter={t:02d} mean={mean_s:.4f} elite_best={elite_best_score:.4f} global_best={best_score:.4f}")

    # Normalize best_w (should already sum to 1)
    with torch.no_grad():
        best_w = best_w / best_w.sum()
    return best_w.detach().cpu(), best_score, hist

# -----------------------------
# RecBole integration: data hooks
# -----------------------------

def build_user_candidates_from_recalls(
    user_ids: List[int],
    recallers: List[Callable[[List[int], int], List[List[int]]]],
    M: int,
) -> List[List[List[int]]]:
    """
    recallers: list of callables, each takes (user_ids, topk) and returns list of item-id lists per user.
              You can wrap RecBole models here.
    Returns user_candidates[u][k] = list of items for user u from channel k.
    """
    per_channel = []
    for rec in recallers:
        cand = rec(user_ids, M)  # List[List[int]] length = len(user_ids)
        assert len(cand) == len(user_ids)
        per_channel.append(cand)

    # transpose to user-major
    user_candidates = []
    for ui in range(len(user_ids)):
        user_candidates.append([per_channel[k][ui] for k in range(len(recallers))])
    return user_candidates
