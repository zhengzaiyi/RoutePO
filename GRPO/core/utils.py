import random
import numpy as np
import torch
import math
import json
from typing import List, Dict, Optional
from collections import defaultdict
from GRPO.core.recallers import RecBoleRecaller

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ndcg_at_k(rec_list: List[int], gt_items: List[int], k: int) -> float:
    """Calculate NDCG@K (Normalized Discounted Cumulative Gain)"""
    if not gt_items: return 0.0
    k = max(1, k)
    if type(gt_items) == int:
        gt_items = [gt_items]
    gt = set(gt_items)
    
    # Calculate DCG@K
    dcg = 0.0
    for i, item in enumerate(rec_list[:k]):
        if item in gt:
            # relevance = 1 for binary relevance (item is relevant or not)
            # position discount = 1 / log2(i + 2) where i is 0-indexed
            dcg += 1.0 / math.log2(i + 2)
    
    # Calculate IDCG@K (Ideal DCG)
    # In binary relevance case, IDCG is the DCG when all relevant items are at the top
    num_relevant = min(len(gt), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(num_relevant))
    
    # Return NDCG@K
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(rec_list: List[int], gt_items: List[int], k: int) -> float:
    if not gt_items: return 0.0
    k = max(1, k)
    gt = set(gt_items)
    hit = sum(1 for x in rec_list[:k] if x in gt)
    return hit / min(len(gt), k)

def merge_candidates(lists: List[List[int]], ws: List[float], final_k: int) -> List[int]:
    merged = []
    # set scores for each item in each list
    scores = []
    for list_ in lists:
        scores.append({item: i for i, item in enumerate(list_)})
    # weight scores
    for i in range(len(scores)):
        for item in scores[i]:
            scores[i][item] *= ws[i]
    # merge items by scores
    merged = []
    for i in range(final_k):
        max_score = -1
        max_item = None
        for list_ in scores:
            if list_ and list_[0] in list_:
                max_score = list_[0]
                max_item = list_[0]
                break
        if max_item is not None:
            merged.append(max_item)
            scores[scores.index(max_score)] = scores[scores.index(max_score)][1:]
    return merged

def average_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics_list: return {}
    keys = [k for k in metrics_list[0].keys()]
    out = {}
    for k in keys:
        out[k] = float(np.mean([m.get(k, 0.0) for m in metrics_list]))
    return out


TOOLS_DESCRIPTION = {
    "sasrec": {
        "description": "Transformer-based sequential model for short-term interest patterns",
        "when_to_use": "Recent interaction sequences and short-term preferences",
    },
    "bpr": {
        "description": "Matrix factorization for general user-item preference rankings", 
        "when_to_use": "Long-term preferences without sequence order",
    },
    "lightgcn": {
        "description": "Graph neural network leveraging user-item connectivity",
        "when_to_use": "Graph structures and higher-order neighbor relations",
    },
    "pop": {
        "description": "Non-personalized popularity-based recommendations",
        "when_to_use": "Baseline or cold-start situations",
    },
    "itemknn": {
        "description": "Item-based collaborative filtering using similarity",
        "when_to_use": "Item similarity patterns and co-interaction signals",
    },
    "fpmc": {
        "description": "Hybrid model combining matrix factorization + Markov chains",
        "when_to_use": "Next-item prediction with both long-term and sequential behavior",
    }
}

def build_prompt(
    profile_json: str, 
    available_models: List[str] = None, 
    models_with_items: List[Dict] = None,
    hint: str = None,
    type: str = 'json'
) -> str:
    if available_models is None:
        available_models = ['sasrec', 'bpr', 'pop']
    
    # Build models section
    if models_with_items:
        models_str = "\n".join([
            f"- {m['name']}: {m['description']}. When to use: {m['when_to_use']}\n  Top retrieved items: {json.dumps(m['top_retrieved_items'], ensure_ascii=False)}"
            for m in models_with_items
        ])
        models_section = f"Available models:\n{models_str}\n\nTie-breaking rule: If multiple models retrieve the correct item, prefer the one with a higher rank (smaller rank number)."
    else:
        models_section = f"Available models: \n{[m for m in available_models]}"
    
    # Build hint section
    hint_section = f"\nNote: Based on this user's historical interactions, {hint} has shown the best performance." if hint else ""
    
    if type == 'json':
        return (
            "You are a multi-channel recall assistant in a recommendation system. Given a user profile, "
            "output ONLY a JSON file describe the score-weights of different models during the multi-channel recall."
            f"Available models: \n{[json.dumps(TOOLS_DESCRIPTION[m.lower()], indent=2) for m in available_models]}\n"
            f"User Profile:\n{profile_json}\n"
            f"Please output the JSON file containing the score-weights of ALL available models."
            "Your JSON response:"
        )
    elif type == 'classification':
        return (
            "You are an assistant in a recommendation system. Given a user profile, "
            "output ONLY the best recaller model name from the available models.\n"
            f"User Profile:\n{profile_json}\n"
            f"{models_section}"
            f"{hint_section}\n"
            "Your response:"
        )
    elif type == 'instruction':
        return (
            "You are an assistant in a recommendation system. Given a user profile, "
            "output ONLY the best recaller model name from the available models.\n"
            f"User Profile:\n{profile_json}\n"
            f"{models_section}"
            f"{hint_section}\n"
            f"Your response (only output the best recaller model name from {available_models}, no other text):"
        )


def ndcg_rewards(
    completions: List[List[dict]], 
    uid: List[int], 
    histories: List[List[int]], 
    target_items: List[List[int]], 
    recallers: Dict[str, RecBoleRecaller], 
    final_k: int, 
    **kwargs
):
    """Abandoned - Calculate NDCG rewards using only score-weights"""
    rewards = []
    
    for i, completion_msgs in enumerate(completions):
        try:
            # Extract routing content
            content = completion_msgs[-1]["content"]
            model_configs = json.loads(content)
            
            # Calculate recommendation results
            candidates = defaultdict(float)
            for recaller in recallers.keys():
                w = model_configs[recaller]['score-weight']
                
                if recaller in recallers:
                    # Get full item list
                    items = recallers[recaller].recall(uid[i], final_k, histories[i])
                    for item in items:
                        candidates[item[0]] += item[1] * w
            
            # Calculate NDCG
            candidates = sorted(candidates.keys(), key=lambda x: candidates[x], reverse=True)[:final_k]
            ndcg = ndcg_at_k(candidates, target_items[i], k=final_k)
            rewards.append(ndcg)
        except Exception as e:
            print(f"Error processing completion: {e}")
            rewards.append(0)
    
    return rewards

def multi_channel_recall(
    completions: List[List[dict]], 
    uid: List[int], 
    histories: List[List[int]], 
    recallers: Dict[str, RecBoleRecaller], 
    total_item: int, 
) -> List[List[int]]:
    """Calculate multi-channel recall using only score-weights"""
    recall_results = []
    for i, completion_msgs in enumerate(completions):
        try:
            if isinstance(completion_msgs, str):
                content = completion_msgs
            else:
                content = completion_msgs[-1]["content"]
            model_configs = json.loads(content)
            
            # Normalize model names to lowercase for consistency
            normalized_configs = {}
            for model_name, config in model_configs.items():
                normalized_configs[model_name.lower()] = config
                
            # Calculate recommendation results
            candidates = defaultdict(float)
            assert set(normalized_configs.keys()) <= set(recallers.keys()), f"Model configs keys {normalized_configs.keys()} are not in recallers keys {recallers.keys()}"
            
            for recaller in normalized_configs.keys():
                w = normalized_configs[recaller]['score-weight']
                
                # Get full item list from recaller
                items = recallers[recaller].recall(uid[i], total_item, histories[i])
                for item in items:
                    candidates[item[0]] += item[1] * w
            
            # Sort by weighted scores and return top items
            candidates = sorted(candidates.keys(), key=lambda x: candidates[x], reverse=True)[:total_item]
            recall_results.append(candidates)
        except Exception as e:
            print(f"Error processing completion: {e}")
            recall_results.append([])
    return recall_results

def average_recall(
    recallers: Dict[str, RecBoleRecaller],
    uid: List[int],
    histories: List[List[int]],
    final_k: int,
    normalize: Optional[str] = None,
) -> List[List[int]]:
    recall_results = []
    for i, user in enumerate(uid):
        total_scores = 0
        for recaller in recallers.keys():
            scores = recallers[recaller].full_sort_predict(user, histories[i])
            scores = torch.tensor(scores)
            if normalize == 'sigmoid':
                scores = torch.sigmoid(scores)
            elif normalize == 'tanh':
                scores = torch.tanh(scores)
            elif normalize == 'softmax':
                scores = torch.softmax(scores, dim=-1)
            scores = scores.numpy()
            total_scores = scores + total_scores
        topk_indices = np.argsort(total_scores)[::-1][:final_k]
        recall_results.append(topk_indices)
    return recall_results

def evaluate(
    completions: List[List[dict]], 
    uid: List[int], 
    histories: List[List[int]], 
    recallers: Dict[str, RecBoleRecaller], 
    final_k: int, 
    target_items: List[List[int]],
    ks: List[int] = [10, 50],
) -> Dict[str, float]:
    recall_results = multi_channel_recall(
        completions=completions, 
        uid=uid, 
        histories=histories, 
        recallers=recallers, 
        total_item=final_k
    )
    metrics = defaultdict(list)
    for i, recall_result in enumerate(recall_results):
        for k in ks:
            metrics[f"ndcg@{k}"].append(ndcg_at_k(recall_result, target_items[i], k=k))
            metrics[f"recall@{k}"].append(recall_at_k(recall_result, target_items[i], k=k))
    return {
        k: sum(v) / len(v) for k, v in metrics.items()
    }

def jaccard_similarity(list1: List[int], list2: List[int], k: int = None) -> float:
    """
    计算两个推荐列表的 Jaccard 相似度
    
    Args:
        list1
        list2
        k
    
    Returns:
        Jaccard 相似度 (0-1)
    """
    if k is not None:
        list1 = list1[:k]
        list2 = list2[:k]
    
    set1 = set(list1)
    set2 = set(list2)
    
    assert set1 and set2, "set1 and set2 are empty"
    
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    return len(intersection) / len(union) if union else 0.0


def rbo_similarity(list1: List[int], list2: List[int], p: float = 0.9) -> float:
    """
    Calculate the RBO (Rank-Biased Overlap) similarity between two sorted lists
    
    Args:
        list1: The first sorted list
        list2: The second sorted list
        p: Continuation probability parameter (0-1),
    
    Returns:
        RBO similarity (0-1)
    """
    def overlap_at_k(list1: List[int], list2: List[int], k: int) -> float:
        return len(set(list1[:k]).intersection(set(list2[:k]))) / k
    
    min_len = min(len(list1), len(list2))
    
    if min_len == 0:
        return 0.0
    
    rbo_sum = 0.0
    for k in range(1, min_len + 1):
        rbo_sum += overlap_at_k(list1, list2, k) * (p ** (k - 1))
    
    # Handle the case of unequal lengths
    if len(list1) != len(list2):
        extrapolated_overlap = overlap_at_k(list1, list2, min_len)
        rbo_sum += extrapolated_overlap * (p ** min_len) * ((1 - p) ** -1)
    else:
        rbo_sum += (p ** min_len) / (1 - p) * overlap_at_k(list1, list2, min_len)
    
    return (1 - p) * rbo_sum

def evaluate_recallers(
    recallers: Dict[str, RecBoleRecaller],
    uid: List[int],
    histories: List[List[int]],
    target_items: List[List[int]],
    final_k: int,
    ks: List[int] = [10, 50],
) -> Dict[str, float]:
    metrics = {recaller: defaultdict(list) for recaller in recallers.keys()}
    metrics['optimal'] = defaultdict(list)
    metrics['average'] = defaultdict(list)
    metrics['static'] = defaultdict(list)
    all_recommendations = {recaller: [] for recaller in recallers.keys()}
    for i in range(len(uid)):
        for recaller_name in recallers.keys():
            items = recallers[recaller_name].recall(uid[i], int(final_k), histories[i])
            item_ids = [item[0] for item in items] if items else []
            all_recommendations[recaller_name].append(item_ids)
            for k in ks:
                ndcg = ndcg_at_k(item_ids, target_items[i], k=k)
                recall = recall_at_k(item_ids, target_items[i], k=k)
                metrics[recaller_name][f"ndcg@{k}"].append(ndcg)
                metrics[recaller_name][f"recall@{k}"].append(recall)
    
    average_recall_results = average_recall(
        recallers=recallers,
        uid=uid,
        histories=histories,
        final_k=final_k,
    )
    static_recall_results = average_recall(
        recallers=recallers,
        uid=uid,
        histories=histories,
        final_k=final_k,
        normalize='softmax'
    )
    
    for i in range(len(uid)):
        for k in ks:
            metrics['average'][f"ndcg@{k}"].append(ndcg_at_k(average_recall_results[i], target_items[i], k=k))
            metrics['average'][f"recall@{k}"].append(recall_at_k(average_recall_results[i], target_items[i], k=k))
            metrics['static'][f"ndcg@{k}"].append(ndcg_at_k(static_recall_results[i], target_items[i], k=k))
            metrics['static'][f"recall@{k}"].append(recall_at_k(static_recall_results[i], target_items[i], k=k))
    for i in range(len(uid)):
        last_k = 10
        best_ndcg = -1
        for recaller_name in recallers.keys():
            if metrics[recaller_name][f"recall@{last_k}"][i] > best_ndcg:
                best_ndcg = metrics[recaller_name][f"ndcg@{last_k}"][i]
                best_recaller = recaller_name
        for k in ks:
            metrics['optimal'][f"ndcg@{k}"].append(metrics[best_recaller][f"ndcg@{k}"][i])
            metrics['optimal'][f"recall@{k}"].append(metrics[best_recaller][f"recall@{k}"][i])


    result = {}
    for recaller_name in metrics.keys():
        result[recaller_name] = {}
        for metric_name, values in metrics[recaller_name].items():
            result[recaller_name][metric_name] = sum(values) / len(values) if values else 0.0

    # compute jaccard similarity and rbo similarity
    recaller_names = list(recallers.keys())
    similarity_metrics = {
        'jaccard': defaultdict(list),
        'rbo': defaultdict(list)
    }
    
    # Compute the similarity between recallers for each user
    from tqdm import tqdm
    # for i in tqdm(range(len(uid)), desc='Computing similarity'):
    #     # Compute the similarity between all recallers
    #     for j, recaller1 in enumerate(recaller_names):
    #         for k, recaller2 in enumerate(recaller_names):
    #             if j < k:  # Avoid duplicate calculations
    #                 list1 = all_recommendations[recaller1][i]
    #                 list2 = all_recommendations[recaller2][i]
                    
    #                 # Compute Jaccard similarity (for the entire list)
    #                 jaccard = jaccard_similarity(list1, list2, k=10)
    #                 similarity_metrics['jaccard'][f"{recaller1}_vs_{recaller2}"].append(jaccard)
                    
    #                 # Compute RBO similarity
    #                 rbo = rbo_similarity(list1, list2, p=0.9)
    #                 similarity_metrics['rbo'][f"{recaller1}_vs_{recaller2}"].append(rbo)
    # result['similarities'] = {
    #     'jaccard': {},
    #     'rbo': {}
    # }
    # for sim_type in ['jaccard', 'rbo']:
    #     for pair_name, values in similarity_metrics[sim_type].items():
    #         avg_similarity = sum(values) / len(values) if values else 0.0
    #         result['similarities'][sim_type][pair_name] = avg_similarity
        
    #     # Compute the overall average similarity for all pairs
    #     all_values = []
    #     for values in similarity_metrics[sim_type].values():
    #         all_values.extend(values)
    #     if all_values:
    #         result['similarities'][sim_type]['average'] = sum(all_values) / len(all_values)
    #     else:
    #         result['similarities'][sim_type]['average'] = 0.0        
    return result