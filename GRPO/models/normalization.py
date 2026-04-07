"""
Score normalization utilities for multi-channel recall.
Extracted to avoid code duplication across different recall functions.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional


def normalize_scores(
    items: List[Tuple[int, float]], 
    method: Optional[str] = None
) -> List[Tuple[int, float]]:
    """
    Normalize scores in a list of (item_id, score) tuples.
    
    Args:
        items: List of (item_id, score) tuples
        method: Normalization method. Options:
            - None: No normalization (returns original items)
            - 'minmax': Min-Max normalization to [0, 1]
            - 'zscore': Z-score normalization (mean=0, std=1)
            - 'softmax': Softmax normalization (sum to 1)
    
    Returns:
        List of (item_id, normalized_score) tuples with same order
    """
    if not items or method is None:
        return items
    
    item_ids = [item[0] for item in items]
    scores = [item[1] for item in items]
    
    if method == 'minmax':
        min_score, max_score = min(scores), max(scores)
        if max_score > min_score:
            normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]
        else:
            # All same score, set to 0.5
            normalized_scores = [0.5] * len(scores)
    elif method == 'zscore':
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        if std_score > 1e-8:
            normalized_scores = [(s - mean_score) / std_score for s in scores]
        else:
            # All same score, set to 0
            normalized_scores = [0.0] * len(scores)
    elif method == 'softmax':
        # Convert to tensor for softmax
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        normalized_scores = torch.softmax(scores_tensor, dim=0).tolist()
    else:
        # Unknown method, return original
        return items
    
    # Return normalized items with same order
    return list(zip(item_ids, normalized_scores))

