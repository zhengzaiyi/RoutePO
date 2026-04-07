"""
Evaluation utilities for recaller evaluation.
Extracted to avoid code duplication across different evaluation functions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict
from tqdm import tqdm
from GRPO.core.recallers import RecBoleRecaller
from GRPO.core.utils import ndcg_at_k, recall_at_k


def extract_eval_data(example: Dict[str, Any], idx: int = 0) -> Tuple[int, List[int], List[int], Optional[List[int]]]:
    """
    Extract evaluation data from an example.
    
    Args:
        example: Example dictionary from dataset
        idx: Index as fallback for user_id
    
    Returns:
        Tuple of (user_id, eval_hist, gt_items, full_hist)
    """
    user_id = example.get("user_id", idx)
    eval_hist = example.get("history") or example.get("eval_hist", [])
    gt_items = example.get("target_items") or example.get("gt_items") or example.get("ground_truth", [])
    full_hist = example.get("full_hist")
    return user_id, eval_hist, gt_items, full_hist


def evaluate_base_recallers(
    instances: List[Dict[str, Any]],
    recallers: Dict[str, RecBoleRecaller],
    recaller_names: List[str],
    final_k: int,
    ks: List[int] = [10, 20, 50],
    use_tqdm: bool = True
) -> Dict[str, Dict[str, List[float]]]:
    """
    Evaluate base recallers on a list of instances.
    
    Args:
        instances: List of evaluation instances, each containing:
            - user_id: int
            - eval_hist or history: List[int] - evaluation history
            - gt_items or ground_truth or target_items: List[int] - ground truth items
            - full_hist: Optional[List[int]] - full history for masking
        recallers: Dictionary of recaller name -> RecBoleRecaller
        recaller_names: List of recaller names to evaluate
        final_k: Final k for recall
        ks: List of k values for metrics calculation
        use_tqdm: Whether to use tqdm progress bar
    
    Returns:
        Dictionary mapping recaller_name -> {metric_name: [values]}
    """
    metrics = {recaller_name: defaultdict(list) for recaller_name in recaller_names}
    
    for recaller_name in recaller_names:
        if recaller_name not in recallers:
            continue
        
        recaller = recallers[recaller_name]
        iterator = tqdm(instances, desc=f"Evaluating {recaller_name}", leave=False) if use_tqdm else instances
        
        for instance in iterator:
            # Extract data from instance (handle different field names)
            user_id = instance.get('user_id')
            eval_hist = instance.get('eval_hist') or instance.get('history', [])
            gt_items = instance.get('gt_items') or instance.get('ground_truth') or instance.get('target_items', [])
            full_hist = instance.get('full_hist')
            
            if len(eval_hist) < 5:
                continue
            
            # Generate recommendations
            items = recaller.recall(user_id, final_k, eval_hist, full_hist=full_hist, gt_items=gt_items)
            item_ids = [item[0] for item in items] if items else []
            
            # Calculate metrics for each k
            for k in ks:
                metrics[recaller_name][f"ndcg@{k}"].append(ndcg_at_k(item_ids, gt_items, k))
                metrics[recaller_name][f"recall@{k}"].append(recall_at_k(item_ids, gt_items, k))
    
    return metrics


def aggregate_metrics(metrics: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics by taking mean of lists.
    
    Args:
        metrics: Dictionary mapping method_name -> {metric_name: [values]}
    
    Returns:
        Dictionary mapping method_name -> {metric_name: mean_value}
    """
    results = {}
    for method, method_metrics in metrics.items():
        results[method] = {}
        for metric_name, values in method_metrics.items():
            if values:
                results[method][metric_name] = np.mean(values)
    return results


def print_base_recaller_table(
    results: Dict[str, Dict[str, float]],
    recaller_names: List[str],
    title: str = "Base Recaller Performance"
):
    """
    Print a formatted table showing base recaller performance.
    
    Args:
        results: Aggregated results dictionary
        recaller_names: List of recaller names to display
        title: Title for the table section
    """
    print(f"\n--- {title} ---")
    header = f"{'Metric':<15}" + "".join([f"{name:>15}" for name in recaller_names])
    print(header)
    print("-" * (15 + 15 * len(recaller_names)))
    
    for k in [10, 20, 50]:
        for metric in ['ndcg', 'recall']:
            key = f"{metric}@{k}"
            row = f"{key:<15}"
            for recaller_name in recaller_names:
                val = results.get(recaller_name, {}).get(key, 0)
                row += f"{val:>15.4f}"
            print(row)


def find_best_base_model(
    results: Dict[str, Dict[str, float]],
    recaller_names: List[str],
    metric_key: str = "ndcg@50"
) -> Tuple[str, float]:
    """
    Find the best base model based on a metric.
    
    Args:
        results: Aggregated results dictionary
        recaller_names: List of recaller names to consider
        metric_key: Metric key to use for comparison (e.g., "ndcg@50")
    
    Returns:
        Tuple of (best_model_name, best_metric_value)
    """
    best_value = 0
    best_name = None
    
    for recaller_name in recaller_names:
        value = results.get(recaller_name, {}).get(metric_key, 0)
        if value > best_value:
            best_value = value
            best_name = recaller_name
    
    return best_name, best_value


def compute_performance_gaps(
    results: Dict[str, Dict[str, float]],
    recaller_names: List[str],
    eval_k: int,
    method_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute NDCG/Recall gaps vs best base for given methods.
    results: aggregated metrics (method -> {ndcg@k, recall@k}); must include base recaller names.
    Returns dict with eval_k, best_base_ndcg, best_base_recall, and per method_key: ndcg_gap_vs_best_base, recall_gap_vs_best_base.
    """
    method_keys = method_keys or ["single_select", "multi_channel"]
    gap_k = eval_k if eval_k in [10, 20, 50] else 50
    ndcg_key = f"ndcg@{gap_k}"
    recall_key = f"recall@{gap_k}"
    best_base_ndcg = max(results.get(r, {}).get(ndcg_key, 0) for r in recaller_names)
    best_base_recall = max(results.get(r, {}).get(recall_key, 0) for r in recaller_names)
    out = {
        "eval_k": gap_k,
        "best_base_ndcg": best_base_ndcg,
        "best_base_recall": best_base_recall,
    }
    for key in method_keys:
        m = results.get(key, {})
        out[key] = {
            "ndcg_gap_vs_best_base": m.get(ndcg_key, 0) - best_base_ndcg,
            "recall_gap_vs_best_base": m.get(recall_key, 0) - best_base_recall,
        }
    return out


def print_performance_gaps(
    performance_gaps: Dict[str, Any],
    method_labels: Optional[Dict[str, str]] = None,
) -> None:
    """Print a compact 'Performance Gaps vs Best Base' section."""
    method_labels = method_labels or {"single_select": "Single", "multi_channel": "Multi-Ch"}
    gap_k = performance_gaps.get("eval_k", 50)
    best_ndcg = performance_gaps.get("best_base_ndcg", 0)
    best_recall = performance_gaps.get("best_base_recall", 0)
    print(f"\n--- Gaps vs Best Base (k={gap_k}) | NDCG@{gap_k}={best_ndcg:.4f} Recall@{gap_k}={best_recall:.4f} ---")
    for key, label in method_labels.items():
        g = performance_gaps.get(key, {})
        if not g:
            continue
        ndcg_g = g.get("ndcg_gap_vs_best_base", 0)
        rec_g = g.get("recall_gap_vs_best_base", 0)
        print(f"  {label}: NDCG {ndcg_g:+.4f}  Recall {rec_g:+.4f}")


def print_comparison_table(
    results: Dict[str, Dict[str, float]],
    recaller_names: List[str],
    methods: List[str],
    method_labels: Optional[Dict[str, str]] = None,
    title: str = "Comparison"
):
    """
    Print a comparison table between different methods and base recallers.
    
    Args:
        results: Aggregated results dictionary
        recaller_names: List of base recaller names
        methods: List of method names to compare (e.g., ["single_select", "multi_channel", "avg_score_weight"])
        method_labels: Optional mapping from method name to display label
        title: Title for the table section
    """
    if method_labels is None:
        method_labels = {m: m for m in methods}
    
    print(f"\n--- {title} ---")
    
    # Build header - match original format
    if len(methods) == 3:
        # Special format for single_select, multi_channel, avg_score_weight
        header = f"{'Metric':<12} {'Single':>12} {'Multi-Ch':>12} {'Avg-SW':>12} {'Best Base':>12} {'Multi Impr':>12} {'Avg Impr':>12}"
        print(header)
        print("-" * 84)
        
        for k in [10, 20, 50]:
            for metric in ['ndcg', 'recall']:
                key = f"{metric}@{k}"
                single = results.get(methods[0], {}).get(key, 0) if len(methods) > 0 else 0
                multi = results.get(methods[1], {}).get(key, 0) if len(methods) > 1 else 0
                avg_sw = results.get(methods[2], {}).get(key, 0) if len(methods) > 2 else 0
                best_base = max(results.get(name, {}).get(key, 0) for name in recaller_names)
                multi_impr = multi - best_base
                avg_impr = avg_sw - best_base
                print(f"{key:<12} {single:>12.4f} {multi:>12.4f} {avg_sw:>12.4f} {best_base:>12.4f} {multi_impr:>+12.4f} {avg_impr:>+12.4f}")
    else:
        # Generic format
        header_parts = ["Metric"]
        for method in methods:
            header_parts.append(method_labels.get(method, method))
        header_parts.extend(["Best Base", "Improvement"])
        
        header = " ".join([f"{part:>12}" for part in header_parts])
        print(header)
        print("-" * len(header))
        
        for k in [10, 20, 50]:
            for metric in ['ndcg', 'recall']:
                key = f"{metric}@{k}"
                row_parts = [key]
                
                # Get method values
                for method in methods:
                    val = results.get(method, {}).get(key, 0)
                    row_parts.append(f"{val:.4f}")
                
                # Get best base
                best_base = max(results.get(name, {}).get(key, 0) for name in recaller_names)
                row_parts.append(f"{best_base:.4f}")
                
                # Calculate improvement (using first method)
                if methods:
                    first_method_val = results.get(methods[0], {}).get(key, 0)
                    improvement = first_method_val - best_base
                    row_parts.append(f"{improvement:+.4f}")
                
                row = " ".join([f"{part:>12}" for part in row_parts])
                print(row)


def print_cross_checkpoint_table(
    all_multi_results: Dict[str, Dict[str, Dict[str, float]]],
    recaller_names: List[str],
):
    """
    Print a comparison table across multiple checkpoints (e.g., SFT vs GRPO).
    Each checkpoint contributes Single and Multi-Ch columns.
    
    Args:
        all_multi_results: Dict mapping checkpoint_name -> multi_channel evaluation results
        recaller_names: List of base recaller names
    """
    ckpt_names = list(all_multi_results.keys())
    
    print("\n" + "="*60)
    print(f"Cross-Checkpoint Comparison ({' vs '.join(n.upper() for n in ckpt_names)})")
    print("="*60)
    
    # Build header: Metric | <ckpt>-Single <ckpt>-Multi ... | Best Base
    col_w = 12
    header = f"{'Metric':<{col_w}}"
    for name in ckpt_names:
        header += f" {name.upper()+'-Sing':>{col_w}} {name.upper()+'-Multi':>{col_w}}"
    header += f" {'Best Base':>{col_w}}"
    print(header)
    print("-" * len(header))
    
    for k in [10, 20, 50]:
        for metric in ['ndcg', 'recall']:
            key = f"{metric}@{k}"
            row = f"{key:<{col_w}}"
            for name in ckpt_names:
                res = all_multi_results[name]
                single = res.get("single_select", {}).get(key, 0)
                multi = res.get("multi_channel", {}).get(key, 0)
                row += f" {single:>{col_w}.4f} {multi:>{col_w}.4f}"
            best_base = max(
                all_multi_results[ckpt_names[0]].get(rn, {}).get(key, 0)
                for rn in recaller_names
            )
            row += f" {best_base:>{col_w}.4f}"
            print(row)
