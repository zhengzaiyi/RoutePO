#!/usr/bin/env python3
"""
Case Study: Global & user-specific fusion weight analysis with CEM/PG comparison.

Loads model predictions, CEM results, and PG results, then produces:
1. Global fusion weight comparison (Ours vs CEM vs PG)
2. User-specific fusion weight examples (by history length, by preference)
3. Visualization of weight distributions
"""

import argparse
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DATASET_ITEM_CONFIG = {
    "ml-1m":  {"iid_field": "item_id",    "title_field": "movie_title"},
    "steam":  {"iid_field": "product_id",  "title_field": "title"},
    "Food":   {"iid_field": "item_id",     "title_field": "name"},
}


def build_item_title_map(dataset: str, data_path: str = "dataset") -> Dict[int, str]:
    """Build internal_id -> item_title mapping by reproducing RecBole's token remapping.

    RecBole sorts all unique item tokens (as strings) alphabetically,
    then assigns internal IDs starting from 1 (0 = [PAD]).
    """
    cfg = DATASET_ITEM_CONFIG.get(dataset)
    if cfg is None:
        print(f"WARNING: no item config for dataset '{dataset}', item titles unavailable")
        return {}

    iid_field = cfg["iid_field"]
    title_field = cfg["title_field"]

    inter_path = os.path.join(data_path, dataset, f"{dataset}.inter")
    item_path = os.path.join(data_path, dataset, f"{dataset}.item")
    if not os.path.exists(inter_path) or not os.path.exists(item_path):
        print(f"WARNING: dataset files not found at {data_path}/{dataset}/, item titles unavailable")
        return {}

    inter_df = pd.read_csv(inter_path, sep="\t")
    inter_df.columns = [c.split(":")[0] for c in inter_df.columns]
    item_tokens = sorted(set(str(t) for t in inter_df[iid_field].unique()))
    id_to_token = {i + 1: t for i, t in enumerate(item_tokens)}

    item_df = pd.read_csv(item_path, sep="\t")
    item_df.columns = [c.split(":")[0] for c in item_df.columns]
    item_df[iid_field] = item_df[iid_field].astype(str).str.strip().str.split(".").str[0]
    token_to_title = dict(zip(item_df[iid_field].astype(str), item_df[title_field].astype(str)))

    id_to_title: Dict[int, str] = {}
    for internal_id, token in id_to_token.items():
        title = token_to_title.get(token, "")
        if title:
            id_to_title[internal_id] = title

    print(f"Loaded item titles: {len(id_to_title)}/{len(id_to_token)} items mapped for {dataset}")
    return id_to_title


def load_model_predictions(pred_path: str) -> List[dict]:
    """Load per-user predictions from our model (SFT or GRPO)."""
    with open(pred_path) as f:
        return json.load(f)


def _load_arrow_table(test_dir: str):
    """Load the original data-*.arrow files from a HF Dataset directory (ignoring cache files)."""
    import glob as _glob
    import pyarrow as pa
    arrow_files = sorted(_glob.glob(os.path.join(test_dir, "data-*.arrow")))
    if not arrow_files:
        arrow_files = sorted(_glob.glob(os.path.join(test_dir, "*.arrow")))
        arrow_files = [f for f in arrow_files if not os.path.basename(f).startswith("cache-")]
    if not arrow_files:
        return None
    tables = []
    for f in arrow_files:
        reader = pa.ipc.open_stream(f)
        tables.append(reader.read_all())
    return pa.concat_tables(tables) if len(tables) > 1 else tables[0]


def find_test_dataset_dir(data_dir: str, dataset: str, model_name: str,
                          combo: str, profile_cutoff: int) -> Optional[str]:
    """Find the test dataset directory, with flexible combo matching.

    Search order:
    1. Exact combo + cutoff match
    2. Case-insensitive combo match with exact cutoff
    3. Case-insensitive combo match with any cutoff
    4. Superset combo (requested recallers are a subset) with same cutoff
    5. Any directory that has a test/ subdirectory for this dataset+model
    """
    import glob as _glob
    base = os.path.join(data_dir, dataset)
    if not os.path.isdir(base):
        return None

    target_recallers = set(combo.lower().split("_"))

    exact_path = os.path.join(base, f"{model_name}_pure_sft_data_{combo}_{profile_cutoff}", "test")
    if os.path.isdir(exact_path):
        return os.path.dirname(exact_path)

    candidates = sorted(_glob.glob(os.path.join(base, f"{model_name}_pure_sft_data_*")))

    combo_cutoff_hit = None
    combo_any_cutoff_hit = None
    superset_hit = None
    any_hit = None

    for cand in candidates:
        test_sub = os.path.join(cand, "test")
        if not os.path.isdir(test_sub):
            continue
        dir_name = os.path.basename(cand)
        suffix = dir_name.split("_pure_sft_data_", 1)[-1]
        # suffix e.g. "ItemKNN_LightGCN_Pop_500000" or "ItemKNN_LightGCN_Pop_500000_ptk5"
        # Try to split off cutoff: last purely-numeric segment before any _ptk suffix
        base_suffix = suffix.split("_ptk")[0]  # strip ptk variants
        parts = base_suffix.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            dir_combo, dir_cutoff = parts[0], int(parts[1])
        else:
            dir_combo, dir_cutoff = base_suffix, None

        combo_match = dir_combo.lower() == combo.lower()
        cutoff_match = dir_cutoff == profile_cutoff

        if combo_match and cutoff_match and combo_cutoff_hit is None:
            combo_cutoff_hit = cand
        elif combo_match and combo_any_cutoff_hit is None:
            combo_any_cutoff_hit = cand

        dir_recallers = set(dir_combo.lower().split("_"))
        if target_recallers <= dir_recallers and cutoff_match and superset_hit is None:
            superset_hit = cand
        if any_hit is None:
            any_hit = cand

    chosen = combo_cutoff_hit or combo_any_cutoff_hit or superset_hit or any_hit
    if chosen:
        chosen_suffix = os.path.basename(chosen).split("_pure_sft_data_", 1)[-1]
        if chosen_suffix.lower() != f"{combo}_{profile_cutoff}".lower():
            print(f"NOTE: exact '{combo}_{profile_cutoff}' test data not found, "
                  f"using: {os.path.basename(chosen)}")
    return chosen


def load_user_prompts(data_dir: Optional[str]) -> Dict[int, str]:
    """Load the test split and build user_id -> prompt mapping."""
    if data_dir is None:
        return {}
    test_path = os.path.join(data_dir, "test")
    if not os.path.isdir(test_path):
        print(f"WARNING: test dataset not found at {test_path}, prompts will be unavailable")
        return {}
    try:
        table = _load_arrow_table(test_path)
        if table is None:
            raise FileNotFoundError("no .arrow files found")
        uid_col = table.column("user_id")
        prompt_col = table.column("prompt") if "prompt" in table.column_names else table.column("text")
        uid_to_prompt: Dict[int, str] = {}
        for uid, prompt in zip(uid_col.to_pylist(), prompt_col.to_pylist()):
            if uid is not None:
                uid_to_prompt[uid] = prompt
        print(f"Loaded prompts for {len(uid_to_prompt)} users from {test_path}")
        return uid_to_prompt
    except Exception as e:
        print(f"WARNING: failed to load test dataset from {test_path}: {e}")
        return {}


def load_cem_results(result_path: str) -> dict:
    """Load CEM baseline results."""
    with open(result_path) as f:
        return json.load(f)


def load_pg_results(result_path: str) -> dict:
    """Load PG baseline results."""
    with open(result_path) as f:
        return json.load(f)


def compute_global_weights(predictions: List[dict], recaller_names: List[str]) -> np.ndarray:
    """Compute average fusion weights across all users."""
    all_weights = np.array([p["merge_weights"] for p in predictions])
    return all_weights.mean(axis=0)


def compute_weight_stats(predictions: List[dict]) -> dict:
    """Compute weight distribution statistics."""
    all_weights = np.array([p["merge_weights"] for p in predictions])
    return {
        "mean": all_weights.mean(axis=0),
        "std": all_weights.std(axis=0),
        "median": np.median(all_weights, axis=0),
        "min": all_weights.min(axis=0),
        "max": all_weights.max(axis=0),
        "q25": np.percentile(all_weights, 25, axis=0),
        "q75": np.percentile(all_weights, 75, axis=0),
    }


def bucket_users_by_history_length(predictions: List[dict], gt_key="gt_items") -> Dict[str, List[dict]]:
    """Group users into history-length buckets based on number of recalled items."""
    buckets = defaultdict(list)
    for p in predictions:
        n_items = sum(len(v) for v in p.get("recaller_predictions", {}).values())
        avg_items = n_items / max(1, len(p.get("recaller_predictions", {})))
        if avg_items < 20:
            buckets["sparse (<20)"].append(p)
        elif avg_items < 40:
            buckets["medium (20-40)"].append(p)
        else:
            buckets["dense (40+)"].append(p)
    return dict(buckets)


def find_diverse_users(predictions: List[dict], recaller_names: List[str], n: int = 5) -> dict:
    """Find users with the most diverse weight patterns."""
    results = {}

    all_weights = np.array([p["merge_weights"] for p in predictions])
    global_mean = all_weights.mean(axis=0)

    # Users where each recaller dominates
    for i, name in enumerate(recaller_names):
        idx = np.argmax(all_weights[:, i])
        results[f"strongest_{name}"] = predictions[idx]

    # Most balanced user (closest to uniform)
    uniform = np.ones(len(recaller_names)) / len(recaller_names)
    dists = np.linalg.norm(all_weights - uniform, axis=1)
    results["most_balanced"] = predictions[np.argmin(dists)]

    # Most extreme user (farthest from uniform)
    results["most_extreme"] = predictions[np.argmax(dists)]

    # Users farthest from global mean
    dists_from_mean = np.linalg.norm(all_weights - global_mean, axis=1)
    top_diverse = np.argsort(dists_from_mean)[-n:]
    for rank, idx in enumerate(reversed(top_diverse)):
        results[f"diverse_{rank+1}"] = predictions[idx]

    return results


def snack_merge(recaller_items: Dict[str, list], weights: Dict[str, float], total_k: int) -> List[int]:
    """Simplified Snack Fusion (prefix-quota merge) for case study evaluation."""
    names = list(weights.keys())
    s = sum(weights.values())
    w = {n: v / s for n, v in weights.items()} if s > 0 else {n: 1.0 / len(names) for n in names}

    merged, seen = [], set()
    ptr = {n: 0 for n in names}
    selected = {n: 0 for n in names}

    for _ in range(total_k * 50):
        if len(merged) >= total_k:
            break
        t_next = len(merged) + 1
        deficits = {n: round(w[n] * t_next) - selected[n] for n in names}
        best = max(names, key=lambda n: deficits[n])
        items = recaller_items.get(best, [])
        found = False
        while ptr[best] < len(items):
            item_id = items[ptr[best]][0] if isinstance(items[ptr[best]], (list, tuple)) else items[ptr[best]]
            ptr[best] += 1
            if item_id not in seen:
                merged.append(item_id)
                seen.add(item_id)
                selected[best] += 1
                found = True
                break
        if not found:
            break
    return merged[:total_k]


def compute_per_user_metrics(
    predictions: List[dict],
    recaller_names: List[str],
    our_weights_key: str = "merge_weights",
    cem_weights: Optional[dict] = None,
    pg_weights: Optional[dict] = None,
    pg_per_user_weights: Optional[dict] = None,
    k: int = 50,
) -> List[dict]:
    """Compute per-user NDCG/Recall under FusionPO, CEM, and PG weights.
    
    pg_per_user_weights: if available, maps str(user_id) -> {recaller: weight}.
    Falls back to pg_weights (global average) when per-user entry is missing.
    """
    import math

    def _ndcg(rec_list, gt_items, k):
        if not gt_items:
            return 0.0
        gt = set(gt_items)
        dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(rec_list[:k]) if item in gt)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt), k)))
        return dcg / idcg if idcg > 0 else 0.0

    def _to_weight_dict(weight_vec, names):
        if isinstance(weight_vec, dict):
            lower_map = {k.lower(): v for k, v in weight_vec.items()}
            return {n.lower(): lower_map.get(n.lower(), 0) for n in names}
        return {names[i].lower(): float(weight_vec[i]) for i in range(len(names))}

    cem_w = _to_weight_dict(cem_weights, recaller_names) if cem_weights else None
    pg_w_fallback = _to_weight_dict(pg_weights, recaller_names) if pg_weights else None

    results = []
    for pred in predictions:
        rp = pred["recaller_predictions"]
        gt = pred.get("gt_items", [])
        uid = pred.get("user_id", "?")
        our_w = _to_weight_dict(pred[our_weights_key], recaller_names)

        our_list = snack_merge(rp, our_w, k)
        our_ndcg = _ndcg(our_list, gt, k)

        entry = {"user_id": uid, "gt_items": gt, "our_weights": pred[our_weights_key],
                 "our_ndcg": our_ndcg, "our_rec_list": our_list}

        if cem_w:
            cem_list = snack_merge(rp, cem_w, k)
            entry["cem_ndcg"] = _ndcg(cem_list, gt, k)
            entry["cem_weights"] = cem_w
            entry["cem_rec_list"] = cem_list

        if pg_per_user_weights or pg_w_fallback:
            pw = None
            if pg_per_user_weights:
                pw_raw = pg_per_user_weights.get(str(uid)) or pg_per_user_weights.get(int(uid) if isinstance(uid, str) else uid)
                if pw_raw:
                    pw = _to_weight_dict(pw_raw, recaller_names)
            if pw is None:
                pw = pg_w_fallback
            if pw:
                pg_list = snack_merge(rp, pw, k)
                entry["pg_ndcg"] = _ndcg(pg_list, gt, k)
                entry["pg_weights"] = pw
                entry["pg_rec_list"] = pg_list

        results.append(entry)
    return results


def find_fusionpo_winners(
    per_user: List[dict],
    recaller_names: List[str],
    n: int = 5,
) -> List[dict]:
    """Find users where FusionPO outperforms both CEM and PG the most (absolute gap)."""
    scored = []
    for u in per_user:
        cem_gap = u["our_ndcg"] - u.get("cem_ndcg", u["our_ndcg"])
        pg_gap = u["our_ndcg"] - u.get("pg_ndcg", u["our_ndcg"])
        min_gap = min(cem_gap, pg_gap)
        if min_gap > 0:
            scored.append((min_gap, u))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [u for _, u in scored[:n]]


def find_fusionpo_winners_relative(
    per_user: List[dict],
    recaller_names: List[str],
    n: int = 5,
    eps: float = 1e-8,
) -> List[dict]:
    """Find users where FusionPO outperforms both CEM and PG by the largest relative ratio.

    Relative improvement = (our - baseline) / max(baseline, eps).
    Winner score = min(relative_improvement_over_cem, relative_improvement_over_pg).
    """
    scored = []
    for u in per_user:
        cem_ndcg = u.get("cem_ndcg", None)
        pg_ndcg = u.get("pg_ndcg", None)

        ratios = []
        if cem_ndcg is not None:
            ratios.append((u["our_ndcg"] - cem_ndcg) / max(cem_ndcg, eps))
        if pg_ndcg is not None:
            ratios.append((u["our_ndcg"] - pg_ndcg) / max(pg_ndcg, eps))

        if not ratios:
            continue
        min_ratio = min(ratios)
        if min_ratio > 0:
            scored.append((min_ratio, u))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [u for _, u in scored[:n]]


def print_global_comparison(
    recaller_names: List[str],
    our_weights: np.ndarray,
    our_metrics: dict,
    cem_weights: Optional[dict] = None,
    cem_metrics: Optional[dict] = None,
    pg_weights: Optional[dict] = None,
    pg_metrics: Optional[dict] = None,
):
    """Print a comparison table of global fusion weights and metrics."""
    print("\n" + "=" * 80)
    print("GLOBAL FUSION WEIGHT COMPARISON")
    print("=" * 80)

    header = f"{'Recaller':<15}"
    methods = [("Ours", our_weights)]
    if cem_weights:
        methods.append(("CEM", np.array([cem_weights.get(n, 0) for n in recaller_names])))
    if pg_weights:
        methods.append(("PG", np.array([pg_weights.get(n, 0) for n in recaller_names])))
    for name, _ in methods:
        header += f" {name:>10}"
    print(header)
    print("-" * len(header))

    for i, rn in enumerate(recaller_names):
        row = f"{rn:<15}"
        for _, w in methods:
            row += f" {w[i]:>10.4f}"
        print(row)

    # Metrics comparison
    print("\n" + "-" * 80)
    print("PERFORMANCE COMPARISON")
    print("-" * 80)
    metric_header = f"{'Metric':<15}"
    metric_sources = [("Ours", our_metrics)]
    if cem_metrics:
        metric_sources.append(("CEM", cem_metrics))
    if pg_metrics:
        metric_sources.append(("PG", pg_metrics))
    for name, _ in metric_sources:
        metric_header += f" {name:>10}"
    print(metric_header)
    print("-" * len(metric_header))

    for k in [10, 20, 50]:
        for metric in ["ndcg", "recall"]:
            key = f"{metric}@{k}"
            row = f"{key:<15}"
            for name, m in metric_sources:
                val = m.get(key, 0)
                row += f" {val:>10.4f}"
            print(row)


def print_user_examples(diverse_users: dict, recaller_names: List[str]):
    """Print user-specific weight examples."""
    print("\n" + "=" * 80)
    print("USER-SPECIFIC FUSION WEIGHT EXAMPLES")
    print("=" * 80)

    for label, pred in diverse_users.items():
        weights = pred["merge_weights"]
        uid = pred.get("user_id", "?")
        gt = pred.get("gt_items", [])
        print(f"\n--- {label} (user_id={uid}, gt_items={len(gt)}) ---")
        for i, rn in enumerate(recaller_names):
            bar = "#" * int(weights[i] * 40)
            print(f"  {rn:<15} {weights[i]:.4f}  {bar}")


def print_bucket_analysis(buckets: Dict[str, List[dict]], recaller_names: List[str]):
    """Print weight analysis grouped by user buckets."""
    print("\n" + "=" * 80)
    print("WEIGHT ANALYSIS BY USER GROUP")
    print("=" * 80)

    header = f"{'Group':<20} {'Count':>6}"
    for rn in recaller_names:
        header += f" {rn:>12}"
    print(header)
    print("-" * len(header))

    for bucket_name, preds in sorted(buckets.items()):
        weights = np.array([p["merge_weights"] for p in preds])
        means = weights.mean(axis=0)
        row = f"{bucket_name:<20} {len(preds):>6}"
        for m in means:
            row += f" {m:>12.4f}"
        print(row)


def plot_weight_comparison(
    recaller_names: List[str],
    our_weights: np.ndarray,
    cem_weights: Optional[dict] = None,
    pg_weights: Optional[dict] = None,
    save_path: str = "results/case_study_weights.pdf",
):
    """Bar chart comparing global fusion weights across methods."""
    n = len(recaller_names)
    methods = {"Ours (GRPO)": our_weights}
    if cem_weights:
        methods["CEM"] = np.array([cem_weights.get(rn, 0) for rn in recaller_names])
    if pg_weights:
        methods["PG"] = np.array([pg_weights.get(rn, 0) for rn in recaller_names])

    x = np.arange(n)
    width = 0.8 / len(methods)
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (method_name, w) in enumerate(methods.items()):
        ax.bar(x + i * width, w, width, label=method_name, color=colors[i % len(colors)])

    ax.set_xlabel("Recaller")
    ax.set_ylabel("Fusion Weight")
    ax.set_title("Global Fusion Weight Comparison")
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(recaller_names)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nWeight comparison plot saved to: {save_path}")
    plt.close()

    # Save raw data
    data_path = save_path.replace('.pdf', '_data.json')
    raw_data = {
        "recaller_names": recaller_names,
        "methods": {name: w.tolist() if hasattr(w, 'tolist') else list(w) for name, w in methods.items()}
    }
    with open(data_path, 'w') as f:
        json.dump(raw_data, f, indent=2)
    print(f"Weight comparison data saved to: {data_path}")


def plot_weight_distribution(
    predictions: List[dict],
    recaller_names: List[str],
    pg_per_user_weights: Optional[dict] = None,
    pg_weights: Optional[dict] = None,
    save_path: str = "results/case_study_weight_dist.pdf",
):
    """Violin/box plot showing per-user weight distribution for each recaller."""
    all_weights = np.array([p["merge_weights"] for p in predictions])
    n = len(recaller_names)

    has_pg = pg_per_user_weights is not None or pg_weights is not None
    pg_all_weights = None
    if has_pg:
        pg_rows = []
        for p in predictions:
            uid = p.get("user_id", "?")
            pw = None
            if pg_per_user_weights:
                pw = pg_per_user_weights.get(str(uid)) or pg_per_user_weights.get(
                    int(uid) if isinstance(uid, str) else uid
                )
            if pw is None and pg_weights:
                pw = pg_weights
            if pw:
                pg_rows.append([pw.get(rn, 0) for rn in recaller_names])
        if pg_rows:
            pg_all_weights = np.array(pg_rows)

    if pg_all_weights is not None:
        fig, ax = plt.subplots(figsize=(max(8, n * 2.2), 4))
        width = 0.35
        positions_ours = np.arange(n) - width / 2
        positions_pg = np.arange(n) + width / 2

        parts_ours = ax.violinplot(
            [all_weights[:, i] for i in range(n)],
            positions=positions_ours, widths=width,
            showmeans=True, showmedians=True,
        )
        for pc in parts_ours['bodies']:
            pc.set_facecolor('#2196F3')
            pc.set_alpha(0.6)
        for key in ('cmeans', 'cmedians', 'cmins', 'cmaxes', 'cbars'):
            if key in parts_ours:
                parts_ours[key].set_color('#1565C0')

        parts_pg = ax.violinplot(
            [pg_all_weights[:, i] for i in range(n)],
            positions=positions_pg, widths=width,
            showmeans=True, showmedians=True,
        )
        for pc in parts_pg['bodies']:
            pc.set_facecolor('#4CAF50')
            pc.set_alpha(0.6)
        for key in ('cmeans', 'cmedians', 'cmins', 'cmaxes', 'cbars'):
            if key in parts_pg:
                parts_pg[key].set_color('#2E7D32')

        from matplotlib.patches import Patch
        ax.legend(
            handles=[Patch(facecolor='#2196F3', alpha=0.6, label='Ours (GRPO)'),
                     Patch(facecolor='#4CAF50', alpha=0.6, label='PG')],
            loc='upper right',
        )
        ax.set_title("Per-User Fusion Weight Distribution")
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        parts = ax.violinplot(
            [all_weights[:, i] for i in range(n)],
            positions=range(n),
            showmeans=True, showmedians=True,
        )
        for pc in parts['bodies']:
            pc.set_alpha(0.6)
        ax.set_title("Per-User Fusion Weight Distribution (Ours)")

    ax.set_xticks(range(n))
    ax.set_xticklabels(recaller_names)
    ax.set_ylabel("Fusion Weight")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Weight distribution plot saved to: {save_path}")
    plt.close()

    # Save raw data
    data_path = save_path.replace('.pdf', '_data.json')
    raw_data = {
        "recaller_names": recaller_names,
        "num_users": len(predictions),
        "ours": {
            "per_user_weights": all_weights.tolist(),
            "stats": {
                rn: {
                    "mean": float(all_weights[:, i].mean()),
                    "std": float(all_weights[:, i].std()),
                    "min": float(all_weights[:, i].min()),
                    "max": float(all_weights[:, i].max()),
                }
                for i, rn in enumerate(recaller_names)
            }
        }
    }
    if pg_all_weights is not None:
        raw_data["pg"] = {
            "per_user_weights": pg_all_weights.tolist(),
            "stats": {
                rn: {
                    "mean": float(pg_all_weights[:, i].mean()),
                    "std": float(pg_all_weights[:, i].std()),
                    "min": float(pg_all_weights[:, i].min()),
                    "max": float(pg_all_weights[:, i].max()),
                }
                for i, rn in enumerate(recaller_names)
            }
        }
    with open(data_path, 'w') as f:
        json.dump(raw_data, f, indent=2)
    print(f"Weight distribution data saved to: {data_path}")


def plot_user_heatmap(
    diverse_users: dict,
    recaller_names: List[str],
    save_path: str = "results/case_study_user_heatmap.pdf",
):
    """Heatmap of fusion weights for selected diverse users."""
    labels = list(diverse_users.keys())
    data = np.array([diverse_users[l]["merge_weights"] for l in labels])

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.4)))
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(recaller_names)))
    ax.set_xticklabels(recaller_names, rotation=45, ha='right')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Fusion Weights for Selected Users")

    for i in range(len(labels)):
        for j in range(len(recaller_names)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha='center', va='center', fontsize=8)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"User heatmap saved to: {save_path}")
    plt.close()

    # Save raw data
    data_path = save_path.replace('.pdf', '_data.json')
    raw_data = {
        "recaller_names": recaller_names,
        "user_labels": labels,
        "weights_matrix": data.tolist(),
        "users": {
            label: {
                "user_id": diverse_users[label].get("user_id"),
                "weights": {rn: float(data[i, j]) for j, rn in enumerate(recaller_names)}
            }
            for i, label in enumerate(labels)
        }
    }
    with open(data_path, 'w') as f:
        json.dump(raw_data, f, indent=2)
    print(f"User heatmap data saved to: {data_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Case Study: Fusion Weight Analysis")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--recaller_combo", type=str, required=True,
                        help="Underscore-joined sorted recaller names, e.g. ItemKNN_LightGCN_Pop")
    parser.add_argument("--model_name", type=str, default="Llama-3.2-1B-Instruct",
                        help="Short model name (basename) for file lookup")
    parser.add_argument("--checkpoint", type=str, default="grpo",
                        choices=["sft", "grpo"],
                        help="Which checkpoint's predictions to analyze")
    parser.add_argument("--pred_dir", type=str, default="results",
                        help="Directory containing pure_predictions_*.json files")
    parser.add_argument("--cem_dir", type=str, default="results/cem",
                        help="Directory containing CEM result files")
    parser.add_argument("--pg_dir", type=str, default="results/pg",
                        help="Directory containing PG result files")
    parser.add_argument("--output_dir", type=str, default="results/case_study",
                        help="Directory to save analysis outputs")
    parser.add_argument("--data_dir", type=str, default="GRPO/data/pure_models",
                        help="Base directory for SFT datasets (contains {dataset}/{model}_pure_sft_data_...)")
    parser.add_argument("--profile_cutoff", type=int, default=500000,
                        help="Profile cutoff used during data generation (for dataset path)")
    parser.add_argument("--data_path", type=str, default="dataset",
                        help="Path to raw dataset files (contains {dataset}/{dataset}.inter etc.)")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip generating plots (text output only)")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = args.dataset
    combo = args.recaller_combo
    recaller_names = sorted(combo.split("_"))

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load item title mapping ---
    id_to_title = build_item_title_map(dataset, args.data_path)

    # --- Load our model predictions ---
    pred_file = f"{args.pred_dir}/pure_predictions_{dataset}_{args.model_name}_{combo}_{args.checkpoint}.json"
    if not os.path.exists(pred_file):
        print(f"ERROR: Prediction file not found: {pred_file}")
        print("Make sure you've run testing with --do_test_grpo / --do_test_sft first.")
        sys.exit(1)
    predictions = load_model_predictions(pred_file)
    print(f"Loaded {len(predictions)} user predictions from {pred_file}")

    def _find_file(directory, prefix, dataset, combo):
        """Find a result file trying exact match first, then case-insensitive glob."""
        exact = os.path.join(directory, f"{prefix}_{dataset}_{combo}.json")
        if os.path.exists(exact):
            return exact
        import glob
        pattern = os.path.join(directory, f"{prefix}_{dataset}_*.json")
        for candidate in glob.glob(pattern):
            basename = os.path.splitext(os.path.basename(candidate))[0]
            file_combo = basename.split(f"{prefix}_{dataset}_", 1)[-1]
            if file_combo.lower() == combo.lower():
                return candidate
        return None

    # --- Load CEM results ---
    cem_weights, cem_metrics = None, None
    cem_file = _find_file(args.cem_dir, "cem_results", dataset, combo)
    if cem_file:
        cem_data = load_cem_results(cem_file)
        cem_weights = {k.lower(): v for k, v in cem_data.get("cem_fusion", {}).get("optimized_weights", {}).items()}
        cem_metrics = cem_data.get("cem_fusion", {}).get("cem_optimized", {})
        print(f"Loaded CEM results from {cem_file}")
    else:
        print(f"CEM results not found for {dataset}_{combo}, skipping CEM comparison")

    # --- Load PG results ---
    pg_weights, pg_per_user_weights, pg_metrics = None, None, None
    pg_file = _find_file(args.pg_dir, "pg_results", dataset, combo)
    if pg_file:
        pg_data = load_pg_results(pg_file)
        pg_weights = {k.lower(): v for k, v in pg_data.get("pg_fusion", {}).get("avg_weights", {}).items()}
        raw_puw = pg_data.get("pg_fusion", {}).get("per_user_weights", None)
        pg_per_user_weights = {uid: {k.lower(): v for k, v in w.items()} for uid, w in raw_puw.items()} if raw_puw else None
        pg_metrics = pg_data.get("pg_fusion", {}).get("pg_optimized", {})
        print(f"Loaded PG results from {pg_file}"
              f" (per-user weights: {'yes' if pg_per_user_weights else 'no'})")
    else:
        print(f"PG results not found for {dataset}_{combo}, skipping PG comparison")

    # --- Compute our global weights and metrics ---
    our_global_weights = compute_global_weights(predictions, recaller_names)
    weight_stats = compute_weight_stats(predictions)

    # Load our metrics from the result file
    our_result_file = f"{args.pred_dir}/pure_model_results_{dataset}_{args.model_name}_{combo}_{args.checkpoint}.json"
    our_metrics = {}
    if os.path.exists(our_result_file):
        with open(our_result_file) as f:
            our_result_data = json.load(f)
        mce = our_result_data.get("multi_channel_evaluation", {})
        mc = mce.get("multi_channel", {})
        our_metrics = mc
        print(f"Loaded our metrics from {our_result_file}")

    # --- 1. Global comparison ---
    print_global_comparison(
        recaller_names, our_global_weights, our_metrics,
        cem_weights, cem_metrics,
        pg_weights, pg_metrics,
    )

    # --- 2. Weight statistics ---
    print("\n" + "=" * 80)
    print("FUSION WEIGHT STATISTICS (Our Model)")
    print("=" * 80)
    for i, rn in enumerate(recaller_names):
        print(f"  {rn:<15}  mean={weight_stats['mean'][i]:.4f}  std={weight_stats['std'][i]:.4f}  "
              f"median={weight_stats['median'][i]:.4f}  "
              f"[{weight_stats['q25'][i]:.4f}, {weight_stats['q75'][i]:.4f}]")

    # --- 3. User-specific examples ---
    diverse_users = find_diverse_users(predictions, recaller_names)
    print_user_examples(diverse_users, recaller_names)

    # --- 4. Bucket analysis ---
    buckets = bucket_users_by_history_length(predictions)
    if buckets:
        print_bucket_analysis(buckets, recaller_names)

    # --- 5. Load user prompts from dataset ---
    sft_data_path = find_test_dataset_dir(
        args.data_dir, dataset, args.model_name, combo.lower(), args.profile_cutoff,
    )
    uid_to_prompt = load_user_prompts(sft_data_path)

    # --- 6. Per-user comparison: find FusionPO winners ---
    fusionpo_winners = []
    fusionpo_winners_relative = []
    if cem_weights or pg_weights or pg_per_user_weights:
        per_user = compute_per_user_metrics(
            predictions, recaller_names,
            cem_weights=cem_weights, pg_weights=pg_weights,
            pg_per_user_weights=pg_per_user_weights, k=50,
        )
        fusionpo_winners = find_fusionpo_winners(per_user, recaller_names, n=10)
        fusionpo_winners_relative = find_fusionpo_winners_relative(per_user, recaller_names, n=10)

        def _format_rec_item(item_id, gt_set):
            title = id_to_title.get(item_id, "")
            hit = "*" if item_id in gt_set else ""
            return f"{item_id}{hit}({title})" if title else f"{item_id}{hit}"

        def _format_rec_list(rec_list, gt_set, top_n=10):
            return "[" + ", ".join(_format_rec_item(item, gt_set) for item in rec_list[:top_n]) + "]"

        def _gt_hit_positions(rec_list, gt_set, max_k=50):
            hits = []
            for rank, item in enumerate(rec_list[:max_k]):
                if item in gt_set:
                    title = id_to_title.get(item, "")
                    hits.append(f"@{rank+1}:{item}({title})" if title else f"@{rank+1}:{item}")
            return hits

        def _print_winners(title, winners, uid_to_prompt):
            print("\n" + "=" * 80)
            print(title)
            print("=" * 80)
            for i, u in enumerate(winners):
                uid = u["user_id"]
                gt_set = set(u.get("gt_items", []))
                cem_str = f"  CEM={u['cem_ndcg']:.4f}" if 'cem_ndcg' in u else ""
                pg_str = f"  PG={u['pg_ndcg']:.4f}" if 'pg_ndcg' in u else ""
                rel_parts = []
                if 'cem_ndcg' in u and u['cem_ndcg'] > 0:
                    rel_parts.append(f"vs CEM: {(u['our_ndcg'] - u['cem_ndcg'])/u['cem_ndcg']*100:+.1f}%")
                if 'pg_ndcg' in u and u['pg_ndcg'] > 0:
                    rel_parts.append(f"vs PG: {(u['our_ndcg'] - u['pg_ndcg'])/u['pg_ndcg']*100:+.1f}%")
                rel_str = f"  Relative: [{', '.join(rel_parts)}]" if rel_parts else ""

                print(f"\n--- Winner #{i+1} (user_id={uid}) ---")
                print(f"  NDCG@50:  Ours={u['our_ndcg']:.4f}{cem_str}{pg_str}{rel_str}")

                # Weight comparison table
                cem_w = u.get("cem_weights", {})
                pg_w = u.get("pg_weights", {})
                header = f"  {'Recaller':<15} {'Ours':>8}"
                if cem_w:
                    header += f" {'CEM':>8}"
                if pg_w:
                    header += f" {'PG':>8}"
                print(header)
                print(f"  {'-'*(len(header)-2)}")
                w = u["our_weights"]
                for j, rn in enumerate(recaller_names):
                    wval = w[j] if isinstance(w, list) else w.get(rn, 0)
                    row = f"  {rn:<15} {float(wval):>8.4f}"
                    if cem_w:
                        cv = cem_w.get(rn.lower(), cem_w.get(rn, 0))
                        row += f" {float(cv):>8.4f}"
                    if pg_w:
                        pv = pg_w.get(rn.lower(), pg_w.get(rn, 0))
                        row += f" {float(pv):>8.4f}"
                    print(row)

                # Top-10 recommendation lists (* = GT hit)
                gt_items = u.get("gt_items", [])
                gt_display = [f"{g}({id_to_title.get(g, '')})" if id_to_title.get(g) else str(g) for g in gt_items]
                print(f"  Ground truth items: [{', '.join(gt_display)}]")
                our_rec = u.get("our_rec_list", [])
                if our_rec:
                    print(f"  Ours  top-10: {_format_rec_list(our_rec, gt_set)}")
                    print(f"    GT hit @positions: {_gt_hit_positions(our_rec, gt_set)}")
                cem_rec = u.get("cem_rec_list", [])
                if cem_rec:
                    print(f"  CEM   top-10: {_format_rec_list(cem_rec, gt_set)}")
                    print(f"    GT hit @positions: {_gt_hit_positions(cem_rec, gt_set)}")
                pg_rec = u.get("pg_rec_list", [])
                if pg_rec:
                    print(f"  PG    top-10: {_format_rec_list(pg_rec, gt_set)}")
                    print(f"    GT hit @positions: {_gt_hit_positions(pg_rec, gt_set)}")

                if uid in uid_to_prompt:
                    print(f"  Prompt: {uid_to_prompt[uid][:200]}...")
                elif str(uid) in uid_to_prompt:
                    print(f"  Prompt: {uid_to_prompt[str(uid)][:200]}...")

        _print_winners("FUSIONPO WINNERS - ABSOLUTE GAP (our method >> CEM & PG)", fusionpo_winners, uid_to_prompt)
        _print_winners("FUSIONPO WINNERS - RELATIVE RATIO (our method >> CEM & PG)", fusionpo_winners_relative, uid_to_prompt)

    # --- 8. Plots ---
    if not args.no_plot:
        prefix = f"{args.output_dir}/{dataset}_{combo}_{args.checkpoint}"

        plot_weight_comparison(
            recaller_names, our_global_weights,
            cem_weights, pg_weights,
            save_path=f"{prefix}_weight_comparison.pdf",
        )
        plot_weight_distribution(
            predictions, recaller_names,
            pg_per_user_weights=pg_per_user_weights,
            pg_weights=pg_weights,
            save_path=f"{prefix}_weight_distribution.pdf",
        )
        plot_user_heatmap(
            diverse_users, recaller_names,
            save_path=f"{prefix}_user_heatmap.pdf",
        )

    # --- 9. Save structured summary ---
    def _build_winner_entry(u, uid_to_prompt):
        uid = u["user_id"]
        gt_set = set(u.get("gt_items", []))

        def _hit_positions(rec_list):
            return [{"position": rank + 1, "id": item, "title": id_to_title.get(item, "")}
                    for rank, item in enumerate(rec_list[:50]) if item in gt_set]

        def _rec_top_with_titles(rec_list, n=10):
            return [{"id": item, "title": id_to_title.get(item, ""), "gt_hit": item in gt_set}
                    for item in rec_list[:n]]

        entry = {
            "user_id": uid,
            "our_weights": u["our_weights"] if isinstance(u["our_weights"], list)
                           else list(u["our_weights"]),
            "gt_items": [{"id": g, "title": id_to_title.get(g, "")} for g in u["gt_items"]],
            "our_ndcg": u["our_ndcg"],
            "cem_ndcg": u.get("cem_ndcg"),
            "pg_ndcg": u.get("pg_ndcg"),
        }
        if u.get("our_rec_list"):
            entry["our_rec_top10"] = _rec_top_with_titles(u["our_rec_list"])
            entry["our_gt_hit_positions"] = _hit_positions(u["our_rec_list"])
        if u.get("cem_weights"):
            entry["cem_weights"] = u["cem_weights"]
        if u.get("cem_rec_list"):
            entry["cem_rec_top10"] = _rec_top_with_titles(u["cem_rec_list"])
            entry["cem_gt_hit_positions"] = _hit_positions(u["cem_rec_list"])
        if u.get("pg_weights"):
            entry["pg_weights"] = u["pg_weights"]
        if u.get("pg_rec_list"):
            entry["pg_rec_top10"] = _rec_top_with_titles(u["pg_rec_list"])
            entry["pg_gt_hit_positions"] = _hit_positions(u["pg_rec_list"])
        if u.get("cem_ndcg") and u["cem_ndcg"] > 0:
            entry["relative_vs_cem"] = (u["our_ndcg"] - u["cem_ndcg"]) / u["cem_ndcg"]
        if u.get("pg_ndcg") and u["pg_ndcg"] > 0:
            entry["relative_vs_pg"] = (u["our_ndcg"] - u["pg_ndcg"]) / u["pg_ndcg"]
        prompt = uid_to_prompt.get(uid) or uid_to_prompt.get(str(uid))
        if prompt:
            entry["prompt"] = prompt
        return entry

    summary = {
        "dataset": dataset,
        "recaller_combo": combo,
        "recaller_names": recaller_names,
        "model": args.model_name,
        "checkpoint": args.checkpoint,
        "num_users": len(predictions),
        "our_global_weights": {rn: float(our_global_weights[i]) for i, rn in enumerate(recaller_names)},
        "our_weight_stats": {
            rn: {k: float(v[i]) for k, v in weight_stats.items()}
            for i, rn in enumerate(recaller_names)
        },
        "our_metrics": our_metrics,
        "cem_weights": cem_weights,
        "cem_metrics": cem_metrics,
        "pg_weights": pg_weights,
        "pg_metrics": pg_metrics,
        "diverse_user_examples": {
            label: {
                "user_id": p.get("user_id"),
                "merge_weights": p["merge_weights"],
                "gt_items": p.get("gt_items", []),
            }
            for label, p in diverse_users.items()
        },
        "bucket_analysis": {
            bucket_name: {
                "count": len(preds),
                "avg_weights": np.array([p["merge_weights"] for p in preds]).mean(axis=0).tolist(),
            }
            for bucket_name, preds in buckets.items()
        } if buckets else {},
        "fusionpo_winners_absolute": [
            _build_winner_entry(u, uid_to_prompt) for u in fusionpo_winners
        ],
        "fusionpo_winners_relative": [
            _build_winner_entry(u, uid_to_prompt) for u in fusionpo_winners_relative
        ],
    }

    summary_path = f"{args.output_dir}/{dataset}_{combo}_{args.checkpoint}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
