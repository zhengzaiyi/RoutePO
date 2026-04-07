"""
Generate dataset with per-position recaller metrics for each user.

For each user, this script saves:
- The full interaction sequence
- For each position (item as ground truth), the metrics of all base recallers

Output format:
{
    user_id: {
        "sequence": [item1, item2, item3, ...],
        "position_metrics": {
            "1": {"bpr": {"ndcg@10": 0.1, ...}, "sasrec": {...}, ...},
            "2": {"bpr": {...}, "sasrec": {...}, ...},
            ...
        }
    }
}

Supported base recallers: BPR, SASRec, Pop, ItemKNN, FPMC, GRU4Rec, LightGCN, SimpleX
"""

import argparse
import json
import os
import sys

# Disable distributed training for RecBole to avoid torch.distributed errors
os.environ['MASTER_ADDR'] = ''
os.environ['MASTER_PORT'] = ''
os.environ['RANK'] = ''
os.environ['WORLD_SIZE'] = ''
os.environ['LOCAL_RANK'] = ''
from collections import defaultdict
from typing import Dict, List
from datetime import datetime
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GRPO.core.data import load_dataset, get_base_config_dict
from GRPO.models.main import initialize_recallers
from GRPO.core.recallers import RecBoleRecaller
from GRPO.core.utils import set_seed, ndcg_at_k, recall_at_k

try:
    from recbole.config import Config
    from recbole.data import create_dataset
    from recbole.utils import init_seed as recbole_init_seed
except ImportError:
    pass

# All supported recaller models
ALL_RECALLER_MODELS = ['BPR', 'SASRec', 'Pop', 'ItemKNN', 'FPMC', 'GRU4Rec', 'LightGCN', 'SimpleX']


def hit_at_k(rec_list: List[int], gt_items: List[int], k: int) -> float:
    """Calculate Hit@K"""
    if not gt_items:
        return 0.0
    if isinstance(gt_items, int):
        gt_items = [gt_items]
    gt = set(gt_items)
    for item in rec_list[:k]:
        if item in gt:
            return 1.0
    return 0.0


def mrr_at_k(rec_list: List[int], gt_items: List[int], k: int) -> float:
    """Calculate MRR@K"""
    if not gt_items:
        return 0.0
    if isinstance(gt_items, int):
        gt_items = [gt_items]
    gt = set(gt_items)
    for i, item in enumerate(rec_list[:k]):
        if item in gt:
            return 1.0 / (i + 1)
    return 0.0


def compute_metrics(
    rec_list: List[int], 
    gt_items: List[int], 
    ks: List[int] = [5, 10, 20, 50]
) -> Dict[str, float]:
    """Compute all metrics for a recommendation list"""
    metrics = {}
    for k in ks:
        metrics[f"ndcg@{k}"] = ndcg_at_k(rec_list, gt_items, k)
        metrics[f"recall@{k}"] = recall_at_k(rec_list, gt_items, k)
        metrics[f"hit@{k}"] = hit_at_k(rec_list, gt_items, k)
        metrics[f"mrr@{k}"] = mrr_at_k(rec_list, gt_items, k)
    return metrics


def get_user_full_sequences(dataset_name: str, data_path: str, seed: int = 42) -> Dict[int, List[int]]:
    """
    Get full interaction sequence for each user from raw dataset.
    Returns: {user_id: [item1, item2, item3, ...]} in chronological order
    """
    config_dict = get_base_config_dict(dataset_name, data_path, seed)
    cfg = Config(
        model="SASRec",
        dataset=dataset_name,
        config_dict=config_dict,
    )
    recbole_init_seed(cfg['seed'], reproducibility=True)
    ds = create_dataset(cfg)
    
    uid_field = ds.uid_field
    iid_field = ds.iid_field
    time_field = ds.time_field if hasattr(ds, 'time_field') and ds.time_field else None
    
    # Get all interactions
    inter_feat = ds.inter_feat
    
    # Build user -> items mapping (in chronological order)
    user_sequences = defaultdict(list)
    
    # Get data as numpy arrays
    user_ids = inter_feat[uid_field].numpy() if hasattr(inter_feat[uid_field], 'numpy') else np.array(inter_feat[uid_field])
    item_ids = inter_feat[iid_field].numpy() if hasattr(inter_feat[iid_field], 'numpy') else np.array(inter_feat[iid_field])
    
    if time_field and time_field in inter_feat.columns:
        timestamps = inter_feat[time_field].numpy() if hasattr(inter_feat[time_field], 'numpy') else np.array(inter_feat[time_field])
        # Sort by timestamp for each user
        for uid, iid, ts in zip(user_ids, item_ids, timestamps):
            user_sequences[int(uid)].append((int(iid), ts))
        
        # Sort each user's sequence by timestamp
        for uid in user_sequences:
            user_sequences[uid] = [item for item, _ in sorted(user_sequences[uid], key=lambda x: x[1])]
    else:
        # No timestamp, assume data is already in order
        for uid, iid in zip(user_ids, item_ids):
            user_sequences[int(uid)].append(int(iid))
    
    return dict(user_sequences), ds.item_num


def generate_user_metrics_data(
    user_sequences: Dict[int, List[int]],
    recallers: Dict[str, RecBoleRecaller],
    final_k: int = 50,
    min_history_len: int = 1,
    ks: List[int] = [5, 10, 20, 50],
) -> Dict[int, Dict]:
    """
    Generate metrics data for all users.
    
    For each user, for each position (item as ground truth),
    compute metrics for all recallers.
    
    Returns:
        {
            user_id: {
                "sequence": [item1, item2, ...],
                "position_metrics": {
                    "pos": {
                        "recaller_name": {"ndcg@10": ..., ...},
                        ...
                        "best_recaller": "xxx"
                    },
                    ...
                }
            }
        }
    """
    recaller_names = sorted(recallers.keys())
    user_data = {}
    
    # Statistics
    total_positions = 0
    best_recaller_counts = defaultdict(int)
    recaller_metrics_sum = {name: defaultdict(float) for name in recaller_names}
    recaller_metrics_count = {name: 0 for name in recaller_names}
    
    print(f"Processing {len(user_sequences)} users...")
    
    for uid, sequence in tqdm(user_sequences.items(), desc="Processing users"):
        if len(sequence) < min_history_len + 1:
            # Not enough items for meaningful evaluation
            continue
        
        user_entry = {
            "sequence": sequence,
            "position_metrics": {}
        }
        
        # For each position (starting from min_history_len), use that position's item as ground truth
        # History = items before that position
        for pos in range(min_history_len, len(sequence)):
            history = sequence[:pos]
            gt_item = sequence[pos]
            full_sequence = sequence  # All items for masking
            
            position_metrics = {}
            best_ndcg = -1
            best_recaller = None
            
            for recaller_name in recaller_names:
                recaller = recallers[recaller_name]
                
                # Generate recommendations
                # Pass full_sequence and gt_items for proper masking
                items = recaller.recall(
                    uid, final_k, history, 
                    full_hist=full_sequence, 
                    gt_items=[gt_item]
                )
                item_ids = [item[0] for item in items] if items else []
                
                # Compute metrics
                metrics = compute_metrics(item_ids, [gt_item], ks)
                position_metrics[recaller_name] = metrics
                
                # Track best recaller
                if metrics.get("ndcg@10", 0) > best_ndcg:
                    best_ndcg = metrics.get("ndcg@10", 0)
                    best_recaller = recaller_name
                
                # Aggregate for statistics
                for metric_name, value in metrics.items():
                    recaller_metrics_sum[recaller_name][metric_name] += value
                recaller_metrics_count[recaller_name] += 1
            
            position_metrics["best_recaller"] = best_recaller
            position_metrics["ground_truth"] = gt_item
            
            user_entry["position_metrics"][str(pos)] = position_metrics
            
            best_recaller_counts[best_recaller] += 1
            total_positions += 1
        
        if user_entry["position_metrics"]:
            user_data[uid] = user_entry
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary: {len(user_data)} users, {total_positions} positions")
    print(f"{'='*60}")
    
    print(f"\nAverage Metrics by Recaller:")
    header = f"{'Recaller':<15}" + "".join([f"{'NDCG@'+str(k):>12}" for k in ks])
    print(header)
    print("-" * (15 + 12 * len(ks)))
    
    for recaller_name in recaller_names:
        count = recaller_metrics_count[recaller_name]
        if count > 0:
            row = f"{recaller_name:<15}"
            for k in ks:
                avg = recaller_metrics_sum[recaller_name][f"ndcg@{k}"] / count
                row += f"{avg:>12.4f}"
            print(row)
    
    print(f"\nBest Recaller Distribution:")
    for name, count in sorted(best_recaller_counts.items(), key=lambda x: -x[1]):
        pct = count / total_positions * 100 if total_positions else 0
        print(f"  {name}: {count} ({pct:.1f}%)")
    
    return user_data


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate per-user per-position recaller metrics data'
    )
    
    parser.add_argument('--dataset', type=str, default='Amazon_All_Beauty')
    parser.add_argument('--data_path', type=str, default='./dataset')
    parser.add_argument('--output_dir', type=str, default='./GRPO/recaller_metrics_data')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    
    parser.add_argument('--recbole_models', type=str, nargs='+', 
                        default=['BPR', 'SASRec', 'LightGCN', 'Pop', 'ItemKNN', 'GRU4Rec'],
                        help='List of recaller models to use')
    parser.add_argument('--use_all_recallers', action='store_true',
                        help='Use all available recaller models')
    
    parser.add_argument('--final_k', type=int, default=50)
    parser.add_argument('--min_history_len', type=int, default=5)
    parser.add_argument('--ks', type=int, nargs='+', default=[5, 10, 20, 50])
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    print("="*60)
    print("Per-User Per-Position Recaller Metrics Generation")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {args.output_dir}")
    
    # Determine which recallers to use
    if args.use_all_recallers:
        recaller_models = ALL_RECALLER_MODELS
    else:
        recaller_models = args.recbole_models
    
    print(f"Recallers: {recaller_models}")
    print(f"K values: {args.ks}")
    
    # Get full user sequences from raw dataset
    print("\n" + "="*60)
    print("Loading user sequences from raw dataset...")
    print("="*60)
    
    user_sequences, num_items = get_user_full_sequences(
        args.dataset, args.data_path, args.seed
    )
    
    print(f"Total users: {len(user_sequences)}")
    print(f"Total items: {num_items}")
    
    seq_lengths = [len(seq) for seq in user_sequences.values()]
    print(f"Sequence length - min: {min(seq_lengths)}, max: {max(seq_lengths)}, avg: {np.mean(seq_lengths):.1f}")
    
    # Initialize recallers
    print("\n" + "="*60)
    print("Initializing recallers...")
    print("="*60)
    
    recallers = initialize_recallers(
        model_names=recaller_models,
        dataset_name=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        data_path=args.data_path,
        seed=args.seed,
        use_latest_checkpoint=True,
        num_items=num_items
    )
    
    # Generate data
    print("\n" + "="*60)
    print("Generating per-position metrics for all users...")
    print("="*60)
    
    user_data = generate_user_metrics_data(
        user_sequences,
        recallers,
        final_k=args.final_k,
        min_history_len=args.min_history_len,
        ks=args.ks,
    )
    
    # Save data
    print("\n" + "="*60)
    print("Saving data...")
    print("="*60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    recaller_combo = "_".join(sorted(recallers.keys()))
    
    output_file = os.path.join(
        args.output_dir,
        f"{args.dataset}_user_position_metrics_{recaller_combo}.json"
    )
    
    # Convert int keys to string for JSON serialization
    serializable_data = {str(uid): data for uid, data in user_data.items()}
    
    with open(output_file, 'w') as f:
        json.dump(serializable_data, f)
    
    print(f"Saved to {output_file}")
    
    # Also save a compact version with just essential info
    compact_file = os.path.join(
        args.output_dir,
        f"{args.dataset}_user_position_metrics_{recaller_combo}_compact.json"
    )
    
    # Compute summary stats
    total_positions = sum(len(d["position_metrics"]) for d in user_data.values())
    best_counts = defaultdict(int)
    for user_entry in user_data.values():
        for pos_data in user_entry["position_metrics"].values():
            best_counts[pos_data["best_recaller"]] += 1
    
    summary = {
        "dataset": args.dataset,
        "recallers": list(recallers.keys()),
        "num_users": len(user_data),
        "total_positions": total_positions,
        "ks": args.ks,
        "best_recaller_distribution": dict(best_counts),
    }
    
    summary_file = os.path.join(
        args.output_dir,
        f"{args.dataset}_summary_{recaller_combo}.json"
    )
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved summary to {summary_file}")
    print("\nDone!")


if __name__ == "__main__":
    main()
