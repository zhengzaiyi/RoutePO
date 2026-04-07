"""
分析数据集标签噪声的脚本

诊断问题：
1. 不同 recaller 之间 NDCG 差距有多大？
2. 同一用户在不同历史长度下标签一致性如何？
3. 标签的置信度分布如何？
"""

import argparse
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datasets import Dataset
from collections import defaultdict
import matplotlib.pyplot as plt

def analyze_dataset_noise(data_path: str):
    """分析数据集的标签噪声"""
    
    print(f"\n{'='*60}")
    print("Label Noise Analysis")
    print(f"{'='*60}")
    print(f"Data path: {data_path}")
    
    # 加载数据
    train_dataset = Dataset.load_from_disk(f'{data_path}/train')
    
    with open(f'{data_path}/label_mapping.json', 'r') as f:
        labels = json.load(f)
        id2label = {int(k): v for k, v in labels["id2label"].items()}
    
    print(f"\nDataset size: {len(train_dataset)} samples")
    print(f"Classes: {list(id2label.values())}")
    
    # 统计类别分布
    class_counts = defaultdict(int)
    for item in train_dataset:
        class_counts[item["best_recaller"]] += 1
    
    print(f"\n--- Class Distribution ---")
    total = len(train_dataset)
    for name, count in sorted(class_counts.items()):
        print(f"  {name}: {count} ({count/total*100:.1f}%)")
    
    # 分析 NDCG 差距
    if "best_ndcg" in train_dataset.column_names:
        ndcg_values = train_dataset["best_ndcg"]
        print(f"\n--- Best NDCG Statistics ---")
        print(f"  Mean: {np.mean(ndcg_values):.4f}")
        print(f"  Std:  {np.std(ndcg_values):.4f}")
        print(f"  Min:  {np.min(ndcg_values):.4f}")
        print(f"  Max:  {np.max(ndcg_values):.4f}")
        
        # 低 NDCG 样本占比（可能是噪声标签）
        low_ndcg_threshold = 0.05
        low_ndcg_count = sum(1 for n in ndcg_values if n < low_ndcg_threshold)
        print(f"\n  Samples with NDCG < {low_ndcg_threshold}: {low_ndcg_count} ({low_ndcg_count/total*100:.1f}%)")
        print(f"  (These samples may have noisy labels since all recallers perform poorly)")
    
    # 分析同一用户的标签一致性
    print(f"\n--- User Label Consistency ---")
    user_labels = defaultdict(list)
    for item in train_dataset:
        user_labels[item["user_id"]].append(item["best_recaller"])
    
    # 统计有多少用户有不一致的标签
    users_with_multiple_samples = [uid for uid, labels in user_labels.items() if len(labels) > 1]
    inconsistent_users = []
    
    for uid in users_with_multiple_samples:
        labels_set = set(user_labels[uid])
        if len(labels_set) > 1:
            inconsistent_users.append(uid)
    
    print(f"  Total users: {len(user_labels)}")
    print(f"  Users with multiple samples: {len(users_with_multiple_samples)}")
    print(f"  Users with inconsistent labels: {len(inconsistent_users)} ({len(inconsistent_users)/max(1,len(users_with_multiple_samples))*100:.1f}%)")
    
    if inconsistent_users:
        # 展示一些不一致的例子
        print(f"\n  Example inconsistent users (showing first 5):")
        for uid in inconsistent_users[:5]:
            labels = user_labels[uid]
            unique_labels = list(set(labels))
            print(f"    User {uid}: {len(labels)} samples -> labels: {unique_labels}")
    
    # 分析标签的边界情况
    # 计算每个样本的"置信度"（best_ndcg 与 second_best_ndcg 的差距）
    # 这需要原始的 recaller_scores，如果数据中没有保存，则跳过
    
    print(f"\n--- Noise Mitigation Suggestions ---")
    
    # 计算熵（类别分布的不确定性）
    probs = np.array(list(class_counts.values())) / total
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(len(class_counts))
    
    print(f"\n  Class distribution entropy: {entropy:.4f} (max: {max_entropy:.4f})")
    
    if entropy < max_entropy * 0.8:
        print(f"  WARNING: Class imbalance detected! Consider using --balance_classes")
    
    if len(inconsistent_users) / max(1, len(users_with_multiple_samples)) > 0.3:
        print(f"  WARNING: High user inconsistency! Consider:")
        print(f"    - Using --filter_high_variance_users")
        print(f"    - Reducing data augmentation (increase MIN_HISTORY_FOR_AUGMENTATION)")
    
    if "best_ndcg" in train_dataset.column_names:
        avg_ndcg = np.mean(ndcg_values)
        if avg_ndcg < 0.1:
            print(f"  WARNING: Average NDCG is very low ({avg_ndcg:.4f})!")
            print(f"    - This suggests recallers perform similarly (high noise)")
            print(f"    - Consider using --filter_close_recallers with higher threshold")
    
    return {
        "total_samples": total,
        "class_distribution": dict(class_counts),
        "users_with_inconsistent_labels": len(inconsistent_users),
        "entropy": entropy,
    }


def analyze_recaller_score_gap(data_path: str):
    """
    更深入地分析 recaller 之间的分数差距
    需要重新计算（因为原始 recaller_scores 没有保存）
    """
    # 这里可以加载原始数据并重新计算每个样本的所有 recaller scores
    # 但这需要较长时间，所以作为可选功能
    pass


def suggest_hyperparameters(analysis_results: dict):
    """基于分析结果建议超参数"""
    
    print(f"\n{'='*60}")
    print("Recommended Hyperparameters")
    print(f"{'='*60}")
    
    total = analysis_results["total_samples"]
    entropy = analysis_results["entropy"]
    inconsistent_ratio = analysis_results["users_with_inconsistent_labels"] / max(1, total)
    
    # 基础建议
    suggestions = []
    
    # 1. 类别平衡
    class_dist = analysis_results["class_distribution"]
    max_class = max(class_dist.values())
    min_class = min(class_dist.values())
    imbalance_ratio = max_class / max(1, min_class)
    
    if imbalance_ratio > 2:
        suggestions.append("--balance_classes --balance_strategy undersample")
        print(f"\n1. Class imbalance ratio: {imbalance_ratio:.1f}x")
        print(f"   -> Use: --balance_classes --balance_strategy undersample")
    
    # 2. 过滤接近的 recaller
    suggestions.append("--filter_close_recallers --close_threshold 0.1")
    print(f"\n2. Filter ambiguous samples (high noise)")
    print(f"   -> Use: --filter_close_recallers --close_threshold 0.1")
    print(f"   (Try 0.05, 0.1, 0.15 and compare)")
    
    # 3. 过滤高方差用户
    if inconsistent_ratio > 0.1:
        suggestions.append("--filter_high_variance_users --variance_threshold 0.15")
        print(f"\n3. User inconsistency is high ({inconsistent_ratio*100:.1f}%)")
        print(f"   -> Use: --filter_high_variance_users --variance_threshold 0.15")
    
    # 4. 学习率建议
    print(f"\n4. Training suggestions:")
    print(f"   -> Lower learning rate: --learning_rate 5e-6 (instead of 1e-5)")
    print(f"   -> More epochs with early stopping")
    print(f"   -> Larger batch size for gradient stability")
    
    # 5. Label smoothing（需要代码修改）
    print(f"\n5. Label smoothing (requires code change):")
    print(f"   -> Add label smoothing in loss function")
    print(f"   -> Or use soft labels based on NDCG differences")
    
    # 生成推荐的完整命令
    print(f"\n{'='*60}")
    print("Recommended Command")
    print(f"{'='*60}")
    print("./pure_sft_v2_test.sh <dataset> <gpu> custom \\")
    print("    --filter_close_recallers --close_threshold 0.1 \\")
    print("    --filter_high_variance_users --variance_threshold 0.15 \\")
    print("    --balance_classes --balance_strategy undersample")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to the dataset folder (e.g., GRPO/pure_models/Amazon_All_Beauty/...)")
    args = parser.parse_args()
    
    results = analyze_dataset_noise(args.data_path)
    suggest_hyperparameters(results)
















