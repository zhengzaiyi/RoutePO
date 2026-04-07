"""
Utilities for plotting evaluation results
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict


def plot_recaller_pair_improvement_heatmap(all_results: Dict, all_recallers: List[str], score_norm: str, metric: str = 'recall@50'):
    """
    绘制recaller pair的improvement heatmap
    
    Args:
        all_results: evaluate_score_norms返回的结果字典
        all_recallers: 所有recaller名称列表
        score_norm: 要绘制的score_norm方法
        metric: 要显示的metric（默认'recall@50'）
    """
    # 创建矩阵存储improvement值
    n = len(all_recallers)
    improvement_matrix = np.full((n, n), np.nan)
    
    # 填充矩阵
    for combo_key, combo_results in all_results.items():
        if score_norm not in combo_results:
            continue
        
        # 解析combo_key (格式: "recaller1_recaller2")
        parts = combo_key.split('_')
        if len(parts) == 2:
            r1, r2 = parts[0], parts[1]
            if r1 in all_recallers and r2 in all_recallers:
                result = combo_results[score_norm]
                improvements = result.get('improvements', {})
                improvement_value = improvements.get(metric, np.nan)
                
                # 找到索引
                idx1 = all_recallers.index(r1)
                idx2 = all_recallers.index(r2)
                
                # 填充对称矩阵
                improvement_matrix[idx1, idx2] = improvement_value
                improvement_matrix[idx2, idx1] = improvement_value
    
    # 创建mask来隐藏对角线（自己与自己的pair没有意义）
    mask = np.eye(n, dtype=bool)
    
    # 绘制heatmap
    plt.figure(figsize=(max(8, n * 0.8), max(6, n * 0.7)))
    sns.heatmap(
        improvement_matrix,
        xticklabels=all_recallers,
        yticklabels=all_recallers,
        annot=True,
        fmt='.4f',
        cmap='RdYlGn',
        center=0,
        mask=mask,
        cbar_kws={'label': f'{metric} Improvement'},
        square=True,
        linewidths=0.5
    )
    plt.title(f'Recaller Pair {metric} Improvement Heatmap\n(Score Norm: {score_norm})', fontsize=14, pad=20)
    plt.xlabel('Recaller 2', fontsize=12)
    plt.ylabel('Recaller 1', fontsize=12)
    plt.tight_layout()
    plt.show()
