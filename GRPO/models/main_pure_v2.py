"""
main_pure_v2.py - 数据集质量改进版本

新增开关：
- --filter_close_recallers: 抛弃不同 recallers 非常接近的 sample
- --close_threshold: 判断"接近"的阈值 (default: 0.05)
- --filter_high_variance_users: 抛弃同一用户变化很大的 sample
- --variance_threshold: 用户方差阈值 (default: 0.1)
- --max_ground_truth: ground truth item 能多就多 (default: 10, 原来是 5)
- --add_recaller_metrics_to_prompt: 在 prompt 中声明该用户在 evaluation set 上的最佳 recaller

新增 V2.1 - 标签噪声处理：
- --use_soft_labels: 使用基于 NDCG 分布的软标签而非硬标签
- --soft_label_temperature: 软标签的 temperature (越小越接近硬标签)
- --label_smoothing: 标签平滑系数 (0.0 = 无平滑, 0.1 = 常用值)
"""

import argparse
import json
import os
from functools import partial
from typing import List, Dict, Tuple
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datetime import datetime
import numpy as np
from datasets import Dataset
from tqdm import tqdm
from collections import defaultdict
from transformers import (
    AutoModelForSequenceClassification, 
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer, 
    TrainingArguments, 
    DataCollatorWithPadding, 
    DataCollatorForSeq2Seq,
    Trainer
)
from transformers.trainer_utils import get_last_checkpoint
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from GRPO.core.agents import UserProfileAgent
from GRPO.core.data import load_dataset
from GRPO.models.main import initialize_recallers
from GRPO.core.recallers import RecBoleRecaller
from GRPO.core.utils import set_seed, build_prompt, ndcg_at_k, recall_at_k
from GRPO.models.soft_utils import multi_channel_recall_score, compute_ndcg_at_k as soft_ndcg
from GRPO.trainers.trl_trainer import GRPOTrainer
from GRPO.models.soft_utils import multi_channel_recall_score, compute_ndcg_at_k
from trl import GRPOConfig


# Dataset generation configuration
MIN_HISTORY_FOR_AUGMENTATION = 30
AUGMENTATION_STEP = 10


def build_prompt_with_hint(
    user_profile: str,
    available_models: List[str],
    best_recaller: str,
    type: str = 'classification'
) -> str:
    """
    构建包含最佳 recaller 提示的 prompt
    
    Args:
        user_profile: 用户画像文本
        available_models: 可用的 recaller 名称列表
        best_recaller: 该用户在 evaluation set 上的最佳 recaller 名称
        type: prompt 类型
    """
    # 构建提示信息
    if best_recaller:
        hint_str = f"\n\nNote: Based on this user's historical interactions, {best_recaller} has shown the best performance.\n"
    else:
        hint_str = ""
    
    # 使用原始的 build_prompt 并追加提示信息
    base_prompt = build_prompt(user_profile, available_models=available_models, type=type)
    
    # 在 prompt 的适当位置插入提示信息（在任务描述之前）
    insert_marker = "Based on"
    if insert_marker in base_prompt:
        idx = base_prompt.find(insert_marker)
        enhanced_prompt = base_prompt[:idx] + hint_str + "\n" + base_prompt[idx:]
    else:
        enhanced_prompt = base_prompt + hint_str
    
    return enhanced_prompt


def _clean_history(hist: List[int]) -> List[int]:
    """Remove padding zeros from history."""
    return hist[:hist.index(0)] if 0 in hist else hist


def _find_best_recaller(uid: int, recallers: Dict, recall_hist: List[int], 
                        label_gt: List[int], final_k: int) -> Tuple[str, float, Dict]:
    """Find best recaller: use recall_hist to recall, evaluate against label_gt."""
    scores = {}
    best_ndcg, best_name = -1, None
    for name, recaller in recallers.items():
        items = recaller.recall(uid, final_k, recall_hist)
        item_ids = [item[0] for item in items] if items else []
        ndcg = ndcg_at_k(item_ids, label_gt, k=final_k)
        scores[name] = {"ndcg": ndcg, "recall": recall_at_k(item_ids, label_gt, k=final_k)}
        if ndcg > best_ndcg:
            best_ndcg, best_name = ndcg, name
    return best_name, best_ndcg, scores


def create_sft_dataset_v3(
    profile_agent: UserProfileAgent,
    user_ids: List[int],
    histories: List[List[int]],
    target_items: List[int],
    recallers: Dict[str, RecBoleRecaller],
    final_k: int,
    profile_cutoff: int = 20,
    train_ratio: float = 0.7,
    eval_ratio: float = 0.2,
    inner_label_ratio: float = 0.1,  # 从 recall_hist 内部抽取的比例 (用于 prompt hint)
    test_gt_ratio: float = 0.1,      # 用于真正 label 的比例
) -> Tuple[Dataset, Dataset, Dataset, Dict[str, int], Dict[int, str]]:
    """
    Create train/eval/test datasets WITHOUT data leakage.
    
    历史划分:
    ┌─────────────────────────────────┬─────────────┐
    │  recall_hist (90%)              │  test_gt    │
    │                                 │  (10%)      │
    └─────────────────────────────────┴─────────────┘
    
    recall_hist 内部再划分 (用于 prompt hint，无泄露):
    ┌────────────────────┬────────────┐
    │  inner_recall      │ inner_gt   │
    │  (90%)             │ (10%)      │
    └────────────────────┴────────────┘
    
    1. prompt_hint: 用 inner_recall 召回，用 inner_gt 评估 → 放在 prompt 中
    2. label: 用 recall_hist 召回，用 test_gt 评估 → 真正的 ground truth
    """
    recaller_names = sorted(recallers.keys())
    label2id = {name: i for i, name in enumerate(recaller_names)}
    id2label = {i: name for name, i in label2id.items()}
    
    all_samples = []
    # Collect per-recaller metrics on the label split (recall_hist vs test_gt)
    base_metrics = {name: {"ndcg": [], "recall": []} for name in recaller_names}
    
    for i, uid in enumerate(tqdm(user_ids, desc="Building samples")):
        hist = _clean_history(histories[i])
        if len(hist) < 15:
            continue
        
        # 第一层划分: recall_hist (90%) | test_gt (10%)
        n_test = max(2, int(len(hist) * test_gt_ratio))
        test_gt = hist[-n_test:]
        recall_hist = hist[:-n_test]
        
        if len(recall_hist) < 10:
            continue
        
        # 第二层划分: 用于 prompt hint (无泄露)
        n_inner = max(2, int(len(recall_hist) * inner_label_ratio))
        inner_gt = recall_hist[-n_inner:]
        inner_recall = recall_hist[:-n_inner]
        
        if len(inner_recall) < 5:
            continue
        
        # 1. Prompt hint: 用 inner_recall 召回，用 inner_gt 评估
        hint_best, _, _ = _find_best_recaller(uid, recallers, inner_recall, inner_gt, final_k)
        
        # 2. Label: 用 recall_hist 召回，用 test_gt 评估 (真正的 ground truth)
        label_best, _, label_scores = _find_best_recaller(uid, recallers, recall_hist, test_gt, final_k)
        # 聚合各个 base recaller 的指标
        for name, score_dict in label_scores.items():
            base_metrics[name]["ndcg"].append(score_dict["ndcg"])
            base_metrics[name]["recall"].append(score_dict["recall"])
        
        # 构建 prompt（基于 recall_hist，包含 hint）
        profile = profile_agent.forward(uid, recall_hist, cut_off=profile_cutoff)
        prompt = build_prompt_with_hint(profile, recaller_names, hint_best, type='classification')
        
        all_samples.append({
            "text": prompt,
            "prompt": prompt,
            "labels": label2id[label_best],      # 真正的 label (基于 test_gt)
            "best_recaller": label_best,
            "prompt_hint": hint_best,            # prompt 中的 hint (基于 inner_gt)
            "user_id": uid,
            "recall_hist": recall_hist,
            "test_gt": test_gt,
        })
    
    # 划分 train/eval/test
    np.random.shuffle(all_samples)
    n = len(all_samples)
    n_train = int(n * train_ratio)
    n_eval = int(n * eval_ratio)
    
    train_samples = all_samples[:n_train]
    eval_samples = all_samples[n_train:n_train + n_eval]
    test_samples = all_samples[n_train + n_eval:]
    
    # 打印统计
    print(f"\n{'='*60}\nDataset Statistics (V3 - No Leakage)\n{'='*60}")
    print(f"Total: {n}, Train: {len(train_samples)}, Eval: {len(eval_samples)}, Test: {len(test_samples)}")
    
    dist = defaultdict(int)
    for s in all_samples:
        dist[s["best_recaller"]] += 1
    print(f"\nLabel Distribution:")
    for m, c in sorted(dist.items()):
        print(f"  {m}: {c} ({c/n*100:.1f}%)")
    
    # 打印各个 base recaller 在 label split 上的平均指标
    print(f"\nBase Recaller Evaluation (recall_hist -> test_gt):")
    for name in recaller_names:
        ndcg_vals = base_metrics[name]["ndcg"]
        recall_vals = base_metrics[name]["recall"]
        if ndcg_vals:
            print(f"  {name}: NDCG@{final_k}={np.mean(ndcg_vals):.4f}, Recall@{final_k}={np.mean(recall_vals):.4f} (n={len(ndcg_vals)})")
        else:
            print(f"  {name}: no valid samples")
    
    return (
        Dataset.from_list(train_samples),
        Dataset.from_list(eval_samples),
        Dataset.from_list(test_samples),
        label2id,
        id2label,
    )


def compute_metrics(eval_pred):
    """Compute metrics for classification"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    # 处理 soft labels 的情况
    if len(eval_pred.label_ids.shape) > 1:
        # soft labels: 取 argmax
        labels = np.argmax(eval_pred.label_ids, axis=1)
    else:
        labels = eval_pred.label_ids
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1_macro': f1_score(labels, predictions, average='macro'),
        'f1_weighted': f1_score(labels, predictions, average='weighted')
    }


class SoftLabelTrainer(Trainer):
    """支持软标签的 Trainer"""
    
    def __init__(self, *args, use_soft_labels=False, label_smoothing=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_soft_labels = use_soft_labels
        self.label_smoothing = label_smoothing
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """自定义 loss 计算，支持软标签和标签平滑"""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.use_soft_labels and labels.dim() > 1:
            # 软标签: 使用 KL 散度或 cross entropy
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            loss = torch.nn.functional.kl_div(log_probs, labels, reduction='batchmean')
        elif self.label_smoothing > 0:
            # 标签平滑
            num_classes = logits.size(-1)
            smooth_labels = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
            smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - self.label_smoothing)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            loss = -(smooth_labels * log_probs).sum(dim=-1).mean()
        else:
            # 标准交叉熵
            loss = torch.nn.functional.cross_entropy(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


def compute_metrics_seq2seq(eval_pred, tokenizer, label2id):
    """Compute metrics for seq2seq generation"""
    predictions, labels = eval_pred
    
    if predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=-1)
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    pred_label_ids = []
    true_label_ids = []
    
    for pred_text, true_text in zip(decoded_preds, decoded_labels):
        pred_text = pred_text.strip()
        true_text = true_text.strip()
        
        pred_id = label2id.get(pred_text, -1)
        true_id = label2id.get(true_text, -1)
        
        pred_label_ids.append(pred_id)
        true_label_ids.append(true_id)
    
    pred_label_ids = np.array(pred_label_ids)
    true_label_ids = np.array(true_label_ids)
    
    valid_mask = (pred_label_ids != -1) & (true_label_ids != -1)
    
    if np.sum(valid_mask) == 0:
        return {'accuracy': 0.0, 'f1_macro': 0.0, 'f1_weighted': 0.0, 'valid_predictions': 0}
    
    valid_preds = pred_label_ids[valid_mask]
    valid_labels = true_label_ids[valid_mask]
    
    return {
        'accuracy': accuracy_score(valid_labels, valid_preds),
        'f1_macro': f1_score(valid_labels, valid_preds, average='macro'),
        'f1_weighted': f1_score(valid_labels, valid_preds, average='weighted'),
        'valid_predictions': np.sum(valid_mask) / len(pred_label_ids)
    }


def evaluate_pure_model(model, tokenizer, test_dataset, id2label, recallers=None, final_k=50):
    """Evaluate the pure classification model (V3: uses test_gt from dataset)."""
    device = model.device
    model.eval()
    
    all_predictions, all_labels = [], []
    recommendation_results = []
    
    with torch.no_grad():
        for example in tqdm(test_dataset, desc="Evaluating"):
            inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, 
                             max_length=1536, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]
            
            all_predictions.append(prediction)
            all_labels.append(example["labels"])
            
            if recallers is not None:
                pred_name = id2label[prediction]
                true_name = example["best_recaller"]
                uid = example["user_id"]
                recall_hist = example["recall_hist"]
                test_gt = example["test_gt"]  # V3: 用 test_gt 评估
                
                # Predicted recaller
                if pred_name in recallers:
                    items = recallers[pred_name].recall(uid, final_k, recall_hist)
                    pred_ids = [item[0] for item in items] if items else []
                    pred_ndcg = ndcg_at_k(pred_ids, test_gt, k=final_k)
                    pred_recall = recall_at_k(pred_ids, test_gt, k=final_k)
                else:
                    pred_ids, pred_ndcg, pred_recall = [], 0.0, 0.0
                
                # True (label) recaller
                if true_name in recallers:
                    items = recallers[true_name].recall(uid, final_k, recall_hist)
                    true_ids = [item[0] for item in items] if items else []
                    true_ndcg = ndcg_at_k(true_ids, test_gt, k=final_k)
                    true_recall = recall_at_k(true_ids, test_gt, k=final_k)
                else:
                    true_ids, true_ndcg, true_recall = [], 0.0, 0.0
                
                recommendation_results.append({
                    "user_id": uid,
                    "predicted_recaller": pred_name,
                    "true_recaller": true_name,
                    "predicted_ndcg": pred_ndcg,
                    "true_ndcg": true_ndcg,
                    "predicted_recall": pred_recall,
                    "true_recall": true_recall,
                    "correct_prediction": pred_name == true_name,
                })
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    
    print(f"\nAccuracy: {accuracy:.4f}, F1 Macro: {f1_macro:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=list(id2label.values())))
    
    result = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "predictions": all_predictions.tolist(),
        "labels": all_labels.tolist(),
        "confusion_matrix": confusion_matrix(all_labels, all_predictions).tolist()
    }

    if recommendation_results:
        avg_pred_ndcg = np.mean([r["predicted_ndcg"] for r in recommendation_results])
        avg_true_ndcg = np.mean([r["true_ndcg"] for r in recommendation_results])
        avg_pred_recall = np.mean([r["predicted_recall"] for r in recommendation_results])
        avg_true_recall = np.mean([r["true_recall"] for r in recommendation_results])
        correct_predictions = sum([r["correct_prediction"] for r in recommendation_results])
        
        print(f"\nRecommendation Metrics:")
        print(f"Average Predicted NDCG@{final_k}: {avg_pred_ndcg:.4f}")
        print(f"Average True Best NDCG@{final_k}: {avg_true_ndcg:.4f}")
        print(f"Average Predicted Recall@{final_k}: {avg_pred_recall:.4f}")
        print(f"Average True Best Recall@{final_k}: {avg_true_recall:.4f}")
        print(f"Correct Recaller Predictions: {correct_predictions}/{len(recommendation_results)} ({correct_predictions/len(recommendation_results)*100:.1f}%)")
        
        result.update({
            "recommendation_results": recommendation_results,
            "avg_predicted_ndcg": avg_pred_ndcg,
            "avg_true_ndcg": avg_true_ndcg,
            "avg_predicted_recall": avg_pred_recall,
            "avg_true_recall": avg_true_recall,
            "recaller_prediction_accuracy": correct_predictions / len(recommendation_results)
        })
    
    return result


def multi_channel_recall_average(
    recallers: Dict,
    recaller_names: List[str],
    user_id: int,
    history: List[int],
    total_k: int
) -> List[Tuple[int, float]]:
    """Multi-channel recall using average (uniform) weights."""
    candidates = defaultdict(float)
    num_recallers = len(recaller_names)
    weight = 1.0 / num_recallers
    
    for name in recaller_names:
        name_lower = name.lower()
        if name_lower in recallers:
            items = recallers[name_lower].recall(user_id, total_k, history)
            for item_id, score in items:
                candidates[item_id] += score * weight
    
    sorted_items = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:total_k]


def evaluate_multi_channel_recall(
    model, 
    tokenizer, 
    test_dataset, 
    recallers: Dict[str, RecBoleRecaller],
    recaller_names: List[str],
    final_k: int = 50,
    use_softmax_weights: bool = True
):
    """Evaluate model using multi-channel recall (V3: uses test_gt from dataset)."""
    device = model.device
    model.eval()
    
    metrics = {
        "single_select": defaultdict(list),
        "multi_channel": defaultdict(list),
        "avg_score_weight": defaultdict(list),
    }
    for name in recaller_names:
        metrics[name] = defaultdict(list)
    
    with torch.no_grad():
        for example in tqdm(test_dataset, desc="Multi-Channel Eval"):
            inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, 
                             max_length=1536, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            logits = outputs.logits[0]
            softmax_weights = torch.softmax(logits, dim=-1)
            
            uid = example["user_id"]
            recall_hist = example["recall_hist"]
            test_gt = example["test_gt"]  # V3: 用 test_gt 评估
            
            # 1. Single recaller selection (argmax)
            pred_name = recaller_names[logits.argmax().item()]
            if pred_name in recallers:
                items = recallers[pred_name].recall(uid, final_k, recall_hist)
                single_rec = [item[0] for item in items]
                for k in [10, 20, 50]:
                    metrics["single_select"][f"ndcg@{k}"].append(ndcg_at_k(single_rec, test_gt, k))
                    metrics["single_select"][f"recall@{k}"].append(recall_at_k(single_rec, test_gt, k))
            
            # 2. Multi-channel recall with softmax weights
            if use_softmax_weights:
                candidates = multi_channel_recall_score(
                    softmax_weights, recallers, recaller_names, uid, recall_hist, final_k
                )
                multi_rec = [item_id for item_id, _ in candidates]
                for k in [10, 20, 50]:
                    metrics["multi_channel"][f"ndcg@{k}"].append(ndcg_at_k(multi_rec, test_gt, k))
                    metrics["multi_channel"][f"recall@{k}"].append(recall_at_k(multi_rec, test_gt, k))
            
            # 3. Multi-channel recall with average weights
            avg_candidates = multi_channel_recall_average(recallers, recaller_names, uid, recall_hist, final_k)
            avg_rec = [item_id for item_id, _ in avg_candidates]
            for k in [10, 20, 50]:
                metrics["avg_score_weight"][f"ndcg@{k}"].append(ndcg_at_k(avg_rec, test_gt, k))
                metrics["avg_score_weight"][f"recall@{k}"].append(recall_at_k(avg_rec, test_gt, k))
            
            # 4. Evaluate each base recaller
            for name in recaller_names:
                if name in recallers:
                    items = recallers[name].recall(uid, final_k, recall_hist)
                    base_rec = [item[0] for item in items] if items else []
                    for k in [10, 20, 50]:
                        metrics[name][f"ndcg@{k}"].append(ndcg_at_k(base_rec, test_gt, k))
                        metrics[name][f"recall@{k}"].append(recall_at_k(base_rec, test_gt, k))
    
    results = {}
    for method, method_metrics in metrics.items():
        results[method] = {}
        for metric_name, values in method_metrics.items():
            if values:
                results[method][metric_name] = np.mean(values)
    
    print("\n" + "="*60)
    print("Multi-Channel Recall Evaluation")
    print("="*60)
    
    print(f"\n--- Base Recaller Performance ---")
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
    
    if results.get("single_select") and results.get("multi_channel"):
        print(f"\n--- Model Selection vs Multi-Channel vs Avg Score-Weight ---")
        print(f"{'Metric':<12} {'Single':>12} {'Multi-Ch':>12} {'Avg-SW':>12} {'Best Base':>12} {'Multi Impr':>12}")
        print("-" * 72)
        for k in [10, 20, 50]:
            for metric in ['ndcg', 'recall']:
                key = f"{metric}@{k}"
                single = results["single_select"].get(key, 0)
                multi = results["multi_channel"].get(key, 0)
                avg_sw = results["avg_score_weight"].get(key, 0)
                best_base = max(results.get(name, {}).get(key, 0) for name in recaller_names)
                multi_impr = multi - best_base
                print(f"{key:<12} {single:>12.4f} {multi:>12.4f} {avg_sw:>12.4f} {best_base:>12.4f} {multi_impr:>+12.4f}")
    
    return results


def tokenize_function(examples, tokenizer, max_length=1536, seq2seq=False):
    if seq2seq:
        inputs = []
        targets = []
        
        for text, recaller in zip(examples["text"], examples["best_recaller"]):
            input_text = text + f"\n\nBest recaller:"
            target_text = f" {recaller}"
            
            inputs.append(input_text)
            targets.append(input_text + target_text)
        
        model_inputs = tokenizer(targets, padding="max_length", truncation=True, max_length=max_length)
        
        labels = []
        input_lengths = []
        
        for input_text in inputs:
            input_tokens = tokenizer(input_text, truncation=True, max_length=max_length)
            input_lengths.append(len(input_tokens["input_ids"]))
        
        for i, length in enumerate(input_lengths):
            label = model_inputs["input_ids"][i].copy()
            label[:length] = [-100] * length
            for j, token_id in enumerate(label):
                if token_id == tokenizer.pad_token_id:
                    label[j] = -100
            labels.append(label)
        
        model_inputs["labels"] = labels
        return model_inputs
    else:
        tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)
        tokenized["labels"] = examples["labels"]
        return tokenized


def get_paths(args):
    """Get standardized paths"""
    model_name = args.model_name.split('/')[-1]
    base = f"{args.output_dir}/{args.dataset}/{model_name}"
    recbole_models = args.recbole_models
    recbole_models.sort()
    recbole_models = '_'.join(recbole_models)
    
    # 添加 v2 后缀以区分
    suffix = "_v2"
    if args.filter_close_recallers:
        suffix += f"_fc{args.close_threshold}"
    if args.filter_high_variance_users:
        suffix += f"_fv{args.variance_threshold}"
    if args.max_ground_truth != 5:
        suffix += f"_gt{args.max_ground_truth}"
    if args.add_recaller_metrics_to_prompt:
        suffix += "_metrics"
    if args.balance_classes:
        suffix += f"_bal{args.balance_strategy[:2]}"  # "un" for undersample, "ov" for oversample
    # V2.1: 软标签和标签平滑
    if args.use_soft_labels:
        suffix += f"_soft{args.soft_label_temperature}"
    if args.label_smoothing > 0:
        suffix += f"_ls{args.label_smoothing}"
    
    return {
        "sft": f"{base}_pure_{f'seq2seq_' if args.seq2seq else ''}sft_{recbole_models}{suffix}",
        "data": f"{base}_pure_sft_data_{recbole_models}_{args.profile_cutoff}{suffix}",
        "grpo": f"{base}_pure_grpo_{recbole_models}{suffix}",
        "grpo_data": f"{base}_pure_grpo_data_{recbole_models}_{args.profile_cutoff}{suffix}"
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Pure Text SFT Training for Model Selection (V2 - Data Quality Improvements)')
    
    # Data
    parser.add_argument('--dataset', type=str, default='Amazon_All_Beauty')
    parser.add_argument('--data_path', type=str, default='./dataset')
    parser.add_argument('--output_dir', type=str, default='GRPO/pure_models')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    
    # Model
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--recbole_models', type=str, nargs='+', default=['BPR', 'SASRec', 'LightGCN'])
    
    # Tasks
    parser.add_argument('--gen_sft_data', action='store_true')
    parser.add_argument('--do_sft', action='store_true')
    parser.add_argument('--do_test_sft', action='store_true')
    parser.add_argument('--do_test_grpo', action='store_true')
    
    # ============== 新增: 数据质量改进开关 ==============
    parser.add_argument('--filter_close_recallers', action='store_true',
                       help='过滤掉不同 recaller 表现非常接近的样本')
    parser.add_argument('--close_threshold', type=float, default=0.05,
                       help='判断 recaller 表现"接近"的 NDCG 差异阈值')
    
    parser.add_argument('--filter_high_variance_users', action='store_true',
                       help='过滤掉同一用户在不同历史长度下表现变化很大的样本')
    parser.add_argument('--variance_threshold', type=float, default=0.1,
                       help='用户 NDCG 方差阈值')
    
    parser.add_argument('--max_ground_truth', type=int, default=5,
                       help='Ground truth item 数量上限 (原来是 5，可以增大)')
    
    parser.add_argument('--add_recaller_metrics_to_prompt', action='store_true',
                       help='在 prompt 中插入各 recaller 在该用户上的历史 metric')
    
    parser.add_argument('--balance_classes', action='store_true',
                       help='平衡各类别的样本数量')
    parser.add_argument('--balance_strategy', type=str, default='undersample',
                       choices=['undersample', 'oversample'],
                       help='类别平衡策略: undersample (下采样到最小类) 或 oversample (上采样到最大类)')
    
    # ============== V2.1: 标签噪声处理 ==============
    parser.add_argument('--use_soft_labels', action='store_true',
                       help='使用基于 NDCG 分布的软标签进行训练')
    parser.add_argument('--soft_label_temperature', type=float, default=0.1,
                       help='软标签 temperature (越小分布越尖锐，默认0.1)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                       help='标签平滑系数 (0.0=无平滑, 推荐0.1)')
    # ================================================
    
    # Training
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--max_length', type=int, default=1536)
    
    # Other
    parser.add_argument('--final_k', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--padding_side', type=str, default='right', choices=['left', 'right'])
    parser.add_argument('--profile_cutoff', type=int, default=20)
    parser.add_argument('--seq2seq', action='store_true')
    
    # GRPO training
    parser.add_argument('--do_grpo', action='store_true')
    parser.add_argument('--tau_gumbel', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--noise_scale', type=float, default=0.3)
    parser.add_argument('--use_soft_grpo_loss', action='store_true', default=True)
    parser.add_argument('--use_ppo_loss', action='store_true')
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--sync_ref_model', action='store_true')
    parser.add_argument('--ref_model_sync_steps', type=int, default=100)
    parser.add_argument('--num_generations', type=int, default=4)
    parser.add_argument('--grpo_lr', type=float, default=1e-6)
    parser.add_argument('--grpo_epochs', type=int, default=3)
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    paths = get_paths(args)
    
    print(f"\n{'='*60}")
    print(f"Main Pure V2.1 - Data Quality + Label Noise Handling")
    print(f"{'='*60}")
    print(f"Data Quality Settings:")
    print(f"  - filter_close_recallers: {args.filter_close_recallers} (threshold={args.close_threshold})")
    print(f"  - filter_high_variance_users: {args.filter_high_variance_users} (threshold={args.variance_threshold})")
    print(f"  - max_ground_truth: {args.max_ground_truth}")
    print(f"  - add_recaller_metrics_to_prompt: {args.add_recaller_metrics_to_prompt}")
    print(f"  - balance_classes: {args.balance_classes} (strategy={args.balance_strategy})")
    print(f"\nLabel Noise Handling (V2.1):")
    print(f"  - use_soft_labels: {args.use_soft_labels} (temperature={args.soft_label_temperature})")
    print(f"  - label_smoothing: {args.label_smoothing}")
    print(f"{'='*60}\n")

    grpo_config = GRPOConfig(
        output_dir=paths["grpo"],
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.grpo_epochs,
        learning_rate=args.grpo_lr,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=1000,
        save_total_limit=3,
        bf16=args.bf16,
        num_generations=args.num_generations,
        epsilon=args.epsilon,
        beta=args.beta,
        sync_ref_model=args.sync_ref_model,
        ref_model_sync_steps=args.ref_model_sync_steps,
        scale_rewards="group",
        report_to="wandb",
        run_name=paths["grpo"] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        seed=args.seed,
    )
    
    # Initialize components if needed
    if args.gen_sft_data or args.do_test_sft or args.do_test_grpo:
        inter_dataset = load_dataset(args.dataset, args.data_path, seed=args.seed)
        profile_agent = UserProfileAgent(inter_dataset, args.dataset)
        recallers = initialize_recallers(
            model_names=args.recbole_models,
            dataset_name=args.dataset,
            checkpoint_dir=args.checkpoint_dir,
            data_path=args.data_path,
            seed=args.seed,
            use_latest_checkpoint=True,
            num_items=inter_dataset.ds.item_num
        )
    
    # Generate SFT data (V3: 无数据泄露，一次生成 train/eval/test)
    if args.gen_sft_data:
        train_dataset, eval_dataset, test_dataset, label2id, id2label = create_sft_dataset_v3(
            profile_agent, 
            inter_dataset.test_user_ids[:100000],
            inter_dataset.test_histories[:100000],
            inter_dataset.test_target_items[:100000],
            recallers, 
            args.final_k, 
            args.profile_cutoff,
            train_ratio=0.7,
            eval_ratio=0.2,
            inner_label_ratio=0.1,
            test_gt_ratio=0.1,
        )
        
        # Save
        os.makedirs(paths["data"], exist_ok=True)
        train_dataset.save_to_disk(f'{paths["data"]}/train')
        eval_dataset.save_to_disk(f'{paths["data"]}/eval')
        test_dataset.save_to_disk(f'{paths["data"]}/test')
        with open(f'{paths["data"]}/label_mapping.json', 'w') as f:
            json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
        
        # 保存配置
        config = {
            "version": "v3",
            "train_ratio": 0.7,
            "eval_ratio": 0.2,
            "inner_label_ratio": 0.1,
            "test_gt_ratio": 0.1,
        }
        with open(f'{paths["data"]}/data_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nSaved: Train={len(train_dataset)}, Eval={len(eval_dataset)}, Test={len(test_dataset)}")
        print(f"Data path: {paths['data']}")
        return
    
    # Train model
    if args.do_sft:
        # Load data
        train_dataset = Dataset.load_from_disk(f'{paths["data"]}/train')
        eval_dataset = Dataset.load_from_disk(f'{paths["data"]}/eval')
        with open(f'{paths["data"]}/label_mapping.json', 'r') as f:
            labels = json.load(f)
            label2id = labels["label2id"]
            id2label = {int(k): v for k, v in labels["id2label"].items()}
        
        # Setup model
        dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
        
        if args.seq2seq:
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    args.model_name,
                    torch_dtype=dtype,
                    device_map="auto"
                )
            except:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name,
                    torch_dtype=dtype,
                    device_map="auto"
                )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name,
                num_labels=len(label2id),
                id2label=id2label,
                label2id=label2id,
                torch_dtype=dtype,
                device_map="auto"
            )
            
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        tokenizer.padding_side = args.padding_side
        
        # Tokenize
        tokenize_fn = partial(tokenize_function, tokenizer=tokenizer, max_length=args.max_length, seq2seq=args.seq2seq)
        # V3: 新的列名
        columns_to_remove = ["text", "best_recaller", "prompt_hint", "user_id", "recall_hist", "test_gt"]
        columns_to_remove = [c for c in columns_to_remove if c in train_dataset.column_names]
        
        tokenized_train = train_dataset.map(tokenize_fn, batched=True, 
                                           remove_columns=columns_to_remove)
        tokenized_eval = eval_dataset.map(tokenize_fn, batched=True, 
                                         remove_columns=[c for c in columns_to_remove if c in eval_dataset.column_names])
        
        # V2.1: 如果使用软标签，将 soft_labels 作为 labels
        if args.use_soft_labels and "soft_labels" in tokenized_train.column_names:
            tokenized_train = tokenized_train.rename_column("soft_labels", "labels")
            tokenized_eval = tokenized_eval.rename_column("soft_labels", "labels")
        
        # Train
        if args.seq2seq:
            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
            compute_metrics_fn = partial(compute_metrics_seq2seq, tokenizer=tokenizer, label2id=label2id)
        else:
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            compute_metrics_fn = compute_metrics
        
        training_args = TrainingArguments(
            output_dir=paths["sft"],
            save_total_limit=1,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            save_steps=args.save_steps,
            eval_steps=args.save_steps,
            logging_steps=args.logging_steps,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            bf16=args.bf16,
            fp16=args.fp16 and not args.bf16,
            gradient_checkpointing=args.gradient_checkpointing,
            eval_strategy="steps",
            save_strategy="steps",
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            load_best_model_at_end=True,
            report_to="wandb",
            run_name=paths["sft"] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
            seed=args.seed,
            # V2.1: 如果不使用软标签但使用标签平滑，通过 TrainingArguments 传入
            label_smoothing_factor=args.label_smoothing if not args.use_soft_labels else 0.0,
        )
        
        # V2.1: 根据是否使用软标签选择 Trainer
        if args.use_soft_labels:
            trainer = SoftLabelTrainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_eval,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics_fn,
                use_soft_labels=True,
                label_smoothing=0.0,  # 软标签模式下不额外平滑
            )
            print("Using SoftLabelTrainer with soft labels")
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_eval,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics_fn,
            )
            if args.label_smoothing > 0:
                print(f"Using standard Trainer with label smoothing = {args.label_smoothing}")
        
        print("Training...")
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(paths["sft"])
    
    # GRPO Training
    if args.do_grpo:
        print("\n" + "="*60)
        print("Starting Pure GRPO Training (V2)")
        print("="*60)
        
        if 'recallers' not in dir() or recallers is None:
            inter_dataset = load_dataset(args.dataset, args.data_path, seed=args.seed)
            recallers = initialize_recallers(
                model_names=args.recbole_models,
                dataset_name=args.dataset,
                checkpoint_dir=args.checkpoint_dir,
                data_path=args.data_path,
                seed=args.seed,
                use_latest_checkpoint=True,
                num_items=inter_dataset.ds.item_num
            )
        
        with open(f'{paths["data"]}/label_mapping.json', 'r') as f:
            labels = json.load(f)
            label2id = labels["label2id"]
            id2label = {int(k): v for k, v in labels["id2label"].items()}
        
        recaller_names = sorted(args.recbole_models)
        
        sft_model_path = paths["sft"]
        if not os.path.exists(sft_model_path):
            print(f"Error: SFT model not found at {sft_model_path}. Run --do_sft first.")
            return
        
        if not os.path.exists(os.path.join(sft_model_path, "config.json")):
            last_ckpt = get_last_checkpoint(sft_model_path)
            if last_ckpt:
                sft_model_path = last_ckpt
        
        dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
        
        model = AutoModelForSequenceClassification.from_pretrained(
            sft_model_path,
            num_labels=len(label2id),
            torch_dtype=dtype,
            device_map="cuda"
        )
        
        tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        grpo_train_dataset = Dataset.load_from_disk(f'{paths["data"]}/train')
        grpo_eval_dataset_full = Dataset.load_from_disk(f'{paths["data"]}/eval')
        grpo_eval_dataset = grpo_eval_dataset_full.select(range(min(1000, len(grpo_eval_dataset_full))))
        print(f"GRPO training dataset: {len(grpo_train_dataset)} samples")
        print(f"GRPO eval dataset: {len(grpo_eval_dataset)} samples")
        
        def grpo_reward_fn(prompts, completions, completion_ids, **kwargs):
            return [0.0] * len(prompts)
        
        grpo_trainer = GRPOTrainer(
            model=model,
            reward_funcs=grpo_reward_fn,
            args=grpo_config,
            train_dataset=grpo_train_dataset,
            eval_dataset=grpo_eval_dataset,
            processing_class=tokenizer,
        )
        
        if args.beta > 0:
            ref_model = AutoModelForSequenceClassification.from_pretrained(
                sft_model_path,
                num_labels=len(label2id),
                torch_dtype=dtype,
                device_map="cuda"
            )
            ref_model.eval()
            for param in ref_model.parameters():
                param.requires_grad = False
            grpo_trainer.ref_model = ref_model
        
        grpo_trainer.use_pure_classification = True
        grpo_trainer.pure_recallers = recallers
        grpo_trainer.pure_recaller_names = recaller_names
        grpo_trainer.pure_final_k = args.final_k
        grpo_trainer.pure_noise_scale = args.noise_scale
        grpo_trainer.use_soft_grpo_loss = not args.use_ppo_loss
        grpo_trainer._signature_columns = ["prompt", "user_id", "history", "target_items"]
        
        print("Starting GRPO training...")
        grpo_trainer.train()
        grpo_trainer.save_model()
        tokenizer.save_pretrained(paths["grpo"])
        print(f"GRPO model saved to {paths['grpo']}")
    
    # Test model
    if args.do_test_sft or args.do_test_grpo:
        if args.do_test_sft:
            model_path = paths["sft"] if os.path.exists(paths["sft"]) else args.model_name
        elif args.do_test_grpo:
            model_path = paths["grpo"] if os.path.exists(paths["grpo"]) else args.model_name
        
        if os.path.exists(model_path) and not os.path.exists(os.path.join(model_path, "config.json")):
            last_checkpoint = get_last_checkpoint(model_path)
            if last_checkpoint:
                model_path = last_checkpoint
                print(f"Loading from last checkpoint: {model_path}")
            else:
                print(f"Warning: No checkpoint found in {model_path}")
        
        with open(f'{paths["data"]}/label_mapping.json', 'r') as f:
            labels = json.load(f)
            id2label = {int(k): v for k, v in labels["id2label"].items()}
        
        dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
        
        if args.seq2seq:
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=dtype, device_map="auto")
            except:
                model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map="auto")
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_path, torch_dtype=dtype, device_map="auto")
            
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        tokenizer.padding_side = args.padding_side
        
        # Load test dataset from disk (generated in gen_sft_data phase)
        test_dataset = Dataset.load_from_disk(f'{paths["data"]}/test')
        print(f"Loaded test dataset: {len(test_dataset)} samples")
        
        if not args.seq2seq:
            results = evaluate_pure_model(model, tokenizer, test_dataset, id2label, recallers, args.final_k)
            
            recaller_names = sorted(recallers.keys())
            multi_results = evaluate_multi_channel_recall(
                model, tokenizer, test_dataset, recallers, recaller_names,
                args.final_k, use_softmax_weights=True
            )
            results["multi_channel_evaluation"] = multi_results
            
            # Add configuration info to results
            recaller_combo = "_".join(sorted(args.recbole_models))
            results["config"] = {
                "dataset": args.dataset,
                "recbole_models": args.recbole_models,
                "recaller_combo": recaller_combo,
                "model_name": args.model_name,
                "final_k": args.final_k,
                "profile_cutoff": args.profile_cutoff,
                "test_samples": len(test_dataset),
                "model_path": model_path,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                # V2 specific options
                "filter_close_recallers": args.filter_close_recallers,
                "close_threshold": args.close_threshold,
                "max_ground_truth": args.max_ground_truth,
                "use_soft_labels": args.use_soft_labels,
                "label_smoothing": args.label_smoothing,
            }
            
            # Save results with recaller combo in filename
            os.makedirs("results", exist_ok=True)
            result_filename = f"results/pure_v2_results_{args.dataset}_{recaller_combo}.json"
            with open(result_filename, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {result_filename}")
            
            # Print summary for easy comparison
            print("\n" + "="*60)
            print("SUMMARY")
            print("="*60)
            print(f"Dataset: {args.dataset}")
            print(f"Recaller Combo: {recaller_combo}")
            print(f"Classification Accuracy: {results['accuracy']:.4f}")
            print(f"Classification F1 Macro: {results['f1_macro']:.4f}")
            if 'avg_predicted_ndcg' in results:
                print(f"Predicted NDCG@{args.final_k}: {results['avg_predicted_ndcg']:.4f}")
                print(f"True Best NDCG@{args.final_k}: {results['avg_true_ndcg']:.4f}")
            if 'avg_predicted_recall' in results:
                print(f"Predicted Recall@{args.final_k}: {results['avg_predicted_recall']:.4f}")
                print(f"True Best Recall@{args.final_k}: {results['avg_true_recall']:.4f}")
            if 'base_model_results' in results:
                print(f"\nBase Model Performance (NDCG@{args.final_k}):")
                for name, res in results['base_model_results'].items():
                    print(f"  {name}: {res['avg_ndcg']:.4f}")
        else:
            print("Seq2seq evaluation not implemented yet. Model saved successfully.")


if __name__ == "__main__":
    main()

