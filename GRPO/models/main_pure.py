import argparse
import json
import os
from functools import partial
from typing import Any, List, Dict, Tuple
import sys
import warnings
import math
from typing import Dict, List, Tuple, Optional, Union
# Suppress pandas FutureWarning from recbole
warnings.filterwarnings('ignore', category=FutureWarning, message='.*A value is trying to be set on a copy of a DataFrame.*')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.distributions import Dirichlet
from datetime import datetime
import numpy as np
from datasets import Dataset
from tqdm import tqdm
from collections import defaultdict
from transformers import (
    AutoModelForSequenceClassification, 
    AutoModelForCausalLM,
    AutoTokenizer, 
    TrainingArguments, 
    DataCollatorWithPadding, 
    DataCollatorForSeq2Seq,
    Trainer
)
from transformers.trainer_utils import get_last_checkpoint
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import wandb

from GRPO.core.agents import UserProfileAgent
from GRPO.core.data import load_dataset
from GRPO.models.main import initialize_recallers
from GRPO.core.recallers import RecBoleRecaller
from GRPO.core.utils import set_seed, build_prompt, ndcg_at_k, recall_at_k, TOOLS_DESCRIPTION
from GRPO.models.soft_utils import multi_channel_recall_score, multi_channel_recall_top_k, compute_ndcg_at_k as soft_ndcg
from GRPO.trainers.trl_trainer import GRPOTrainer
from GRPO.models.soft_utils import multi_channel_recall_score, compute_ndcg_at_k
from GRPO.models.normalization import normalize_scores
from GRPO.models.evaluation_utils import (
    evaluate_base_recallers, aggregate_metrics, print_base_recaller_table,
    find_best_base_model, print_comparison_table, print_cross_checkpoint_table,
    compute_performance_gaps, print_performance_gaps,
    extract_eval_data
)
from GRPO.models.model_utils import load_model_and_tokenizer, load_label_mapping, load_hint_map, load_model_only
from trl import GRPOConfig


# Dataset generation configuration
MIN_HISTORY_FOR_AUGMENTATION = 30
AUGMENTATION_STEP = 10


def safe_load_tokenizer(model_path: str):
    """
    Safely load tokenizer, handling DeBERTa-v2 fast tokenizer issues.
    For DeBERTa models, directly use DebertaV2Tokenizer (slow) to avoid fast tokenizer bugs.
    """
    if "deberta" in model_path.lower():
        try:
            from transformers import DebertaV2Tokenizer
            tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
            return tokenizer
        except Exception as e:
            raise RuntimeError(f"Failed to load DeBERTa tokenizer from {model_path}. Error: {e}")
    else:
        # For other models, use AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return tokenizer


def create_sft_dataset(
    profile_agent: UserProfileAgent,
    user_ids: List[int],
    histories: List[List[int]],
    target_items: List[int],
    recallers: Dict[str, RecBoleRecaller],
    args,  # 主函数的 args 对象
    user_hint_map: Dict[int, str] = None,  # 用户->eval set最佳recaller的映射
    use_self_hint: bool = False,  # 是否使用当前样本的best_recaller作为hint (用于eval set)
    use_augmentation: bool = True,
    fallback_recaller: str = None,  # 所有recaller均为0时的默认label (来自eval set最佳recaller)
    min_thres: float = -0.001,
    max_thres: float = 1.001,
    prompt_type: str = 'classification',
) -> Tuple[Dataset, Dict[str, int], Dict[int, str], Dict[int, str]]:
    """Create dataset where the model predicts the best recaller class
    
    Args from args object:
        - train_k: Top-k for recaller recall
        - profile_cutoff: Cutoff length for user profiles
        - use_soft_label: Use soft labels based on score difference
        - soft_label_temperature: Temperature for soft label
        - random_history_selection: Randomly select history items
        - prompt_top_k: Number of top items to display per recaller
        - autoregressive: Use 'instruction' prompt type for seq2seq
    """
    # Extract args
    train_k = args.train_k
    profile_cutoff = args.profile_cutoff
    use_soft_label = getattr(args, 'use_soft_label', False)
    soft_label_temperature = getattr(args, 'soft_label_temperature', 5.0)
    random_history_selection = getattr(args, 'random_history_selection', False)
    prompt_top_k = getattr(args, 'prompt_top_k', 3)
    
    dataset = []
    metrics = {recaller: defaultdict(list) for recaller in recallers.keys()}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Create label mappings
    recaller_names = sorted(list(recallers.keys()))
    label2id = {name: i for i, name in enumerate(recaller_names)}
    id2label = {i: name for name, i in label2id.items()}
    max_tokens_cls = 0
    max_tokens_ar = 0
    _has_chat_tpl = hasattr(tokenizer, 'apply_chat_template')
    for i, uid in tqdm(enumerate(user_ids)):
        hist = histories[i]
        gt_items = target_items[i]
        if 0 in hist:
            hist = hist[:hist.index(0)]
        
        fixed_hist_len = min(profile_cutoff, int(len(hist)))
        start_pos = max(0, len(hist) - fixed_hist_len)
        start_positions = [start_pos]
        
        for start_pos in start_positions:
            
            if random_history_selection:
                # 随机选择历史项：从 gt_items 之前的所有项中随机选择 profile_cutoff 个，保持时间顺序
                available_hist = hist
                if len(available_hist) >= profile_cutoff:
                    selected_indices = sorted(np.random.choice(len(available_hist), profile_cutoff, replace=False))
                    eval_hist = [available_hist[i] for i in selected_indices]
                else:
                    eval_hist = available_hist
            else:
                eval_hist = hist[start_pos:]
            
            # if len(gt_items) < 1:
            #     continue
            
            # Find best recaller and collect scores
            # Mask all interacted items except gt_items for fair evaluation
            recaller_scores = {}
            recaller_top_items = {}  # Store top items for each recaller
            best_ndcg, best_recaller = -1, None
            for recaller_name, recaller in recallers.items():
                items = recaller.recall(uid, 5, hist, full_hist=hist + gt_items, gt_items=gt_items)
                item_ids = [item[0] for item in items] if items else []
                ndcg = ndcg_at_k(item_ids, gt_items, k=train_k)
                metrics[recaller_name]["ndcg"].append(ndcg)
                recaller_scores[recaller_name] = ndcg
                # Store top items with their metadata and rank
                recaller_top_items[recaller_name] = [
                    {"rank": idx + 1, **profile_agent.item_feat.get(iid, {})} 
                    for idx, iid in enumerate(item_ids[:prompt_top_k])
                ]
                
                if ndcg > best_ndcg:
                    best_ndcg = ndcg
                    best_recaller = recaller_name
            
            # 所有recaller均为0时, 使用eval set最佳recaller作为fallback label
            if best_ndcg <= 0 and fallback_recaller:
                best_recaller = fallback_recaller
            
            if best_ndcg < min_thres or best_ndcg > max_thres:
                continue
            
            # 获取 ground truth items 的最早交互时间戳作为参考
            gt_timestamps = []
            for gt_id in gt_items:
                inter_key = f'{uid}_{gt_id}'
                if inter_key in profile_agent.inter_feat:
                    ts = profile_agent.inter_feat[inter_key].get('timestamp')
                    if ts is not None:
                        try:
                            ts = float(ts)
                            if ts > 1e12:
                                ts = ts / 1000
                            if ts > 86400:
                                gt_timestamps.append(ts)
                        except (ValueError, TypeError):
                            pass
            reference_ts = min(gt_timestamps) if gt_timestamps else None
            
            # Create sample
            user_profile = profile_agent.forward(uid, eval_hist, cut_off=profile_cutoff, reference_timestamp=reference_ts)
            # 确定 hint: use_self_hint 优先使用当前 best_recaller，否则使用 user_hint_map
            if use_self_hint:
                hint = best_recaller
            elif user_hint_map:
                hint = user_hint_map.get(uid, "")
            else:
                hint = ""
            
            # Build available models description with top retrieved items
            models_with_items = [
                {
                    "name": m,
                    "description": TOOLS_DESCRIPTION.get(m.lower(), {}).get("description", ""),
                    "when_to_use": TOOLS_DESCRIPTION.get(m.lower(), {}).get("when_to_use", "depends on the user profile and interaction history."),
                    "top_retrieved_items": recaller_top_items.get(m, [])
                }
                for m in recaller_names
            ]
            
            prompt = build_prompt(
                user_profile, 
                available_models=recaller_names,
                models_with_items=models_with_items,
                hint=hint if hint else None,
                type=prompt_type
            )
            if _has_chat_tpl:
                _chat_text = tokenizer.apply_chat_template(
                    [{"role": "assistant", "content": prompt}],
                    tokenize=False, add_generation_prompt=False,
                )
            else:
                _chat_text = prompt
            _n_cls = len(tokenizer(_chat_text).input_ids)
            _ar_text = _chat_text + f"\n\nBest recaller: {best_recaller}"
            _n_ar = len(tokenizer(_ar_text).input_ids)
            max_tokens_cls = max(max_tokens_cls, _n_cls)
            max_tokens_ar = max(max_tokens_ar, _n_ar)
            sample = {
                "text": prompt,
                "prompt": prompt,  # For GRPO compatibility
                "labels": label2id[best_recaller],
                "best_recaller": best_recaller,
                "best_ndcg": best_ndcg,
                "user_id": uid,
                "history": eval_hist,  # For GRPO reward computation
                "target_items": gt_items,  # For GRPO reward computation
                "full_hist": hist + gt_items,  # All interacted items for masking during evaluation
                "history_len_used": len(eval_hist),
            }
            
            # Compute soft labels if enabled
            if use_soft_label:
                scores = torch.tensor([recaller_scores[name] for name in recaller_names])
                # Temperature-scaled softmax: higher temp = sharper, lower = smoother
                soft_labels = torch.softmax(scores * soft_label_temperature, dim=0).tolist()
                sample["soft_labels"] = soft_labels
            
            dataset.append(sample)
    
    # Print statistics
    print(f"\nDataset created: {len(dataset)} samples from {len(set(d['user_id'] for d in dataset))} users")
    for recaller_name in recallers.keys():
        if metrics[recaller_name]["ndcg"]:
            avg_ndcg = np.mean(metrics[recaller_name]["ndcg"])
            print(f"{recaller_name}: avg NDCG = {avg_ndcg:.4f}")
    
    # Best model distribution
    best_model_counts = defaultdict(int)
    for item in dataset:
        best_model_counts[item["best_recaller"]] += 1
    for model, count in sorted(best_model_counts.items()):
        print(f"{model}: {count} ({count/len(dataset)*100:.1f}%)")
    
    # 构建 user -> best_recaller 映射 (取每个用户最后一条样本的 best_recaller)
    user_best_recaller = {}
    for item in dataset:
        user_best_recaller[item["user_id"]] = item["best_recaller"]
    
    # 全局最佳 recaller (avg NDCG 最高)
    global_best_recaller = None
    best_avg = -1
    for recaller_name in recallers.keys():
        if metrics[recaller_name]["ndcg"]:
            avg = np.mean(metrics[recaller_name]["ndcg"])
            if avg > best_avg:
                best_avg = avg
                global_best_recaller = recaller_name
    print(f"Global best recaller (highest avg NDCG): {global_best_recaller} ({best_avg:.4f})")
    
    token_stats = {"max_tokens_cls": max_tokens_cls, "max_tokens_ar": max_tokens_ar}
    print(f"Max prompt tokens (after chat template): cls={max_tokens_cls}, ar={max_tokens_ar}")
    
    return Dataset.from_list(dataset), label2id, id2label, user_best_recaller, global_best_recaller, token_stats


def compute_metrics(eval_pred):
    """Compute metrics for classification"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return {
        'accuracy': accuracy_score(eval_pred.label_ids, predictions),
        'f1_macro': f1_score(eval_pred.label_ids, predictions, average='macro'),
        'f1_weighted': f1_score(eval_pred.label_ids, predictions, average='weighted')
    }


class DirichletSequenceClassification(nn.Module):
    """用狄利克雷分布替代 softmax 头的分类模型"""
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.num_labels = base_model.config.num_labels
        hidden_size = getattr(base_model.config, 'hidden_size', 
                             getattr(base_model.config, 'd_model', 768))
        # 保存原始分类头以便替换
        self.original_classifier = None
        if hasattr(base_model, 'classifier'):
            self.original_classifier = base_model.classifier
            # 创建新的浓度参数头
            self.concentration_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.num_labels)
            )
            # 替换分类头
            base_model.classifier = nn.Identity()
        elif hasattr(base_model, 'score'):
            self.original_classifier = base_model.score
            self.concentration_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.num_labels)
            )
            base_model.score = nn.Identity()
        else:
            # 如果没有找到分类头，创建一个新的
            self.concentration_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.num_labels)
            )
    
    def forward(self, **inputs):
        outputs = self.base_model(**inputs, output_hidden_states=True)
        # 获取 pooler output 或 last hidden state
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            hidden = outputs.pooler_output
        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden = outputs.hidden_states[-1][:, 0]  # CLS token
        elif hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state[:, 0]  # CLS token
        else:
            raise ValueError("Cannot extract hidden state from model outputs")
        
        # 计算浓度参数（确保为正）
        log_concentration = self.concentration_head(hidden)
        concentration = torch.softplus(log_concentration) + 1e-6  # 确保 > 0
        
        # 创建狄利克雷分布
        dirichlet_dist = Dirichlet(concentration)
        
        # 计算期望值用于预测（归一化的浓度参数）
        probs = concentration / concentration.sum(dim=-1, keepdim=True)
        
        # 返回格式兼容 transformers
        class ModelOutput:
            def __init__(self, logits, probs, dirichlet_dist):
                self.logits = logits  # 用于兼容性，实际是 probs 的 log
                self.probs = probs
                self.dirichlet_dist = dirichlet_dist
        
        return ModelOutput(
            logits=torch.log(probs + 1e-10),  # 兼容性：提供 logits
            probs=probs,
            dirichlet_dist=dirichlet_dist
        )
    
    def eval(self):
        self.base_model.eval()
        self.concentration_head.eval()
        return self
    
    def train(self, mode=True):
        self.base_model.train(mode)
        self.concentration_head.train(mode)
        return self


class SoftLabelTrainer(Trainer):
    """Trainer that supports soft labels using KL divergence loss instead of hard CE loss"""
    
    def __init__(self, *args, use_soft_label=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_soft_label = use_soft_label
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.use_soft_label and "soft_labels" in inputs:
            soft_labels = inputs.pop("soft_labels")  # (batch, num_classes)
            # Keep hard labels for metrics computation
            _ = inputs.pop("labels", None)
            
            outputs = model(**inputs)
            
            # 如果模型是 DirichletSequenceClassification，使用狄利克雷分布的负对数似然
            if isinstance(model, DirichletSequenceClassification):
                dirichlet_dist = outputs.dirichlet_dist
                soft_labels_tensor = torch.tensor(soft_labels, device=dirichlet_dist.concentration.device, 
                                                 dtype=dirichlet_dist.concentration.dtype)
                # 归一化确保是有效的概率分布
                soft_labels_tensor = soft_labels_tensor / (soft_labels_tensor.sum(dim=-1, keepdim=True) + 1e-10)
                # 添加小的平滑以避免数值问题
                num_classes = soft_labels_tensor.size(-1)
                soft_labels_tensor = soft_labels_tensor * (1 - 1e-5) + 1e-5 / num_classes
                # 负对数似然：-log p(soft_labels | concentration)
                loss = -dirichlet_dist.log_prob(soft_labels_tensor).mean()
            else:
                logits = outputs.logits  # (batch, num_classes)
                # KL divergence: KL(soft_target || model_output)
                log_probs = torch.log_softmax(logits, dim=-1)
                soft_labels_tensor = torch.tensor(soft_labels, device=logits.device, dtype=logits.dtype)
                # Soft cross entropy: -sum(soft_target * log_prob)
                loss = -(soft_labels_tensor * log_probs).sum(dim=-1).mean()
            
            if return_outputs:
                return loss, outputs
            return loss
        else:
            # 标准损失计算
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            
            if isinstance(model, DirichletSequenceClassification):
                # 对于硬标签，将标签转换为 one-hot，然后计算负对数似然
                num_classes = model.num_labels
                one_hot = torch.zeros(labels.size(0), num_classes, device=labels.device, dtype=outputs.dirichlet_dist.concentration.dtype)
                one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
                dirichlet_dist = outputs.dirichlet_dist
                # 添加小的平滑以避免数值问题
                one_hot = one_hot * (1 - 1e-5) + 1e-5 / num_classes
                loss = -dirichlet_dist.log_prob(one_hot).mean()
            else:
                loss = nn.functional.cross_entropy(outputs.logits, labels)
            
            if return_outputs:
                return loss, outputs
            return loss


def compute_metrics_seq2seq(eval_pred, tokenizer, label2id):
    """Compute metrics for seq2seq generation"""
    predictions, labels = eval_pred
    
    # If predictions are logits, take argmax to get token ids
    if predictions.ndim == 3:  # (batch_size, seq_len, vocab_size)
        predictions = np.argmax(predictions, axis=-1)
    
    # Replace -100 with pad_token_id for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Convert text predictions to label ids
    pred_label_ids = []
    true_label_ids = []
    
    for pred_text, true_text in zip(decoded_preds, decoded_labels):
        # Clean up text
        pred_text = pred_text.strip()
        true_text = true_text.strip()
        
        # Map to label ids
        pred_id = label2id.get(pred_text, -1)  # -1 for unknown predictions
        true_id = label2id.get(true_text, -1)
        
        pred_label_ids.append(pred_id)
        true_label_ids.append(true_id)
    
    pred_label_ids = np.array(pred_label_ids)
    true_label_ids = np.array(true_label_ids)
    
    # Only evaluate on valid predictions
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


def evaluate_pure_model(model, tokenizer, test_dataset, id2label, recallers=None, eval_k=50, use_chat_format=False, max_length=1536):
    """Evaluate the pure classification model and generate recommendations"""
    device = model.device
    model.eval()
    
    all_predictions, all_labels = [], []
    recommendation_results = []
    with torch.no_grad():
        for idx, example in enumerate(tqdm(test_dataset, desc="Evaluating")):
            text = example["text"]
            # Apply chat format if enabled
            if use_chat_format and hasattr(tokenizer, 'apply_chat_template'):
                text = apply_chat_format([text], tokenizer)[0]
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                             max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]
            
            all_predictions.append(prediction)
            all_labels.append(example["labels"])
            
            # Generate recommendations if recallers are provided
            predicted_recaller_name = id2label[prediction]
            true_recaller_name = example["best_recaller"]
                
            # Get data directly from example (already computed in create_sft_dataset)
            user_id, eval_hist, gt_items, full_hist = extract_eval_data(example, idx)
                    
            # Generate recommendations using predicted recaller
            predicted_recaller = recallers[predicted_recaller_name]
            pred_items = predicted_recaller.recall(user_id, eval_k, eval_hist, full_hist=full_hist, gt_items=gt_items)
            pred_item_ids = [item[0] for item in pred_items] if pred_items else []
            pred_ndcg = ndcg_at_k(pred_item_ids, gt_items, k=eval_k)
            pred_recall = recall_at_k(pred_item_ids, gt_items, k=eval_k)
                
                # Generate recommendations using true recaller for comparison
            if true_recaller_name in recallers and len(eval_hist) >= 5:
                true_recaller = recallers[true_recaller_name]
                true_items = true_recaller.recall(user_id, eval_k, eval_hist, full_hist=full_hist, gt_items=gt_items)
                true_item_ids = [item[0] for item in true_items] if true_items else []
                true_ndcg = ndcg_at_k(true_item_ids, gt_items, k=eval_k)
                true_recall = recall_at_k(true_item_ids, gt_items, k=eval_k)
            else:
                print(f"True recaller {true_recaller_name} not in recallers")
                true_item_ids = []
                true_ndcg = 0.0
                true_recall = 0.0
                
            recommendation_results.append({
                    "user_id": user_id,
                    "predicted_recaller": predicted_recaller_name,
                    "true_recaller": true_recaller_name,
                    "predicted_items": pred_item_ids[:10],  # Top 10 for display
                    "true_items": true_item_ids[:10],  # Top 10 for display
                    "ground_truth": gt_items,
                    "eval_hist": eval_hist,  # Store eval_hist for base model evaluation
                    "full_hist": full_hist,  # All interacted items for masking
                    "predicted_ndcg": pred_ndcg,
                    "true_ndcg": true_ndcg,
                    "predicted_recall": pred_recall,
                    "true_recall": true_recall,
                    "correct_prediction": predicted_recaller_name == true_recaller_name,
                    "history_length": len(eval_hist)
            })
    
    # Calculate classification metrics
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

    # Add recommendation metrics if available
    if recommendation_results:
        avg_pred_ndcg = np.mean([r["predicted_ndcg"] for r in recommendation_results])
        avg_true_ndcg = np.mean([r["true_ndcg"] for r in recommendation_results])
        avg_pred_recall = np.mean([r["predicted_recall"] for r in recommendation_results])
        avg_true_recall = np.mean([r["true_recall"] for r in recommendation_results])
        correct_predictions = sum([r["correct_prediction"] for r in recommendation_results])
        
        ndcg_improvement = avg_pred_ndcg - avg_true_ndcg
        recall_improvement = avg_pred_recall - avg_true_recall
        acc = correct_predictions / len(recommendation_results)
        print(f"\nRecommendation @{eval_k}: Pred NDCG={avg_pred_ndcg:.4f} Recall={avg_pred_recall:.4f} | TrueBest NDCG={avg_true_ndcg:.4f} Recall={avg_true_recall:.4f} | Acc={acc:.2%} ({correct_predictions}/{len(recommendation_results)})")
        result.update({
            "recommendation_results": recommendation_results,
            "avg_predicted_ndcg": avg_pred_ndcg,
            "avg_true_ndcg": avg_true_ndcg,
            "avg_predicted_recall": avg_pred_recall,
            "avg_true_recall": avg_true_recall,
            "ndcg_improvement": ndcg_improvement,
            "recall_improvement": recall_improvement,
            "recaller_prediction_accuracy": correct_predictions / len(recommendation_results)
        })
    
    # Evaluate base recallers and compute gaps (single-select vs best base)
    if recallers is not None and recommendation_results:
        valid_eval_instances = [r for r in recommendation_results if r["history_length"] >= 5]
        base_model_results = {}
        for recaller_name, recaller in recallers.items():
            ndcgs, recalls = [], []
            for res in tqdm(valid_eval_instances, desc=f"Base {recaller_name}", leave=False):
                items = recaller.recall(res["user_id"], eval_k, res["eval_hist"], full_hist=res.get("full_hist"), gt_items=res["ground_truth"])
                ids = [item[0] for item in items] if items else []
                ndcgs.append(ndcg_at_k(ids, res["ground_truth"], k=eval_k))
                recalls.append(recall_at_k(ids, res["ground_truth"], k=eval_k))
            if ndcgs:
                base_model_results[recaller_name] = {"avg_ndcg": np.mean(ndcgs), "avg_recall": np.mean(recalls), "num_evaluations": len(ndcgs)}
        if base_model_results:
            best_base_ndcg = max(m["avg_ndcg"] for m in base_model_results.values())
            best_base_recall = max(m["avg_recall"] for m in base_model_results.values())
            best_ndcg_model = max(base_model_results, key=lambda k: base_model_results[k]["avg_ndcg"])
            best_recall_model = max(base_model_results, key=lambda k: base_model_results[k]["avg_recall"])
            ndcg_gap = avg_pred_ndcg - best_base_ndcg
            recall_gap = avg_pred_recall - best_base_recall
            print(f"Base recallers ({len(valid_eval_instances)} evals): best_ndcg={best_ndcg_model}={best_base_ndcg:.4f} best_recall={best_recall_model}={best_base_recall:.4f} | Model gaps: NDCG {ndcg_gap:+.4f} Recall {recall_gap:+.4f}")
            result.update({
                "base_model_results": base_model_results,
                "best_base_ndcg": best_base_ndcg, "best_base_recall": best_base_recall,
                "best_ndcg_model": best_ndcg_model, "best_recall_model": best_recall_model,
                "ndcg_gap_vs_best_base": ndcg_gap, "recall_gap_vs_best_base": recall_gap,
            })
    return result

def evaluate_baseline_recallers(
    test_dataset,
    recallers: Dict[str, RecBoleRecaller],
    recaller_names: List[str],
    eval_k: int = 50,
    save_predictions: bool = True,
    merge_method: str = "average",
    score_norm: Optional[str] = None,
    score_norm_kwargs: Optional[dict] = None,
):
    """
    Evaluate baseline recallers and fusion without model.
    This is a standalone baseline evaluation that doesn't require a model.
    
    Args:
        merge_method: Method for merging recaller results. Options:
            - 'average': Use multi_channel_recall_score with uniform weights
            - 'top_k': Use multi_channel_recall_top_k (keep highest scores)
    """
    metrics = {
        "avg_score_weight": defaultdict(list),  # Average score-weight (uniform)
    }
    # Add metrics for each base recaller
    for recaller_name in recaller_names:
        metrics[recaller_name] = defaultdict(list)
    
    # Store predictions
    predictions = []
    
    for idx, example in enumerate(tqdm(test_dataset, desc="Evaluating Baselines")):
        # Get data directly from example
        user_id, eval_hist, gt_items, full_hist = extract_eval_data(example, idx)
        
        if len(eval_hist) < 5:
            continue
        
        # Collect predictions from all recallers
        recaller_predictions = {}
        for recaller_name in recaller_names:
            if recaller_name in recallers:
                items = recallers[recaller_name].recall(user_id, eval_k, eval_hist, full_hist=full_hist, gt_items=gt_items)
                # Store as list of [item_id, score] tuples
                recaller_predictions[recaller_name] = [[int(item_id), float(score)] for item_id, score in items] if items else []
        
        # 1. Multi-channel recall fusion
        if merge_method == "top_k":
            avg_candidates = multi_channel_recall_top_k(
                recallers, recaller_names, user_id, eval_hist, eval_k,
                full_hist=full_hist, gt_items=gt_items
            )
        else:  # default: "average"
            # Use multi_channel_recall_score with uniform weights
            num_recallers = len(recaller_names)
            uniform_weights = torch.ones(num_recallers) / num_recallers
            avg_candidates = multi_channel_recall_score(
                softmax_weights=uniform_weights,
                recallers=recallers,
                recaller_names=recaller_names,
                user_id=user_id,
                history=eval_hist,
                total_k=eval_k,
                full_hist=full_hist,
                gt_items=gt_items,
                score_norm=score_norm,
                score_norm_kwargs=score_norm_kwargs
            )
        avg_rec = [item_id for item_id, _ in avg_candidates]
        for k in [10, 20, 50]:
            metrics["avg_score_weight"][f"ndcg@{k}"].append(ndcg_at_k(avg_rec, gt_items, k))
            metrics["avg_score_weight"][f"recall@{k}"].append(recall_at_k(avg_rec, gt_items, k))
        
        # Save predictions
        if save_predictions:
            # For baseline, merge weights are uniform (1/n for each recaller)
            num_recallers = len(recaller_names)
            uniform_weights_list = [1.0 / num_recallers] * num_recallers
            predictions.append({
                "user_id": user_id,
                "recaller_predictions": recaller_predictions,
                "gt_items": gt_items,
                "merge_weights": uniform_weights_list  # Uniform weights for baseline
            })
        
        # 2. Evaluate each base recaller
        for recaller_name in recaller_names:
            if recaller_name in recallers:
                base_rec = [item[0] for item in recaller_predictions[recaller_name]]
                for k in [10, 20, 50]:
                    metrics[recaller_name][f"ndcg@{k}"].append(ndcg_at_k(base_rec, gt_items, k))
                    metrics[recaller_name][f"recall@{k}"].append(recall_at_k(base_rec, gt_items, k))
    
    # Aggregate results
    results = aggregate_metrics(metrics)
    
    print("\n" + "="*60)
    print("Baseline Recaller Evaluation (No Model)")
    print("="*60)
    print_base_recaller_table(results, recaller_names)
    if results.get("avg_score_weight"):
        fusion_label = "Fusion" if merge_method == "top_k" else "Avg-SW"
        print(f"\n--- {fusion_label} vs Best Base ---")
        print(f"{'Metric':<12} {fusion_label:>10} {'BestBase':>10} {'Gap':>10}")
        print("-" * 44)
        for k in [10, 20, 50]:
            for metric in ["ndcg", "recall"]:
                key = f"{metric}@{k}"
                val = results["avg_score_weight"].get(key, 0)
                best = max(results.get(n, {}).get(key, 0) for n in recaller_names)
                print(f"{key:<12} {val:>10.4f} {best:>10.4f} {val - best:>+10.4f}")
    if save_predictions:
        results["predictions"] = predictions
    return results


def evaluate_multi_channel_recall(
    model, 
    tokenizer, 
    test_dataset, 
    recallers: Dict[str, RecBoleRecaller],
    recaller_names: List[str],
    eval_k: int = 50,
    merge_method: str = "average",
    save_predictions: bool = True,
    score_norm: Optional[str] = None,
    score_norm_kwargs: Optional[dict] = None,
    id2label: Optional[Dict[int, str]] = None,
    use_chat_format: bool = False,
    max_length: int = 1536,
):
    """
    Evaluate model using multi-channel recall with softmax weights.
    
    Instead of selecting a single recaller, uses the softmax output
    as weights for aggregating candidates from all recallers.
    
    Args:
        id2label: Optional mapping from label id to recaller name. If provided,
                  recaller_names will be ordered according to id2label to ensure
                  softmax_weights[i] corresponds to recaller_names[i].
        use_chat_format: If True, apply chat template to inputs (for Instruct models).
    """
    device = model.device
    model.eval()
    
    # Ensure recaller_names order matches id2label if provided
    # This ensures softmax_weights[i] corresponds to recaller_names[i]
    if id2label is not None:
        # Create ordered list based on id2label: [id2label[0], id2label[1], ...]
        ordered_recaller_names = [id2label[i] for i in sorted(id2label.keys())]
        # Verify all recaller_names are in id2label
        if set(ordered_recaller_names) == set(recaller_names):
            recaller_names = ordered_recaller_names
    
    metrics = {
        "single_select": defaultdict(list),  # Best single recaller
        "multi_channel": defaultdict(list),   # Weighted multi-channel (softmax)
        "avg_score_weight": defaultdict(list),  # Average score-weight (uniform)
        "avg_top_k": defaultdict(list),  # Average top-k (uniform weights with quota-based scheduling)
    }
    # Add metrics for each base recaller
    for recaller_name in recaller_names:
        metrics[recaller_name] = defaultdict(list)
    
    # Store predictions
    predictions = []
    
    with torch.no_grad():
        for idx, example in enumerate(tqdm(test_dataset, desc="Evaluating Multi-Channel")):
            text = example["text"]
            # Apply chat format if enabled
            if use_chat_format and hasattr(tokenizer, 'apply_chat_template'):
                text = apply_chat_format([text], tokenizer)[0]
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                             max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            logits = outputs.logits[0]  # (num_classes,)
            
            # Get softmax weights
            softmax_weights = torch.softmax(logits, dim=-1)
            
            # Get data directly from example (already computed in create_sft_dataset)
            user_id, eval_hist, gt_items, full_hist = extract_eval_data(example, idx)
            
            if len(eval_hist) < 5:
                continue
            
            # Collect predictions from all recallers
            recaller_predictions = {}
            for recaller_name in recaller_names:
                items = recallers[recaller_name].recall(user_id, eval_k, eval_hist, full_hist=full_hist, gt_items=gt_items)
                # Store as list of [item_id, score] tuples
                recaller_predictions[recaller_name] = [[int(item_id), float(score)] for item_id, score in items] if items else []
            
            # 1. Single recaller selection (argmax)
            pred_idx = logits.argmax().item()
            pred_recaller = recaller_names[pred_idx]
            # Use original name for lookup (consistent with evaluate_pure_model)
            single_rec = [item[0] for item in recaller_predictions[pred_recaller]]
            for k in [10, 20, 50]:
                metrics["single_select"][f"ndcg@{k}"].append(ndcg_at_k(single_rec, gt_items, k))
                metrics["single_select"][f"recall@{k}"].append(recall_at_k(single_rec, gt_items, k))
            
            # 2. Multi-channel recall with softmax weights
            if merge_method == "average":
                candidates = multi_channel_recall_score(
                    softmax_weights, recallers, recaller_names,
                    user_id, eval_hist, eval_k,
                    full_hist=full_hist, gt_items=gt_items,
                    score_norm=score_norm,
                    score_norm_kwargs=score_norm_kwargs
                )
            elif merge_method == "top_k":
                candidates = multi_channel_recall_top_k(
                    recallers, recaller_names, user_id, eval_hist, eval_k,
                    full_hist=full_hist, gt_items=gt_items, weights=softmax_weights
                )
            multi_rec = [item_id for item_id, _ in candidates]
            for k in [10, 20, 50]:
                metrics["multi_channel"][f"ndcg@{k}"].append(ndcg_at_k(multi_rec, gt_items, k))
                metrics["multi_channel"][f"recall@{k}"].append(recall_at_k(multi_rec, gt_items, k))
            
            # 3. Multi-channel recall with average (uniform) score-weight
            # Use multi_channel_recall_score with uniform weights
            num_recallers = len(recaller_names)
            uniform_weights = torch.ones(num_recallers) / num_recallers
            avg_candidates = multi_channel_recall_score(
                softmax_weights=uniform_weights,
                recallers=recallers,
                recaller_names=recaller_names,
                user_id=user_id,
                history=eval_hist,
                total_k=eval_k,
                full_hist=full_hist,
                gt_items=gt_items,
                score_norm=score_norm,
                score_norm_kwargs=score_norm_kwargs
            )
            avg_rec = [item_id for item_id, _ in avg_candidates]
            for k in [10, 20, 50]:
                metrics["avg_score_weight"][f"ndcg@{k}"].append(ndcg_at_k(avg_rec, gt_items, k))
                metrics["avg_score_weight"][f"recall@{k}"].append(recall_at_k(avg_rec, gt_items, k))
            
            # 4. Multi-channel recall with average (uniform) top-k scheduling
            # Use multi_channel_recall_top_k with uniform weights
            avg_topk_candidates = multi_channel_recall_top_k(
                recallers=recallers,
                recaller_names=recaller_names,
                user_id=user_id,
                history=eval_hist,
                total_k=eval_k,
                full_hist=full_hist,
                gt_items=gt_items,
                weights=uniform_weights
            )
            avg_topk_rec = [item_id for item_id, _ in avg_topk_candidates]
            for k in [10, 20, 50]:
                metrics["avg_top_k"][f"ndcg@{k}"].append(ndcg_at_k(avg_topk_rec, gt_items, k))
                metrics["avg_top_k"][f"recall@{k}"].append(recall_at_k(avg_topk_rec, gt_items, k))
            
            # Save predictions
            if save_predictions:
                # Save merge weights (softmax weights from model)
                merge_weights = softmax_weights.tolist() if hasattr(softmax_weights, 'tolist') else softmax_weights
                predictions.append({
                    "user_id": user_id,
                    "recaller_predictions": recaller_predictions,
                    "gt_items": gt_items,
                    "merge_weights": merge_weights  # Softmax weights from model
                })
            
            # 5. Evaluate each base recaller
            for recaller_name in recaller_names:
                base_rec = [item[0] for item in recaller_predictions[recaller_name]]
                for k in [10, 20, 50]:
                    metrics[recaller_name][f"ndcg@{k}"].append(ndcg_at_k(base_rec, gt_items, k))
                    metrics[recaller_name][f"recall@{k}"].append(recall_at_k(base_rec, gt_items, k))
    
    # Aggregate results
    results = aggregate_metrics(metrics)
    
    # Print comparison
    print("\n" + "="*60)
    print("Multi-Channel Recall Evaluation")
    print("="*60)
    
    # Print base recaller performance
    print_base_recaller_table(results, recaller_names)
    
    if results.get("single_select") and results.get("multi_channel"):
        print_comparison_table(
            results, recaller_names,
            methods=["single_select", "multi_channel", "avg_score_weight", "avg_top_k"],
            method_labels={"single_select": "Single", "multi_channel": "Multi-Ch", "avg_score_weight": "Avg-SW", "avg_top_k": "Avg-TK"},
            title="Single vs Multi-Ch vs Avg-SW vs Avg-TK"
        )
    performance_gaps = compute_performance_gaps(results, recaller_names, eval_k, method_keys=["single_select", "multi_channel"])
    results["performance_gaps"] = performance_gaps
    print_performance_gaps(performance_gaps, method_labels={"single_select": "Single", "multi_channel": "Multi-Ch"})
    # Add predictions to results
    if save_predictions:
        results["predictions"] = predictions
    
    return results


def apply_chat_format(texts, tokenizer):
    """Apply chat template to texts if tokenizer supports it"""
    chat_texts = []
    for text in texts:
        messages = [{"role": "assistant", "content": text}]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        chat_texts.append(chat_text)
    return chat_texts


def tokenize_function(examples, tokenizer, max_length=1536, autoregressive=False, use_chat_format=False):
    texts = examples["text"]
    
    # Apply chat format if enabled
    if use_chat_format and hasattr(tokenizer, 'apply_chat_template'):
        texts = apply_chat_format(texts, tokenizer)
    
    if autoregressive:
        # Build full sequence: prompt + label
        full_texts = []
        for text, recaller in zip(texts, examples["best_recaller"]):
            full_text = text + f"\n\nBest recaller: {recaller}"
            full_texts.append(full_text)
        
        # Tokenize full sequences
        model_inputs = tokenizer(full_texts, padding="max_length", truncation=True, max_length=max_length)
        
        # Autoregressive: labels = input_ids (next token prediction on full sequence)
        # Only mask padding tokens
        labels = []
        for input_ids in model_inputs["input_ids"]:
            label = input_ids.copy()
            for j, token_id in enumerate(label):
                if token_id == tokenizer.pad_token_id:
                    label[j] = -100
            labels.append(label)
        model_inputs["labels"] = labels
        return model_inputs
    else:
        # For classification: keep numeric labels
        tokenized = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length)
        tokenized["labels"] = examples["labels"]
        return tokenized


def _save_token_stats(data_dir: str, split_name: str, token_stats: dict, num_samples: int):
    """Save per-split token statistics and recompute global max across splits."""
    meta_path = os.path.join(data_dir, "token_stats.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {"splits": {}}
    meta["splits"][split_name] = {**token_stats, "num_samples": num_samples}
    meta["max_tokens_cls"] = max(s["max_tokens_cls"] for s in meta["splits"].values())
    meta["max_tokens_ar"] = max(s["max_tokens_ar"] for s in meta["splits"].values())
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Token stats saved: {meta_path} (global max: cls={meta['max_tokens_cls']}, ar={meta['max_tokens_ar']})")


def get_paths(args):
    """Get standardized paths"""
    model_name = args.model_name.split('/')[-1]
    base = f"{args.output_dir}/{args.dataset}/{model_name}"
    recbole_models = list(args.recbole_models)
    recbole_models.sort()
    recbole_models = '_'.join(recbole_models)
    # Determine training mode prefix
    mode_prefix = "ar_" if args.autoregressive else ""
    # Include profile_cutoff / prompt_top_k in paths when they differ from defaults
    prompt_top_k = getattr(args, 'prompt_top_k', 3)
    pc_suffix = f"_pc{args.profile_cutoff}" if args.profile_cutoff != 20 else ""
    ptk_suffix = f"_ptk{prompt_top_k}" if prompt_top_k != 3 else ""
    param_suffix = f"{pc_suffix}{ptk_suffix}"
    data_path = f"{base}_pure_sft_data_{recbole_models}_{args.profile_cutoff}{ptk_suffix}"
    return {
        "model_name": args.model_name,
        "sft": f"{base}_pure_{mode_prefix}sft_{recbole_models}{param_suffix}",
        "ar_sft": f"{base}_pure_ar_sft_{recbole_models}{param_suffix}",
        "data": data_path,
        "grpo": f"{base}_pure_grpo_{recbole_models}{param_suffix}",
        "grpo_data": f"{base}_pure_grpo_data_{recbole_models}_{args.profile_cutoff}{ptk_suffix}"
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Pure Text SFT Training for Model Selection')
    # Data
    parser.add_argument('--dataset', type=str, default='Amazon_All_Beauty')
    parser.add_argument('--data_path', type=str, default='./dataset')
    parser.add_argument('--output_dir', type=str, default='GRPO/pure_models')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    # Model
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--recbole_models', type=str, nargs='+', default=['BPR', 'SASRec', 'LightGCN'])
    # Tasks
    parser.add_argument('--gen_sft_train', action='store_true', help='Generate SFT training dataset')
    parser.add_argument('--gen_sft_eval', action='store_true', help='Generate SFT evaluation dataset')
    parser.add_argument('--gen_sft_test', action='store_true', help='Generate SFT test dataset')
    parser.add_argument('--do_sft', action='store_true')
    parser.add_argument('--do_test_sft', action='store_true')
    parser.add_argument('--do_test_grpo', action='store_true')
    parser.add_argument('--test_baseline', action='store_true', 
                        help='Run baseline tests (recallers and avg) separately from model testing')
    # Training
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--max_length', type=int, default=1536)
    # Other
    parser.add_argument('--train_k', type=int, default=50,
                        help='Top-k for recaller recall during dataset creation and GRPO training.')
    parser.add_argument('--eval_k', type=int, default=50,
                        help='Top-k for recaller recall during evaluation (NDCG@k, Recall@k).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--padding_side', type=str, default='right', choices=['left', 'right'],
                       help='Padding side for tokenizer. Use "left" for decoder-only models in classification.')
    parser.add_argument('--profile_cutoff', type=int, default=20,
                       help='Cutoff length for user profiles and minimum history length for augmentation.')
    parser.add_argument('--autoregressive', action='store_true',
                       help='Use autoregressive (next token prediction) on full sequence including prompt. Learns domain knowledge from entire sequence.')
    parser.add_argument('--use_dirichlet_head', action='store_true',
                       help='Use Dirichlet distribution head instead of softmax. Requires SoftLabelTrainer.')
    # GRPO training
    parser.add_argument('--do_grpo', action='store_true',
                       help='Run SofT-GRPO training on top of SFT model.')
    parser.add_argument('--tau_gumbel', type=float, default=1.0,
                       help='Gumbel-Softmax temperature for SofT-GRPO.')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Nucleus sampling threshold for SofT-GRPO.')
    parser.add_argument('--noise_scale', type=float, default=0.3,
                       help='Scale factor for Gumbel noise (0.0=no noise, 1.0=full noise).')
    parser.add_argument('--use_soft_grpo_loss', action='store_true', default=True,
                       help='Use SofT-GRPO loss (Gumbel reparameterization). Default: True.')
    parser.add_argument('--use_ppo_loss', action='store_true',
                       help='Use PPO-style loss instead of SofT-GRPO loss.')
    parser.add_argument('--epsilon', type=float, default=0.2,
                       help='PPO clipping coefficient.')
    parser.add_argument('--beta', type=float, default=0.01,
                       help='KL penalty weight.')
    parser.add_argument('--sync_ref_model', action='store_true',
                       help='Periodically sync ref_model with current model.')
    parser.add_argument('--ref_model_sync_steps', type=int, default=100,
                       help='Steps between ref_model syncs.')
    parser.add_argument('--num_generations', type=int, default=4,
                       help='Number of generations per prompt (G in GRPO).')
    parser.add_argument('--grpo_lr', type=float, default=1e-6,
                       help='Learning rate for GRPO training.')
    parser.add_argument('--grpo_epochs', type=int, default=3,
                        help='Number of GRPO training epochs.')
    # Soft label options
    parser.add_argument('--use_soft_label', action='store_true',
                        help='Use soft labels based on score difference instead of hard winner labels.')
    parser.add_argument('--soft_label_temperature', type=float, default=5.0,
                        help='Temperature for soft label: higher = sharper distribution, lower = smoother.')
    parser.add_argument('--random_history_selection', action='store_true',
                        help='Randomly select profile_cutoff items from history instead of using the most recent ones.')
    parser.add_argument('--prompt_top_k', type=int, default=3,
                        help='Number of top items to display per recaller in the prompt.')
    parser.add_argument('--score_norm', type=str, default='minmax',
                        choices=[None, 'none', 'minmax', 'zscore', 'softmax', 'percentile', 'rank_reciprocal', 'rank_exp', 'platt'],
                        help='Rescoring method applied per channel before merging in multi-channel recall. '
                             'Options: None/"none" (no change), minmax (per-channel min-max to [0,1]), '
                             'zscore (per-channel z-score), softmax (per-channel softmax distribution), '
                             'percentile (per-channel empirical CDF to [0,1]), '
                             'rank_reciprocal (1/(offset+rank) using rank only), '
                             'rank_exp (exp(-alpha*rank) using rank only), '
                             'platt (sigmoid(a*s+b), requires score_norm_kwargs).')
    parser.add_argument('--score_norm_kwargs', type=str, default=None,
                        help='JSON string for additional kwargs for the chosen score_norm method. '
                             'Example: \'{"temperature": 1.0}\' for softmax, \'{"offset": 1.0}\' for rank_reciprocal, '
                             '\'{"alpha": 1.0}\' for rank_exp, \'{"a": 1.0, "b": 0.0}\' for platt.')
    parser.add_argument('--merge_method', type=str, default='average',
                        choices=['average', 'top_k'],
                        help='Method for merging recaller results in baseline evaluation. '
                             'Options: average (uniform weights), top_k (keep highest scores).')
    parser.add_argument('--skip_sft_init', action='store_true',
                        help='Skip loading SFT checkpoint for GRPO; initialize from base model instead. '
                             'Used for w/o SFT ablation study.')
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    paths = get_paths(args)
    
    # Parse score_norm_kwargs JSON string if provided
    score_norm_kwargs = None
    if args.score_norm_kwargs:
        try:
            score_norm_kwargs = json.loads(args.score_norm_kwargs)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse --score_norm_kwargs as JSON: {e}. Ignoring it.")
            score_norm_kwargs = None
    
    # Determine which score normalization to use
    # Priority: score_norm > normalize_scores (for backward compatibility)
    score_norm = args.score_norm
    recaller_names = sorted(args.recbole_models)
    recaller_names = [name.lower() for name in recaller_names]
    

    grpo_config = GRPOConfig(
        output_dir=paths["grpo"],
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.grpo_epochs,
        learning_rate=args.grpo_lr,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="constant",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=1000,
        save_total_limit=1,
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        # num_generations=2,
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
    if args.gen_sft_train or args.gen_sft_eval or args.gen_sft_test or args.do_test_sft or args.do_test_grpo or args.test_baseline:
        inter_dataset = load_dataset(
            args.dataset, args.data_path, seed=args.seed,
        )
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
    
    # Generate SFT data
    os.makedirs(paths["data"], exist_ok=True)
    hint_map_path = f'{paths["data"]}/eval_user_hint_map.json'
    label_mapping_path = f'{paths["data"]}/label_mapping.json'
    
    # Generate Eval dataset (must be generated first to get hints)
    if args.gen_sft_eval:
        print("="*60)
        print("Generating Eval dataset to get user hints...")
        print("="*60)
        eval_dataset, label2id, id2label, eval_user_hint_map, eval_global_best, eval_token_stats = create_sft_dataset(
            profile_agent,
            inter_dataset.eval_user_ids,
            inter_dataset.eval_histories,
            inter_dataset.eval_target_items,
            recallers, args,
            use_augmentation=False,
            min_thres=0.001,
        )
        print(f"Collected hints for {len(eval_user_hint_map)} users from eval set")
        eval_dataset.save_to_disk(f'{paths["data"]}/eval')
        _save_token_stats(paths["data"], "eval", eval_token_stats, len(eval_dataset))
        with open(label_mapping_path, 'w') as f:
            json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
        with open(hint_map_path, 'w') as f:
            json.dump({str(k): v for k, v in eval_user_hint_map.items()}, f, indent=2)
        # Save eval-set best recaller (highest avg NDCG) for fallback in train/test
        eval_best_path = f'{paths["data"]}/eval_best_recaller.json'
        with open(eval_best_path, 'w') as f:
            json.dump({"best_recaller": eval_global_best}, f)
        print(f"Saved Eval dataset: {len(eval_dataset)} samples")
    
    # Load eval-set best recaller (highest avg NDCG) for fallback labels
    eval_best_path = f'{paths["data"]}/eval_best_recaller.json'
    eval_best_recaller = json.load(open(eval_best_path)).get("best_recaller") if os.path.exists(eval_best_path) else None
    
    # Generate Train dataset (requires eval hints)
    if args.gen_sft_train:
        print("="*60)
        print("Generating Train dataset with eval hints...")
        print("="*60)
        # Load eval hint map if available
        eval_user_hint_map = load_hint_map(hint_map_path)
        if eval_user_hint_map:
            print(f"Loaded eval hints for {len(eval_user_hint_map)} users")
        train_dataset, _, _, _, _, train_token_stats = create_sft_dataset(
            profile_agent, 
            inter_dataset.train_user_ids,
            inter_dataset.train_histories,
            inter_dataset.train_target_items,
            recallers, args,
            user_hint_map=eval_user_hint_map,
            fallback_recaller=eval_best_recaller,
            min_thres=0.001,
        )
        train_dataset.save_to_disk(f'{paths["data"]}/train')
        _save_token_stats(paths["data"], "train", train_token_stats, len(train_dataset))
        print(f"Saved Train dataset: {len(train_dataset)} samples")
    
    # Generate Test dataset (requires eval hints)
    if args.gen_sft_test:
        print("="*60)
        print("Generating Test dataset with eval hints...")
        print("="*60)
        # Load eval hint map if available
        eval_user_hint_map = load_hint_map(hint_map_path)
        if eval_user_hint_map:
            print(f"Loaded eval hints for {len(eval_user_hint_map)} users")
        test_dataset, _, _, _, _, test_token_stats = create_sft_dataset(
            profile_agent,
            inter_dataset.test_user_ids,
            inter_dataset.test_histories,
            inter_dataset.test_target_items,
            recallers, args,
            user_hint_map=eval_user_hint_map,
            use_augmentation=False,
            fallback_recaller=eval_best_recaller,
            min_thres=0.001,
        )
        test_dataset.save_to_disk(f'{paths["data"]}/test')
        _save_token_stats(paths["data"], "test", test_token_stats, len(test_dataset))
        print(f"Saved Test dataset: {len(test_dataset)} samples")
    
    if args.gen_sft_train or args.gen_sft_eval or args.gen_sft_test:
        return
    
    # Load token stats to auto-set max_length
    token_stats_path = os.path.join(paths["data"], "token_stats.json")
    if os.path.exists(token_stats_path):
        with open(token_stats_path) as f:
            _ts = json.load(f)
        _key = "max_tokens_ar" if args.autoregressive else "max_tokens_cls"
        saved_max = _ts[_key]
        print(f"[INFO] Loaded token stats from dataset: {_key}={saved_max} (cli max_length={args.max_length})")
        args.max_length = saved_max
    
    # Train model
    if args.do_sft:
        # Load data
        train_dataset = Dataset.load_from_disk(f'{paths["data"]}/train')
        eval_dataset = Dataset.load_from_disk(f'{paths["data"]}/eval')
        test_dataset = Dataset.load_from_disk(f'{paths["data"]}/test')
        
        # Load label mapping
        label2id, id2label = load_label_mapping(f'{paths["data"]}/label_mapping.json')
        
        # For non-AR mode: use AR checkpoint as initialization if exists
        base_model = args.model_name
        if not args.autoregressive and os.path.exists(paths["ar_sft"]):
            ar_ckpt = get_last_checkpoint(paths["ar_sft"]) or paths["ar_sft"]
            if os.path.exists(os.path.join(ar_ckpt, "config.json")):
                print(f"[SFT] Loading from AR checkpoint: {ar_ckpt}")
                base_model = ar_ckpt
        
        # Load model and tokenizer
        # Use device_map=None for distributed training (Trainer handles device placement)
        sft_device_map = None if int(os.environ.get("WORLD_SIZE", 1)) > 1 else "auto"
        model, tokenizer = load_model_and_tokenizer(
            base_model, args, label2id, id2label,
            use_dirichlet_head=args.use_dirichlet_head,
            device_map=sft_device_map
        )
        
        # Disable gradient checkpointing for DeBERTa models (not compatible)
        use_gradient_checkpointing = args.gradient_checkpointing
        
        # Check if model is an Instruct model (case insensitive)
        use_chat_format = "instruct" in args.model_name.lower()
        if use_chat_format:
            print(f"[INFO] Detected Instruct model, using chat format for tokenization")
        
        # Tokenize
        tokenize_fn = partial(tokenize_function, tokenizer=tokenizer, max_length=args.max_length, 
                             autoregressive=args.autoregressive, use_chat_format=use_chat_format)
        if args.autoregressive:
            columns_to_remove = ["text", "labels", "best_ndcg", "user_id", "history_len_used"]
        else:
            columns_to_remove = ["text", "best_recaller", "best_ndcg", "user_id", "history_len_used"]
        
        # Remove soft_labels column if not using soft label training
        if not args.use_soft_label and "soft_labels" in train_dataset.column_names:
            columns_to_remove.append("soft_labels")
        
        tokenized_train = train_dataset.map(tokenize_fn, batched=True, 
                                           remove_columns=columns_to_remove)
        tokenized_eval = eval_dataset.map(tokenize_fn, batched=True, 
                                         remove_columns=columns_to_remove)
        tokenized_test = test_dataset.map(tokenize_fn, batched=True, 
                                         remove_columns=columns_to_remove)
        # Train
        if args.autoregressive:
            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
            compute_metrics_fn = partial(compute_metrics_seq2seq, tokenizer=tokenizer, label2id=label2id)
        else:
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            compute_metrics_fn = compute_metrics
        
        # Use SoftLabelTrainer if soft labels are enabled or using Dirichlet head
        use_soft_trainer = args.use_soft_label or args.use_dirichlet_head
        TrainerClass = SoftLabelTrainer if use_soft_trainer else Trainer
        # For autoregressive: use smaller eval batch and only compute loss (logits are huge)
        eval_batch_size = 1 if args.autoregressive else 2
        trainer = TrainerClass(
            model=model,
            args=TrainingArguments(
                output_dir=paths["sft"],
                save_total_limit=1,
                per_device_train_batch_size=args.per_device_train_batch_size,
                per_device_eval_batch_size=eval_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                num_train_epochs=args.num_train_epochs,
                save_steps=1000,
                eval_steps=args.eval_steps,
                logging_steps=args.logging_steps,
                learning_rate=args.learning_rate,
                warmup_steps=args.warmup_steps,
                lr_scheduler_type="constant",
                bf16=args.bf16,
                fp16=args.fp16 and not args.bf16,
                gradient_checkpointing=use_gradient_checkpointing,
                eval_strategy="steps",
                save_strategy="steps",
                metric_for_best_model="eval_loss",
                greater_is_better=False,  # loss 越低越好
                load_best_model_at_end=True,  # 训练结束时加载最佳模型
                report_to="wandb",
                run_name=paths["sft"] + ("_soft" if args.use_soft_label else "") + ("_dirichlet" if args.use_dirichlet_head else "") + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
                seed=args.seed,
                prediction_loss_only=args.autoregressive,  # Skip collecting huge logits for AR mode
            ),
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn,
            **({"use_soft_label": args.use_soft_label} if args.use_soft_label else {}),
        )
        
        print("Training...")
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(paths["sft"])
    
    # GRPO Training
    if args.do_grpo:
        
        print("\n" + "="*60)
        print("Starting Pure GRPO Training")
        print("="*60)
        
        # Initialize recallers if not already done (only needed when running --do_grpo alone)
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
        
        # Load label mapping
        label2id, id2label = load_label_mapping(f'{paths["data"]}/label_mapping.json')
        
        # Determine GRPO init model: SFT checkpoint if available, else base model
        sft_model_path = paths["sft"]
        skip_sft_init = getattr(args, 'skip_sft_init', False)
        if skip_sft_init or not os.path.exists(sft_model_path):
            grpo_init_path = args.model_name
            print(f"[GRPO] Initializing from base model: {grpo_init_path}")
        else:
            grpo_init_path = sft_model_path
            print(f"[GRPO] Initializing from SFT checkpoint: {grpo_init_path}")
        
        # Create a minimal args-like object for GRPO loading
        class GRPOArgs:
            bf16 = args.bf16
            fp16 = args.fp16
            padding_side = "left"
            autoregressive = False
        
        grpo_args = GRPOArgs()
        model, tokenizer = load_model_and_tokenizer(
            grpo_init_path, grpo_args, label2id, id2label,
            device_map="cuda"
        )
        model.config.pad_token_id = tokenizer.eos_token_id
        
        # Load SFT dataset directly for GRPO (already contains prompt, history, target_items)
        grpo_train_dataset = Dataset.load_from_disk(f'{paths["data"]}/train')
        grpo_eval_dataset_full = Dataset.load_from_disk(f'{paths["data"]}/eval')
        grpo_eval_dataset = grpo_eval_dataset_full.select(range(min(1000, len(grpo_eval_dataset_full))))
        print(f"GRPO training dataset: {len(grpo_train_dataset)} samples")
        print(f"GRPO eval dataset: {len(grpo_eval_dataset)} samples")
        
        # Reward function (placeholder - actual rewards computed in trainer)
        def grpo_reward_fn(prompts, completions, completion_ids, **kwargs):
            return [0.0] * len(prompts)
        
        # GRPO Config
        
        # Initialize GRPOTrainer with pure classification mode (like main_soft.py)
        grpo_trainer = GRPOTrainer(
            model=model,
            reward_funcs=grpo_reward_fn,
            args=grpo_config,
            train_dataset=grpo_train_dataset,
            eval_dataset=grpo_eval_dataset,
            processing_class=tokenizer,
        )
        
        # Create reference model for KL penalty (override the default CausalLM ref_model)
        if args.beta > 0:
            ref_model = load_model_only(
                grpo_init_path, grpo_args, label2id, id2label, device_map="cuda"
            )
            ref_model.config.pad_token_id = tokenizer.eos_token_id
            ref_model.eval()
            for param in ref_model.parameters():
                param.requires_grad = False
            # Use standard ref_model attribute (enables SyncRefModelCallback)
            grpo_trainer.ref_model = ref_model
        
        # Enable pure classification GRPO mode (similar to use_beta_sampling in main_soft.py)
        grpo_trainer.use_pure_classification = True
        grpo_trainer.pure_recallers = recallers
        grpo_trainer.pure_recaller_names = recaller_names
        # Use k=5 for reward ndcg@k during GRPO (keep other stages unchanged)
        grpo_trainer.pure_final_k = 5
        grpo_trainer.pure_noise_scale = args.noise_scale
        # Merge method: 'score' (weighted sum) or 'topk' (prefix-quota scheduling)
        grpo_trainer.pure_merge_method = 'topk' if args.merge_method == 'top_k' else 'score'
        # SofT-GRPO loss (default) vs PPO-style loss
        grpo_trainer.use_soft_grpo_loss = not args.use_ppo_loss
        # Keep extra columns for reward computation (not removed by _remove_unused_columns)
        grpo_trainer._signature_columns = ["prompt", "user_id", "history", "target_items"]
        
        print("Starting GRPO training...")
        grpo_trainer.train()
        grpo_trainer.save_model()
        tokenizer.save_pretrained(paths["grpo"])
        print(f"GRPO model saved to {paths['grpo']}")
    
    # Test model
    if args.do_test_sft or args.do_test_grpo or args.test_baseline:
        # Load test dataset from disk (generated with --gen_sft_test)
        if os.path.exists(f'{paths["data"]}/test'):
            test_dataset = Dataset.load_from_disk(f'{paths["data"]}/test')
            print(f"Loaded test dataset: {len(test_dataset)} samples")
        else:
            test_dataset, _, _, _, _, _ = create_sft_dataset(
                profile_agent,
                inter_dataset.test_user_ids,
                inter_dataset.test_histories,
                inter_dataset.test_target_items,
                recallers, args,
                use_augmentation=False,
            )
        
        if args.autoregressive:
            print("Autoregressive evaluation not implemented yet. Use training eval metrics instead.")
            return
        
        recaller_combo = "_".join(recaller_names)
        model_name_short = args.model_name.split("/")[-1]  # e.g. Qwen2.5-0.5B-Instruct
        _ptk = getattr(args, 'prompt_top_k', 3)
        _param_suffix = (f"_pc{args.profile_cutoff}" if args.profile_cutoff != 20 else "") + \
                        (f"_ptk{_ptk}" if _ptk != 3 else "")
        os.makedirs("results", exist_ok=True)
        base_config = {
            "dataset": args.dataset,
            "recbole_models": recaller_names,
            "recaller_combo": recaller_combo,
            "train_k": args.train_k,
            "eval_k": args.eval_k,
            "profile_cutoff": args.profile_cutoff,
            "test_samples": len(test_dataset),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # Build list of checkpoints to test
        test_configs = []
        if args.do_test_sft:
            test_configs.append(("sft", paths["sft"]))
        if args.do_test_grpo:
            test_configs.append(("grpo", paths["grpo"]))
        
        # Load label mapping once (shared across checkpoints)
        label2id, id2label = None, None
        if test_configs:
            label2id, id2label = load_label_mapping(f'{paths["data"]}/label_mapping.json')
        
        # Test each checkpoint
        all_multi_results = {}
        for test_name, model_base_path in test_configs:
            print("\n" + "="*60)
            print(f"MODEL TESTING: {test_name.upper()}")
            print("="*60)
            
            # Resolve model path
            model_path = model_base_path if os.path.exists(model_base_path) else args.model_name
            if os.path.exists(model_path) and not os.path.exists(os.path.join(model_path, "config.json")):
                last_checkpoint = get_last_checkpoint(model_path)
                if last_checkpoint:
                    model_path = last_checkpoint
                    print(f"Loading from last checkpoint: {model_path}")
                else:
                    print(f"Warning: No checkpoint found in {model_path}")
            
            model, tokenizer = load_model_and_tokenizer(
                model_path, args, label2id=label2id, id2label=id2label
            )
            model.config.pad_token_id = tokenizer.eos_token_id
            
            # Check if model is an Instruct model (case insensitive)
            use_chat_format = "instruct" in args.model_name.lower()
            
            results = evaluate_pure_model(model, tokenizer, test_dataset, id2label, recallers, args.eval_k, use_chat_format=use_chat_format, max_length=args.max_length)
            multi_results = evaluate_multi_channel_recall(
                model, tokenizer, test_dataset, recallers, recaller_names,
                args.eval_k, merge_method=args.merge_method,
                score_norm=score_norm,
                score_norm_kwargs=score_norm_kwargs,
                id2label=id2label,
                use_chat_format=use_chat_format,
                max_length=args.max_length,
            )
            all_multi_results[test_name] = multi_results
            
            results["multi_channel_evaluation"] = multi_results
            results["config"] = {**base_config, "model_name": args.model_name, "model_path": model_path}
            
            if "predictions" in multi_results:
                pred_file = f"results/pure_predictions_{args.dataset}_{model_name_short}_{recaller_combo}_{test_name}{_param_suffix}.json"
                with open(pred_file, "w") as f:
                    json.dump(multi_results.pop("predictions"), f)
                print(f"Predictions saved to: {pred_file}")
            
            result_file = f"results/pure_model_results_{args.dataset}_{model_name_short}_{recaller_combo}_{test_name}{_param_suffix}.json"
            with open(result_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Model results saved to: {result_file}")
            
            # Free GPU memory before loading next checkpoint
            del model, tokenizer
            torch.cuda.empty_cache()
        
        # Print cross-checkpoint comparison table
        if len(all_multi_results) > 1:
            print_cross_checkpoint_table(all_multi_results, recaller_names)
        
        # Baseline testing
        if args.test_baseline:
            print("\n" + "="*60)
            print("BASELINE TESTING (Recallers + Avg)")
            print("="*60)
            baseline_results = evaluate_baseline_recallers(
                test_dataset, recallers, recaller_names, args.eval_k,
                save_predictions=True, 
                merge_method=args.merge_method,
                score_norm=score_norm,
                score_norm_kwargs=score_norm_kwargs
            )
            baseline_results["config"] = {**base_config}
            
            if "predictions" in baseline_results:
                pred_file = f"results/baseline_predictions_{args.dataset}_{recaller_combo}.json"
                with open(pred_file, "w") as f:
                    json.dump(baseline_results.pop("predictions"), f)
                print(f"Baseline predictions saved to: {pred_file}")
            
            result_file = f"results/baseline_results_{args.dataset}_{recaller_combo}.json"
            with open(result_file, "w") as f:
                json.dump(baseline_results, f, indent=2)
            print(f"Baseline results saved to: {result_file}")


if __name__ == "__main__":
    main()