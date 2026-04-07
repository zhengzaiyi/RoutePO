#!/usr/bin/env python3
"""
Inference latency benchmark for GRPO, CEM, and PG methods.

Measures per-sample average latency for:
  1. Recall channel (RecBole model) inference
  2. Extra overhead introduced by each fusion method (GRPO / CEM / PG)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from GRPO.core.data import load_dataset
from GRPO.core.recallers import RecBoleRecaller
from GRPO.core.utils import set_seed
from GRPO.models.main import initialize_recallers
from GRPO.models.evaluation_utils import extract_eval_data
from GRPO.baselines.cem_utils import fuse_by_quota
from GRPO.baselines.baseline_pg import PersonalizedFusionPG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _stats(times_ms: List[float]) -> Dict[str, float]:
    """Compute summary statistics for a list of latencies (in ms)."""
    a = np.array(times_ms)
    return {
        "mean_ms": float(np.mean(a)),
        "std_ms": float(np.std(a)),
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "n_samples": len(a),
    }


# ---------------------------------------------------------------------------
# 1. Recall-channel latency
# ---------------------------------------------------------------------------

def benchmark_recall_channels(
    recallers: Dict[str, RecBoleRecaller],
    recaller_names: List[str],
    user_ids: List[int],
    eval_hists: List[List[int]],
    gt_items_list: List,
    full_hists: List,
    eval_k: int,
    warmup: int = 10,
) -> Tuple[Dict, List]:
    """
    Time each RecBole recall() call per user, returning per-channel and
    aggregate (sum-of-K-channels) per-sample latencies.

    Also returns the cached candidate lists so downstream methods
    don't need to re-invoke recall().
    """
    K = len(recaller_names)
    n_users = len(user_ids)

    per_channel_times: Dict[str, List[float]] = {n: [] for n in recaller_names}
    per_sample_total_times: List[float] = []

    # user_candidates[u][k] = list of (item_id, score) tuples
    user_candidates_raw: List[Dict[str, list]] = []

    for u in tqdm(range(n_users), desc="Recall channels"):
        uid = user_ids[u]
        hist = eval_hists[u]
        gt = list(gt_items_list[u]) if gt_items_list[u] else None
        fh = full_hists[u] if full_hists[u] else None

        sample_cands: Dict[str, list] = {}
        sample_total = 0.0

        for name in recaller_names:
            _sync_cuda()
            t0 = time.perf_counter()
            items = recallers[name].recall(uid, eval_k, hist, full_hist=fh, gt_items=gt)
            _sync_cuda()
            dt = (time.perf_counter() - t0) * 1000  # ms

            sample_cands[name] = items if items else []

            if u >= warmup:
                per_channel_times[name].append(dt)
                sample_total += dt

        user_candidates_raw.append(sample_cands)
        if u >= warmup:
            per_sample_total_times.append(sample_total)

    results = {
        "per_channel": {n: _stats(per_channel_times[n]) for n in recaller_names},
        "total_per_sample": _stats(per_sample_total_times),
    }
    return results, user_candidates_raw


# ---------------------------------------------------------------------------
# 2. GRPO (LLM) extra overhead
# ---------------------------------------------------------------------------

def benchmark_grpo_extra(
    model,
    tokenizer,
    test_dataset,
    recaller_names: List[str],
    user_candidates_raw: List[Dict[str, list]],
    eval_k: int,
    max_length: int = 1536,
    use_chat_format: bool = False,
    warmup: int = 10,
) -> Dict:
    """
    Measure per-sample extra latency of GRPO:
      tokenize + LLM forward + softmax -> weights + fusion merge.
    Recall results are pre-cached; only the decision overhead is timed.
    """
    device = model.device
    model.eval()

    llm_forward_times: List[float] = []
    fusion_times: List[float] = []
    total_extra_times: List[float] = []

    K = len(recaller_names)

    with torch.no_grad():
        valid_idx = 0
        for idx, example in enumerate(tqdm(test_dataset, desc="GRPO extra")):
            text = example["text"]
            if use_chat_format and hasattr(tokenizer, "apply_chat_template"):
                messages = [{"role": "assistant", "content": text}]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )

            hist = example.get("history") or example.get("eval_hist", [])
            if isinstance(hist, list) and 0 in hist:
                hist = hist[: hist.index(0)]
            if len(hist) < 5:
                continue

            cands = user_candidates_raw[idx]

            # --- LLM forward ---
            _sync_cuda()
            t0 = time.perf_counter()

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits[0]
            softmax_weights = torch.softmax(logits, dim=-1)

            _sync_cuda()
            t_llm = (time.perf_counter() - t0) * 1000

            # --- Fusion (top-k quota scheduling, matching evaluate_multi_channel_recall) ---
            _sync_cuda()
            t1 = time.perf_counter()

            # Build per-channel item-id lists for fuse_by_quota
            per_channel_lists = []
            for name in recaller_names:
                ids = [item[0] for item in cands[name]] if cands[name] else []
                per_channel_lists.append(ids)
            fuse_by_quota([per_channel_lists], softmax_weights.cpu(), eval_k)

            _sync_cuda()
            t_fusion = (time.perf_counter() - t1) * 1000

            if valid_idx >= warmup:
                llm_forward_times.append(t_llm)
                fusion_times.append(t_fusion)
                total_extra_times.append(t_llm + t_fusion)

            valid_idx += 1

    return {
        "llm_forward": _stats(llm_forward_times),
        "fusion": _stats(fusion_times),
        "total_extra": _stats(total_extra_times),
    }


# ---------------------------------------------------------------------------
# 3. CEM extra overhead
# ---------------------------------------------------------------------------

def benchmark_cem_extra(
    recaller_names: List[str],
    user_candidates_raw: List[Dict[str, list]],
    eval_hists: List[List[int]],
    eval_k: int,
    warmup: int = 10,
) -> Dict:
    """
    CEM uses globally fixed weights -> only fuse_by_quota at inference.
    We use uniform weights here (timing is weight-independent).
    """
    K = len(recaller_names)
    weights = torch.ones(K) / K

    fusion_times: List[float] = []
    measured = 0

    for u in tqdm(range(len(user_candidates_raw)), desc="CEM extra"):
        hist = eval_hists[u]
        if isinstance(hist, list) and 0 in hist:
            hist = hist[: hist.index(0)]
        if len(hist) < 5:
            continue

        cands = user_candidates_raw[u]
        per_channel_lists = []
        for name in recaller_names:
            ids = [item[0] for item in cands[name]] if cands[name] else []
            per_channel_lists.append(ids)

        t0 = time.perf_counter()
        fuse_by_quota([per_channel_lists], weights, eval_k)
        dt = (time.perf_counter() - t0) * 1000

        if measured >= warmup:
            fusion_times.append(dt)
        measured += 1

    return {"fusion": _stats(fusion_times), "total_extra": _stats(fusion_times)}


# ---------------------------------------------------------------------------
# 4. PG extra overhead
# ---------------------------------------------------------------------------

def benchmark_pg_extra(
    recaller_names: List[str],
    user_candidates_raw: List[Dict[str, list]],
    user_ids: List[int],
    eval_hists: List[List[int]],
    num_users: int,
    num_items: int,
    eval_k: int,
    device: str = "cuda",
    warmup: int = 10,
) -> Dict:
    """
    PG: neural forward (predict_weights) + fuse_by_quota per sample.
    We instantiate the model with random weights (same architecture)
    because inference latency depends only on tensor shapes, not values.
    """
    K = len(recaller_names)

    pg_model = PersonalizedFusionPG(
        num_users=num_users,
        num_items=num_items,
        num_channels=K,
        embedding_dim=64,
        hidden_dim=128,
        device=device,
    )

    # Filter valid users (history >= 5)
    valid_indices = []
    for u in range(len(user_ids)):
        hist = eval_hists[u]
        if isinstance(hist, list) and 0 in hist:
            hist = hist[: hist.index(0)]
        if len(hist) >= 5:
            valid_indices.append(u)

    v_uids = [user_ids[i] for i in valid_indices]
    v_hists = [eval_hists[i] for i in valid_indices]
    v_cands_raw = [user_candidates_raw[i] for i in valid_indices]

    # Convert to user_candidates format: list[u][k] = list of item_ids
    v_cands = []
    for cands_dict in v_cands_raw:
        per_ch = []
        for name in recaller_names:
            ids = [item[0] for item in cands_dict[name]] if cands_dict[name] else []
            per_ch.append(ids)
        v_cands.append(per_ch)

    n = len(v_uids)

    # --- Per-sample timing (batch_size=1 to reflect online latency) ---
    forward_times: List[float] = []
    fusion_times: List[float] = []
    total_extra_times: List[float] = []

    pg_model.user_module.eval()
    pg_model.channel_module.eval()
    pg_model.policy.eval()

    with torch.no_grad():
        for i in tqdm(range(n), desc="PG extra"):
            uid_list = [v_uids[i]]
            hist_list = [v_hists[i]]
            cand_list = [v_cands[i]]

            # --- PG forward ---
            _sync_cuda()
            t0 = time.perf_counter()
            w = pg_model.predict_weights(uid_list, hist_list, cand_list)
            _sync_cuda()
            t_fwd = (time.perf_counter() - t0) * 1000

            # --- Fusion ---
            t1 = time.perf_counter()
            fuse_by_quota(cand_list, w[0], eval_k)
            t_fus = (time.perf_counter() - t1) * 1000

            if i >= warmup:
                forward_times.append(t_fwd)
                fusion_times.append(t_fus)
                total_extra_times.append(t_fwd + t_fus)

    return {
        "pg_forward": _stats(forward_times),
        "fusion": _stats(fusion_times),
        "total_extra": _stats(total_extra_times),
    }


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_test_users(test_dataset) -> Tuple[List[int], List[list], List[set], List]:
    """Extract user_ids, histories, ground truths, and full_hists from test dataset."""
    user_ids, eval_hists, gt_items_list, full_hists = [], [], [], []

    for example in test_dataset:
        user_id, eval_hist, gt_items, full_hist = extract_eval_data(example)
        if isinstance(eval_hist, list) and 0 in eval_hist:
            eval_hist = eval_hist[: eval_hist.index(0)]
        user_ids.append(user_id)
        eval_hists.append(eval_hist)
        if isinstance(gt_items, int):
            gt_items = [gt_items]
        gt_items_list.append(set(gt_items) if gt_items else set())
        full_hists.append(full_hist)

    return user_ids, eval_hists, gt_items_list, full_hists


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_summary_table(recall_res, grpo_res, cem_res, pg_res, dataset_name, combo_name):
    hdr = f"\nDataset: {dataset_name} | Recallers: {combo_name}"
    sep = "=" * 72
    print(sep)
    print(hdr)
    print(sep)

    fmt = "{:<30s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}"
    print(fmt.format("Component", "Mean(ms)", "Std(ms)", "Median(ms)", "P95(ms)"))
    print("-" * 72)

    def _row(label: str, s: Dict):
        print(fmt.format(
            label,
            f"{s['mean_ms']:.3f}",
            f"{s['std_ms']:.3f}",
            f"{s['median_ms']:.3f}",
            f"{s['p95_ms']:.3f}",
        ))

    # Recall channels
    _row("Recall Total (K channels)", recall_res["total_per_sample"])
    for ch_name, ch_stats in recall_res["per_channel"].items():
        _row(f"  {ch_name}", ch_stats)

    print("-" * 72)

    # GRPO
    if grpo_res:
        _row("GRPO Extra (total)", grpo_res["total_extra"])
        _row("  LLM forward", grpo_res["llm_forward"])
        _row("  Fusion", grpo_res["fusion"])

    # CEM
    _row("CEM Extra (total)", cem_res["total_extra"])

    # PG
    _row("PG Extra (total)", pg_res["total_extra"])
    _row("  PG forward", pg_res["pg_forward"])
    _row("  Fusion", pg_res["fusion"])

    print(sep + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Inference latency benchmark")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--data_path", type=str, default="./dataset")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--recbole_models", type=str, nargs="+", required=True)
    p.add_argument("--eval_k", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup_samples", type=int, default=10)
    p.add_argument("--max_samples", type=int, default=0,
                   help="Limit number of test samples (0 = all)")
    p.add_argument("--output_dir", type=str, default="results/latency")

    # GRPO model (optional; skipped if not provided)
    p.add_argument("--grpo_model_path", type=str, default=None,
                   help="Path to trained GRPO / SFT classification checkpoint")
    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                   help="Base model name (for tokenizer and config)")
    p.add_argument("--max_length", type=int, default=1536)
    p.add_argument("--bf16", action="store_true")

    # Test dataset path (pre-generated by main_pure.py)
    p.add_argument("--test_dataset_path", type=str, default=None,
                   help="Path to pre-generated test dataset directory")

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    recaller_names = sorted(args.recbole_models)
    combo_name = "_".join(recaller_names)

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load dataset & recallers
    # ------------------------------------------------------------------
    print("Loading dataset ...")
    inter_dataset = load_dataset(args.dataset, args.data_path, seed=args.seed)

    print("Initializing recallers ...")
    recallers = initialize_recallers(
        model_names=args.recbole_models,
        dataset_name=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        data_path=args.data_path,
        seed=args.seed,
        use_latest_checkpoint=True,
        num_items=inter_dataset.ds.item_num,
    )

    # ------------------------------------------------------------------
    # 2. Load test dataset
    # ------------------------------------------------------------------
    if args.test_dataset_path and os.path.isdir(args.test_dataset_path):
        test_dataset = Dataset.load_from_disk(args.test_dataset_path)
        print(f"Loaded test dataset from {args.test_dataset_path}: {len(test_dataset)} samples")
    else:
        raise FileNotFoundError(
            f"Test dataset not found at {args.test_dataset_path}. "
            "Generate it first with main_pure.py --gen_sft_test."
        )

    if args.max_samples > 0:
        test_dataset = test_dataset.select(range(min(args.max_samples, len(test_dataset))))
        print(f"Using {len(test_dataset)} samples")

    user_ids, eval_hists, gt_items_list, full_hists = extract_test_users(test_dataset)

    # ------------------------------------------------------------------
    # 3. Benchmark recall channels
    # ------------------------------------------------------------------
    print("\n>>> Benchmarking recall channels ...")
    recall_res, user_candidates_raw = benchmark_recall_channels(
        recallers, recaller_names, user_ids, eval_hists,
        gt_items_list, full_hists, args.eval_k, warmup=args.warmup_samples,
    )

    # ------------------------------------------------------------------
    # 4. Benchmark GRPO extra (optional)
    # ------------------------------------------------------------------
    grpo_res = None
    if args.grpo_model_path and os.path.isdir(args.grpo_model_path):
        print("\n>>> Benchmarking GRPO extra overhead ...")
        from GRPO.models.model_utils import load_model_and_tokenizer, load_label_mapping

        # Build a minimal args namespace for load_model_and_tokenizer
        class _Args:
            bf16 = args.bf16
            fp16 = False
            padding_side = "left"
            autoregressive = False
            use_dirichlet_head = False
            model_name = args.model_name
            max_length = args.max_length

        label_map_path = os.path.join(
            os.path.dirname(args.test_dataset_path), "label_mapping.json"
        )
        if os.path.exists(label_map_path):
            label2id, id2label = load_label_mapping(label_map_path)
        else:
            label2id = {n: i for i, n in enumerate(recaller_names)}
            id2label = {i: n for i, n in enumerate(recaller_names)}

        model, tokenizer = load_model_and_tokenizer(
            args.grpo_model_path, _Args(), label2id=label2id, id2label=id2label
        )
        model.config.pad_token_id = tokenizer.eos_token_id

        use_chat_format = "instruct" in args.model_name.lower()

        grpo_res = benchmark_grpo_extra(
            model, tokenizer, test_dataset, recaller_names,
            user_candidates_raw, args.eval_k,
            max_length=args.max_length,
            use_chat_format=use_chat_format,
            warmup=args.warmup_samples,
        )

        del model, tokenizer
        torch.cuda.empty_cache()
    else:
        print("\n>>> Skipping GRPO benchmark (no --grpo_model_path provided)")

    # ------------------------------------------------------------------
    # 5. Benchmark CEM extra
    # ------------------------------------------------------------------
    print("\n>>> Benchmarking CEM extra overhead ...")
    cem_res = benchmark_cem_extra(
        recaller_names, user_candidates_raw, eval_hists,
        args.eval_k, warmup=args.warmup_samples,
    )

    # ------------------------------------------------------------------
    # 6. Benchmark PG extra
    # ------------------------------------------------------------------
    print("\n>>> Benchmarking PG extra overhead ...")
    pg_res = benchmark_pg_extra(
        recaller_names, user_candidates_raw, user_ids, eval_hists,
        num_users=inter_dataset.ds.user_num,
        num_items=inter_dataset.ds.item_num,
        eval_k=args.eval_k, device=device,
        warmup=args.warmup_samples,
    )

    # ------------------------------------------------------------------
    # 7. Print & save
    # ------------------------------------------------------------------
    print_summary_table(recall_res, grpo_res, cem_res, pg_res, args.dataset, combo_name)

    output = {
        "config": {
            "dataset": args.dataset,
            "recbole_models": recaller_names,
            "eval_k": args.eval_k,
            "warmup_samples": args.warmup_samples,
            "max_samples": args.max_samples,
            "grpo_model_path": args.grpo_model_path,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "recall_channel": recall_res,
        "grpo_extra": grpo_res,
        "cem_extra": cem_res,
        "pg_extra": pg_res,
    }

    out_file = os.path.join(
        args.output_dir,
        f"latency_{args.dataset}_{combo_name}.json",
    )
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
