#!/usr/bin/env python3
"""
GRPO examples using RecBole models
"""

import os
import sys

def main():
    # Example 1: Use RecBole models for router recommendation (Router-only mode)
    print("Example 1: Router-only mode using trained RecBole models")
    
    cmd1 = [
        "python", "GRPO.py",
        "--dataset", "All_Beauty",  # Adjust according to your dataset name
        "--data_path", "./data",
        "--recbole_models", "SASRec", "BPR", "Pop",  # RecBole models to use
        "--checkpoint_dir", "./checkpoints",
        "--use_latest_checkpoint",  # Use the latest checkpoint
        "--router_only",  # Use router only
        "--final_k", "50",
        "--group_size", "4",
        "--seed", "42"
    ]
    
    print("Command 1:", " ".join(cmd1))
    print()
    
    # Example 2: Train router (GRPO)
    print("Example 2: Training mode, using GRPO to train router")
    
    cmd2 = [
        "python", "GRPO.py",
        "--dataset", "All_Beauty",
        "--data_path", "./data", 
        "--recbole_models", "SASRec", "BPR",
        "--checkpoint_dir", "./checkpoints",
        "--use_latest_checkpoint",
        "--router_only",
        "--epochs", "5",
        "--router_batch_size", "64",
        "--group_size", "4",
        "--final_k", "50",
        "--seed", "42"
    ]
    
    print("Command 2:", " ".join(cmd2))
    print()
    
    # Example 3: Use local HuggingFace model for routing
    print("Example 3: Using local HuggingFace model for routing")
    
    cmd3 = [
        "python", "GRPO.py",
        "--dataset", "All_Beauty",
        "--data_path", "./data",
        "--recbole_models", "SASRec", "BPR", "Pop",
        "--checkpoint_dir", "./checkpoints", 
        "--use_latest_checkpoint",
        "--router_only",
        "--use_hf_local",  # Use local HF model
        "--hf_model", "microsoft/DialoGPT-medium",  # Specify model
        "--final_k", "50"
    ]
    
    print("Command 3:", " ".join(cmd3))
    print()
    
    print("Notes:")
    print("1. Ensure your dataset is in ./data directory and follows RecBole format requirements")
    print("2. Ensure trained model weight files exist in ./checkpoints directory")
    print("3. Model names should match checkpoint file name prefixes, e.g., 'SASRec-Aug-26-2025_16-28-37.pth'")
    print("4. If no checkpoint exists, the system will use untrained models")
    print("5. Available RecBole models include: SASRec, BPR, Pop, ItemKNN, etc.")

if __name__ == "__main__":
    main()
