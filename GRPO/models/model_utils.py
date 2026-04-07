"""
Model and tokenizer loading utilities.
Extracted to avoid code duplication across different training and testing stages.
"""

import torch
import json
import os
from typing import Dict, Optional, Tuple, Any
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer
)
# Import from main_pure at function level to avoid circular import


def load_label_mapping(path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Load label mapping from JSON file.
    
    Args:
        path: Path to label_mapping.json file
    
    Returns:
        Tuple of (label2id, id2label) dictionaries
    """
    with open(path, 'r') as f:
        labels = json.load(f)
        label2id = labels["label2id"]
        id2label = {int(k): v for k, v in labels["id2label"].items()}
    return label2id, id2label


def load_hint_map(path: str) -> Dict[int, str]:
    """
    Load eval user hint map from JSON file.
    
    Args:
        path: Path to eval_user_hint_map.json file
    
    Returns:
        Dictionary mapping user_id -> best_recaller_name
    """
    if not os.path.exists(path):
        return {}
    
    with open(path, 'r') as f:
        hint_map = json.load(f)
        return {int(k): v for k, v in hint_map.items()}


def load_model_and_tokenizer(
    model_path: str,
    args: Any,
    label2id: Optional[Dict[str, int]] = None,
    id2label: Optional[Dict[int, str]] = None,
    use_dirichlet_head: bool = False,
    device_map: str = "auto"
) -> Tuple[Any, Any]:
    """
    Load model and tokenizer with consistent configuration.
    
    Args:
        model_path: Path to model or model name
        args: Arguments object containing bf16, fp16, padding_side, autoregressive flags
        label2id: Optional label2id mapping for classification models
        id2label: Optional id2label mapping for classification models
        use_dirichlet_head: Whether to wrap model with DirichletSequenceClassification
        device_map: Device map for model loading
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # Import here to avoid circular import
    from GRPO.models.main_pure import safe_load_tokenizer, DirichletSequenceClassification
    
    # Calculate dtype
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    
    # Load model based on type
    if getattr(args, 'autoregressive', False):
        # Autoregressive: use CausalLM for next token prediction
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map
        )
    else:
        # Classification model
        if label2id is None or id2label is None:
            raise ValueError("label2id and id2label must be provided for classification models")
        
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            torch_dtype=dtype,
            device_map=device_map
        )
        
        # Wrap with Dirichlet head if requested
        if use_dirichlet_head:
            model = DirichletSequenceClassification(base_model)
            print("Using DirichletSequenceClassification with Dirichlet distribution head")
        else:
            model = base_model
    
    # Load tokenizer
    tokenizer = safe_load_tokenizer(model_path)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = args.padding_side
    
    # Set pad_token_id in model config if needed
    if hasattr(model, 'config'):
        model.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    
    return model, tokenizer


def load_model_only(
    model_path: str,
    args: Any,
    label2id: Optional[Dict[str, int]] = None,
    id2label: Optional[Dict[int, str]] = None,
    device_map: str = "auto"
) -> Any:
    """
    Load model only (without tokenizer). Useful for loading reference models.
    
    Args:
        model_path: Path to model or model name
        args: Arguments object containing bf16, fp16, autoregressive flags
        label2id: Optional label2id mapping for classification models
        id2label: Optional id2label mapping for classification models
        device_map: Device map for model loading
    
    Returns:
        Model object
    """
    # Calculate dtype
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    
    # Load model based on type
    if getattr(args, 'autoregressive', False):
        # Autoregressive: use CausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map
        )
    else:
        # Classification model
        if label2id is None or id2label is None:
            raise ValueError("label2id and id2label must be provided for classification models")
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            torch_dtype=dtype,
            device_map=device_map
        )
    
    return model

