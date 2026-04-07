import torch
import torch.nn as nn
from typing import Dict, Union, Any, Optional, Tuple
from transformers import PreTrainedModel
from trl import SFTTrainer


class SoftSFTTrainer(SFTTrainer):
    """Custom SFT Trainer that handles different loss functions for [num] and [soft_token] predictions."""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss with different strategies:
        - For [num] tokens: use cross-entropy loss
        - For [soft_token] tokens: use MSE loss from value head (no CE loss since it's deterministic)
        
        Since each sample has only one token as completion:
        - If completion is [num]: compute CE loss
        - If completion is [soft_token]: compute MSE loss only
        
        Args:
            model: The model to compute loss for
            inputs: Dictionary of inputs
            return_outputs: Whether to return outputs along with loss
            num_items_in_batch: Number of items in batch (for gradient accumulation)
        """
        value_labels = inputs.pop("value_labels", None)
        value_mask = inputs.pop("value_mask", None)
        
        outputs = model(value_labels=value_labels, value_mask=value_mask, **inputs)
        
        loss = outputs.get("loss", None)
        if loss is None:
            raise ValueError("Model did not return a loss. Ensure the model supports mixed token losses.")
        
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Override prediction step to handle value predictions."""
        
        inputs = self._prepare_inputs(inputs)
        
        # Pop auxiliary tensors to avoid duplication during inference
        value_labels = inputs.pop("value_labels", None)
        value_mask = inputs.pop("value_mask", None)
        
        with torch.no_grad():
            outputs = model(value_labels=value_labels, value_mask=value_mask, **inputs)
            loss = outputs.get("loss")
            if loss is None:
                raise ValueError("Model did not return a loss during prediction.")
            
        if prediction_loss_only:
            return (loss, None, None)
            
        # Return logits and labels (use lm_logits if available)
        logits = outputs.get("lm_logits")
        return (loss, logits, inputs.get("labels"))
