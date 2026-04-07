import os
from dataclasses import dataclass
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput


def beta_nll_loss(alpha, beta, target, apply_transform=True):
    """
    Compute negative log-likelihood for Beta distribution.
    
    Args:
        alpha: Alpha parameter of Beta distribution (shape: [batch_size])
        beta: Beta parameter of Beta distribution (shape: [batch_size])
        target: Target values in [0, 1] (shape: [batch_size])
        apply_transform: Whether to apply softplus+1.0 transform to alpha/beta
    
    Returns:
        Negative log-likelihood loss
    """
    if apply_transform:
        # Ensure parameters are positive and > 1 for well-defined Beta distribution
        # Using softplus and adding 1.0 to ensure alpha, beta > 1
        alpha = F.softplus(alpha) + 1.0
        beta = F.softplus(beta) + 1.0
    
    # Ensure target is in valid range with epsilon for numerical stability
    # Using larger epsilon to avoid extreme values
    target = target.clamp(0.01, 0.99)
    
    # Compute log Beta function: log B(alpha, beta) = log Gamma(alpha) + log Gamma(beta) - log Gamma(alpha + beta)
    log_beta_fn = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    
    # Compute negative log-likelihood
    # -log p(y|α,β) = -[(α-1)log y + (β-1)log(1-y)] + log B(α,β)
    nll = -((alpha - 1) * torch.log(target) + (beta - 1) * torch.log(1 - target)) + log_beta_fn
    
    return nll.mean()

def logit_normal_nll(mu, sigma, target, eps=1e-4):
    """
    target: [0,1]，内部做轻微夹取到 (eps, 1-eps)
    mu, sigma: 实数与正数，分别是 logit(y) 的均值/标准差
    """
    y = target.clamp(eps, 1 - eps)
    z = torch.logit(y)                 # R
    sigma = sigma.clamp_min(1e-6)
    # 高斯 NLL：0.5 * [ ((z-mu)/sigma)^2 + 2*log(sigma) + log(2π) ]
    nll = 0.5 * (((z - mu) / sigma) ** 2 + 2.0 * torch.log(sigma) + math.log(2 * math.pi))  # log(2π)=1.8378...
    return nll.mean()

class LLMWithValueHead(PreTrainedModel):
    """Wraps a language model with an additional scalar regression head."""
    def __init__(self, base_model, tokenizer):
        # Initialize with base model's config
        super().__init__(base_model.config)
        self.transformer = base_model  # Store the base model (avoid 'base_model' which conflicts with parent class)
        hidden_size = self.config.hidden_size
        self.tokenizer = tokenizer
        # 2. [MOD] Value head for Beta distribution parameters
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)  # Output alpha and beta parameters
        )
        # Initialize value_head weights
        for module in self.value_head:
            if isinstance(module, nn.Linear):
                # use a simple initialization
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def get_input_embeddings(self):
        """Returns the model's input embeddings."""
        return self.transformer.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        """Set model's input embeddings."""
        self.transformer.set_input_embeddings(value)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, tokenizer=None, *args, **kwargs):
        """Load pretrained model with non-strict mode to allow missing weights."""
        # Load the base model first
        model_kwargs = {k: v for k, v in kwargs.items() if k != 'tokenizer'}
        model_kwargs['attn_implementation'] = 'eager'  # Avoid issues with custom model
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, 
            *args, 
            **model_kwargs
        )
        
        # Load or use provided tokenizer
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
            special_tokens = {"additional_special_tokens": ["[num]", "[soft_token]"]}
            tokenizer.add_special_tokens(special_tokens)
        
        # Create the model instance
        model = cls(base_model, tokenizer)
        
        # Try to load additional weights if they exist (non-strict mode)
        weights_file = None
        if os.path.isdir(pretrained_model_name_or_path):
            # Check for different weight file names
            for filename in ["pytorch_model.bin", "model.safetensors", "adapter_model.bin"]:
                filepath = os.path.join(pretrained_model_name_or_path, filename)
                if os.path.exists(filepath):
                    weights_file = filepath
                    break
        
        if weights_file:
            try:
                if weights_file.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    state_dict = load_file(weights_file)
                else:
                    state_dict = torch.load(weights_file, map_location="cpu")
                
                # Load with strict=False to allow missing keys
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    print(f"Missing keys (will be randomly initialized): {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys (ignored): {unexpected_keys}")
            except Exception as e:
                print(f"Warning: Could not load additional weights from {weights_file}: {e}")
                print("Using randomly initialized value_head weights.")
        
        return model
    
    def forward(self, input_ids, attention_mask=None, labels=None, value_labels=None, **kwargs):
        # Classify each example: value-branch if the LAST PROMPT token is [num], else lm-branch.
        # We detect the last prompt token per sample using labels when available; otherwise use last valid token.
        batch_size, seq_len = input_ids.size(0), input_ids.size(1)
        device = input_ids.device
        num_token_id = self.tokenizer.convert_tokens_to_ids("[num]")
        soft_token_id = self.tokenizer.convert_tokens_to_ids("[soft_token]")
        
        if attention_mask is not None:
            last_pos = attention_mask.long().sum(dim=1) - 1  # (B,)
        else:
            last_pos = torch.full((batch_size,), seq_len - 1, device=device, dtype=torch.long)
        
        # Determine the last prompt token index:
        # - If labels provided: first completion index is first idx where labels != -100; prompt_end = that - 1
        # - Else (inference): prompt is the entire sequence → prompt_end = last_pos
        if labels is not None:
            is_completion = (labels != -100)
            has_completion = is_completion.any(dim=1)
            # First completion position (undefined if no completion; guard below)
            first_comp_pos = is_completion.float().argmax(dim=1)
            prompt_end = torch.where(
                has_completion,
                first_comp_pos - 1,
                last_pos
            )
        else:
            prompt_end = last_pos
        
        # Value branch iff the last prompt token is exactly [num]
        gather_ids = input_ids[torch.arange(batch_size, device=device), prompt_end.clamp(min=0)]
        is_value = (prompt_end >= 0) & (gather_ids == num_token_id)
        is_lm = ~is_value
        
        # Split into two sub-batches
        idx_value = is_value.nonzero(as_tuple=True)[0]
        idx_lm = is_lm.nonzero(as_tuple=True)[0]
        
        lm_logits = None
        lm_loss = None
        value_loss = None
        value_preds = None
        
        # Process LM branch
        if idx_lm.numel() > 0:
            out_lm = self._forward_lm(
                input_ids=input_ids.index_select(0, idx_lm),
                attention_mask=None if attention_mask is None else attention_mask.index_select(0, idx_lm),
                labels=None if labels is None else labels.index_select(0, idx_lm),
                **kwargs,
            )
            lm_logits = out_lm["logits"]
            if "loss" in out_lm and out_lm["loss"] is not None:
                lm_loss = out_lm["loss"]
        
        # Process VALUE branch
        if idx_value.numel() > 0:
            out_val = self._forward_value(
                input_ids=input_ids.index_select(0, idx_value),
                attention_mask=None if attention_mask is None else attention_mask.index_select(0, idx_value),
                labels=None if labels is None else labels.index_select(0, idx_value),
                value_labels=None if value_labels is None else value_labels.index_select(0, idx_value),
                soft_token_id=soft_token_id,
                **kwargs,
            )
            # Gather value loss/preds
            if "loss" in out_val and out_val["loss"] is not None:
                value_loss = out_val["loss"]
            value_preds = out_val.get("value_preds", None)
        # No whole-batch second forward. We let the trainer consume per-branch tensors directly.
        
        # Combine losses
        total_loss = None
        if lm_loss is not None:
            total_loss = lm_loss
        if value_loss is not None:
            total_loss = value_loss if total_loss is None else total_loss + value_loss
        return {
            "loss": total_loss,
            "lm_logits": lm_logits,
            "lm_indices": idx_lm,
            "value_indices": idx_value,
            "value_preds": value_preds,
            "lm_loss": lm_loss,
            "value_loss": value_loss,
        }

    def _forward_lm(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Run base model without built-in CE reduction to control masking
        out = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            **{k: v for k, v in kwargs.items() if k != "labels"},
        )
        lm_loss = None
        if labels is not None:
            shift_logits = out.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return {"loss": lm_loss, "logits": out.logits}

    def _forward_value(self, input_ids, attention_mask=None, labels=None, value_labels=None, soft_token_id=None, **kwargs):
        # Use hidden state at the LAST PROMPT token (which must be [num]) to predict the soft value.
        out = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **{k: v for k, v in kwargs.items() if k != "labels"},
        )
        last_hidden = out.hidden_states[-1]
        batch_size, seq_len = input_ids.size(0), input_ids.size(1)
        device = input_ids.device
        num_token_id = self.tokenizer.convert_tokens_to_ids("[num]")
        
        if attention_mask is not None:
            last_pos = attention_mask.long().sum(dim=1) - 1
        else:
            last_pos = torch.full((batch_size,), seq_len - 1, device=device, dtype=torch.long)

        # Determine prompt_end per sample
        if labels is not None:
            is_comp = (labels != -100)
            has_comp = is_comp.any(dim=1)
            first_comp = is_comp.float().argmax(dim=1)
            prompt_end = torch.where(has_comp, first_comp - 1, last_pos)
        else:
            prompt_end = last_pos

        # Ensure the last prompt token is [num]
        is_num_at_prompt_end = input_ids[torch.arange(batch_size, device=device), prompt_end.clamp(min=0)] == num_token_id
        if not is_num_at_prompt_end.any():
            return {"loss": None, "logits": out.logits, "value_preds": None}
        rows = is_num_at_prompt_end.nonzero(as_tuple=True)[0]
        pos = prompt_end.index_select(0, rows)
        hidden_at_num = last_hidden[rows, pos, :]
        # value_head outputs [batch_size, 2] for alpha and beta
        value_preds = self.value_head(hidden_at_num)
        
        value_loss = None
        if value_labels is not None:
            # alpha_preds = value_preds[:, 0]
            # beta_preds = value_preds[:, 1]
            # target = value_labels.index_select(0, rows).float()
            # # Use Beta distribution negative log-likelihood loss
            # value_loss = beta_nll_loss(alpha_preds, beta_preds, target)
            raw_m, raw_s = value_preds[:, 0], value_preds[:, 1]
            m = torch.sigmoid(raw_m)
            s = F.softplus(raw_s) + 1e-2
            alpha = m * s + 1
            beta = (1 - m) * s + 1
            target = value_labels.index_select(0, rows).float()
            # Alpha and beta are already positive from mean-concentration parameterization
            value_loss = beta_nll_loss(alpha, beta, target, apply_transform=False)
        
        # Optionally, force next token at [num] to be [soft_token] by manipulating logits
        if soft_token_id is not None:
            out.logits[rows, pos, :] = -1e10
            out.logits[rows, pos, soft_token_id] = 10.0
        
        return {"loss": value_loss, "logits": out.logits, "value_preds": value_preds}


@dataclass
class MixedLossCausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    lm_loss: Optional[torch.FloatTensor] = None
    value_loss: Optional[torch.FloatTensor] = None
    value_preds: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class LLMWithMixedTokenLoss(PreTrainedModel):
    """
    Language model wrapper that supports mixed token-level objectives.

    - Tokens whose labels are not `-100` contribute to a standard next-token cross-entropy loss.
    - Tokens flagged via `value_mask` contribute to an auxiliary regression loss computed on their hidden states.

    This allows a single sequence to interleave discrete and continuous supervision signals without splitting the batch.
    """

    def __init__(self, base_model: PreTrainedModel, tokenizer):
        super().__init__(base_model.config)
        self.transformer = base_model
        self.tokenizer = tokenizer
        hidden_size = self.config.hidden_size
        # Value head now outputs 2 parameters for Beta distribution (alpha, beta)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),  # Output alpha and beta parameters
        )

        for module in self.value_head:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_input_embeddings(self):
        return self.transformer.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.transformer.set_input_embeddings(value)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        value_labels=None,
        value_mask=None,
        return_dict=None,
        **kwargs,
    ):
        if return_dict is None:
            return_dict = self.config.use_return_dict

        output_hidden_states = kwargs.pop("output_hidden_states", None)
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states
        if value_mask is not None:
            output_hidden_states = True

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        logits = transformer_outputs.logits
        hidden_states_all = transformer_outputs.hidden_states
        past_key_values = transformer_outputs.past_key_values
        attentions = transformer_outputs.attentions

        lm_loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        value_loss = None
        value_preds = None
        if value_mask is not None:
            if value_labels is None:
                raise ValueError("`value_mask` was provided but `value_labels` is None.")
            if value_mask.shape != input_ids.shape:
                raise ValueError("`value_mask` must have the same shape as `input_ids`.")
            if value_labels.shape != input_ids.shape:
                raise ValueError("`value_labels` must have the same shape as `input_ids`.")

            mask = value_mask.to(dtype=torch.bool, device=input_ids.device)
            if mask.any():
                last_hidden = hidden_states_all[-1]
                batch_idx, token_idx = mask.nonzero(as_tuple=True)
                # Use the hidden state from the previous position ([num] instead of [soft_token])
                # Ensure we don't go below index 0
                prev_token_idx = (token_idx - 1).clamp(min=0)
                
                # Assert that the previous token is [num]
                num_token_id = self.tokenizer.convert_tokens_to_ids("[num]")
                prev_tokens = input_ids[batch_idx, prev_token_idx]
                assert (prev_tokens == num_token_id).all(), \
                    f"Expected all previous tokens of [soft_token] to be [num] (id={num_token_id}), but got {prev_tokens.tolist()}"
                
                selected_hidden = last_hidden[batch_idx, prev_token_idx, :]
                # value_head now outputs [batch_size, 2] for alpha and beta
                value_preds = self.value_head(selected_hidden)
                mu, rho = value_preds[:, 0], value_preds[:, 1]
                sigma = F.softplus(rho) + 1e-2

                target = value_labels.to(value_preds.device)[batch_idx, token_idx].float()
                # Use Beta distribution negative log-likelihood loss
                # value_loss = logit_normal_nll(mu, sigma, target)
                # value_loss = beta_nll_loss(alpha, beta, target, apply_transform=False)

                with torch.no_grad():
                    pos = (target > 0.5).float()
                    n_pos = pos.sum().clamp(min=1.0)
                    n_neg = (1 - pos).sum().clamp(min=1.0)
                    pos_weight = n_neg / n_pos              # 稀有正样本 → 大权重
                value_loss = torch.nn.functional.binary_cross_entropy_with_logits(mu, target, pos_weight=pos_weight)
            else:
                value_preds = torch.empty(0, device=input_ids.device)

        total_loss = None
        if lm_loss is not None:
            total_loss = lm_loss
        if value_loss is not None:
            total_loss = value_loss if total_loss is None else total_loss + value_loss
        if total_loss is None:
            total_loss = logits.sum() * 0.0

        total_loss = value_loss

        if not return_dict:
            output = (logits, past_key_values, hidden_states_all, attentions)
            return (total_loss,) + output

        return MixedLossCausalLMOutput(
            loss=total_loss,
            logits=logits,
            lm_loss=lm_loss,
            value_loss=value_loss,
            value_preds=value_preds,
            past_key_values=past_key_values,
            hidden_states=hidden_states_all,
            attentions=attentions,
        )


def get_mixed_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens = {"additional_special_tokens": ["[num]", "[soft_token]"]}
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.padding_side = "right"

    if "qwen2" in model_name.lower():
        from transformers import Qwen2ForCausalLM

        base_model = Qwen2ForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation="eager",
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation="eager",
        )

    base_model.resize_token_embeddings(len(tokenizer))
    try:
        base_model.config.use_cache = False
    except Exception:
        pass

    model = LLMWithMixedTokenLoss(base_model, tokenizer)
    return model, tokenizer


def get_model_and_tokenizer(model_name: str):
    from transformers import AutoConfig
    # Load tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens = {"additional_special_tokens": ["[num]", "[soft_token]"]}
    num_added = tokenizer.add_special_tokens(special_tokens)
    
    # Set padding side to right to match the SoftDataCollator's assumptions
    tokenizer.padding_side = "right"
    
    # Get the config to determine model class
    config = AutoConfig.from_pretrained(model_name)
    
    # Load the actual base model directly using its class
    if 'qwen2' in model_name.lower():
        from transformers import Qwen2ForCausalLM
        base_model = Qwen2ForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation="eager"
        )
    else:
        # Fallback for other models
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation="eager"
        )
    
    # Resize token embeddings on the base model before wrapping
    base_model.resize_token_embeddings(len(tokenizer))
    # Disable KV cache to reduce memory during training
    try:
        base_model.config.use_cache = False
    except Exception:
        pass
    
    # Then wrap with value head
    model = LLMWithValueHead(base_model, tokenizer)
    return model, tokenizer
