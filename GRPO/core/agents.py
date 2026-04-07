import json
import math
import os
from typing import List, Optional, Union
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from GRPO.core.utils import ndcg_at_k
from collections import defaultdict
from pydantic.config import ConfigDict
import outlines
from tqdm import tqdm
import datetime
import yaml


import torch
import torch.nn as nn
from typing import Dict
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, AutoModel
from GRPO.core.utils import build_prompt
from pydantic import BaseModel, Field, create_model
from typing import Dict, List, Type
from GRPO.core.data import InteractionData
from recbole.data.dataset import SequentialDataset
from GRPO.core.constant import dataset_feat_alias, dataset_rating_scale


class UserProfileAgent:
    def __init__(
        self, 
        inter_dataset: InteractionData,
        dataset_name: str,
        reviews: Dict[str, List[dict]] = None,
        recaller_metrics_path: str = None,  # Path to recaller metrics JSON file
    ): 
        self.num_items = inter_dataset.ds.item_num
        self.inter_feat, self.item_feat, self.user_feat = {}, {}, {}
        self.iid_field, self.uid_field = inter_dataset.ds.iid_field, inter_dataset.ds.uid_field
        self.dataset_name = dataset_name
        
        # Load recaller metrics if provided
        self.recaller_metrics = None
        if recaller_metrics_path and os.path.exists(recaller_metrics_path):
            with open(recaller_metrics_path, 'r') as f:
                self.recaller_metrics = json.load(f)
            print(f"Loaded recaller metrics for {len(self.recaller_metrics)} users")
        if inter_dataset.ds.user_feat is not None:
            for row in tqdm(inter_dataset.ds.user_feat.iterrows(), desc='Preprocessing user_feat'):
                row_dict = row[1].to_dict()
                uid = int(row_dict.pop(self.uid_field))
                self.user_feat[uid] = self._preprocess_feat(row_dict, inter_dataset.ds)
        import pandas as pd
        item_feat_pd = pd.read_csv(f'dataset/{self.dataset_name}/{self.dataset_name}.item', sep='\t')
        item_feat_pd.columns = item_feat_pd.columns.str.split(':').str[0]
        for row in tqdm(item_feat_pd.iterrows(), desc='Preprocessing item_feat'):
            row_dict = row[1].to_dict()
            # Clean string values that might have Python unicode prefix or escape sequences
            for k, v in row_dict.items():
                if isinstance(v, str):
                    # Remove Python 2 unicode prefix (u"..." or u'...')
                    if (v.startswith("u'") and v.endswith("'")) or (v.startswith('u"') and v.endswith('"')):
                        v = v[2:-1]
                    # Decode unicode escape sequences like \u2122 -> ™
                    try:
                        v = v.encode('utf-8').decode('unicode_escape').encode('latin1').decode('utf-8')
                    except (UnicodeDecodeError, UnicodeEncodeError):
                        try:
                            v = v.encode('utf-8').decode('unicode_escape')
                        except:
                            pass  # Keep original if decoding fails
                    row_dict[k] = v
            itoken = row_dict.pop(self.iid_field)
            if str(itoken) in inter_dataset.ds.field2token_id[self.iid_field].keys():
                self.item_feat[inter_dataset.ds.field2token_id[self.iid_field][str(itoken)]] = self._preprocess_feat(row_dict, inter_dataset.ds)
        for row in tqdm(inter_dataset.ds.inter_feat.iterrows(), desc='Preprocessing inter_feat'):
            row_dict = row[1].to_dict()
            uid = int(row_dict.pop(self.uid_field))
            iid = int(row_dict.pop(self.iid_field))
            self.inter_feat[f'{uid}_{iid}'] = self._preprocess_feat(row_dict, inter_dataset.ds)
        # TODO:self.reviews = self._preprocess_reviews(reviews)

    def _preprocess_feat(self, feat: dict, ds: SequentialDataset):
        field2id_token = ds.field2id_token
        res = {}

        def change_type(var):
            if type(var) == float:
                return int(var)
            return var
        
        def clean_value(val):
            """Clean problematic values for natural language prompts"""
            if val is None:
                return None
            # Handle [PAD] tokens
            if isinstance(val, str) and val == '[PAD]':
                return None  # Will be filtered out
            # Handle NaN values
            if isinstance(val, float) and (np.isnan(val) or val != val):
                return None
            # Handle string 'nan' or '.nan'
            if isinstance(val, str) and val.lower() in ['nan', '.nan']:
                return None
            # Clean Python unicode prefix (u"..." or u'...')
            if isinstance(val, str) and (val.startswith("u'") or val.startswith('u"')):
                val = val[2:-1] if val.endswith(val[1]) else val[1:]
            return val
        
        for key in feat.keys():
            if key not in field2id_token.keys():
                res[key] = feat[key]
            elif type(feat[key]) == np.ndarray:
                res[key] = ' '.join([field2id_token[key][change_type(item)] for item in feat[key]])
            else:
                res[key] = field2id_token[key][change_type(feat[key])]
            if key in dataset_feat_alias[self.dataset_name].keys():
                res[key] = dataset_feat_alias[self.dataset_name][key][change_type(res[key])]
            # Clean the value
            res[key] = clean_value(res[key])
        
        # Remove keys with None values (filtered out [PAD], NaN, etc.)
        res = {k: v for k, v in res.items() if v is not None}
        return res


    def _preprocess_reviews(self, reviews: List[dict]):
        processed_reviews = defaultdict(list)
        for user, user_reviews in reviews.items():
            for review in user_reviews:
                processed_reviews[user].append({
                    "rating": review["rating"],
                    "item": self.id2meta[str(review["item_id"])],
                    "review title": review["title"],
                    "review text": review["text"],
                    "purchased": review["verified_purchase"],
                })
        return processed_reviews

    def _get_item_metadata(self, item_id: int):
        return self.id2meta[str(item_id)]

    @staticmethod
    def _format_relative_time(seconds_diff):
        """将秒数差值转换为相对时间字符串（只显示最大单位）"""
        if seconds_diff < 0:
            return "in the future"
        
        minutes = seconds_diff / 60
        hours = minutes / 60
        days = hours / 24
        weeks = days / 7
        months = days / 30
        years = days / 365
        
        if years >= 1:
            return f"{int(years)} year{'s' if years >= 2 else ''} ago"
        elif months >= 1:
            return f"{int(months)} month{'s' if months >= 2 else ''} ago"
        elif weeks >= 1:
            return f"{int(weeks)} week{'s' if weeks >= 2 else ''} ago"
        elif days >= 1:
            return f"{int(days)} day{'s' if days >= 2 else ''} ago"
        elif hours >= 1:
            return f"{int(hours)} hour{'s' if hours >= 2 else ''} ago"
        elif minutes >= 1:
            return f"{int(minutes)} minute{'s' if minutes >= 2 else ''} ago"
        else:
            return "just now"

    def forward(self, user_id: int, history: List[int], cut_off: int = 5, use_meta_data: bool = False, 
                include_recaller_metrics: bool = False, available_recallers: List[str] = None,
                reference_timestamp: float = None):
        if 0 in history:
            history = history[:history.index(0)]
        if len(history) > cut_off:
            history = history[-cut_off:]
        user_profile_dict = {}
        if user_id in self.user_feat.keys():
            user_profile_dict['Demographic'] = self.user_feat[user_id]
        user_profile_dict['purchased item numbers'] = len(history)
        user_profile_dict['purchase history'] = [
            {**self.item_feat.get(item_id, {}), **self.inter_feat.get(f'{user_id}_{item_id}', {})} 
            for item_id in history
        ]
        
        # Add recaller metrics if enabled and available
        if include_recaller_metrics and self.recaller_metrics:
            user_metrics = self.recaller_metrics.get(str(user_id), {})
            if user_metrics:
                seq = user_metrics.get("sequence", [])
                pos_metrics = user_metrics.get("position_metrics", {})
                # Normalize available_recallers to lowercase for matching
                valid_recallers = set(r.lower() for r in available_recallers) if available_recallers else None
                
                for i, item_id in enumerate(history):
                    if item_id in seq:
                        pos = seq.index(item_id)
                        if str(pos) in pos_metrics:
                            pm = pos_metrics[str(pos)]
                            # Find best recaller from available subset
                            if valid_recallers:
                                best, best_score = None, -1
                                for r in valid_recallers:
                                    if r in pm and isinstance(pm[r], dict):
                                        score = pm[r].get('ndcg@10', 0)
                                        if score > best_score:
                                            best_score, best = score, r
                                if best:
                                    user_profile_dict['purchase history'][i]['best_recaller'] = best
                            else:
                                user_profile_dict['purchase history'][i]['best_recaller'] = pm.get('best_recaller', 'unknown')
               
        def _format_timestamp_field(hist_item, field_name, reference_timestamp):
            if field_name not in hist_item:
                return
            ts = hist_item[field_name]
            try:
                ts = float(ts)
                if ts > 1e12:
                    ts = ts / 1000
                if ts > 86400:
                    if reference_timestamp:
                        diff = reference_timestamp - ts
                        hist_item[field_name] = self._format_relative_time(diff)
                    else:
                        hist_item[field_name] = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    hist_item[field_name] = 'unknown'
            except (ValueError, OSError, OverflowError, TypeError):
                hist_item[field_name] = 'unknown'
        
        max_rating = dataset_rating_scale.get(self.dataset_name, 5)
        for hist_item in user_profile_dict['purchase history']:
            if 'rating' in hist_item:
                hist_item['rating'] = f"{hist_item['rating']}/{max_rating}"
            _format_timestamp_field(hist_item, 'timestamp', reference_timestamp)
            _format_timestamp_field(hist_item, 'submitted_timestamp', reference_timestamp)
        
        # Convert numpy types to Python native types for yaml serialization
        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.str_, np.bytes_)):
                return str(obj)
            return obj
        
        user_profile_dict = convert_numpy(user_profile_dict)
        
        # Convert purchase history list to numbered dict for better readability
        user_profile_dict['purchase history'] = {
            str(i+1): item for i, item in enumerate(user_profile_dict['purchase history'])
        }
        
        # Use yaml.dump with safe_dump behavior and proper unicode handling
        return yaml.dump(user_profile_dict, default_flow_style=False, allow_unicode=True, default_style=None, sort_keys=False)


class LLMRouterAgent:
    def __init__(
        self, 
        n_per_user: int = 4, 
        available_models: List[str] = None
    ):
        self.n_per_user = n_per_user
        self.available_models = available_models or ['sasrec', 'bpr', 'pop']
            
    
    @staticmethod
    def _extract_json_array(text: str) -> Optional[List[dict]]:
        """Extract JSON array from generated text"""
        try:
            s = text.find("["); e = text.rfind("]")
            if s != -1 and e != -1 and e > s:
                js = text[s:e+1]
                data = json.loads(js)
                if isinstance(data, list):
                    return data
        except Exception:
            return None
        return None
    
    def get_logprob(self, text: str, label: str, model, tokenizer) -> dict:
        """Compute logprob using provided model (policy model)"""
        full_text = text + label
        
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        labels = inputs["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        with torch.no_grad():
            outputs = model(**inputs, labels=labels, return_dict=True)
            loss = outputs.loss
            avg_logprob = -loss.item()
            total_logprob = avg_logprob * (labels != -100).sum().item()

        return {"avg_logprob": avg_logprob, "total_logprob": total_logprob}
    
    def generate(self, profile_json: str, model, tokenizer, 
                 available_models: List[str] = None,
                 temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9, 
                 num_return_sequences: int = 1) -> dict:
        """Generate routes using provided HuggingFace model"""
        prompt = build_prompt(profile_json, available_models or self.available_models)
        
        all_routes = []
        all_logprobs = []
        
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                min_new_tokens=10,
                do_sample=True, 
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        data = [self._extract_json_array(t[len(prompt):]) for t in text]
        all_routes.extend(data)
        
        # Calculate total logprobs directly
        for i in range(num_return_sequences):
            total_logprob = 0.0
            for j, score in enumerate(outputs.scores):
                log_probs = torch.log_softmax(score[i], dim=-1)
                token_id = outputs.sequences[i][len(inputs.input_ids[0]) + j]
                total_logprob += log_probs[token_id].item()
            all_logprobs.append(total_logprob)
        
        return {
            "routes": all_routes,
            "logprobs": all_logprobs
        }

    def _select_diverse_routes(self, candidate_routes: List[dict], n: int) -> List[dict]:
        """
        choose n different routes from candidate routes
        """
        if not candidate_routes:
            return []
        
        unique_routes = []
        seen_keys = set()
        
        for route in candidate_routes:
            if "models" in route and isinstance(route["models"], list):
                models_key = tuple(
                    (m.get("name", ""), m.get("k", 0), round(float(m.get("weight", 0.0)), 3))
                    for m in route["models"]
                )
                route_key = ("new_format", models_key)
            else:
                route_key = (
                    "old_format",
                    route.get("model_1", ""),
                    route.get("model_2", ""),
                    route.get("k_1", 0),
                    route.get("k_2", 0),
                    round(float(route.get("w_1", 0.0)), 3)
                )
            
            if route_key not in seen_keys:
                seen_keys.add(route_key)
                unique_routes.append(route)
                
                if len(unique_routes) >= n:
                    break
        
        return unique_routes

    def _create_fallback_routes(self, n: int) -> List[dict]:
        """Create fallback routes when LLM generation fails"""
        fallback = []
        num_models = len(self.available_models) or 3
        
        for j in range(n):
            models_per_route = min(3, max(2, num_models))
            selected_models = []
            
            start_idx = j % num_models
            for i in range(models_per_route):
                model_idx = (start_idx + i) % num_models
                model_name = self.available_models[model_idx] if self.available_models else f"model_{model_idx}"
                k_value = min(500, max(1, 20 + (i * 15) + (j * 5)))
                
                # Simple weight assignment
                weight = 0.5 if i == 0 else 0.5 / (models_per_route - 1)
                
                selected_models.append({
                    "name": model_name,
                    "k": k_value,
                    "weight": weight
                })
            
            fallback.append({"models": selected_models})
        return fallback

    def _clean_and_validate_routes(self, routes: List[dict], n: int) -> List[dict]:
        """Clean and validate route formats"""
        return routes[:n] # TODO: validate routes

    def forward(self, profile_json: str, model=None, tokenizer=None, n_candidates: int = None, 
                temperature: float = 0.8) -> List[dict]:
        n = n_candidates or self.n_per_user
        
        # Try LLM generation if model is provided
        routes = []
        if model and tokenizer:
            result = self.generate(
                profile_json, 
                model,
                tokenizer,
                self.available_models,
                temperature=temperature,
                num_return_sequences=n_candidates
            )
            routes = result.get("routes", []) if isinstance(result, dict) else result or []
        
        # Use fallback if needed
        if len(routes) < n:
            routes = self._create_fallback_routes(n)
        
        # Clean and validate
        return self._clean_and_validate_routes(routes, n)


def generate_diverse_routes(
    router: LLMRouterAgent, 
    prof_json: str, 
    n_candidates: int, 
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizerBase, 
    temperature=0.8, 
    include_logprobs=False
):
    """Generate diverse routes using a single model"""
    if not include_logprobs:
        # Simple generation without logprobs
        return router.forward(
            prof_json, 
            model=model,
            tokenizer=tokenizer,
            n_candidates=n_candidates, 
            temperature=temperature, 
            ensure_diversity=True
        )
    
    # Generate with logprobs
    result = router.generate(
        prof_json,
        n_candidates,
        model,
        tokenizer,
        router.available_models,
        temperature=temperature,
        num_return_sequences=n_candidates
    )
    
    routes = result["routes"]
    logprobs = result["logprobs"]
    
    # Apply diversity selection if needed
    if len(routes) > n_candidates:
        diverse_routes = router._select_diverse_routes(routes, n_candidates)
        route_indices = [routes.index(r) for r in diverse_routes]
        logprobs = [logprobs[i] for i in route_indices]
        routes = diverse_routes
    
    return {
        "routes": routes,
        "logprobs": logprobs
    }


def calculate_route_logprobs(
    router: LLMRouterAgent,
    prof_json: str,
    routes: List[dict],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase
) -> List[float]:
    """Calculate log probabilities for given routes"""
    logprobs = []
    for route in routes:
        result = router.get_logprob(prof_json, json.dumps(route), model, tokenizer)
        logprobs.append(result['total_logprob'])
    return logprobs

class PerModelConfig(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        # populate_by_name=True
    )
    k: int = Field(default=50, alias='top-k', description="Number of top items to retrieve")
    w: float = Field(default=1.0, alias='score-weight', description="Weight for scoring")

def create_model_config(
    available_models: List[str], 
    default_configs: Optional[Dict[str, BaseModel]] = None,
    class_name: str = "ModelConfigs"
) -> Type[BaseModel]:
    """
    Create an advanced dynamic Pydantic model with support for custom default configurations
    
    Args:
        available_models: List of available model names
        default_configs: Optional dictionary of default configurations
        class_name: Name of the generated class
    
    Returns:
        Dynamically created Pydantic model class
    """
    default_configs = default_configs or {}
    
    fields = {}
    for model_name in available_models:
        default_config = default_configs.get(model_name, PerModelConfig()) if default_configs else PerModelConfig()
        
        fields[model_name] = (
            PerModelConfig, 
            Field(default=default_config, description=f"Configuration for {model_name} model")
        )
    
    ModelConfigs = create_model(class_name, **fields, __base__=BaseModel)
    
    return ModelConfigs

# Example usage for GRPO:
# 1. Generate routes with ref_model
# ref_result = generate_diverse_routes(router, prof_json, n_candidates, ref_model, tokenizer, include_logprobs=True)
# ref_routes = ref_result["routes"]
# ref_logprobs = ref_result["logprobs"]
#
# 2. Calculate policy model logprobs for the same routes
# policy_logprobs = calculate_route_logprobs(router, prof_json, ref_routes, policy_model, tokenizer)


