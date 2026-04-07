from typing import Dict, List, Optional, Union
import os
import torch
import numpy as np
import warnings
# Suppress pandas FutureWarning from recbole
warnings.filterwarnings('ignore', category=FutureWarning, message='.*A value is trying to be set on a copy of a DataFrame.*')

# Fix compatibility: scipy >= 1.14 removed dok_matrix._update that recbole depends on
from scipy.sparse import dok_matrix as _dok_matrix
if not hasattr(_dok_matrix, '_update'):
    _dok_matrix._update = _dok_matrix.update

from recbole.quick_start.quick_start import load_data_and_model

from .data import InteractionData, get_base_config_dict

def find_latest_checkpoint(model_name: str, checkpoint_dir: str) -> str:
    import glob
    pattern = os.path.join(checkpoint_dir, f"{model_name}*.pth")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]


class BaseRecaller:
    def __init__(self, name: str, num_items: int):
        self.name = name
        self.num_items = num_items
    def recall(self, user_id: int, topk: int, history: List[int]) -> List[int]:
        raise NotImplementedError


class RecBoleRecaller(BaseRecaller):
    def __init__(self, model_name: str, dataset_name: str, checkpoint_path: str,
                 data_path: str = "./data", config_dict: dict = None, num_items: int = 0):
        super().__init__(model_name.lower(), num_items=num_items)
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path
        self.config_dict = config_dict or {}
        self._init_recbole_model()

    def _get_model_config(self):
        """Get specific configuration for each model type"""
        # Base configuration
        base_config = get_base_config_dict(self.dataset_name, self.data_path)
        # Model-specific configurations based on RecBole documentation
        import os
        import torch
        
        # # 检查是否在分布式环境中
        # local_rank = int(os.environ.get('LOCAL_RANK', 0))
        # world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # # 为每个进程设置不同的设备
        # if world_size > 1:
        #     # 在分布式环境中，使用 LOCAL_RANK 作为设备 ID
        #     base_config['device'] = f'cuda:{local_rank}'
        # elif torch.cuda.is_available():
        #     base_config['device'] = 'cuda'
        # else:
        #     base_config['device'] = 'cpu'
        if self.model_name == 'BPR':
            model_config = {
                **base_config,
                'epochs': 100,
                'train_neg_sample_args': {
                    'distribution': 'uniform',
                    'sample_num': 1,
                    'alpha': 1.0,
                    'dynamic': False,
                    'candidate_num': 0
                },
                'loss_type': 'BPR',
                'embedding_size': 64,
                'learning_rate': 0.001,
                'train_batch_size': 4096,
                'eval_batch_size': 4096 * self.num_items,
            }
            
        elif self.model_name == 'SASRec':
            model_config = {
                **base_config,
                'epochs': 100,
                'train_neg_sample_args': None,
                'loss_type': 'CE',
                'learning_rate': 0.001,
                'train_batch_size': 4096,
                'eval_batch_size': 4096,
                'max_seq_length': 50,
                'hidden_size': 64,
                'n_layers': 2,
                'n_heads': 2,
                'inner_size': 256,
                'hidden_dropout_prob': 0.5,
                'attn_dropout_prob': 0.5,
                'hidden_act': 'gelu',
                'layer_norm_eps': 1e-12,
                'initializer_range': 0.02,
            }
            
        elif self.model_name == 'Pop':
            model_config = {
                **base_config,
                'train_neg_sample_args': None,
                'epochs': 1,  # Pop model doesn't need many epochs
            }
            
        elif self.model_name == 'ItemKNN':
            model_config = {
                **base_config,
                'train_neg_sample_args': None,
                'k': 20,  # Number of similar items
                # 'shrink': 0.0,  # Shrinkage parameter
                'eval_batch_size': 4096 * self.num_items,
            }
            
        elif self.model_name == 'FPMC':
            model_config = {
                **base_config,
                'epochs': 100,
                'train_neg_sample_args': {
                    'distribution': 'uniform',
                    'sample_num': 1
                },
                'loss_type': 'BPR',
                'embedding_size': 64,
                'learning_rate': 0.001,
            }
            
        elif self.model_name == 'GRU4Rec':
            model_config = {
                **base_config,
                'epochs': 100,
                'train_neg_sample_args': None,
                'loss_type': 'CE',
                'embedding_size': 64,
                'hidden_size': 128,
                'num_layers': 1,
                'dropout_prob': 0.3,
                'learning_rate': 0.001,
            }
        elif self.model_name == 'LightGCN':
            model_config = {
                **base_config,
                'epochs': 100,
                'train_neg_sample_args': {
                    'distribution': 'uniform',
                    'sample_num': 1,
                },
                'loss_type': 'BPR',
                'embedding_size': 64,
                'n_layers': 3,
                'reg_weight': 1e-5,
                'learning_rate': 0.001,
                'train_batch_size': 2048,
                'eval_batch_size': 2048 * 20000,
            }
        elif self.model_name == 'SimpleX':
            model_config = {
                **base_config,
                'epochs': 100,
                'train_neg_sample_args': {
                    'distribution': 'uniform',
                    'sample_num': 1,
                },
                'loss_type': 'BPR',
                'embedding_size': 64,
                'aggregator': 'mean',
                'gamma': 0.5,
                'margin': 0.9,
                'negative_weight': 0.5,
                'reg_weight': 1e-5,
                'learning_rate': 0.001,
                'train_batch_size': 2048,
                'eval_batch_size': 2048 * 20000,
            }
        elif self.model_name == 'RaCT':
            model_config = {
                **base_config,
                'epochs': 100,
                'train_neg_sample_args': None,
                'latent_dimension': 64,
                'mlp_hidden_size': [600],
                'dropout_prob': 0.5,
                'anneal_cap': 0.2,
                'total_anneal_steps': 200000,
                'critic_layers': [100, 1],
                'beta': 0.2,
                'learning_rate': 0.001,
                'train_batch_size': 512,
                'eval_batch_size': 512,
            }
        elif self.model_name == 'EASE':
            model_config = {
                **base_config,
                'train_neg_sample_args': None,
                'epochs': 1,
                'reg_weight': 250.0,
            }
        elif self.model_name == 'RecVAE':
            model_config = {
                **base_config,
                'epochs': 100,
                'train_neg_sample_args': None,
                'hidden_dimension': 600,
                'latent_dimension': 64,
                'dropout_prob': 0.5,
                'beta': None,
                'gamma': 0.005,
                'mixture_weights': [0.15, 0.75, 0.1],
                'n_enc_epochs': 3,
                'n_dec_epochs': 1,
                'learning_rate': 0.001,
                'train_batch_size': 512,
                'eval_batch_size': 512,
            }
        elif self.model_name == 'SLIMElastic':
            model_config = {
                **base_config,
                'train_neg_sample_args': None,
                'epochs': 1,
                'alpha': 0.2,
                'l1_ratio': 0.02,
                'positive_only': True,
                'hide_item': False,
            }
        elif self.model_name == 'NeuMF':
            model_config = {
                **base_config,
                'epochs': 100,
                'train_neg_sample_args': {
                    'distribution': 'uniform',
                    'sample_num': 1,
                },
                'loss_type': 'BCE',
                'mf_embedding_size': 64,
                'mlp_embedding_size': 64,
                'mlp_hidden_size': [128, 64],
                'dropout_prob': 0.1,
                'mf_train': True,
                'mlp_train': True,
                'learning_rate': 0.001,
                'train_batch_size': 2048,
                'eval_batch_size': 2048,
            }
        else:
            # Default configuration for other models
            model_config = {
                **base_config,
                'epochs': 100,
                'train_neg_sample_args': {
                    'distribution': 'uniform',
                    'sample_num': 1
                },
                'loss_type': 'BPR',
                'embedding_size': 64,
                'learning_rate': 0.001,
            }
        
        # Update with user-provided config
        model_config.update(self.config_dict)
        model_config['dataset_save_path'] = f'dataset/{self.dataset_name}/5core_{self.model_name}.pth'
        return model_config

    def _init_recbole_model(self):
        # Import RecBole modules (no try-catch, assume RecBole is installed)
        from recbole.utils import get_trainer, get_model as get_recbole_model
        from recbole.config import Config
        from recbole.data import create_dataset, data_preparation
        from recbole.utils import init_seed as recbole_init_seed
        # Get model class
        self.model_class = get_recbole_model(self.model_name)
        
        # Get model-specific configuration
        model_config = self._get_model_config()
        model_config['checkpoint_dir'] = self.checkpoint_path
        # Create RecBole config
        self.config = Config(
            model=self.model_name, 
            dataset=self.dataset_name, 
            config_dict=model_config
        )
        
        # Initialize seed
        recbole_init_seed(self.config['seed'], self.config['reproducibility'])
        
        # Create dataset and prepare data
        self.dataset = create_dataset(self.config)
        self.train_data, self.valid_data, self.test_data = data_preparation(self.config, self.dataset)
        
        # Initialize model
        self.model = self.model_class(self.config, self.dataset).to(self.config['device'])

        # Load checkpoint if provided
        # TODO: 看看能不能用xxx加速
        if self.checkpoint_path and find_latest_checkpoint(self.model_name, self.checkpoint_path) and self.model_name not in ['Pop']:
            # Load checkpoint
            model_path = find_latest_checkpoint(self.model_name, self.checkpoint_path)
            print(f"Loading checkpoint from {model_path} for model {self.model_name}")
            checkpoint = torch.load(model_path, map_location=self.config['device'], weights_only=False)
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.model.eval()
        else: # not in training free models
            print(f"Training model {self.model_name} from scratch and saving checkpoint to {self.checkpoint_path}")
            # Train model
            self.trainer = get_trainer(self.config['MODEL_TYPE'], self.config['model'])(self.config, self.model)
            best_valid_score, best_valid_result = self.trainer.fit(
                self.train_data, self.valid_data, saved=True, show_progress=True
            )
            
            print(f"{self.model_name} training completed!")
            print(f"Best validation result: {best_valid_result}")
        
        # Build user-item mappings
        self.iid_list_field = f'{self.dataset.iid_field}_list'
        self._build_user_item_mappings()

    def _build_user_item_mappings(self):
        """Build user-item mappings following the same logic as data.py"""
        self.user2items_train = {}
        self.user2items_test = {}
        self.user2item_list_train = {}
        self.user2item_list_test = {}
        
        uid_field = self.dataset.uid_field
        iid_field = self.dataset.iid_field
        
        # Build training mappings
        if hasattr(self.train_data.dataset.inter_feat, 'interaction'):
            # For sequential models with item_id_list
            interaction = self.train_data.dataset.inter_feat.interaction
            train_user_ids = interaction[uid_field]
            train_item_id_lists = interaction[self.iid_list_field] if self.iid_list_field in interaction.keys() else [[]]*len(train_user_ids)
            train_item_ids = interaction[iid_field]
                
            for i in range(len(self.train_data.dataset.inter_feat)):
                user_id = int(train_user_ids[i])
                item_id = int(train_item_ids[i])
                item_id_list = train_item_id_lists[i]

                # Build historical sequence (item_id_list)
                self.user2item_list_train[user_id] = [int(it) for it in item_id_list if it > 0]

                # Build target items (item_id)
                self.user2items_train.setdefault(user_id, []).append(item_id)
        else:
            # Fallback for older RecBole versions
            for inter in self.train_data.dataset.inter_feat:
                user_id = int(inter[uid_field])
                item_id = int(inter[iid_field])
                self.user2items_train.setdefault(user_id, []).append(item_id)
                self.user2item_list_train[user_id] = self.user2items_train[user_id][:-1]
        
        # Build test mappings  
        if hasattr(self.test_data.dataset.inter_feat, 'interaction'):
            # For sequential models with item_id_list
            inter = self.test_data.dataset.inter_feat.interaction
            test_user_ids = inter[uid_field]
            test_item_id_lists = inter[self.iid_list_field] if self.iid_list_field in inter.keys() else [[]]*len(test_user_ids)
            test_item_ids = inter[iid_field]
                
            for i in range(len(self.test_data.dataset.inter_feat)):
                user_id = int(test_user_ids[i])
                item_id = int(test_item_ids[i])
                item_id_list = test_item_id_lists[i]
                
                # Build historical sequence (item_id_list)
                self.user2item_list_test[user_id] = [int(it) for it in item_id_list if it > 0]
                
                # Build target items (item_id)
                self.user2items_test.setdefault(user_id, []).append(item_id)
        else:
            # Fallback for older RecBole versions
            for inter in self.test_data.dataset.inter_feat:
                user_id = int(inter[uid_field])
                item_id = int(inter[iid_field])
                self.user2items_test.setdefault(user_id, []).append(item_id)
                self.user2item_list_test[user_id] = self.user2items_train.get(user_id, [])

    def _create_user_interaction(self, user_id: int, history: List[int] = None) -> Optional[object]:
        """Create a proper Interaction object for the given user"""
        from recbole.data.interaction import Interaction
        
        # Create interaction dict with user information
        interaction_dict = {}
        uid_field = self.dataset.uid_field
        interaction_dict[uid_field] = torch.tensor([user_id], device=self.config['device'])
        
        # For sequential models, add item_id_list if needed
        
        if hasattr(self.dataset, 'field2type') and self.iid_list_field in self.dataset.field2type:
            # Use provided history or fall back to internal history
            if history is not None:
                hist_items = history
            else:
                hist_items = self.user2item_list_train.get(user_id, [])
                
            if hist_items:
                # Pad or truncate to max_seq_length if needed
                max_seq_len = 50
                if len(hist_items) > max_seq_len:
                    hist_items = hist_items[-max_seq_len:]
                # Ensure proper tensor shape [batch_size, seq_len]
                hist_tensor = torch.tensor([hist_items], dtype=torch.long, device=self.config['device'])
            else:
                # Empty sequence with proper shape
                hist_tensor = torch.tensor([[0]], dtype=torch.long, device=self.config['device'])
            interaction_dict[self.iid_list_field] = hist_tensor
            
            # Use truncated length for item_length (must match hist_tensor)
            actual_len = len(hist_items)
        else:
            actual_len = 0
        
        # Add other fields that might be required
        # Length field for sequential models
        if 'item_length' in getattr(self.dataset, 'field2type', {}):
            # Use actual_len (after truncation) to match the tensor length
            if self.iid_list_field in self.dataset.field2type:
                hist_len = actual_len
            elif history is not None:
                hist_len = min(len(history), 50)
            else:
                hist_len = min(len(self.user2item_list_train.get(user_id, [])), 50)
            interaction_dict['item_length'] = torch.tensor([hist_len], device=self.config['device'])
        
        # Create the Interaction object
        return Interaction(interaction_dict)


    def full_sort_predict(self, user_id: int, history: List[int], 
                           full_hist: List[int] = None, gt_items: List[int] = None) -> torch.Tensor:
        """Generate scores for all items.
        
        Args:
            user_id: The user ID
            history: History items used for prediction (eval_hist)
            full_hist: All interacted items (optional). If provided along with gt_items,
                      scores of items in full_hist but not in gt_items will be masked to -inf
            gt_items: Ground truth items to keep unmasked (optional)
        """
        interaction = self._create_user_interaction(user_id, history)
        self.model.eval()
        with torch.no_grad():
            # Check if model supports full sort prediction
            if not hasattr(self.model, 'full_sort_predict'):
                return np.ndarray([])
            
            # Generate scores for all items using proper Interaction object
            scores = self.model.full_sort_predict(interaction)
            if scores.dim() > 1:
                scores_np = scores[0].cpu().numpy().flatten()  # Take first batch element
            else:
                scores_np = scores.cpu().numpy().flatten()
            
            # Mask scores of interacted items (except gt_items) if full_hist is provided
            if full_hist is not None:
                gt_set = set(gt_items) if gt_items else set()
                for item_id in full_hist:
                    if item_id not in gt_set and 0 <= item_id < len(scores_np):
                        scores_np[item_id] = float('-inf')
        
        return scores_np

    def recall(self, user_id: int, topk: int, history: List[int], 
                full_hist: List[int] = None, gt_items: List[int] = None) -> List[int]:
        """Generate recommendations for a user
        
        Args:
            user_id: The user ID to generate recommendations for
            topk: Number of recommendations to return
            history: Optional explicit history list. If not provided, will use internal user history
            full_hist: All interacted items (optional). If provided along with gt_items,
                      items in full_hist but not in gt_items will be excluded from candidates
            gt_items: Ground truth items to keep as valid candidates (optional)
            
        Returns:
            List of recommended item IDs
        """
        if not hasattr(self, 'model') or self.model is None:
            return []
            
        # Check if user exists in training data
        if user_id not in self.user2items_train:
            return []
            
        self.model.eval()
        with torch.no_grad():
            # Check if model supports full sort prediction
            if not hasattr(self.model, 'full_sort_predict'):
                return []
            
            # Create proper Interaction object for this user
            interaction = self._create_user_interaction(user_id, history)
            if interaction is None:
                return []
            
            # Generate scores for all items using proper Interaction object
            scores = self.model.full_sort_predict(interaction)
            
            # Determine which items to exclude from candidates
            if full_hist is not None:
                # Use full_hist for filtering, but keep gt_items as valid candidates
                gt_set = set(gt_items) if gt_items else set()
                exclude_set = set(item for item in full_hist if item not in gt_set)
            elif history:
                # Use explicitly provided history
                exclude_set = set(history)
            else:
                # Use internal training history + historical sequence
                user_train_items = set(self.user2items_train.get(user_id, []))
                user_history_items = set(self.user2item_list_train.get(user_id, []))
                exclude_set = user_train_items.union(user_history_items)
            
            # Convert scores to numpy and filter out excluded items
            if scores.dim() > 1:
                scores_np = scores[0].cpu().numpy().flatten()  # Take first batch element
            else:
                scores_np = scores.cpu().numpy().flatten()
            
            # Create candidate list excluding items in exclude_set
            candidates = []
            for item_id in range(len(scores_np)):
                if item_id not in exclude_set:
                    candidates.append((item_id, scores_np[item_id]))
            
            # Sort by score and return top-k
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[:topk]

    def get_user_history(self, user_id: int, split: str = 'train') -> List[int]:
        """Get user's historical interaction sequence
        
        Args:
            user_id: The user ID
            split: 'train' or 'test' split
            
        Returns:
            List of historical item IDs
        """
        if split == 'train':
            return self.user2item_list_train.get(user_id, [])
        elif split == 'test':
            return self.user2item_list_test.get(user_id, [])
        else:
            return []


# PopularityRecaller and ItemCFRecaller are now unified through RecBoleRecaller
# Usage: RecBoleRecaller(model_name='Pop', ...) or RecBoleRecaller(model_name='ItemKNN', ...)
