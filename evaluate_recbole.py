# Import all required libraries
import warnings
# Suppress pandas FutureWarning from recbole
warnings.filterwarnings('ignore', category=FutureWarning, message='.*A value is trying to be set on a copy of a DataFrame.*')

import torch
from datasets import load_dataset
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR, Pop, ItemKNN, LightGCN
from recbole.model.sequential_recommender import SASRec
from recbole.utils import init_seed, init_logger, get_trainer
import wandb

# Set torch.load compatibility
torch.serialization.add_safe_globals([dict, list, tuple, set])

# Define unified model training function
def train_model(model_type, dataset_name='All_Beauty', epochs=10, wandb_run_name=None, learning_rate=0.001, **kwargs):
    """
    Unified function to train recommendation models
    
    Args:
        model_type: Model type ('BPR', 'SASRec', 'Pop')
        dataset_name: Dataset name
        epochs: Training epochs·
        **kwargs: Additional model-specific parameters
    
    Returns:
        dict: Dictionary containing model, trainer, config and results
    """
    
    print(f"\n=== Training {model_type} Model ===")
    
    # Base configuration
    base_config = {
        # 'data_path': 'seq_rec_results/dataset/processed/',
        # 'benchmark_filename': ['train', 'valid', 'test'],
        'epochs': epochs,
        'stopping_step': epochs,
        'eval_step': 1,
        'metrics': ['Recall', 'NDCG'],
        'topk': [10, 50],
        'valid_metric': 'NDCG@10',
        'checkpoint_dir': './checkpoints/',
        'show_progress': True,
        'log_wandb': True,
        'wandb_project': wandb_run_name,
        'learning_rate': learning_rate
    }
    ITEM_ID_FIELD = 'product_id' if dataset_name == 'steam' else 'item_id'
    base_config.update({
        'data_path': 'dataset',
        'load_col': {
            # 'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
            'inter': ['user_id', ITEM_ID_FIELD, 'timestamp']
        },
        'user_inter_num_interval': "[5,inf)",
        'item_inter_num_interval': "[5,inf)",
        'train_neg_sample_args': None,
        'loss_type': 'CE',
        # 'val_interval': {
        #     'rating': '[3,inf)'  # 只保留rating >= 4的交互
        # },
        'eval_args': {
            'split': {'RS': [0.8,0.1,0.1]},
            'order': 'TO',  # Temporal Order
            'group_by': 'user'
        },
        'ITEM_ID_FIELD': ITEM_ID_FIELD,
    })
    
    # Model-specific configurations
    if model_type == 'BPR':
        model_class = BPR
        model_config = {
            **base_config,
            'train_neg_sample_args': {
                'distribution': 'uniform',
                'sample_num': 1,
                'alpha': 1.0,
                'dynamic': False,
                'candidate_num': 0
            },
            'loss_type': 'BPR',
            'train_batch_size': 2048,
            'eval_batch_size': 2048 * 20000,
            'learning_rate': 2e-4,
        }
        
    elif model_type == 'SASRec':
        model_class = SASRec
        model_config = {
            **base_config,
            'train_neg_sample_args': None,
            'loss_type': 'CE',
            'train_batch_size': 256,
            'max_seq_length': 50,
            'hidden_size': 64,
            'n_layers': 2,
            'n_heads': 2,
            'inner_size': 256,
            'hidden_dropout_prob': 0.5,
            'attn_dropout_prob': 0.5,
        }
        
    elif model_type == 'Pop':
        model_class = Pop
        model_config = {
            **base_config,
            'train_neg_sample_args': None,
        }
    elif model_type == 'ItemKNN':
        model_class = ItemKNN
        model_config = {
            **base_config,
            'train_neg_sample_args': None,
            'eval_batch_size': 2048 * 20000,
        }
    elif model_type == 'LightGCN':
        model_class = LightGCN
        model_config = {
            **base_config,
            # LightGCN需要负采样
            'train_neg_sample_args': {
                'distribution': 'uniform',
                'sample_num': 1,  # 每个正样本配1个负样本
            },
            'loss_type': 'BPR',
            'embedding_size': 64,
            'n_layers': 3,  # GCN层数
            'reg_weight': 1e-5,  # 正则化系数
            'train_batch_size': 2048,
            'eval_batch_size': 2048 * 20000,
            'learning_rate': 5e-3,
        }
    elif model_type == 'SimpleX':
        from recbole.model.general_recommender import SimpleX
        model_class = SimpleX
        model_config = {
            **base_config,
            'train_neg_sample_args': {
                'distribution': 'uniform',
                'sample_num': 1,
            },
            'loss_type': 'BPR',
            'embedding_size': 64,
            'aggregator': 'mean',  # 或 'user_attention', 'self_attention'
            'gamma': 0.5,
            'margin': 0.9,
            'negative_weight': 0.5,
            'reg_weight': 1e-5,
            'train_batch_size': 2048,
            'eval_batch_size': 2048 * 20000,
            'learning_rate': 2e-3,
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Merge user-defined parameters
    model_config.update(kwargs)
    
    # Create config and dataset
    config = Config(
        model=model_type,
        dataset=dataset_name,
        config_dict=model_config
    )
    
    # Create dataset
    model_dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, model_dataset)
    
    print(f"{model_type} dataset stats:")
    print(f"Users: {model_dataset.user_num}")
    print(f"Items: {model_dataset.item_num}")
    print(f"Interactions: {model_dataset.inter_num}")
    
    # Initialize model and trainer
    init_seed(config['seed'], config['reproducibility'])
    model = model_class(config, model_dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    
    print(f"Training {model_type} model...")
    
    # torch.load compatibility settings
    original_load = torch.load
    def safe_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = safe_load
    
    try:
        # Train model
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=True, show_progress=True
        )
        
        print(f"{model_type} training completed!")
        print(f"Best validation result: {best_valid_result}")
        
        # Test model
        test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=True)
        print(f"{model_type} test result: {test_result}")
        
        return {
            'model_type': model_type,
            'model': model,
            'trainer': trainer,
            'config': config,
            'dataset': model_dataset,
            'train_data': train_data,
            'valid_data': valid_data,
            'test_data': test_data,
            'best_valid_result': best_valid_result,
            'test_result': test_result
        }
        
    finally:
        # Restore original torch.load function
        torch.load = original_load

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--models_to_train', type=str, nargs='+', default=['BPR'])
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    # Train all models using unified function
    print("=== Training All Models with Unified Function ===")

    # Store all model results
    model_results = {}
    dataset_name = args.dataset
    wandb_run_name = f"evaluate_recbole_{dataset_name}_{'_'.join([f'{k}_{v}' for k, v in vars(args).items()])}"
    wandb.init(project='recaller_evaluation_1028', name=wandb_run_name)
    wandb.config.update(args)
    # Train all models
    for model_type in args.models_to_train:
            result = train_model(
                dataset_name=dataset_name,
                model_type=model_type,
                epochs=args.epochs,
                wandb_run_name=wandb_run_name,
                learning_rate=args.learning_rate
            )
            model_results[model_type] = result
            print(f"✅ {model_type} training successful")

    print(f"\nTraining completed! Successfully trained {len([r for r in model_results.values() if r is not None])} models")