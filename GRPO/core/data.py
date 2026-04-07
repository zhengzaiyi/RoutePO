from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import torch
import os
import copy
import warnings
from tqdm import tqdm
# Suppress pandas FutureWarning from recbole
warnings.filterwarnings('ignore', category=FutureWarning, message='.*A value is trying to be set on a copy of a DataFrame.*')

from GRPO.core.constant import dataset2item_feat_fields, dataset2user_feat_fields, dataset2inter_feat_fields

try:
    from recbole.config import Config
    from recbole.data.dataset import SequentialDataset
    from recbole.data import create_dataset, data_preparation
    from recbole.data.dataloader import TrainDataLoader, AbstractDataLoader, FullSortEvalDataLoader
    from recbole.utils import init_seed as recbole_init_seed
    from recbole.utils import get_model as get_recbole_model
    RECB = True
except Exception:
    RECB = False
from datasets import Dataset


@dataclass
class InteractionData:
    ds: SequentialDataset
    train_user_ids: List[int]
    train_histories: List[List[int]]
    train_target_items: List[int]
    eval_user_ids: List[int]
    eval_histories: List[List[int]]
    eval_target_items: List[int]
    test_user_ids: List[int]
    test_histories: List[List[int]]
    test_target_items: List[int]

def keep_longest_per_user(
    dataloader,
    dataset: SequentialDataset,
    config: Config = None,
    use_all_gt: bool = True,
):
    """
    Filter dataset to keep only one record per user (the one with maximum sequence length).
    
    Args:
        dataloader: DataLoader to filter (can be TrainDataLoader, FullSortEvalDataLoader, etc.)
        dataset: SequentialDataset object (for field names)
        config: Config object (not used anymore, kept for compatibility)
        use_all_gt: Whether to use all ground truth items (not used, kept for compatibility)
        
    Returns:
        Tuple of (user_ids, item_id_lists, item_ids) numpy arrays
    """
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    iid_list_field = f'{iid_field}_list'
    len_field = dataset.item_list_length_field  # 一般是 f"{iid_field}_list_length"
    
    # Get the sub-dataset from dataloader
    sub_dataset = dataloader.dataset
    inter = sub_dataset.inter_feat  # Interaction 对象（增广后的多条样本）
    
    # 取出 uid 与序列长度（numpy 数组）
    uids = inter[uid_field]
    lengths = inter[len_field]
    n = len(uids)
    idx = np.arange(n)
    
    # 为了"每个用户保留长度最大的那条"，我们按 (uid, length, idx) 升序排序，
    # 然后取每个 uid 在排序后出现的最后一个位置（即最大 length；若并列取最后一条）
    order = np.lexsort((idx, lengths, uids))  # 先按 uid，再按 length，再按原始 idx
    uids_sorted = uids[order]
    # 获取每个用户的起始位置和计数
    unique_uids, first_pos, counts = np.unique(uids_sorted, return_index=True, return_counts=True)
    
    last_pos = []
    for i, (start, count) in enumerate(zip(first_pos, counts)):
        pos_in_user = min(int(count), count - 1)
        last_pos.append(start + pos_in_user)
    
    keep_indices_sorted = order[last_pos]  # 映射回原始样本下标
    
    # 构造布尔掩码并切片出过滤后的 Interaction
    keep_mask = np.zeros(n, dtype=bool)
    keep_mask[keep_indices_sorted] = True
    print("sum of keep_mask", sum(keep_mask))
    # 直接返回过滤后的三元组
    filtered_inter = inter[keep_mask]
    user_ids = filtered_inter[uid_field].tolist()
    item_id_lists = filtered_inter[iid_list_field].tolist()
    for i in range(len(item_id_lists)):
        if 0 in item_id_lists[i]:
            index_0 = item_id_lists[i].index(0)
            item_id_lists[i] = item_id_lists[i][:index_0]
    item_ids = filtered_inter[iid_field].tolist()
    
    return user_ids, item_id_lists, item_ids


def get_base_config_dict(
    dataset_name: str, 
    data_path: str = 'dataset', 
    seed: int = 42,
    train_ratio: float = 0.8,
    eval_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict:
    """
    Get base config dict for RecBole dataset loading.
    
    Args:
        dataset_name: Name of the dataset
        data_path: Path to dataset directory
        seed: Random seed
        train_ratio: Ratio for training set (default: 0.8, matching main_pure.py)
        eval_ratio: Ratio for validation set (default: 0.1, matching main_pure.py)
        test_ratio: Ratio for test set (default: 0.1, matching main_pure.py)
    
    Returns:
        Config dictionary for RecBole
    """
    # Normalize ratios to ensure they sum to 1.0
    total = train_ratio + eval_ratio + test_ratio
    train_ratio = train_ratio / total
    eval_ratio = eval_ratio / total
    test_ratio = test_ratio / total
    
    config_dict = {
        'data_path': data_path,
        'seed': seed,
        'load_col': {
            'inter': dataset2inter_feat_fields[dataset_name], # required
            # 'item': dataset2item_feat_fields[dataset_name], # required
        },
        'user_inter_num_interval': "[5,inf)",
        'item_inter_num_interval': "[5,inf)",
        'train_neg_sample_args': None,
        'loss_type': 'CE',
        'val_interval': {
            'rating': '[3,inf)'
        },
        'eval_args': {
            # 'split': {'RS': [train_ratio, eval_ratio, test_ratio]},  
            'split': {'LS': "valid_and_test"},  
            'order': 'TO',
            'group_by': 'user'
        },
        'save_dataset': True,
        'save_dataloaders': True,
    }
    if dataset2user_feat_fields[dataset_name] is not None:
        config_dict['load_col']['user'] = dataset2user_feat_fields[dataset_name] # optional
    if dataset_name == 'steam':
        config_dict.update({
            'ITEM_ID_FIELD': 'product_id',
        })
        del config_dict['val_interval']
    elif dataset_name == 'anime':
        del config_dict['val_interval']
    return config_dict

def get_mask(time_order: bool, rate: List[float], length: int):
    if sum(rate) != 1.0:
        raise ValueError(f"Sum of rates must be 1.0, but got {sum(rate)}")
    
    train_rate, eval_rate, test_rate = rate
    indices = np.arange(length)
    if not time_order:
        indices = np.random.permutation(indices)
    test_size = max(1, int(length * test_rate))
    eval_size = max(1, int(length * eval_rate))
    train_gt_size = min(test_size, eval_size)
    train_size = length - test_size - eval_size - train_gt_size
    
    train_gt_start = train_size
    eval_start = train_gt_start + train_gt_size
    test_start = eval_start + eval_size
    return {
        'train': indices[:train_size],
        'train_gt': indices[train_gt_start:train_gt_start+train_gt_size],
        'eval': indices[eval_start:eval_start+eval_size],
        'test': indices[test_start:test_start+test_size],
    }
    
def load_dataset(
    dataset: str, 
    data_path: str, 
    seed: int = 42, 
    train_ratio: float = 0.8,
    eval_ratio: float = 0.1,
    test_ratio: float = 0.1,
    dataset_type: str = 'P5',
    # recbole configs
    # P5 configs
    P5_TO=True,
    
) -> InteractionData:
    """
    Load dataset with train/eval/test split (matching main_pure.py style).
    
    Args:
        dataset: Dataset name
        data_path: Path to dataset directory
        seed: Random seed
        filter_train: Whether to filter train set (keep one record per user)
        filter_eval: Whether to filter eval set (keep one record per user)
        filter_test: Whether to filter test set (keep one record per user)
        train_ratio: Ratio for training set (default: 0.8, matching main_pure.py)
        eval_ratio: Ratio for validation set (default: 0.1, matching main_pure.py)
        test_ratio: Ratio for test set (default: 0.1, matching main_pure.py)
    
    Returns:
        InteractionData with train/eval/test splits
    """
    config_dict = get_base_config_dict(dataset, data_path, seed, train_ratio, eval_ratio, test_ratio)
    # Disable caching to ensure fresh data processing (avoids stale float32 data)
    config_dict['save_dataset'] = False
    config_dict['save_dataloaders'] = False
    model = 'SASRec'
    cfg = Config(
        model=model, 
        dataset=dataset, 
        config_dict=config_dict,
    )
    # assert 
    recbole_init_seed(cfg['seed'], reproducibility=True) # TODO: check if this is correct
    ds = create_dataset(cfg)
    print(ds)
    raw_ds = copy.deepcopy(ds)
    train, valid, test = data_preparation(cfg, ds)

    # Verify float64 timestamp precision
    _tf = raw_ds.time_field
    _tl = f'{_tf}_list'
    _sample_ts = train.dataset.inter_feat[_tl][0][:3].tolist()
    print(f"[DTYPE CHECK] {_tl} dtype={train.dataset.inter_feat[_tl].dtype}, sample={_sample_ts}")

    uid_field, iid_field = ds.uid_field, ds.iid_field
    iid_list_field = f'{iid_field}_list'
    
    if dataset_type == 'recbole':
        # every item serves as a target item in one dataset sample
        train_user_ids = train.dataset.inter_feat.interaction[uid_field].tolist()
        train_item_id_lists = train.dataset.inter_feat.interaction[iid_list_field].tolist()
        train_item_ids = train.dataset.inter_feat.interaction[iid_field].tolist()
        
        eval_user_ids = valid.dataset.inter_feat.interaction[uid_field].tolist()
        eval_item_id_lists = valid.dataset.inter_feat.interaction[iid_list_field].tolist()
        eval_item_ids = valid.dataset.inter_feat.interaction[iid_field].tolist()
        
        test_user_ids = test.dataset.inter_feat.interaction[uid_field].tolist()
        test_item_id_lists = test.dataset.inter_feat.interaction[iid_list_field].tolist()
        test_item_ids = test.dataset.inter_feat.interaction[iid_field].tolist()        
        
        
    elif dataset_type == 'P5':
        full_user_ids, full_item_id_lists, full_item_ids = keep_longest_per_user(test, ds, cfg)
        full_histories = [full_item_id_lists[i] + [full_item_ids[i]] for i in range(len(full_item_id_lists))]
        train_user_ids, train_item_id_lists, train_item_ids = [], [], []
        eval_user_ids, eval_item_id_lists, eval_item_ids = [], [], []
        test_user_ids, test_item_id_lists, test_item_ids = [], [], []
        
        for user_id, full_history in zip(full_user_ids, full_histories):
            mask = get_mask(P5_TO, [train_ratio, eval_ratio, test_ratio], len(full_history))
            # Convert to numpy array for efficient indexing
            item_id_arr = np.array(full_history)
            
            # Build masks using numpy concatenation
            train_mask = mask['train']
            train_eval_mask = np.concatenate([mask['train'], mask['train_gt']])
            train_eval_test_mask = np.concatenate([mask['train'], mask['train_gt'], mask['eval']])
            
            train_user_ids.append(user_id)
            train_item_id_lists.append(item_id_arr[train_mask].tolist())
            train_item_ids.append(item_id_arr[mask['train_gt']].tolist())
            eval_user_ids.append(user_id)
            eval_item_id_lists.append(item_id_arr[train_eval_mask].tolist())
            eval_item_ids.append(item_id_arr[mask['eval']].tolist())
            test_user_ids.append(user_id)
            test_item_id_lists.append(item_id_arr[train_eval_test_mask].tolist())
            test_item_ids.append(item_id_arr[mask['test']].tolist())
        
        
        
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    return InteractionData(
        raw_ds, 
        train_user_ids, 
        train_item_id_lists, 
        train_item_ids, 
        eval_user_ids, 
        eval_item_id_lists, 
        eval_item_ids, 
        test_user_ids, 
        test_item_id_lists, 
        test_item_ids,
    )

