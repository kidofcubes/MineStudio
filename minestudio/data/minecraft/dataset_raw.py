'''
Date: 2024-11-10 10:26:32
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-01-21 23:03:15
FilePath: /MineStudio/minestudio/data/minecraft/dataset_raw.py
'''
import io
import re
import os
import math
import lmdb
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as L

from rich import print
from rich.console import Console
from pathlib import Path
from typing import Union, Tuple, List, Dict, Callable, Any, Optional, Literal

from minestudio.data.minecraft.utils import MineDistributedBatchSampler, batchify
from minestudio.data.minecraft.core import KernelManager
from minestudio.data.minecraft.callbacks import ModalKernelCallback
from minestudio.utils.register import Registers

class RawDataset(Dataset):
    """Raw dataset for training and testing. """
    def __init__(self, 
                 dataset_dirs: List[str], 
                 modal_kernel_callbacks: List[Union[str, ModalKernelCallback]], 
                 modal_kernel_config: Optional[Dict]=None,
                 seed: int=0, 
                 # below are parameters for spliting dataset and building items
                 win_len: int=1, 
                 skip_frame: int=1, 
                 split: Literal['train', 'val']='train',
                 split_ratio: float=0.8,
                 verbose: bool=True,
                 shuffle_episodes: bool=False):
        super().__init__()
        self.win_len = win_len
        self.skip_frame = skip_frame
        self.split = split
        self.split_ratio = split_ratio
        self.verbose = verbose
        self.shuffle_episodes = shuffle_episodes
        self.seed = seed
        
        assert len(modal_kernel_callbacks) > 0, "At least one modal kernel callback is required. "
        if isinstance(modal_kernel_callbacks[0], str):
            assert modal_kernel_config is not None, "Modal kernel config is required. "
            modal_kernel_callbacks = [
                Registers.modal_kernel_callback[name].create_from_config(modal_kernel_config) 
                    for name in modal_kernel_callbacks
            ]
        
        self.kernel_manager = KernelManager(
            dataset_dirs=dataset_dirs, 
            modal_kernel_callbacks=modal_kernel_callbacks,
        )
        self.build_items()
    
    def build_items(self) -> None:
        self.episodes_with_length = self.kernel_manager.get_episodes_with_length()
        _episodes_with_length = list(self.episodes_with_length.items())

        if self.shuffle_episodes:
            print(f"[Raw Dataset] Shuffling episodes with seed {self.seed}. ")
            random.seed(self.seed) # ensure the same shuffle order for all workers
            random.shuffle(_episodes_with_length)

        divider = int(len(_episodes_with_length) * self.split_ratio)
        if self.split == 'train':
            _episodes_with_length = _episodes_with_length[:divider]
        else:
            _episodes_with_length = _episodes_with_length[divider:]
        
        self.items = []
        self.num_items = 0
        self.episodes_with_items = []
        for episode, length in _episodes_with_length:
            num_episode_items = (length + self.win_len - 1) // self.win_len 
            self.episodes_with_items.append( (episode, num_episode_items, self.num_items) )
            self.num_items += num_episode_items
            self.items.append( (self.num_items, episode) )

    def locate_item(self, idx: int) -> Tuple[str, int]:
        """Find the first episode that idx > acc[episode]"""
        left, right = 0, len(self.items)
        while left < right:
            mid = (left + right) // 2
            if self.items[mid][0] <= idx:
                left = mid + 1
            else:
                right = mid
        if left == 0:
            relative_idx = idx
        else:
            relative_idx = idx - self.items[left-1][0]
        episode = self.items[left][1]
        return episode, relative_idx

    def __len__(self) -> int:
        return self.num_items
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        assert idx < len(self), f"Index <{idx}> out of range <{len(self)}>"
        episode, relative_idx = self.locate_item(idx)
        start = max(0, relative_idx * self.win_len) # if start > 0 is the prequest for previous action
        item = self.kernel_manager.read(episode, start, self.win_len, self.skip_frame)
        item["mask"] = item['action_mask']
        item['text'] = 'raw'
        item['timestamp'] = np.arange(start, start+self.win_len, self.skip_frame)
        item['episode'] = episode
        episode_samples = math.ceil(self.episodes_with_length[episode] / self.win_len)
        item['progress'] = f"{relative_idx}/{episode_samples}"
        item = self.to_tensor(item)
        return item

    def to_tensor(self, item: Union[np.ndarray, List, Dict]) -> Union[np.ndarray, List, Dict]:
        """Convert numpy array to torch tensor."""
        if isinstance(item, np.ndarray):
            return torch.from_numpy(item)
        elif isinstance(item, List):
            return [self.to_tensor(val) for val in item]
        elif isinstance(item, Dict):
            return {key: self.to_tensor(val) for key, val in item.items()}
        else:
            return item

class RawDataModule(L.LightningDataModule):
    
    def __init__(self, 
                 data_params: Dict, 
                 batch_size: int=1, 
                 num_workers: int=0, 
                 prefetch_factor: Optional[int] = None,
                 episode_continuous_batch: bool = False):
        super().__init__()
        self.data_params = data_params
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.episode_continuous_batch = episode_continuous_batch
    
    def setup(self, stage: Optional[str]=None):
        self.train_dataset = RawDataset(split='train', **self.data_params)
        self.val_dataset = RawDataset(split='val', **self.data_params)

    def train_dataloader(self):
        if self.episode_continuous_batch:
            # using MineDistributedBatchSampler for loading continuous video frames
            batch_sampler = MineDistributedBatchSampler(
                dataset=self.train_dataset, 
                batch_size=self.batch_size, 
            )
            train_loader = DataLoader(
                dataset=self.train_dataset, 
                batch_sampler=batch_sampler,
                num_workers=self.num_workers, 
                collate_fn=batchify,
                prefetch_factor=self.prefetch_factor,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                dataset=self.train_dataset, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                shuffle=True, 
                collate_fn=batchify,
                prefetch_factor=self.prefetch_factor,
                pin_memory=True,
                drop_last=True,
            )
        return train_loader

    def val_dataloader(self):
        if self.episode_continuous_batch:
            # using MineDistributedBatchSampler for loading continuous video frames
            batch_sampler = MineDistributedBatchSampler(
                dataset=self.val_dataset, 
                batch_size=self.batch_size, 
            )
            val_loader = DataLoader(
                dataset=self.val_dataset, 
                batch_sampler=batch_sampler,
                num_workers=self.num_workers, 
                collate_fn=batchify,
                prefetch_factor=self.prefetch_factor,
                pin_memory=True,
            )
        else:
            val_loader = DataLoader(
                dataset=self.val_dataset, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                shuffle=False, 
                collate_fn=batchify,
                prefetch_factor=self.prefetch_factor,
                pin_memory=True,
                drop_last=True,
            )
        return val_loader

if __name__ == '__main__':

    from tqdm import tqdm
    from minestudio.data.minecraft.callbacks import (
        ImageKernelCallback, 
        ActionKernelCallback, VectorActionKernelCallback, 
        MetaInfoKernelCallback, 
        SegmentationKernelCallback
    )

    data_module = RawDataModule(
        data_params=dict(
            dataset_dirs=[
                '/nfs-shared-2/data/contractors/dataset_10xx', 
            ],
            modal_kernel_callbacks=[
                ImageKernelCallback(frame_width=224, frame_height=224, enable_video_aug=False), 
                ActionKernelCallback(enable_prev_action=True, win_bias=1, read_bias=-1),
                VectorActionKernelCallback(action_chunk_size=32), 
                MetaInfoKernelCallback(),
                SegmentationKernelCallback(frame_width=224, frame_height=224), 
            ],
            win_len=128, 
            split_ratio=0.9,
            shuffle_episodes=True,
        ),
        batch_size=3,
        num_workers=8,
        prefetch_factor=None,
        episode_continuous_batch=True,
    )
    data_module.setup()
    loader = data_module.train_dataloader()
    for idx, batch in enumerate(loader):
        print(
            "\t".join(
                [f"{a} {b}" for a, b in zip(batch['episode'], batch['progress'])]
            )
        )
        # if idx > 50:
        #     break
