'''
Date: 2024-11-10 10:26:52
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2025-01-09 16:57:08
FilePath: /MineStudio/minestudio/data/minecraft/dataset_event.py
'''
import io
import re
import os
import lmdb
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as L

from rich.console import Console
from pathlib import Path
from typing import Union, Tuple, List, Dict, Callable, Sequence, Mapping, Any, Optional, Literal

from minestudio.data.minecraft.utils import batchify
from minestudio.data.minecraft.core import KernelManager
from minestudio.data.minecraft.callbacks import ModalKernelCallback
from minestudio.utils.register import Registers

class EventKernel:
    
    def __init__(self, event_path: Union[str, Path], event_regex: str, min_nearby: Optional[int] = None, max_within: Optional[int] = None) -> None:
        if isinstance(event_path, str):
            event_path = Path(event_path)
        assert event_path.is_dir(), f"Event lmdb file {event_path} does not exist. "
        self.lmdb_stream = lmdb.open(str(event_path), max_readers=128, readonly=True, lock=False)

        with self.lmdb_stream.begin(write=False) as txn:
            __event_info__ = pickle.loads(txn.get(b'__event_info__'))
            # check if codebook exists
            __codebook_bytes__ = txn.get(b'__codebook__', None)
            if __codebook_bytes__ is None:
                self.__codebook__ = None
            else:
                self.__codebook__ = {v: k for k, v in pickle.loads(__codebook_bytes__).items()}
            self.event_info = {}
            for event, value in __event_info__.items():
                if re.match(event_regex, event):
                    self.event_info[event] = value
        
        self.event_list = sorted(list(self.event_info.keys()))
        # if min_nearby is not None or max_within is not None:
        self.filter_out(min_nearby, max_within)
    
    def filter_out(self, min_nearby: Optional[int] = None, max_within: Optional[int] = None):
        episode_event_last = {}
        remaining_events = {}
        for event in self.event_list:
            num_events = self.get_event_size(event)
            remaining_events[event] = []
            for i in range(num_events):
                episode, event_time, value = self.get_event_item(event, i)
                if event_time < 128: # remove dirty events
                    continue
                episode_event_key = f"{episode}:{event}"
                if episode_event_key not in episode_event_last:
                    episode_event_last[episode_event_key] = -100000

                if min_nearby is not None \
                    and event_time - episode_event_last[episode_event_key] <= min_nearby:
                    continue
                
                if max_within is not None \
                    and len(remaining_events[event]) >= max_within:
                    break
                
                episode_event_last[episode_event_key] = event_time
                remaining_events[event].append(i)
            self.event_info[event]['__num_items__'] = len(remaining_events[event])
        self.remaining_events = remaining_events
    
    def get_event_list(self) -> List[str]:
        return self.event_list
    
    def get_event_size(self, event: str) -> int:
        if event not in self.event_info:
            return 0
        return self.event_info[event]['__num_items__']

    def get_event_item(self, event: str, item_idx: int) -> Tuple[str, int, int]:
        assert item_idx < self.get_event_size(event), f"Item index {item_idx} out of range. "
        if hasattr(self, 'remaining_events'):
            item_idx = self.remaining_events[event][item_idx] # remap the index
        key = str((event, item_idx))
        with self.lmdb_stream.begin(write=False) as txn:
            item = pickle.loads(txn.get(key.encode()))
        episode, event_time, value = item
        if self.__codebook__ is not None:
            episode = self.__codebook__[episode]
        return episode, event_time, value

class EventKernelManager:
    
    def __init__(self, event_path: List[Union[str, Path]], event_regex: str, verbose: bool = True, **kwargs) -> None:
        self.verbose = verbose
        self.event_kernels = [EventKernel(event, event_regex, **kwargs) for event in event_path]
        event_set = set()
        for kernel in self.event_kernels:
            event_set.update(kernel.get_event_list())
        self.event_list = sorted(list(event_set))
        if verbose:
            Console().log(f"[Event Kernel Manager] Number of loaded events: {len(self.event_list)}")
    
    def get_event_list(self) -> List[str]:
        return self.event_list
    
    def get_event_size(self, event: str) -> int:
        if event not in self.event_list:
            return 0
        return sum([kernel.get_event_size(event) for kernel in self.event_kernels])
    
    def get_event_item(self, event: str, item_idx: int) -> Tuple[str, int, int]:
        for kernel in self.event_kernels:
            size = kernel.get_event_size(event)
            if item_idx < size:
                return kernel.get_event_item(event, item_idx)
            item_idx -= size
        raise ValueError(f"Item index {item_idx} out of range. ")

class EventDataset(Dataset):
    
    def __init__(self, 
        dataset_dirs: List[str], 
        modal_kernel_callbacks: List[Union[str, ModalKernelCallback]], 
        modal_kernel_config: Optional[Dict]=None,
        # below are parameters for spliting dataset and building items
        win_len: int=1, 
        skip_frame: int=1,
        split: Literal['train', 'val']='train',
        split_ratio: float=0.8, 
        verbose: bool=True,
        # below are event dataset specific parameters
        event_paths: Optional[List[str]]=None,
        bias: int=0,
        event_regex: str='', 
        min_nearby: Optional[int]=None, # minimal avaliable distance between two selected events
        max_within: Optional[int]=None, # maximum number of samples within each event
    ) -> Any:
        super().__init__()
        self.win_len = win_len
        self.skip_frame = skip_frame
        self.split = split
        self.split_ratio = split_ratio
        self.verbose = verbose
        self.bias = bias
        self.event_regex = event_regex

        if event_paths is None:
            event_paths = [Path(x) / "event" for x in dataset_dirs]
        else:
            event_paths = [Path(x) for x in event_paths]

        self.event_kernel = EventKernelManager(
            event_path=event_paths,
            event_regex=event_regex,
            verbose=verbose,
            min_nearby=min_nearby, 
            max_within=max_within,
        )
        
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
        self.event_list = self.event_kernel.get_event_list()
        
        self.num_items = 0
        event_with_items = []
        for event in self.event_list:
            num_event_items = self.event_kernel.get_event_size(event)
            if self.split == 'train':
                bias = 0
                num_event_items = int(num_event_items * self.split_ratio)
            elif self.split == 'val':
                bias = int(num_event_items * self.split_ratio)
                num_event_items = num_event_items - bias
            else:
                raise ValueError(f"Split type <{self.split}> not supported. ")
            self.num_items += num_event_items
            event_with_items.append((self.num_items, event, bias))
        self.items = event_with_items
        
        if self.verbose:
            Console().log(f"[Event Dataset] Regex: {self.event_regex}, Number of events: {len(self.event_list)}, number of items: {self.num_items}")
    
    def locate_item(self, idx: int) -> Tuple[str, int]:
        """Find the first event that idx > acc[event]"""
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
        event = self.items[left][1]
        bias = self.items[left][2]
        return event, relative_idx + bias
    
    def __len__(self) -> int:
        return self.num_items
    
    def __getitem__(self, idx: int) -> Mapping[str, torch.Tensor]:
        assert idx < len(self), f"Index <{idx}> out of range <{len(self)}>"
        event, relative_idx = self.locate_item(idx)
        episode, event_time, value = self.event_kernel.get_event_item(event, relative_idx)
        start = max(event_time - self.win_len + self.bias, 0)
        item = self.kernel_manager.read(episode, start=start, win_len=self.win_len, skip_frame=self.skip_frame)

        for key in list(item.keys()):
            if key.endswith('mask'):
                mask = item.pop(key)
        item["mask"] = mask

        item['text'] = event.replace('minecraft.', '')
        item['episode'] = episode
        item['timestamp'] = np.arange(start, start+self.win_len, self.skip_frame)
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


class EventDataModule(L.LightningDataModule):
    
    def __init__(self, 
                 data_params: Dict, 
                 batch_size: int=1, 
                 num_workers: int=0, 
                 prefetch_factor: Optional[int] = None):
        super().__init__()
        self.data_params = data_params
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
    
    def setup(self, stage: Optional[str]=None):
        self.train_dataset = EventDataset(split='train', **self.data_params)
        self.val_dataset = EventDataset(split='val', **self.data_params)

    def train_dataloader(self):
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
    from torch.utils.data import DataLoader
    from minestudio.data.minecraft.callbacks import (
        ImageKernelCallback, ActionKernelCallback, MetaInfoKernelCallback, SegmentationKernelCallback
    )


    data_module = EventDataModule(
        data_params=dict(
            dataset_dirs=[
                '/nfs-shared-2/data/contractors/dataset_10xx', 
            ], 
            modal_kernel_callbacks=[
                ImageKernelCallback(frame_width=224, frame_height=224, enable_video_aug=False), 
                ActionKernelCallback(),
                MetaInfoKernelCallback(),
                SegmentationKernelCallback(frame_width=224, frame_height=224), 
            ],
            win_len=128, 
            split_ratio=0.9, 
            event_regex='minecraft.kill_entity:.*', 
            min_nearby=64,
            max_within=1000,
        ), 
        batch_size=4, 
        num_workers=4, 
        prefetch_factor=None
    )
    data_module.setup()
    loader = data_module.train_dataloader()
    for idx, batch in enumerate(loader):
        print(
            "\t".join(
                [f"{a} {b}" for a, b in zip(batch['episode'], batch['text'])]
            )
        )
        if idx > 50:
            break
