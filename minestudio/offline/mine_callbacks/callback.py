'''
Date: 2024-11-12 10:57:29
LastEditors: muzhancun muzhancun@126.com
LastEditTime: 2025-01-18 13:53:17
FilePath: /MineStudio/minestudio/offline/mine_callbacks/callback.py
'''
import torch
from typing import Dict, Any
from minestudio.models import MinePolicy

class ObjectiveCallback:

    def __init__(self):
        ...

    def __call__(
        self, 
        batch: Dict[str, Any], 
        batch_idx: int, 
        step_name: str, 
        latents: Dict[str, torch.Tensor], 
        mine_policy: MinePolicy
    ) -> Dict[str, torch.Tensor]:
        return {}

    def before_step(self, batch, batch_idx, step_name):
        return batch
