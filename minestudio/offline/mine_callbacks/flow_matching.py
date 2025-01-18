'''
Date: 2025-01-18 13:49:25
LastEditors: muzhancun muzhancun@126.com
LastEditTime: 2025-01-18 14:11:21
FilePath: /MineStudio/minestudio/offline/mine_callbacks/flow_matching.py
'''
import torch
from typing import Dict, Any
from minestudio.models import MineGenerativePolicy
from minestudio.offline.mine_callbacks.callback import ObjectiveCallback

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

class FlowMatchingCallback(ObjectiveCallback):
    def __init__(self, sigma: float=0.0):
        super().__init__()
        self.sigma = sigma
        self.fm = ConditionalFlowMatcher(sigma=sigma)

    def before_step(self, batch, batch_idx, step_name):
        noise = torch.rand_like(batch['action'])
        batch['noise'] = noise
        t, xt, ut = self.fm.sample(noise, batch['action'])
        batch['sampling_timestep'], batch['xt'], batch['ut'] = t, xt, ut
        return batch
    
    def __call__(
        self, 
        batch: Dict[str, Any], 
        batch_idx: int, 
        step_name: str,
        latents: Dict[str, torch.Tensor], 
        mine_policy: MineGenerativePolicy
    ) -> Dict[str, torch.Tensor]:
        ut = batch['ut']
        vt = latents['vt']
        b, t, d = ut.shape
        mask = batch.get('action_chunk_mask', torch.ones_like(ut))
        mask_ut = ut * mask
        mask_vt = vt * mask
        loss = ((mask_vt - mask_ut) ** 2).sum(-1).mean()
        result = {
            "loss": loss,
        }
        return result