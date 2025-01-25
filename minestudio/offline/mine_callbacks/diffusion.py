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

from diffusers import DDPMScheduler

class DiffusionCallback(ObjectiveCallback):
    def __init__(self, scheduler_kwargs):
        super().__init__()
        self.num_train_timesteps = scheduler_kwargs.get("num_train_timesteps", 1000)
        self.beta_schedule = scheduler_kwargs.get("beta_schedule", "squaredcos_cap_v2")
        self.scheduler = DDPMScheduler(num_train_timesteps=self.num_train_timesteps, beta_schedule=self.beta_schedule)

    def before_step(self, batch, batch_idx, step_name):
        b, t, d = batch['action'].shape
        noise = torch.randn_like(batch['action']).reshape(b*t, d)
        action = batch["action"].reshape(b*t, d)
        timesteps = torch.randint(0, self.num_train_timesteps - 1, (b*t,)).long().to(noise.device)
        #! TODO: implement beta sampling in pi-0.
        noisy_x = self.scheduler.add_noise(action, noise, timesteps)
        batch['sampling_timestep'] = timesteps.reshape(b, t)
        batch['noisy_x'] = noisy_x.reshape(b, t, d)
        batch['noise'] = noise.reshape(b, t, d)
        return batch
    
    def __call__(
        self, 
        batch: Dict[str, Any], 
        batch_idx: int, 
        step_name: str,
        latents: Dict[str, torch.Tensor], 
        mine_policy: MineGenerativePolicy
    ) -> Dict[str, torch.Tensor]:
        noise = batch['noise']
        pred = latents['pred']
        b, t, d = pred.shape # b, 128, 32*22
        mask = batch.get('action_chunk_mask', torch.ones_like(pred)) # b, 128, 32
        action_dim = d // mask.shape[-1]
        # expand mask to the same shape as ut
        mask = mask.unsqueeze(-1).expand(-1, -1, -1, action_dim).reshape(b, t, d)
        mask_noise = noise * mask
        mask_pred = pred * mask
        loss = ((mask_noise - mask_pred) ** 2).sum(-1).mean()
        result = {
            "loss": loss,
        }
        return result
