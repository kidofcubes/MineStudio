'''
Date: 2025-01-18 13:49:25
LastEditors: muzhancun 2100017790@stu.pku.edu.cn
LastEditTime: 2025-01-27 13:53:26
FilePath: /MineStudio/minestudio/offline/mine_callbacks/diffusion.py
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

class DictDiffusionCallback(ObjectiveCallback):
    def __init__(self, scheduler_kwargs):
        super().__init__()
        self.num_train_timesteps = scheduler_kwargs.get("num_train_timesteps", 1000)
        self.beta_schedule = scheduler_kwargs.get("beta_schedule", "squaredcos_cap_v2")
        self.scheduler = DDPMScheduler(**scheduler_kwargs)

    def add_noise(self, batch, type):
        action = batch['action'][type]
        b, t, d = action.shape
        action = action.reshape(b*t, d)
        noise = torch.randn_like(action)
        timesteps = torch.randint(0, self.num_train_timesteps - 1, (b*t,)).long().to(noise.device)
        noisy_x = self.scheduler.add_noise(action, noise, timesteps)
        batch[f"noisy_{type}"] = noisy_x.reshape(b, t, d)
        batch[f"{type}_noise"] = noise.reshape(b, t, d)
        batch[f"{type}_timesteps"] = timesteps.reshape(b, t)
        return batch

    def before_step(self, batch, batch_idx, step_name):
        batch = self.add_noise(batch, 'camera')
        batch = self.add_noise(batch, 'button')
        return batch

    def __call__(
        self, 
        batch: Dict[str, Any], 
        batch_idx: int, 
        step_name: str,
        latents: Dict[str, torch.Tensor], 
        mine_policy: MineGenerativePolicy
    ) -> Dict[str, torch.Tensor]:
        camera_noise = batch['camera_noise']
        button_noise = batch['button_noise']
        camera_pred = latents['camera']
        button_pred = latents['button']
        b, t, d1 = camera_pred.shape
        b, t, d2 = button_pred.shape
        mask = batch.get('action_chunk_mask', torch.ones_like(camera_pred))
        action_chunk_size = mask.shape[-1]
        camera_dim = d1 // action_chunk_size
        button_dim = d2 // action_chunk_size
        camera_mask = mask.unsqueeze(-1).expand(-1, -1, -1, camera_dim).reshape(b, t, d1)
        button_mask = mask.unsqueeze(-1).expand(-1, -1, -1, button_dim).reshape(b, t, d2)
        camera_mask_noise = camera_noise * camera_mask
        button_mask_noise = button_noise * button_mask
        camera_mask_pred = camera_pred * camera_mask
        button_mask_pred = button_pred * button_mask
        camera_loss = ((camera_mask_noise - camera_mask_pred) ** 2).sum(-1).mean()
        button_loss = ((button_mask_noise - button_mask_pred) ** 2).sum(-1).mean()
        result = {
            "loss": 10 * camera_loss + button_loss,
            "camera_loss": camera_loss,
            "button_loss": button_loss,
        }
        return result



    
