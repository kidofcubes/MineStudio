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
    def __init__(self, sigma: float=0.0, sampling="beta", alpha=1.5, beta=1, sigmin=0.001):
        super().__init__()
        self.sigma = sigma
        self.fm = ConditionalFlowMatcher(sigma=sigma)
        assert sampling in ["beta", "normal"], "Sampling method must be 'beta' or 'normal'."
        self.sampling = sampling
        if sampling == "beta":
            self.alpha = alpha
            self.beta = beta
            self.sigmin = sigmin
            self.flow_t_max = 1 - sigmin
            self.flow_beta_dist = torch.distributions.Beta(alpha, beta)

    def before_step(self, batch, batch_idx, step_name):
        b, t, d = batch['action'].shape
        noise = torch.randn_like(batch['action'])
        action = batch['action'].reshape(b*t, d)
        noise = noise.reshape(b*t, d)
        if self.sampling == "beta":
            z = self.flow_beta_dist.sample((noise.shape[0],))
            time = self.flow_t_max * (1 - z).type_as(noise)
            time, xt, ut = self.fm.sample_location_and_conditional_flow(noise, action, time)
        else:
            time, xt, ut = self.fm.sample_location_and_conditional_flow(noise, action)
        batch['sampling_timestep'], batch['xt'], batch['ut'] = time.reshape(b, t), xt.reshape(b, t, d), ut.reshape(b, t, d)
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
        b, t, d = ut.shape # b, 128, 32*22
        mask = batch.get('action_chunk_mask', torch.ones_like(ut)) # b, 128, 32
        action_dim = d // mask.shape[-1]
        # expand mask to the same shape as ut
        mask = mask.unsqueeze(-1).expand(-1, -1, -1, action_dim).reshape(b, t, d)
        mask_ut = ut * mask
        mask_vt = vt * mask
        loss = ((mask_vt - mask_ut) ** 2).sum(-1).mean()
        result = {
            "loss": loss,
        }
        return result
