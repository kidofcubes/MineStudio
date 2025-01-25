'''
Date: 2025-01-18 11:35:59
LastEditors: muzhancun muzhancun@126.com
LastEditTime: 2025-01-18 14:06:49
FilePath: /MineStudio/minestudio/models/vpt_flow/body.py
'''
import os
import pickle
import gymnasium
import numpy as np
import torch
import torch as th
from torch import nn
from torch.nn import functional as F
from pathlib import Path
from copy import deepcopy
from typing import List, Dict, Optional, Callable, Union, Tuple, Any
import torchdiffeq

from huggingface_hub import PyTorchModelHubMixin
from minestudio.utils.vpt_lib.impala_cnn import ImpalaCNN
from minestudio.utils.vpt_lib.util import FanInInitReLULayer, ResidualRecurrentBlocks
from minestudio.utils.flow_lib.mlp import MLP
from minestudio.models.base_policy import MineGenerativePolicy
from minestudio.online.utils import auto_stack, auto_to_torch
from minestudio.utils.register import Registers
from minestudio.models.vpt import ImgPreprocessing, ImgObsProcess, MinecraftPolicy
        
@Registers.model.register
class VPTDiffusionPolicy(MineGenerativePolicy, PyTorchModelHubMixin):

    def __init__(self, policy_kwargs, action_kwargs, scheduler_kwargs, **kwargs):
        super().__init__(**kwargs)
        self.net = MinecraftPolicy(**policy_kwargs)
        self.action_head = MLP(**action_kwargs)
        self.scheduler = DDPMScheduler(**scheduler_kwargs)
        self.cached_init_states = dict()

    def initial_state(self, batch_size: int=None):
        if batch_size is None:
            return [t.squeeze(0).to(self.device) for t in self.net.initial_state(1)]
        else:
            if batch_size not in self.cached_init_states:
                self.cached_init_states[batch_size] = [t.to(self.device) for t in self.net.initial_state(batch_size)]
            return self.cached_init_states[batch_size]

    def forward(self, input, state_in, **kwargs):
        B, T = input["image"].shape[:2]
        first = torch.tensor([[False]], device=self.device).repeat(B, T)
        state_in = self.initial_state(B) if state_in is None else state_in

        #input: 1, 128, 128, 128, 3
        #first: 1, 128
        # state_in[0]: 1, 1, 1, 128
        # state_in[1]: 1, 1, 128, 128
        try:
            (pi_latent, vf_latent), state_out = self.net(input, state_in, context={"first": first})
        except Exception as e:
            import ray
            ray.util.pdb.set_trace()
        t = input["sampling_timestep"]
        noisy_x = input["noisy_x"]
        pred = self.action_head(noise, times=t, cond=pi_latent)
        return {"pred": pred, "pi_logits": pi_latent, "vpred": vf_latent}, state_out
    
    def sample(self, input, state_in = None, **kwargs):
        B, T = input["image"].shape[:2]
        state_in = self.initial_state(B) if state_in is None else state_in
        first = torch.tensor([[False]], device=self.device).repeat(B, T)
        (pi_latent, vf_latent), state_out = self.net(input, state_in, context={"first": first})
        x = input["noise"]
        for i, t in enumerate(self.scheduler.timesteps):
            with torch.no_grad():
                residual = self.action_head(x, times=t, cond=pi_latent)
            x = self.scheduler.step(residual, t, x).prev_sample
        x = x.clip(-1, 1)
        return x, state_out

    def get_state_out(self, input, state_in = None, **kwargs):
        B, T = input["image"].shape[:2]
        state_in = self.initial_state(B) if state_in is None else state_in
        first = torch.tensor([[False]], device=self.device).repeat(B, T)
        (pi_latent, vf_latent), state_out = self.net(input, state_in, context={"first": first})
        return state_out

@Registers.model_loader.register
def load_vpt_diffusion_policy(ckpt_path: str) -> VPTDiffusionPolicy:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    policy_kwargs = ckpt["hyper_parameters"]["policy"]
    action_kwargs = ckpt["hyper_parameters"]["action"]
    model = VPTFlowPolicy(policy_kwargs, action_kwargs)
    print(f"Policy kwargs: {policy_kwargs}")
    print(f"Action kwargs: {action_kwargs}")
    state_dict = {k.replace('mine_policy.', ''): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(state_dict)
    return model