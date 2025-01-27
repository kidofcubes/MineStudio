'''
Date: 2025-01-18 11:35:59
LastEditors: muzhancun 2100017790@stu.pku.edu.cn
LastEditTime: 2025-01-27 13:48:02
FilePath: /MineStudio/minestudio/models/vpt_diffusion/body.py
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

from diffusers import DDPMScheduler
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
        pred = self.action_head(noisy_x, times=t, cond=pi_latent)
        return {"pred": pred, "pi_logits": pi_latent, "vpred": vf_latent}, state_out
    
    def sample(self, input, state_in = None, **kwargs):
        B, T = input["image"].shape[:2]
        state_in = self.initial_state(B) if state_in is None else state_in
        first = torch.tensor([[False]], device=self.device).repeat(B, T)
        (pi_latent, vf_latent), state_out = self.net(input, state_in, context={"first": first})
        x = input["noise"]
        for i, t in enumerate(self.scheduler.timesteps):
            with torch.no_grad():
                residual = self.action_head(x, times=t.to(self.device), cond=pi_latent)
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
class VPTDictDiffusionPolicy(MineGenerativePolicy, PyTorchModelHubMixin):
    """return a dict of camera and button action"""
    def __init__(self, policy_kwargs, camera_kwargs, button_kwargs, scheduler_kwargs, **kwargs):
        super().__init__(**kwargs)
        self.net = MinecraftPolicy(**policy_kwargs)
        self.camera_head = MLP(**camera_kwargs)
        self.button_head = MLP(**button_kwargs)
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
        camera_t = input["camera_timesteps"]
        button_t = input["button_timesteps"]
        noisy_camera = input["noisy_camera"]
        noisy_button = input["noisy_button"]
        camera = self.camera_head(noisy_camera, times=camera_t, cond=pi_latent)
        button = self.button_head(noisy_button, times=button_t, cond=pi_latent)
        return {"camera": camera, "button": button, "pi_logits": pi_latent, "vpred": vf_latent}, state_out

    def sample(self, input, state_in = None, **kwargs):
        B, T = input["image"].shape[:2]
        state_in = self.initial_state(B) if state_in is None else state_in
        first = torch.tensor([[False]], device=self.device).repeat(B, T)
        (pi_latent, vf_latent), state_out = self.net(input, state_in, context={"first": first})
        camera_x = input["camera_noise"]
        button_x = input["button_noise"]
        for i, t in enumerate(self.scheduler.timesteps):
            with torch.no_grad():
                camera_residual = self.camera_head(camera_x, times=t.to(self.device), cond=pi_latent)
                button_residual = self.button_head(button_x, times=t.to(self.device), cond=pi_latent)
            camera_x = self.scheduler.step(camera_residual, t, camera_x).prev_sample
            button_x = self.scheduler.step(button_residual, t, button_x).prev_sample
        camera_x = camera_x.clip(-1, 1)
        button_x = button_x.clip(-1, 1)
        return {"camera": camera_x, "button": button_x}, state_out

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
    scheduler_kwargs = ckpt["hyper_parameters"]["scheduler"]
    model = VPTDiffusionPolicy(policy_kwargs, action_kwargs, scheduler_kwargs)
    print(f"Policy kwargs: {policy_kwargs}")
    print(f"Action kwargs: {action_kwargs}")
    print(f"Scheduler kwargs: {scheduler_kwargs}")
    state_dict = {k.replace('mine_policy.', ''): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(state_dict)
    return model

@Registers.model_loader.register
def load_vpt_dict_diffusion_policy(ckpt_path: str) -> VPTDictDiffusionPolicy:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    policy_kwargs = ckpt["hyper_parameters"]["policy"]
    camera_kwargs = ckpt["hyper_parameters"]["camera"]
    button_kwargs = ckpt["hyper_parameters"]["button"]
    scheduler_kwargs = ckpt["hyper_parameters"]["scheduler"]
    model = VPTDictDiffusionPolicy(policy_kwargs, camera_kwargs, button_kwargs, scheduler_kwargs)
    print(f"Policy kwargs: {policy_kwargs}")
    print(f"Camera kwargs: {camera_kwargs}")
    print(f"Button kwargs: {button_kwargs}")
    print(f"Scheduler kwargs: {scheduler_kwargs}")
    state_dict = {k.replace('mine_policy.', ''): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(state_dict)
    return model
