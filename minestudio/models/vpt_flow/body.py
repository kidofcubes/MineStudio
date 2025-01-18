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
class VPTFlowPolicy(MineGenerativePolicy, PyTorchModelHubMixin):

    def __init__(self, policy_kwargs, action_kwargs, **kwargs):
        super().__init__(**kwargs)
        self.net = MinecraftPolicy(**policy_kwargs)
        self.action_head = MLP(**action_kwargs)
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
        noise = input["noise"]
        vt = self.action_head(noise, times=t, cond=pi_latent)
        return {"vt": vt, "pi_logits": pi_latent, "vpred": vf_latent}, state_out
    
    def sample(self, input, state_in = None, **kwargs):
        (pi_latent, vf_latent), state_out = self.net(input, state_in)
        sampling_timestep = kwargs.get("sampling_timestep", 10)
        times = torch.linspace(0., 1., sampling_timestep, device=self.device)
        traj = torchdiffeq.odeint(
            lambda t, x: self.action_head(x, times=t, cond=pi_latent),
            input["noise"],
            times,
            atol=1e-4,
            rtol=1e-4,
            method="dopri5",
        )
        return traj[-1].clip(-1, 1), state_out

def load_vpt_flow_policy(ckpt_path: str) -> VPTFlowPolicy:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = VPTFlowPolicy.load_from_checkpoint(ckpt_path)
    model.load_state_dict(ckpt["state_dict"])
    return model

if __name__ == "__main__":
    """
    for debugging purpose
    """
    from tqdm import tqdm
    from minestudio.data.minecraft.callbacks import (
        ImageKernelCallback, 
        ActionKernelCallback, VectorActionKernelCallback, 
        MetaInfoKernelCallback, 
        SegmentationKernelCallback
    )
    from minestudio.data import RawDataModule
    from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

    data_module = RawDataModule(
        data_params=dict(
            dataset_dirs=[
                '/nfs-shared-2/data/contractors-new/dataset_10xx', 
            ],
            modal_kernel_callbacks=[
                ImageKernelCallback(frame_width=128, frame_height=128, enable_video_aug=False), 
                # ActionKernelCallback(),
                VectorActionKernelCallback(action_chunk_size=32), 
                # MetaInfoKernelCallback(),
            ],
            win_len=128, 
            split_ratio=0.9,
            shuffle_episodes=True,
        ),
        batch_size=3,
        num_workers=0,
        prefetch_factor=None,
        episode_continuous_batch=True,
    )
    data_module.setup()
    loader = data_module.train_dataloader()
    batch = next(iter(loader))

    model = VPTFlowPolicy(policy_kwargs={
        "attention_heads": 16,
        "attention_mask_style": "clipped_causal",
        "attention_memory_size": 256,
        "diff_mlp_embedding": True,
        "hidsize": 2048,
        "img_shape": [
        128,
        128,
        3
        ],
        "impala_chans": [
        16,
        32,
        32
        ],
        "impala_kwargs": {
        "post_pool_groups": 1
        },
        "impala_width": 8,
        "init_norm_kwargs": {
        "batch_norm": False,
        "group_norm_groups": 1
        },
        "n_recurrence_layers": 4,
        "only_img_input": True,
        "pointwise_ratio": 4,
        "pointwise_use_activation": False,
        "recurrence_is_residual": True,
        "recurrence_type": "transformer",
        "timesteps": 128,
        "use_pointwise_layer": True,
        "use_pre_lstm_ln": False
    },
    action_kwargs={"dim_cond": 2048, "dim_input": 32*22})
    b, t, d = batch['action'].shape
    action = batch['action'].reshape(b*t, d)
    noise = torch.rand_like(action)
    fm = ConditionalFlowMatcher(sigma=0.0)
    time, xt, ut = fm.sample_location_and_conditional_flow(noise, action)
    print(t, xt.shape, ut.shape)
    print(xt.device, ut.device)
    batch['sampling_timestep'], batch['xt'], batch['ut'] = time.reshape(b, t), xt.reshape(b, t, d), ut.reshape(b, t, d)
    batch['noise'] = noise.reshape(b, t, d)

    result, _ = model(batch, state_in=None)
    print(result["vt"].shape)
