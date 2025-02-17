'''
Date: 2025-01-18 11:40:44
LastEditors: muzhancun muzhancun@126.com
LastEditTime: 2025-01-18 11:42:03
FilePath: /MineStudio/minestudio/utils/flow_lib/mlp.py
'''

"""
This implementation is based on https://github.com/lucidrains/autoregressive-diffusion-pytorch/blob/main/autoregressive_diffusion_pytorch/autoregressive_diffusion.py.
"""

import torch
from torch import nn, pi
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from einops import rearrange, repeat, reduce, pack, unpack

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

class AdaptiveLayerNorm(Module):
    def __init__(
        self,
        dim,
        dim_condition = None
    ):
        super().__init__()
        dim_condition = default(dim_condition, dim)

        self.ln = nn.LayerNorm(dim, elementwise_affine = False)
        self.to_gamma = nn.Linear(dim_condition, dim, bias = False)
        nn.init.zeros_(self.to_gamma.weight)

    def forward(self, x, *, condition):
        normed = self.ln(x)
        gamma = self.to_gamma(condition)
        return normed * (gamma + 1.)

class LearnedSinusoidalPosEmb(Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class MLP(nn.Module):
    def __init__(
        self,
        dim_cond,
        dim_input,
        depth = 3,
        width = 512,
        dropout = 0.
    ):
        super().__init__()
        layers = ModuleList([])
        self.to_time_emb = nn.Sequential(
            LearnedSinusoidalPosEmb(dim_cond),
            nn.Linear(dim_cond + 1, dim_cond),
        )

        for _ in range(depth):

            adaptive_layernorm = AdaptiveLayerNorm(
                dim_input,
                dim_condition = dim_cond
            )

            block = nn.Sequential(
                nn.Linear(dim_input, width),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(width, dim_input)
            )

            block_out_gamma = nn.Linear(dim_cond, dim_input, bias = False)
            nn.init.zeros_(block_out_gamma.weight)

            layers.append(ModuleList([
                adaptive_layernorm,
                block,
                block_out_gamma
            ]))

        self.layers = layers

    def forward(
        self,
        noised,
        *,
        times,
        cond
    ):
        ndim = noised.ndim
        if ndim > 2:
            b, t, _ = noised.shape
            noised = rearrange(noised, 'b t ... -> (b t) ...')
            cond = rearrange(cond, 'b t ... -> (b t) ...')
        if times.ndim == 0:
            times = times.unsqueeze(0)
        else:
            times = rearrange(times, 'b t -> (b t)')
        assert noised.ndim == 2

        time_emb = self.to_time_emb(times)
        cond = F.silu(time_emb + cond)

        denoised = noised

        for adaln, block, block_out_gamma in self.layers:
            residual = denoised
            denoised = adaln(denoised, condition = cond)

            block_out = block(denoised) * (block_out_gamma(cond) + 1.)
            denoised = block_out + residual
        if ndim > 2:
            denoised = rearrange(denoised, '(b t) ... -> b t ...', b = b, t = t)
        return denoised
