'''
Date: 2024-12-25 23:39:41
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2024-12-30 14:27:30
FilePath: /MineStudio/minestudio/utils/__init__.py
'''
from .register import Register, Registers
from .temp import get_mine_studio_dir


# A utility function to get the best compute device for PyTorch. 
# Preferring cuda, then mps, ...  and finally cpu.
def get_compute_device():
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device
