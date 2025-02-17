'''
Date: 2024-11-11 19:31:53
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-01-22 23:26:48
FilePath: /MineStudio/minestudio/simulator/callbacks/prev_action.py
'''
import os
import yaml
import numpy as np
from typing import Dict, List, Tuple, Union, Sequence, Mapping, Any, Optional, Literal
from minestudio.simulator.callbacks.callback import MinecraftCallback
from minestudio.utils.register import Registers

# @Registers.simulator_callback.register
class PrevActionCallback(MinecraftCallback):
    
    def __init__(self):
        super().__init__()
        self.prev_action = None

    def before_step(self, sim, action):
        self.prev_action = action
        return action

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        obs["env_prev_action"] = self.prev_action
        return obs, reward, terminated, truncated, info
    
    def after_reset(self, sim, obs, info):
        obs["env_prev_action"] = {
            "attack": np.array(0), "use": np.array(0), "inventory": np.array(0), 
            "forward": np.array(0), "back": np.array(0), "left": np.array(0), "right": np.array(0), 
            "sneak": np.array(0), "sprint": np.array(0), "jump": np.array(0), 
            "hotbar.1": np.array(0), "hotbar.2": np.array(0), "hotbar.3": np.array(0), "hotbar.4": np.array(0), "hotbar.5": np.array(0), 
            "hotbar.6": np.array(0), "hotbar.7": np.array(0), "hotbar.8": np.array(0), "hotbar.9": np.array(0), 
            "camera": np.array([0, 0]),
        }
        return obs, info
