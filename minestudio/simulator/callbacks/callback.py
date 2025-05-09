'''
Date: 2025-01-06 17:32:04
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-05-09 14:54:09
FilePath: /MineStudio/minestudio/simulator/callbacks/callback.py
'''
import os
import yaml
import random
from typing import Dict, List, Tuple, Union, Sequence, Mapping, Any, Optional, Literal
class MinecraftCallback:
    
    def load_data_from_conf(source: Union[str, Dict]) -> Dict:
        """
        source can be a yaml file or a dict. 
        """
        if isinstance(source, Dict):
            data = source
        else:
            assert os.path.exists(source), f"File {source} not exists."
            with open(source, 'r') as f:
                data = yaml.safe_load(f)
        return data

    def create_from_conf(yaml_file: Union[str, Dict]):
        return None

    def before_step(self, sim, action):
        return action

    def before_reset(self, sim, reset_flag: bool) -> bool: # whether need to call env reset
        return reset_flag
    
    def after_reset(self, sim, obs, info):
        return obs, info

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        return obs, reward, terminated, truncated, info

    def before_close(self, sim):
        return

    def after_close(self, sim):
        return

    def before_render(self, sim, image):
        return image

    def after_render(self, sim, image):
        return image


class Compose(MinecraftCallback):
    
    def __init__(self, callbacks:list, options:int=-1):
        self.callbacks = callbacks
        self.options = options
        self.activate_callbacks = []

    def before_reset(self, sim, reset_flag: bool) -> bool:
        if self.options == -1:
            self.activate_callbacks = self.callbacks
        else:
            assert 0 <= self.options <= len(self.callbacks), f"{self.options}"
            self.activate_callbacks = random.sample(self.callbacks, k=self.options)
        for callback in self.activate_callbacks:
            reset_flag = callback.before_reset(sim, reset_flag)
        return reset_flag

    def after_reset(self, sim, obs, info):
        for callback in self.activate_callbacks:
            obs, info = callback.after_reset(sim, obs, info)
        return obs, info

    def before_step(self, sim, action):
        for callback in self.activate_callbacks:
            action = callback.before_step(sim, action)
        return action

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        for callback in self.activate_callbacks:
            obs, reward, terminated, truncated, info = callback.after_step(sim, obs, reward, terminated, truncated, info)
        return obs, reward, terminated, truncated, info

    def before_close(self, sim):
        for callback in self.activate_callbacks:
            callback.before_close(sim)
        return

    def after_close(self, sim):
        for callback in self.activate_callbacks:
            callback.after_close(sim)
        return

    def before_render(self, sim, image):
        for callback in self.activate_callbacks:
            image = callback.before_render(sim, image)
        return image

    def after_render(self, sim, image):
        for callback in self.activate_callbacks:
            image = callback.before_render(sim, image)
        return image

