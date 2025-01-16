'''
Date: 2024-11-11 19:31:53
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-01-16 18:09:12
FilePath: /ROCKET-2/var/nfs-shared/shaofei/nfs-workspace/MineStudio/minestudio/simulator/callbacks/commands.py
'''
import os
import yaml
from typing import Dict, List, Tuple, Union, Sequence, Mapping, Any, Optional, Literal
from minestudio.simulator.callbacks.callback import MinecraftCallback
from minestudio.utils.register import Registers

@Registers.simulator_callback.register
class CommandsCallback(MinecraftCallback):
    
    def create_from_conf(source: Union[str, Dict]):
        data = MinecraftCallback.load_data_from_conf(source)
        available_keys = ['custom_init_commands', 'commands']
        for key in available_keys:
            if key in data:
                commands = data[key]
                return CommandsCallback(commands)
        return None
    
    def __init__(self, commands):
        super().__init__()
        self.commands = commands
    
    def after_reset(self, sim, obs, info):
        for command in self.commands:
            _obs, reward, done, info = sim.env.execute_cmd(command)
            obs.update(_obs)
            info.update(info)
        obs, info = sim._wrap_obs_info(obs, info)
        return obs, info

    def __repr__(self):
        return f"CommandsCallback(commands={self.commands})"
