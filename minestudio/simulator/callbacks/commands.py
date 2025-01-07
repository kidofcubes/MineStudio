'''
Date: 2024-11-11 19:31:53
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2025-01-07 03:19:46
FilePath: /MineStudio/minestudio/simulator/callbacks/commands.py
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
                break
        return CommandsCallback(commands)
    
    def __init__(self, commands):
        super().__init__()
        self.commands = commands
    
    def after_reset(self, sim, obs, info):
        for command in self.commands:
            obs, reward, done, info = sim.env.execute_cmd(command)
        obs, info = sim._wrap_obs_info(obs, info)
        return obs, info

if __name__ == '__main__':
    yaml_file = '/home/caishaofei/tmpdir/MineStudio/task_configs/debug_task/build_gate.yaml'
    commands_callback = CommandsCallback.create_from_conf(yaml_file)
    print(commands_callback)
    
    