'''
Date: 2024-11-11 19:31:53
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2024-11-12 00:11:52
FilePath: /MineStudio/minestudio/simulator/callbacks/commands.py
'''

import json
import random
from minestudio.simulator.callbacks.callback import MinecraftCallback

file_paths = ["MineStudio/minestudio/online/run/equip_main.json", \
              "MineStudio/minestudio/online/run/equip_off.json", \
              "MineStudio/minestudio/online/run/give.json", \
              "MineStudio/minestudio/online/run/summon.json"] 

def load_commands(file_paths):
    commands = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()  
                if line:
                    commands.append(line)
    return commands

# 我可以实现别的函数吗，怎么传进去train 还是 test 还是第几轮的train呢
class CommandsCallback(MinecraftCallback):
    
    def __init__(self, summon_commands, equip_commands):
        super().__init__()
        self.summon_commands = summon_commands
        self.equip_commands = equip_commands
        
    def after_reset(self, sim, obs, info):
        commands = []
        commands.append(random.choice(self.summon_commands))
        commands.append(random.choice(self.equip_commands))
        print('############current commands###########', commands)
        obs, reward, done, info = sim.env.execute_cmd(commands)
        obs, info = sim._wrap_obs_info(obs, info)
        return obs, info



