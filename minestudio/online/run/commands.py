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
    
    def __init__(self, equip_commands, summon_commands, other_summon_commands):
        super().__init__()
        self.summon_commands = summon_commands
        self.equip_commands = equip_commands
        self.other_summon_commands = other_summon_commands
        
    def after_reset(self, sim, obs, info):
        # commands = ['/summon minecraft:sheep ~ ~ ~-1', '/summon minecraft:sheep ~ ~ ~']
        commands = []
        
        k = random.randint(1, 5)
        commands.extend(random.sample(self.summon_commands, k))
        
        n = random.randint(1, 2)
        commands.extend(random.sample(self.other_summon_commands, n))
        commands.append(random.choice(self.equip_commands))

        ### debug!!!!
        print('############current commands###########', commands)

        # Execute each command
        for command in commands:
            obs, reward, done, info = sim.env.execute_cmd(command)

        # Wrap observations and info
        obs, info = sim._wrap_obs_info(obs, info)
        return obs, info



