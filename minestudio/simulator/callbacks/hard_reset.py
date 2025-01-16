'''
Date: 2024-11-11 16:15:32
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-01-16 23:45:32
FilePath: /MineStudio/minestudio/simulator/callbacks/hard_reset.py
'''
import random
import numpy as np
from typing import List
from minestudio.simulator.callbacks.callback import MinecraftCallback
from minestudio.utils.register import Registers


@Registers.simulator_callback.register
class HardResetCallback(MinecraftCallback):

    def create_from_conf(source):
        data = MinecraftCallback.load_data_from_conf(source)
        if 'spawn_positions' in data:
            return HardResetCallback(data['spawn_positions'])
        else:
            return None

    def __init__(self, spawn_positions: List):
        super().__init__()
        """
        position is a list of {
            "seed": int,
            "position": [x, z, y], 
        }
        """
        self.spawn_positions = spawn_positions

    def before_reset(self, sim, reset_flag):
        self.position = random.choice(self.spawn_positions)
        sim.env.seed(self.position['seed'])
        return True

    def after_reset(self, sim, obs, info):
        x, z, y = self.position["position"]
        obs, _, done, info = sim.env.execute_cmd(f"/tp @a {x} {z} {y}")
        for _ in range(50): 
            action = sim.env.action_space.no_op()
            obs, reward, done, info = sim.env.step(action)
        obs, info = sim._wrap_obs_info(obs, info)
        return obs, info