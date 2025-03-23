'''
Date: 2024-11-11 19:31:53
LastEditors: Muyao 2350076251@qq.com
LastEditTime: 2025-03-18 23:06:21
FilePath: /MineStudio/minestudio/simulator/callbacks/commands.py
'''

from minestudio.simulator.callbacks.commands import CommandsCallback

class TeleportCallback(CommandsCallback):
    
    def __init__(self, x:int,y:int,z:int):
        command = f"/tp {x} {y} {z}"
        super().__init__(commands=[command])
        
    def after_reset(self, sim, obs, info):
        super().after_reset(sim, obs, info)
        for kdx in range(30):
            action = sim.env.noop_action()
            obs, reward, done, info = sim.env.step(action)
        obs, info = sim._wrap_obs_info(obs, info)
        return obs, info