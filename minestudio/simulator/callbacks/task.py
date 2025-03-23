'''
Date: 2024-11-11 19:29:45
LastEditors: Muyao 2350076251@qq.com
LastEditTime: 2025-03-18 23:15:12
FilePath: /MineStudio/minestudio/simulator/callbacks/task.py
'''
import random
from minestudio.simulator.callbacks.callback import MinecraftCallback
from rich import print,console

class TaskCallback(MinecraftCallback):
    
    def __init__(self, task_cfg):
        """
        TaskCallback 
        Example:
            task_cfg = [{
                'name': 'chop tree',
                'text': 'chop the tree', 
            }]
        """
        super().__init__()
        self.task_cfg = task_cfg
    
    def after_reset(self, sim, obs, info):
        if self.task_cfg:
            task = random.choice(self.task_cfg)
            console.Console().log(f"Switching to task: {task['name']}.")
            obs["task"] = task
            info["task"] = task
        return obs, info