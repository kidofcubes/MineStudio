'''
Date: 2025-01-05 22:26:22
LastEditors: limuyao 2200017405@stu.pku.edu.cn
LastEditTime: 2025-01-05 22:26:22
FilePath: /MineStudio/minestudio/simulator/callbacks/init_inventory.py
'''

import minecraft_data # https://github.com/SpockBotMC/python-minecraft-data  Provide easy access to minecraft-data in python
from minestudio.simulator.callbacks import MinecraftCallback,InitInventoryCallback
from typing import Union,Literal
import random
import json
import re
from pathlib import Path
from copy import deepcopy
from time import sleep
from rich import console
import uuid

EQUIP_SLOTS = {
    "mainhand": 0,
    "offhand": 40,
    "head": 39,
    "chest": 38,
    "legs": 37,
    "feet": 36,
}
MIN_SLOT_IDX = 0
MAX_INVENTORY_IDX = 35
MAX_SLOT_IDX = 40
SLOT_IDX2NAME = {v: k for k, v in EQUIP_SLOTS.items()}
MIN_ITEMS_NUM = 0
MAX_ITEMS_NUM = 64

INVENTORY_DISTRACTION_LEVEL = {"zero":[0],"one":[1],
                     "easy":range(3,7),"middle":range(7,16),"hard":range(16,36),
                     "normal":range(0,36)}
EQUIP_DISTRACTION_LEVEL = {
    "zero":[0],
    "normal":range(0,6),
}

class SetInventoryCallback(InitInventoryCallback):
    
    def __init__(self, init_inventory:dict,inventory_distraction_level:Union[list,str]=[0],equip_distraction_level:Union[list,str]=[0], change_frequency:Literal["reset","step"]="reset") -> None:
        """
        Examples:
            init_inventory = [{
                    slot: 0
                    type: "oak_planks"
                    quantity: 64  # supporting ">...",">=...","<...","<=...","==...","...",1
                }]
        """
        super().__init__(init_inventory,inventory_distraction_level,equip_distraction_level)
        assert change_frequency in {"reset","step"}
        self.change_frequency = change_frequency
        
        
    def after_step(self, sim, obs, reward, terminated, truncated, info):
        if self.change_frequency == "step":
            obs, info = self._set_inventory(sim,obs,info)
        return obs, reward, terminated, truncated, info
        
    def after_reset(self, sim, obs, info):
        obs, info = self._set_inventory(sim,obs,info)
        return obs, info
    
    def _sample_inventory(self, init_inventory, visited_slots, unvisited_slots):
        distraction_num = min(random.choice(self.inventory_distraction_level),len(unvisited_slots))
        past_item_type = ""
        for idx in range(distraction_num):
            if idx>0 and random.choices([True,False],[0.35,0.65],k=1)[0]:
                item_type = past_item_type
            else:
                item_type = random.choice(self.items_names)
            past_item_type = item_type
            slot = int(random.choice(list(unvisited_slots)))
            unvisited_slots.remove(slot)
            visited_slots.add(slot)
            init_inventory.append({
                "slot":slot,
                "type":item_type,
                "quantity":"random",
        })
        return init_inventory, visited_slots, unvisited_slots

    def _clean_screen(self, sim, obs, info):
        for _ in range(50):
            action = sim.env.noop_action()
            obs, reward, done, info = sim.env.step(action)
        obs, info = sim._wrap_obs_info(obs, info)
        return obs,info
    
if __name__ == "__main__":

    import numpy as np
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import (
        SpeedTestCallback, 
        RecordCallback, 
        RewardsCallback, 
        TaskCallback,
        FastResetCallback
    )
    sim = MinecraftSim(
        action_type="env",
        callbacks=[
            SpeedTestCallback(50), 
            TaskCallback([
                {'name': 'craft', 'text': 'craft crafting_table'}, 
            ]),
            RecordCallback(record_path="./output", fps=30,record_actions=True,record_infos=True,record_origin_observation=True),
            RewardsCallback([{
                'event': 'craft_item', 
                'objects': ['crafting_table'], 
                'reward': 1.0, 
                'identity': 'craft crafting_table', 
                'max_reward_times': 1, 
            }]),
            FastResetCallback(
                biomes=['mountains'],
                random_tp_range=1000,
            ), 
            SetInventoryCallback([
            ],    
            inventory_distraction_level="normal",
            equip_distraction_level="normal")
        ]
    )
    obs, info = sim.reset()
    action = sim.noop_action()
    action["inventory"] = 1
    obs, reward, terminated, truncated, info = sim.step(action)
    for i in range(30):
        action = sim.noop_action()
        obs, reward, terminated, truncated, info = sim.step(action)
    sim.close()