'''
Date: 2024-11-11 17:26:22
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2024-11-12 00:12:08
FilePath: /MineStudio/minestudio/simulator/callbacks/summon_mobs.py
'''

from minestudio.simulator.callbacks.callback import MinecraftCallback
from typing import Union
import random
import json
import re
from pathlib import Path
from copy import deepcopy
from time import sleep
from rich import console

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

DISTRACTION_LEVEL = {"zero":[0],"one":[1],
                     "easy":range(3,7),"middle":range(7,16),"hard":range(16,35),
                     "normal":range(0,35)}



class InitInventoryCallback(MinecraftCallback):
    
    def __init__(self, init_inventory:dict,distraction_level:Union[list,str]=[0]) -> None:
        """
        Examples:
            init_inventory = [{
                    slot: 0
                    type: "oak_planks"
                    quantity: 64  # supporting ">...",">=...","<...","<=...","==...","...",1
                }]
        """
        self.init_inventory = init_inventory
        self.distraction_level = DISTRACTION_LEVEL.get(distraction_level,[0]) if isinstance(distraction_level,str) else distraction_level
        try:
            json_path = Path(__file__).parents[3] / "assets" / "mc_constants.1.16.json"
            with json_path.open('r', encoding='utf-8') as file:
                data = json.load(file)
            self.items_library = data.get("items", [None])[1:] 
        except Exception as e:
            print(f"Failed to load or parse the JSON file: {e}")
            self.items_library = [] 
        with open(str(Path(__file__).parent.parent.parent.parent/"assets"/"mc_constants.1.16.json"), 'r', encoding='utf-8') as file:
            data = json.load(file)
        self.items_library = data.get("items")[1:]
        self.items_names = [item.get("type") for item in self.items_library]
        self.items_name2idx = {item.get("type"):idx for idx,item in enumerate(self.items_library)}

        
    def after_reset(self, sim, obs, info):
        chats = []
        visited_slots = set()
        uncertain_slots = [] 
        init_inventory = []
        for slot_info in self.init_inventory:
            slot = slot_info["slot"]
            if slot == "random":
                uncertain_slots.append(deepcopy(slot_info))
                continue
            visited_slots.add(int(slot))
            init_inventory.append(slot_info)
        unvisited_slots = set(range(MIN_SLOT_IDX, MAX_INVENTORY_IDX + 1)) - visited_slots
        
        # settle uncertain slots
        for uncertain_slot in uncertain_slots:
            slot = int(random.choice(list(unvisited_slots)))
            unvisited_slots.remove(slot)
            uncertain_slot["slot"] = slot
            init_inventory.append(uncertain_slot)
        
        # settle distraction slot
        distraction_num = min(random.choice(self.distraction_level),len(unvisited_slots))
        for _ in range(distraction_num):
            item_type = random.choice(self.items_names)
            slot = int(random.choice(list(unvisited_slots)))
            unvisited_slots.remove(slot)
            init_inventory.append({
                "slot":slot,
                "type":item_type,
                "quantity":"random",
            })
        self.slot_num = len(init_inventory)
        for item_dict in init_inventory:
            slot = item_dict["slot"]
            mc_slot =self._map_slot_number_to_cmd_slot(slot)
            item_type = item_dict["type"]
            assert item_type in self.items_names
            item_quantity = self._item_quantity_parser(item_dict["quantity"],int(self.items_library[self.items_name2idx[item_type]]["stackSize"]))
            chat = f"/replaceitem entity @p {mc_slot} minecraft:{item_type} {item_quantity}"
            if "metadata" in item_dict:
                chat += f" {item_dict['metadata']}"
            chats.append(chat)
        for chat in chats:
            obs, reward, done, info = sim.env.execute_cmd(chat)
        obs, info = sim._wrap_obs_info(obs, info)
        init_flag = False
        
        for _ in range(self.slot_num*2):
            action = sim.env.noop_action()
            obs, reward, done, info = sim.env.step(action)
            init_flag = self._check(obs)
            if init_flag:
                break
        if not init_flag:
            console.Console().log("[red]can't set up init inventory[/red]")
        return obs, info
    
    
    def _map_slot_number_to_cmd_slot(self,slot_number: Union[int,str]) -> str:
        slot_number = int(slot_number)
        assert MIN_SLOT_IDX <= slot_number <= MAX_SLOT_IDX, f"exceed slot index range:{slot_number}"
        if slot_number in {0, 40}:
            return f"weapon.{SLOT_IDX2NAME[slot_number]}"
        elif 36 <= slot_number <= 39:
            return f"armor.{SLOT_IDX2NAME[slot_number]}"
        elif 1 <= slot_number <= 8:
            return f"hotbar.{slot_number}"
        else:
            return f"inventory.{slot_number - 9}"

    def _item_quantity_parser(self,item_quantity: Union[int,str],max_items_num,one_p:float=0.7) -> int:
        """Function to parse item quantity from either an integer or a string command
        """
        
        if isinstance(item_quantity,str):
            
            candidate_nums=set(range(MIN_ITEMS_NUM, max_items_num + 1))
            
            if item_quantity == "random":
                one_flag = random.choices([True, False], weights=[one_p, 1 - one_p], k=1)[0]
                if one_flag:
                    return 1
                else:
                    return random.choice(list(candidate_nums))
            
            
            item_quantity_commands = item_quantity.split(",")
        
            def apply_command(op, val):
                """Apply a command based on the operator and value provided in the string 
                """
                return {
                    '<': set(range(MIN_ITEMS_NUM,val)),
                    '<=': set(range(MIN_ITEMS_NUM,val+1)),
                    '>': set(range(val+1,max_items_num+1)),
                    '>=': set(range(val,max_items_num+1)),
                    '==': {val}
                }[op]
        
            for item_quantity_command in item_quantity_commands:
                match = re.search(r'([<>]=?|==)\s*(\d+)', item_quantity_command.strip()) #matching "<...", ">...", "<=...", ">=...", "==..."
                if match:
                    operator, number = match.groups()
                    number = int(number)
                    candidate_nums &= apply_command(operator,number)
            if candidate_nums:
                item_quantity = random.choice(list(candidate_nums))
            
        elif not isinstance(item_quantity, int):
            raise TypeError("Input must be an integer or a string representing conditions")

        return item_quantity
    
    def _check(self,obs):
        "check whether it set up the init inventory"
        current_slot_num = 0
        for slot_dict in obs["inventory"].values():
            if slot_dict["type"] != "none":
                current_slot_num+=1
        if current_slot_num >= self.slot_num:
            return True
        return False
    
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
            InitInventoryCallback([
                {"slot": 0,
                "type": "oak_planks",
                "quantity":1,},
                {"slot": 1,
                "type": "oak_planks",
                "quantity":">2",},
                {"slot": 2,
                "type": "oak_planks",
                "quantity":"<12,>10",},
                {"slot": "random",
                "type": "oak_planks",
                "quantity":"random",},
            ],distraction_level="normal")
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