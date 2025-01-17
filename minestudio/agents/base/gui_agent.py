from minestudio.agents.base.gui_scripts import CraftScript, SmeltScript, EquipScript
from typing import List, Dict, Union
from minestudio.simulator import MinecraftSim

class GuiAgent:
    def __init__(self, env: MinecraftSim):
        self.env = env
        self.smelt_agent = SmeltScript(env)
        self.craft_agent = CraftScript(env)
        self.equip_agent = EquipScript(env)

    def run(self, skill, object:Union[Dict, str]):
        if skill == 'craft': # support multiple items
            # assert isinstance(object, Dict), "object for craft must be a dictionary!"
            if isinstance(object, str):
                done, info = self.craft_agent.crafting(object, 1)
            elif isinstance(object, Dict):
                obj_name, obj_number = next(iter(object.items()))
                done, info = self.craft_agent.crafting(obj_name, obj_number)
            else:
                raise ValueError(f"Invalid object for craft: {object}!")
            return done, info
        elif skill == 'smelt':
            if isinstance(object, str):
                done, info = self.smelt_agent.smelting(object, 1)
            elif isinstance(object, Dict):
                obj_name, obj_number = next(iter(object.items()))
                done, info = self.smelt_agent.smelting(obj_name, obj_number)
            else:
                raise ValueError(f"Invalid object for smelt: {object}!")
            return done, info
        elif skill == 'equip':
            assert isinstance(object, str), "object for equip must be a string!"
            done, info = self.equip_agent.equip_item(object)
            return done, info
        else:
            raise ValueError(f"Invalid skill: {skill} for GUI Script Agent!")

if __name__ == '__main__':
    import random 
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import RecordCallback, SpeedTestCallback, RewardsCallback, SummonMobsCallback, InitInventoryCallback
    env = MinecraftSim(
        action_type='env',
        obs_size=(128, 128), 
        seed = random.randint(0, 1000000),
        preferred_spawn_biome=random.choice(["forest", "plains"]), 
        callbacks=[
            RecordCallback(record_path='output', fps=30, frame_type="pov"),
            SpeedTestCallback(50),
            InitInventoryCallback(init_inventory=[
                {'slot': 0, 'type': 'oak_log', 'quantity': 64},
                {'slot': 1, 'type': 'cobblestone', 'quantity': 64},
                {'slot': 2, 'type': 'iron_ore', 'quantity': 64},
                {'slot': 4, 'type': 'diamond', 'quantity': 64},
                {'slot': 5, 'type': 'iron_axe', 'quantity': 1},
                ]),
            # SummonMobsCallback([{'name': 'cow', 'number': 10, 'range_x': [-5, 5], 'range_z': [-5, 5]}]),
        ]
    )
    obs, info = env.reset()
    # script = CraftScript(env)
    # done, info = script.crafting('stick', 2)
    # print(done, info)
    # done, info = script.crafting('crafting_table', 1)
    # print(done, info)
    # done, info = script.crafting('wooden_pickaxe', 1)
    # print(done, info)
    # # write_video('crafting.mp4', worker.outframes)
    # env.close()
    agent = GuiAgent(env)
    done, info = agent.run('equip', 'iron_axe')
    print("equip iron axe:", done, info)
    done, info = agent.run('craft', {'oak_planks': 24})
    print("craft oak planks:", done, info)
    done, info = agent.run('craft', {'crafting_table': 1})
    print("craft crafting table:", done, info)
    done, info = agent.run('craft', {'stick': 8})
    print("craft stick:", done, info)
    done, info = agent.run('craft', {'wooden_pickaxe': 1})
    print("craft wooden pickaxe:", done, info)
    done, info = agent.run('craft', {'stone_pickaxe': 1})
    print("craft stone pickaxe:", done, info)
    done, info = agent.run('craft', {'furnace': 1})
    print("craft furnace:", done, info)
    done, info = agent.run('smelt', {'iron_ingot': 3})
    print("smelt iron ingot:", done, info)
    done, info = agent.run('craft', {'iron_pickaxe': 1})
    print("craft iron pickaxe:", done, info)
    done, info = agent.run('craft', {'diamond_hoe': 1})
    print("craft diamond hoe:", done, info)
    env.close()
