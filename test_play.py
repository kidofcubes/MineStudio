'''
Date: 2025-01-07 14:43:24
LastEditors: muzhancun muzhancun@126.com
LastEditTime: 2025-01-08 00:55:03
FilePath: /MineStudio-Nano/project/env_wrapper.py
'''
import minestudio
from minestudio.simulator.callbacks import MinecraftCallback
from rich import print
from typing import Any

MINERL_ACTION_TO_KEYBOARD = {
    "attack":    "*",
    "back":      "S",
    "forward":   "W",
    "left":     "A",
    "right":     "D",
    "jump":      "#",
}

class NanoPlayCallback(MinecraftCallback):
    """
        @__init__: initialize mineflayer param, nano serial, and other parameters
        @after_reset: publish the game public, connect the bot. 
        @before_step: read action from nano, if any
    """
    def __init__(self, bot_name, port, nano_ser):
        super().__init__()
        self.bot_name = bot_name
        self.port = port
        self.nano_ser = nano_ser

    def after_reset(self, sim, obs, info):
        obs, _, _, info = sim.env.execute_cmd(f"/publish {self.port}")
        print(f"Published game on port {self.port}")

        for _ in range(5):
            obs, _, _, _, info = sim.step(sim.noop_action())

        return obs, info

    def before_step(self, sim, action):
        # read action from nano, if any
        try:
            line = self.nano_ser.readline().decode("utf-8")
            action: dict[str, Any] = {
                name: int(key==line[0]) for name, key in MINERL_ACTION_TO_KEYBOARD.items()
            }
            action["camera"] = [0, 0]
        except:
            pass
        return action
    
if __name__ == "__main__":
    from minestudio.simulator import MinecraftSim

    sim = MinecraftSim(
        obs_size=(224, 224),
        action_type="env",
        callbacks=[NanoPlayCallback("bot_name", 8110, "nano_serial")],
    )

    obs, info = sim.reset()

    terminated = False

    import time
    time.sleep(30)
    obs, _, _, info = sim.env.execute_cmd("/gamemode spectator Zhancun")
    print("Switched to spectator mode")

    while not terminated:
        action = None
        obs, reward, terminated, truncated, info = sim.step(sim.noop_action())

    sim.close()
