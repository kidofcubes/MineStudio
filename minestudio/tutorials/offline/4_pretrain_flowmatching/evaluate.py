import numpy as np
from minestudio.simulator import MinecraftSim
from functools import partial
from minestudio.models.vpt_flow import VPTFlowPolicy, load_vpt_flow_policy
from minestudio.simulator.callbacks import MinecraftCallback, RecordCallback
import torch

class NoiseCallback(MinecraftCallback):
    def __init__(self, action_shape=704):
        super().__init__()
        self.action_shape = action_shape

    def after_reset(self, sim, obs, info):
        obs['noise'] = torch.randn(self.action_shape)
        # print(obs['noise'].shape)
        return obs, info

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        obs['noise'] = torch.randn(self.action_shape)
        return obs, reward, terminated, truncated, info

if __name__ == '__main__':
    agent = load_vpt_flow_policy("/nfs-shared-2/zhancun/models/flow.ckpt")
    agent.to('cuda:0')

    sim = MinecraftSim(
        obs_size=(128, 128),
        callbacks=[
            NoiseCallback(),
            RecordCallback(record_path='./output', recording=True),
        ],
        action_type = 'env',
        preferred_spawn_biome = "forest", 
    )

    obs, info = sim.reset()

    memory = None

    for i in range(300):
        action, memory = agent.get_action(obs, memory, input_shape='*')
        print(action)
        obs, reward, terminated, truncated, info = sim.step(action)

    sim.close()