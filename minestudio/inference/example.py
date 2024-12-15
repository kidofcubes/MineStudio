'''
Date: 2024-11-14 19:42:09
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-12-15 11:35:40
FilePath: /MineStudio/minestudio/inference/example.py
'''
import torch
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RecordCallback, SpeedTestCallback
from minestudio.models import VPTPolicy, load_vpt_policy

if __name__ == '__main__':
    
    policy = load_vpt_policy(
        model_path="/nfs-shared/jarvisbase/pretrained/foundation-model-2x.model",
        # weights_path="/nfs-shared/jarvisbase/pretrained/foundation-model-1x.weights", 
        # weights_path="/nfs-shared/jarvisbase/pretrained/rl-from-early-game-2x.weights",
        weights_path="/nfs-shared/jarvisbase/pretrained/bc-early-game-2x.weights", 
    ).to("cuda")
    policy.eval()
    
    env = MinecraftSim(
        obs_size=(128, 128), 
        preferred_spawn_biome="plains", 
        callbacks=[
            RecordCallback(record_path="./output", fps=30, frame_type="pov"),
            SpeedTestCallback(50),
        ]
    )
    memory = None
    obs, info = env.reset()
    with torch.inference_mode():
        for i in range(1200):
            action, memory = policy.get_action(obs, memory, input_shape='*')
            obs, reward, terminated, truncated, info = env.step(action)
    env.reset()
    print("Resetting the environment")
    for i in range(1200):
        action, memory = policy.get_action(obs, memory, input_shape='*')
        obs, reward, terminated, truncated, info = env.step(action)
    env.close()