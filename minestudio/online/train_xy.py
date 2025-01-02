from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RecordCallback, SpeedTestCallback, SummonMobsCallback, MaskActionsCallback, RewardsCallback, JudgeResetCallback, FastResetCallback
from minestudio.models import VPTPolicy, load_vpt_policy
import torch
import os
from minestudio.online.run.commands import CommandsCallback

import numpy as np
base_dir = os.path.dirname(os.path.abspath(__file__))
file_paths = [
    os.path.join(base_dir, "run/equip_main.json"),
    os.path.join(base_dir, "run/equip_off.json"),
    os.path.join(base_dir, "run/give.json")
]
summon_path = [os.path.join(base_dir, "run/sheep_summon.json")]
other_summon_path = [os.path.join(base_dir, "run/other_summon.json")]
def load_commands(file_paths):
    commands = []
    for file_path in file_paths:
        print(file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()  
                if line: 
                    commands.append(line)
    return commands

def find_ckpt_files(directory):
    ckpt_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == "model.ckpt": 
                ckpt_files.append(os.path.join(root, file))
    return ckpt_files


if __name__ == '__main__':
    results = []
    policy = load_vpt_policy(
        model_path="/nfs-shared/jarvisbase/pretrained/foundation-model-2x.model",
        weights_path="/nfs-shared/jarvisbase/pretrained/rl-from-early-game-2x.weights"
    ).to("cuda")

    ckpt_dir = "/scratch2/zhengxinyue/mcu_online/output_1million_4/checkpoints/2024-12-29_02-03-51/4531200"
    ckpt_files = find_ckpt_files(ckpt_dir)
    print(ckpt_files)
    for ckpt_path in ckpt_files:
        print('current ckpt: ',ckpt_path)
        state_dict_model = torch.load(ckpt_path, map_location='cpu')
        policy.load_state_dict(state_dict_model, strict=True)
        print(f'Loaded checkpoint from {ckpt_path}')
        del state_dict_model

        equip_commands = load_commands(file_paths)
        summon_commands = load_commands(summon_path)
        other_summon_commands = load_commands(other_summon_path)

        env = MinecraftSim(
            obs_size=(128, 128), 
            preferred_spawn_biome="plains",
            callbacks=[
                RewardsCallback([{
                    'event': 'kill_entity', 
                    'objects': ['sheep'], 
                    'reward': 5.0, 
                    'identity': 'shoot_sheep', 
                    'max_reward_times': 30, 
                }]),
                CommandsCallback(equip_commands[:100], summon_commands[:100], other_summon_commands[:100]),
                FastResetCallback( 
                    biomes=['plains'],
                    random_tp_range=1000,
                ),
                JudgeResetCallback(600), ##!!!!改600
                RecordCallback(record_path=f"./train_video/", fps=30, frame_type="pov"),

            ]
        )
        
        memory = None
        total_rewards = []
        n = 40
        obs, info = env.reset()
        for _ in range(3):  # 可以根据需要调整循环次数
            terminated = False
            reward_sum = 0
            while not terminated:
                action, memory = policy.get_action(obs, memory, input_shape='*')
                obs, reward, terminated, truncated, info = env.step(action)
                reward_sum += reward
            print('reward_sum', reward_sum)
            obs, info = env.reset()
            total_rewards.append(reward_sum)
        env.close()
            
        print('total_rewards: ', total_rewards)
        mean_reward = np.mean(total_rewards)
        variance_reward = np.var(total_rewards)
        timestep = os.path.basename(os.path.dirname(ckpt_path))  # 提取上层文件夹名字
        # print(f'Timestep: {timestep}, Mean Reward: {mean_reward}, Variance: {variance_reward}')

        results.append((timestep, mean_reward, variance_reward))
        output_file = f"{os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))}_inference.txt"
    
        with open(output_file, "a") as f:
            f.write(f"Timestep: {timestep}, Mean Reward: {mean_reward}, Variance: {variance_reward}\n")