'''
Date: 2024-11-14 19:42:09
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-12-15 13:36:22
FilePath: /MineStudio/minestudio/inference/example.py
'''

from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RecordCallback, SpeedTestCallback
from minestudio.models import VPTPolicy, load_vpt_policy
import torch
import os
import argparse
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="脚本参数示例")
    parser.add_argument("--id", type=str, default="", help="第一个参数")
    args = parser.parse_args()

    policy = load_vpt_policy(
        model_path="/nfs-shared/jarvisbase/pretrained/foundation-model-2x.model",
        weights_path="/nfs-shared/jarvisbase/pretrained/foundation-model-2x.weights"
    ).to("cuda")
    policy.load_state_dict(torch.load("/scratch/hekaichen/workspace/MineStudio/minestudio/online/run/output/checkpoints/2024-12-21_01-27-57/100/model.ckpt"), strict = True)
    from minestudio.simulator.callbacks import (
        MaskActionsCallback, 
        CommandsCallback, 
        JudgeResetCallback,
        FastResetCallback,
        GateRewardsCallback,
        RecordCallback,
        RewardsCallback,
        VoxelsCallback,
        SummonMobsCallback,
    )

    env = MinecraftSim(
        obs_size=(128, 128), 
        preferred_spawn_biome="plains", 
        callbacks=[
            RecordCallback(record_path='output/output', recording=True),
            SummonMobsCallback([{'name': 'sheep', 'number': 15, 'range_x': [-15, 15], 'range_z': [-15, 15]}]),
            MaskActionsCallback(attack = 0), 
            RewardsCallback([{
                'event': 'kill_entity', 
                'objects': ['sheep'], 
                'reward': 5.0, 
                'identity': 'shoot_sheep', 
                'max_reward_times': 30, 
            }]),
            CommandsCallback(commands=[
                '/give @p minecraft:bow 1',
                '/give @p minecraft:arrow 64',
                '/give @p minecraft:arrow 64',
            ]),
            FastResetCallback(
                biomes=['plains'],
                random_tp_range=1000,
            ),
            JudgeResetCallback(600),
        ]
    )
    memory = None
    obs, info = env.reset()

    import shutil
    for j in range(10000):
        try:
            sum_reward = 0
            for i in range(1000):
                action, memory = policy.get_action(obs, memory, input_shape='*')
                obs, reward, terminated, truncated, info = env.step(action)
                sum_reward += reward
            env.reset()
            with open('output/output'+str(args.id)+'/reward.txt', 'a') as f:
                f.write(str(sum_reward)+'\n')
        except Exception as e:
            print(e)
            print("Resetting the environment")
            continue