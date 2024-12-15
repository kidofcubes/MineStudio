'''
Date: 2024-11-14 19:42:09
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-24 08:20:17
FilePath: /MineStudio/minestudio/inference/example.py
'''

from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RecordCallback, SpeedTestCallback, SummonMobsCallback, MaskActionsCallback, RewardsCallback, CommandsCallback, FastResetCallback, JudgeResetCallback
from minestudio.models import VPTPolicy, load_vpt_policy

if __name__ == '__main__':
    
    policy = load_vpt_policy(
        model_path="/nfs-shared/jarvisbase/pretrained/foundation-model-2x.model",
        weights_path="/nfs-shared/jarvisbase/pretrained/foundation-model-2x.weights"
    ).to("cuda")
    
    env = MinecraftSim(
        obs_size=(128, 128), 
        preferred_spawn_biome="plains", 
        callbacks=[
            RecordCallback(record_path="./output", fps=30, frame_type="pov", recording=True),
            SummonMobsCallback([{'name': 'sheep', 'number': 50, 'range_x': [-15, 15], 'range_z': [-15, 15]}]),
            MaskActionsCallback(inventory=0, attack = 0, forward = 0, back = 0, right = 0, left = 0), 
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
    reward_sum = 0
    for j in range(100):
        for i in range(600):
            action, memory = policy.get_action(obs, memory, input_shape='*')
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum+=reward
        env.reset()
        print("Resetting the environment") 
        for i in range(600):
            action, memory = policy.get_action(obs, memory, input_shape='*')
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum+=reward
        env.reset()
        print("reward_sum: ", reward_sum)