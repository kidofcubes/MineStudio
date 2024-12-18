'''
Date: 2024-11-14 19:42:09
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-24 08:20:17
FilePath: /MineStudio/minestudio/inference/example.py
'''

from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RecordCallback, SpeedTestCallback, SummonMobsCallback, MaskActionsCallback, RewardsCallback, CommandsCallback, FastResetCallback, JudgeResetCallback
from minestudio.models import VPTPolicy, load_vpt_policy
import os
import subprocess

def check_and_kill_process():
    # 获取系统内存使用率
    memory_usage = psutil.virtual_memory().percent
    print(f"System memory utilization: {memory_usage:.2f}%")
    
    if memory_usage > 90:  # 超过90%
        current_pid = os.getpid()  # 获取当前进程的PID
        print(f"Memory utilization exceeded 90%. Terminating process with PID {current_pid}...")
        os.kill(current_pid, 9)  # 强制杀掉当前进程


if __name__ == '__main__':
    
    policy = load_vpt_policy(
        model_path="/nfs-shared/jarvisbase/pretrained/foundation-model-2x.model",
        weights_path="/nfs-shared/jarvisbase/pretrained/foundation-model-2x.weights"
    ).to("cuda")
    
    env = MinecraftSim(
        obs_size=(128, 128), 
        preferred_spawn_biome="plains", 
        callbacks=[
            SummonMobsCallback([{'name': 'sheep', 'number': 50, 'range_x': [-15, 15], 'range_z': [-15, 15]}]),
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
    reward_sum = 0
    with open("output/resulttt.txt", "a") as f:
        f.write("------------------------\n")
    for j in range(100):
        for i in range(600):
            action, memory = policy.get_action(obs, memory, input_shape='*')
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum+=reward
        env.reset()
        print("Resetting the environment\n") 
        for i in range(600):
            action, memory = policy.get_action(obs, memory, input_shape='*')
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum+=reward
        env.reset()
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        with open("output/resulttt.txt", "a") as f:
            f.write(f"reward_sum: {reward_sum}\n")
            f.write(f"RSS: {memory_info.rss / 1024**2:.2f} MB\n")
            f.write(f"VMS: {memory_info.vms / 1024**2:.2f} MB\n")
            command = "ps aux | grep 'hekaich.*java' | grep -v grep | wc -l"
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            process_count = int(result.stdout.strip())
            print(f"同时含有 'hekaich' 和 'java' 的进程数量: {process_count}\n")
        check_and_kill_process()