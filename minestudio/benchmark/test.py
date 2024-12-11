'''
Date: 2024-12-06 16:35:39
LastEditors: zhengxinyue
LastEditTime: 2024-12-11 17:47:25
FilePath: /MineStudio/minestudio/benchmark/test.py
'''
import sys
sys.path.append('/home/user/code/MineStudio/')
import os
import ray
from rich import print
from minestudio.inference import EpisodePipeline, MineGenerator, InfoBaseFilter
from minestudio.benchmark.read_conf import convert_yaml_to_callbacks
from functools import partial
from minestudio.models import load_openai_policy
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import (
    SpeedTestCallback, 
    RecordCallback, 
    SummonMobsCallback, 
    MaskActionsCallback, 
    RewardsCallback, 
    CommandsCallback, 
    TaskCallback,
    FastResetCallback
)

import pdb


if __name__ == '__main__':
    ray.init()
    conf_path = './task_configs/simple'
    
    for file_name in os.listdir(conf_path):
        if file_name.endswith('.yaml'):
            file_path = os.path.join(conf_path, file_name)
            commands_callback, task_callback = convert_yaml_to_callbacks(file_path)
            
            env = MinecraftSim(
                obs_size=(128, 128), 
                preferred_spawn_biome="forest", 
                callbacks=[
                    RecordCallback(record_path=f"./output/{file_name}", fps=30, frame_type="pov"),
                    CommandsCallback(commands_callback),
                    TaskCallback(task_callback),
                ]
            )
            # pdb.set_trace()
            policy = load_openai_policy(
                model_path="/nfs-shared/jarvisbase/pretrained/foundation-model-2x.model",
                weights_path="/nfs-shared/jarvisbase/pretrained/foundation-model-2x.weights"
            ).to("cuda")
            
            memory = None
            obs, info = env.reset()
            for i in range(1200):
                action, memory = policy.get_action(obs, memory, input_shape='*')
                obs, reward, terminated, truncated, info = env.step(action)
            env.close()







