'''
Date: 2024-12-13 22:39:49
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2025-01-07 11:26:20
FilePath: /MineStudio/minestudio/tutorials/inference/evaluate_groot/main.py
'''
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RecordCallback, SpeedTestCallback, CommandsCallback, DemonstrationCallback, load_callbacks_from_config
from minestudio.models import GrootPolicy, load_groot_policy
from minestudio.inference import EpisodePipeline, MineGenerator, InfoBaseFilter
from minestudio.benchmark import prepare_task_configs

import ray
import numpy as np
import av
import os
from functools import partial
from rich import print

if __name__ == '__main__':
    ray.init()
    task_configs = prepare_task_configs("simple")
    config_file = task_configs["collect_wood"]
    print(config_file)

    env_generator = partial(
        MinecraftSim,
        obs_size = (224, 224),
        preferred_spawn_biome = "forest", 
        callbacks = [
            SpeedTestCallback(50),
        ] + load_callbacks_from_config(config_file)
    )

    # agent_generator = partial(
    #     load_groot_policy,
    #     # ckpt_path = "/nfs-shared-2/shaofei/minestudio/save/2025-01-05/15-12-43/weights/weight-epoch=2-step=40000.ckpt",
    #     # ckpt_path = "/nfs-shared-2/shaofei/minestudio/save/2025-01-05/15-12-43/weights/weight-epoch=4-step=80000.ckpt"
    #     ckpt_path="/nfs-shared-2/shaofei/minestudio/save/2024-12-12/21-34-37/weights/weight-epoch=9-step=180000-EMA.ckpt", 
    # )
    
    agent_generator = lambda: GrootPolicy.from_pretrained("CraftJarvis/MineStudio_GROOT.18w_EMA")

    worker_kwargs = dict(
        env_generator=env_generator, 
        agent_generator=agent_generator,
        num_max_steps=600,
        num_episodes=2,
        tmpdir="./output",
        image_media="h264",
    )

    pipeline = EpisodePipeline(
        episode_generator=MineGenerator(
            num_workers=4, #! 4
            num_gpus=0.25,
            max_restarts=3,
            **worker_kwargs, 
        ), 
        episode_filter=InfoBaseFilter(
            key="mine_block",
            regex=".*log.*",
            num=1,
        ),
    )
    summary = pipeline.run()
    print(summary)