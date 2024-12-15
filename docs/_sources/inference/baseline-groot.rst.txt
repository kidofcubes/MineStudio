.. _inferece-groot:

Tutorial: Inference with GROOT
--------------------

To inferece with GROOT, you first need to download `reference videos <https://huggingface.co/datasets/CraftJarvis/MinecraftReferenceVideos>`_ and pretrained checkpoints.
The example code is provided in ``minestudio/tutorials/inference/evaluate_groot/main.py``.

.. dropdown:: Evaluating GROOT

    .. code-block:: python

        from minestudio.simulator import MinecraftSim
        from minestudio.simulator.callbacks import RecordCallback, SpeedTestCallback, CommandsCallback, DemonstrationCallback
        from minestudio.models import GrootPolicy, load_groot_policy
        from minestudio.benchmark.utility.read_conf import convert_yaml_to_callbacks
        from minestudio.inference import EpisodePipeline, MineGenerator, InfoBaseFilter

        import ray
        import numpy as np
        import av
        import os
        from functools import partial
        from rich import print

        if __name__ == '__main__':
            ray.init()

            resolution = (224, 224)
            
            file_path = "../../../benchmark/task_configs/simple/collect_wood.yaml"
            commands, task= convert_yaml_to_callbacks(file_path)
            print(f'Task: {task}')
            print(f'Init commands: {commands}')

            env_generator = partial(
                MinecraftSim,
                obs_size = resolution,
                preferred_spawn_biome = "forest", 
                callbacks = [
                    RecordCallback(record_path = "./output", fps = 30, frame_type="pov"),
                    SpeedTestCallback(50),
                    CommandsCallback(commands),
                    DemonstrationCallback("collect_wood")
                ]
            )

            agent_generator = partial(
                load_groot_policy,
                ckpt_path = "/home/zhwang/Desktop/zhancun/jarvisbase/pretrained/groot.ckpt"
            )

            worker_kwargs = dict(
                env_generator=env_generator, 
                agent_generator=agent_generator,
                num_max_steps=1200,
                num_episodes=2,
                tmpdir="./output",
                image_media="h264",
            )

            pipeline = EpisodePipeline(
                episode_generator=MineGenerator(
                    num_workers=4,
                    num_gpus=0.25,
                    max_restarts=3,
                    **worker_kwargs, 
                ), 
                episode_filter=InfoBaseFilter(
                    key="mine_block",
                    val="oak_log",
                    num=1,
                ),
            )
            summary = pipeline.run()
            print(summary)

Since GROOT is an instruction following policy, we need to specify the task, corresponding config and the demonstration video.
Supported tasks and configs can be found in ``minestudio/benchmark/task_configs`` and a detailed explanation can be found in the benchmarking tutorial.

To pass demonstration video to GROOT, we implement a ``DemonstrationCallback`` for the environment.
The ``DemonstrationCallback`` will first try to download the demonstration videos from the hugingface dataset if the local path ``reference_videos`` does not exist.
Then given a task name, a video path will be selected from the downloaded videos.
After the environment is initialized, the demonstration video path will be passed to the ``'ref_video_path'`` field of the observation and then be used to initialize the instruction for the agent.
A line like the following will be printed to the console, indicating the reference video and calculated latent properties.

.. code-block:: text

    =======================================================
    "Ref video is from: ./reference_videos/collect_wood/human/0.mp4."
    "Num frames: 1400"
    =======================================================

    [📚] latent shape: torch.Size([1, 1, 1024]) | mean: -0.028 | std:  1.308

For the inferece pipeline parameters, we need to specify:
    - task, configs and demonstration video for the ``env_generator``.
    - pretrained checkpoint for the ``agent_generator``.
    - rollout steps, number of episodes, output path for ``worker_kwargs``.
    - number of gpus and workers for ``MineGenerator``.
    - An ``episode_filter`` to filter the episode based on the key and value of the observation.

In the above example, we test the GROOT model on the task of collecting wood with 8 episodes and 1200 steps for each episode.
4 workers are used with 0.25 GPU per worker.
The episode will be filtered based on the key ``mine_block`` and value ``oak_log``.

The summary of the pipeline will be printed to the console, showing the success rate and the number of episode.
After the pipeline is finished, the console will print the summary of the pipeline like the following:

.. code-block:: python

    ...    

    (Worker pid=922019) Episode 2 saved at output/episode_2.mp4
    (Worker pid=922013) Speed Test Status:  [repeated 2x across cluster]
    (Worker pid=922013) Average Time: 0.04  [repeated 2x across cluster]
    (Worker pid=922013) Average FPS: 24.28  [repeated 2x across cluster]
    (Worker pid=922013) Total Steps: 2400  [repeated 2x across cluster]
    (Worker pid=922020) Episode 2 saved at output/episode_2.mp4
    (Worker pid=922013) Episode 2 saved at output/episode_2.mp4
    {'num_yes': 6, 'num_episodes': 8, 'yes_rate': '75.00%'}