from numpy import roll
from omegaconf import OmegaConf
import hydra
import logging
from minestudio.online.rollout.rollout_manager import RolloutManager
from minestudio.online.utils.rollout import get_rollout_manager
from minestudio.online.utils.train.training_session import TrainingSession
import ray
import wandb
import uuid
import torch
from minestudio.online.rollout.start_manager import start_rolloutmanager
online_dict = {
    "trainer_name": "PPOTrainer",
    "detach_rollout_manager": True,
    "rollout_config": {
        "num_rollout_workers": 1,
        "num_gpus_per_worker": 1.0,
        "num_cpus_per_worker": 1,
        "fragment_length": 256,
        "to_send_queue_size": 4,
        "worker_config": {
            "num_envs": 4,
            "batch_size": 2,
            "restart_interval": 3600,  # 1h
            "video_fps": 20,
            "video_output_dir": "output/output/videos",
        },
        "replay_buffer_config": {
            "max_chunks": 4800,
            "max_reuse": 1,
            "max_staleness": 2,
            "fragments_per_report": 40,
            "fragments_per_chunk": 1,
            "database_config": {
                "path": "output/replay_buffer_cache",
                "num_shards": 8,
            },
        },
        "episode_statistics_config": {},
    },
    "train_config": {
        "num_workers": 2,
        "num_gpus_per_worker": 1.0,
        "num_iterations": 4000,
        "vf_warmup": 0,
        "learning_rate": 0.00002,
        "anneal_lr_linearly": False,
        "weight_decay": 0.04,
        "adam_eps": 1e-8,
        "batch_size_per_gpu": 1,
        "batches_per_iteration": 20,
        "gradient_accumulation": 10,  # TODO: check
        "epochs_per_iteration": 1,  # TODO: check
        "context_length": 128,
        "discount": 0.999,
        "gae_lambda": 0.95,
        "ppo_clip": 0.2,
        "clip_vloss": False,  # TODO: check
        "max_grad_norm": 5,  # ????
        "zero_initial_vf": True,
        "ppo_policy_coef": 1.0,
        "ppo_vf_coef": 0.5,  # TODO: check
        "kl_divergence_coef_rho": 0.2,
        "entropy_bonus_coef": 0.0,
        "coef_rho_decay": 0.9995,
        "log_ratio_range": 50,  # for numerical stability
        "normalize_advantage_full_batch": True,  # TODO: check!!!
        "use_normalized_vf": True,
        "num_readers": 4,
        "num_cpus_per_reader": 0.1,
        "prefetch_batches": 2,
        "save_interval": 10,
        "keep_interval": 40,
        "record_video_interval": 1,
        "fix_decoder": False,
        "resume": None
    },

    "logger_config": {
        "project": "minestudio_online",
        "name": "bow"
    },
}

def env_generator():
    import numpy as np
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
    sim = MinecraftSim(
        obs_size=(128, 128), 
        preferred_spawn_biome="plains", 
        action_type = "agent",
        callbacks=[
            SpeedTestCallback(50), 
            SummonMobsCallback([{'name': 'cow', 'number': 10, 'range_x': [-5, 5], 'range_z': [-5, 5]}]),
            MaskActionsCallback(inventory=0, camera=np.array([[0]])), 
            #RecordCallback(record_path="./output", fps=30),
            RewardsCallback([{
                'event': 'kill_entity', 
                'objects': ['cow', 'sheep'], 
                'reward': 1.0, 
                'identity': 'kill sheep or cow', 
                'max_reward_times': 5, 
            }]),
            CommandsCallback(commands=[
                '/give @p minecraft:iron_sword 1',
                '/give @p minecraft:diamond 64',
            ]), 
            FastResetCallback(
                biomes=['mountains'],
                random_tp_range=1000,
            ), 
            # TaskCallback([
            #     {'name': 'chop', 'text': 'mine the oak logs'}, 
            #     {'name': 'diamond', 'text': 'mine the diamond ore'},
            # ])
        ]
    )
    return sim

def policy_generator():
    from minestudio.models.openai_vpt.body import load_openai_policy
    model_path = '/nfs-shared/jarvisbase/pretrained/foundation-model-2x.model'
    weights_path = '/nfs-shared/jarvisbase/pretrained/rl-from-early-game-2x.weights'
    policy = load_openai_policy(model_path, weights_path)
    return policy
