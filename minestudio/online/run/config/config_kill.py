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
    "rollout_config": { ##for train env
        "num_rollout_workers": 2,
        "num_gpus_per_worker": 1, #1
        "num_cpus_per_worker": 1,
        "fragment_length": 256,
        "to_send_queue_size": 5, #6!!!
        "worker_config": {
            "num_envs": 10, #12!!!
            "batch_size": 5, #6!!!
            "restart_interval": 3600,  # 1h
            "video_fps": 20,
            "video_output_dir": "output/videos",
        },
        "replay_buffer_config": {
            "max_chunks": 4800,
            "max_reuse": 2,
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
        "num_workers": 2, #2
        "num_gpus_per_worker": 1.0,
        "num_iterations": 10000,  # 4000
        "vf_warmup": 0,
        "learning_rate": 0.00002,
        "anneal_lr_linearly": False,
        "weight_decay": 0.04,
        "adam_eps": 1e-8,
        "batch_size_per_gpu": 1,
        "batches_per_iteration": 100, #可10 200!!! batches_per_iteration/gradient_accumulation = 每个iter更新网络步数
        "gradient_accumulation": 10,  # TODO: check
        "epochs_per_iteration": 1,  # TODO: check
        "context_length": 64,
        "discount": 0.999,
        "gae_lambda": 0.95,
        "ppo_clip": 0.2,
        "clip_vloss": False,  # TODO: check
        "max_grad_norm": 5,  # ????
        "zero_initial_vf": True,
        "ppo_policy_coef": 1.0,
        "ppo_vf_coef": 0.5,  # TODO: check
        "kl_divergence_coef_rho": 0.3, #### 0.2
        "entropy_bonus_coef": 0.0,
        "coef_rho_decay": 1, #0.9995,
        "log_ratio_range": 50,  # for numerical stability
        "normalize_advantage_full_batch": True,  # TODO: check!!!
        "use_normalized_vf": True,
        "num_readers": 4,
        "num_cpus_per_reader": 0.1,
        "prefetch_batches": 2,
        "save_interval": 4, # 1-2
        "keep_interval": 1, #1 40
        "record_video_interval": 2,
        "enable_ref_update": False,
        "fix_decoder": False,
        "resume": "/scratch2/zhengxinyue/mcu_online/10k_seed_1/checkpoints/2025-01-01_14-48-54/1587200",
        "resume_optimizer": True,
        "save_path": "/scratch2/zhengxinyue/mcu_online/10k_seed_2"
    },

    "logger_config": {
        "project": "minestudio_online",
        "name": "10k_seed_2"
    },
}

import os
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)
file_paths = [
    os.path.join(parent_dir, "equip_main.json"),
    os.path.join(parent_dir, "equip_off.json"),
    os.path.join(parent_dir, "give.json")
]
summon_path = [os.path.join(parent_dir, "sheep_summon.json")]
other_summon_path = [os.path.join(parent_dir, "other_summon.json")]
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

equip_commands = load_commands(file_paths)
summon_commands = load_commands(summon_path)
other_summon_commands = load_commands(other_summon_path)



def env_generator(equip_commands, summon_commands, other_summon_commands):
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import (
        SummonMobsCallback, 
        MaskActionsCallback, 
        RewardsCallback,  
        JudgeResetCallback,
        FastResetCallback
    )
    from minestudio.online.run.commands import CommandsCallback
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
            CommandsCallback(equip_commands, summon_commands, other_summon_commands),
            FastResetCallback( 
                biomes=['plains'],
                random_tp_range=1000,
            ),
            JudgeResetCallback(600), ##20s fast reset reward 3
            # lianxu 3 episode
        ]
    )
    return env

def policy_generator():
    from minestudio.models import load_vpt_policy
    policy = load_vpt_policy(
        model_path="/nfs-shared/jarvisbase/pretrained/foundation-model-2x.model",
        weights_path="/nfs-shared/jarvisbase/pretrained/foundation-model-2x.weights"
    ).to("cuda")
    return policy

def new_env_generator():
    return  env_generator(equip_commands[:100], summon_commands[:10], other_summon_commands[:10])
