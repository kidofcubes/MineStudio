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
import json
import random
from minestudio.online.rollout.start_manager import start_rolloutmanager
from minestudio.online.trainer.start_trainer import start_trainer
from minestudio.online.run.config.config import policy_generator, online_dict, new_env_generator

import os


if __name__=='__main__':
    config_name = "gate"
    print("\033[1;32m Starting training session WITH CONFIG: " + config_name + " \033[0m")
    module_name = "minestudio.online.run.config."+config_name

    import importlib
    module = importlib.import_module(module_name)
    env_generator = getattr(module, "env_generator")
    policy_generator = getattr(module, "policy_generator")
    online_dict = getattr(module, "online_dict")
    online_cfg = OmegaConf.create(online_dict)

    start_rolloutmanager(policy_generator, new_env_generator, online_cfg)
    start_trainer(policy_generator, new_env_generator, online_cfg)



#test 是不是直接start一个新的rolloutmanager就行了，给新的random sample的generator


# # =[
#                 '/give @p minecraft:bow 1',
#                 '/give @p minecraft:arrow 64',
#                 '/give @p minecraft:arrow 64',
#             ]



# training_session = None
# try:
#     training_session = ray.get_actor("training_session")
# except ValueError:
#     pass
# if training_session is not None:
#     logger.error("Trainer already running!")
#     exit()

# training_session = TrainingSession.options(name="training_session").remote(hyperparams=cfg, logger_config=cfg.logger_config) # type: ignore
# ray.get(training_session.get_session_id.remote()) # Assure that the session is created before the trainer
# ray.get(rollout_manager.update_training_session.remote())
# print("Making trainer")
# trainer = registry.get_trainer_class(online_cfg.trainer_name)(
#     rollout_manager=rollout_manager,
#     policy_generator=policy_generator,
#     env_generator=env_generator,
#     **online_cfg.train_config
# )
# trainer.fit()


import time
time.sleep(1000000)