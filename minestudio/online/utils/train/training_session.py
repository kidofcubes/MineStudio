'''
Date: 2025-05-20 18:18:38
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-05-20 18:23:37
FilePath: /MineStudio/minestudio/online/utils/train/training_session.py
'''
from numpy import roll
from omegaconf import OmegaConf
from omegaconf import DictConfig
import ray
import wandb
import uuid
import torch

@ray.remote(resources={"wandb": 1})
class TrainingSession:
    def __init__(self, logger_config: DictConfig, hyperparams: DictConfig):
        self.session_id = str(uuid.uuid4())
        hyperparams_dict = OmegaConf.to_container(hyperparams, resolve=True)
        wandb.init(config=hyperparams_dict, **logger_config) # type: ignore
    
    def log(self, *args, **kwargs):
        wandb.log(*args, **kwargs)
    
    def define_metric(self, *args, **kwargs):
        wandb.define_metric(*args, **kwargs)
    
    def log_video(self, data: dict, video_key: str, fps: int):
        data[video_key] = wandb.Video(data[video_key], fps=fps, format="mp4")
        wandb.log(data)

    def get_session_id(self):
        return self.session_id