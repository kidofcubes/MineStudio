from multiprocessing.connection import Connection
from torch.multiprocessing import Process
from typing import Dict, Callable, Optional, Tuple, Union
import cv2
import torch, rich
import traceback
import av
import time
# import tracemalloc
from omegaconf import DictConfig
import uuid
from threading import Thread
from queue import Queue
import numpy as np
from pathlib import Path, PosixPath
import random
import logging
import os
#from minestudio.simulator.entry import MinecraftSim
import copy
from minestudio.simulator import MinecraftSim

class EnvWorker(Process):
    def __init__(self, env_generator: Callable[[], MinecraftSim], conn: Connection, video_output_dir: str, video_fps: int, restart_interval: Optional[int] = None, max_fast_reset: int = 10000, env_id: int = 0, rollout_worker_id: int = 0):
        super().__init__()
        self.max_fast_reset = max_fast_reset
        self.env_generator = copy.deepcopy(env_generator)
        self.env_id = env_id
        self.conn = conn
        self.restart_interval = restart_interval
        self.video_output_dir = Path(video_output_dir)
        self.rollout_worker_id = rollout_worker_id
        if not self.video_output_dir.exists():
            try:
                self.video_output_dir.mkdir(parents=True)
            except FileExistsError:
                pass
        
        self.video_fps = video_fps
    
    def step_agent(self, obs: dict, last_reward: float, last_terminated: bool, last_truncated: bool, episode_uuid: str) -> Tuple[Dict[str, torch.Tensor], float]:
        self.conn.send(("step_agent", obs, last_reward, last_terminated, last_truncated, episode_uuid))
        action, vpred = self.conn.recv()
        return action, vpred

    def reset_state(self) -> Dict[str, torch.Tensor]:
        self.conn.send(("reset_state", None))
        return self.conn.recv()
    
    def report_rewards(self, rewards: np.ndarray):
        self.conn.send(("report_rewards", rewards))
        return self.conn.recv()

    def run(self) -> None:
        while True:
            time.sleep(random.randint(0,20))
            self.env = self.env_generator()
            try:
                start_time = time.time()
                reward = 0.0
                terminated, truncated = True, False # HACK: make sure the 'first' flag of the first step in the first episode is True
                while True:
                    if self.restart_interval is not None and time.time() - start_time >= self.restart_interval:
                        raise Exception("Restart interval reached")
                    self.reset_state()
                    obs, _ = self.env.reset()
                    reward_list = []
                    step = 0
                    v_preds = []
                    episode_uuid = str(uuid.uuid4())
                    while True:
                        if step%100==1:
                            logging.getLogger("ray").info(f"working..., max_fast_reset: {self.max_fast_reset}, env_id: {self.env_id}, rollout_worker_id: {self.rollout_worker_id}")
                        step += 1
                        action, vpred = self.step_agent(obs, 
                                            last_reward=float(reward),
                                            last_terminated=terminated,
                                            last_truncated=truncated,
                                            episode_uuid=episode_uuid
                                        )
                        v_preds.append(vpred)
                        obs, reward, terminated, truncated, _ = self.env.step(action)
                        reward_list.append(reward)
                        if terminated or truncated:
                            break
                    _result = self.report_rewards(np.array(reward_list))

            except Exception as e:
                traceback.print_exc()
                rich.print(f"[bold red]An error occurred in EnvWorker: {e}[/bold red]")