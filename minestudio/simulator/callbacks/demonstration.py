'''
Date: 2025-01-07 05:58:26
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-01-15 20:40:34
FilePath: /ROCKET-2/var/nfs-shared/shaofei/nfs-workspace/MineStudio/minestudio/simulator/callbacks/demonstration.py
'''
import random
import numpy as np
import os

from minestudio.simulator.callbacks.callback import MinecraftCallback
from minestudio.utils import get_mine_studio_dir
from minestudio.utils.register import Registers

def download_reference_videos():
    import huggingface_hub
    root_dir = get_mine_studio_dir()
    local_dir = os.path.join(root_dir, "reference_videos")
    print(f"Downloading reference videos to {local_dir}")
    huggingface_hub.snapshot_download(repo_id='CraftJarvis/MinecraftReferenceVideos', repo_type='dataset', local_dir=local_dir)

@Registers.simulator_callback.register
class DemonstrationCallback(MinecraftCallback):
    """
        This callback is used to provide demonstration data, mainly for GROOT.
    """
    
    def create_from_conf(source):
        data = MinecraftCallback.load_data_from_conf(source)
        if 'reference_video' in data:
            return DemonstrationCallback(data['reference_video'])
        else:
            return None
    
    def __init__(self, task):
        root_dir = get_mine_studio_dir()
        reference_videos_dir = os.path.join(root_dir, "reference_videos")
        if not os.path.exists(reference_videos_dir):
            response = input("Detecting missing reference videos, do you want to download them from huggingface (Y/N)?\n")
            while True:
                if response == 'Y' or response == 'y':
                    download_reference_videos()
                    break
                elif response == 'N' or response == 'n':
                    break
                else:
                    response = input("Please input Y or N:\n")

        self.task = task

        # load the reference video
        ref_video_name = task

        assert os.path.exists(os.path.join(reference_videos_dir, ref_video_name)), f"Reference video {ref_video_name} does not exist."

        ref_video_path = os.path.join(reference_videos_dir, ref_video_name, "human")

        # randomly select a video end with .mp4
        ref_video_list = [f for f in os.listdir(ref_video_path) if f.endswith('.mp4')]

        ref_video_path = os.path.join(ref_video_path, random.choice(ref_video_list))

        self.ref_video_path = ref_video_path

    def after_reset(self, sim, obs, info):
        obs['ref_video_path'] = self.ref_video_path
        return obs, info

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        obs['ref_video_path'] = self.ref_video_path
        return obs, reward, terminated, truncated, info

    def __repr__(self):
        return f"DemonstrationCallback(task={self.task})"
