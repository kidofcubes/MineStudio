'''
Date: 2024-11-10 10:06:28
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2025-01-09 16:33:22
FilePath: /MineStudio/minestudio/data/minecraft/utils.py
'''
import os
import av
import cv2
import requests
import numpy as np
from datetime import datetime
import torch
import torch.distributed as dist
import shutil
from torch.utils.data import Sampler
from tqdm import tqdm
from rich import print
from typing import Union, Tuple, List, Dict, Callable, Sequence, Mapping, Any, Optional, Literal
from huggingface_hub import hf_api, snapshot_download

from minestudio.data.minecraft.callbacks import DrawFrameCallback

def get_repo_total_size(repo_id, repo_type="dataset", branch="main"):

    def fetch_file_list(path=""):
        url = f"https://huggingface.co/api/{repo_type}s/{repo_id}/tree/{branch}/{path}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def calculate_size(path=""):
        files = fetch_file_list(path)
        total_size = 0
        for file in files:
            if file["type"] == "file":
                total_size += file["size"]
            elif file["type"] == "directory":
                total_size += calculate_size(file["path"])
        return total_size
    total_size_bytes = calculate_size()
    total_size_gb = total_size_bytes / (1024 ** 3)
    return total_size_bytes, total_size_gb

def download_dataset_from_huggingface(name: Literal["6xx", "7xx", "8xx", "9xx", "10xx"], base_dir: Optional[str]=None):

    if base_dir is None:
        from minestudio.utils import get_mine_studio_dir
        base_dir = get_mine_studio_dir()
    
    total, used, free = shutil.disk_usage(base_dir)
    repo_id = f"CraftJarvis/minestudio-data-{name}"
    total_size, _ = get_repo_total_size(repo_id)
    print(
        f"""
        [bold]Download Dataset[/bold]
        Dataset: {name}
        Base Dir: {base_dir}
        Total Size: {total_size / 1024 / 1024 / 1024:.2f} GB
        Free Space: {free / 1024 / 1024 / 1024:.2f} GB
        """
    )
    if total_size > free:
        raise ValueError(f"Insufficient space for downloading {name}. ")
    dataset_dir = os.path.join(get_mine_studio_dir(), 'contractors', f'dataset_{name}')
    local_dataset_dir = snapshot_download(repo_id, repo_type="dataset", local_dir=dataset_dir)
    return local_dataset_dir

def pull_datasets_from_remote(dataset_dirs: List[str]) -> List[str]:
    new_dataset_dirs = []
    for path in dataset_dirs:
        if path in ['6xx', '7xx', '8xx', '9xx', '10xx']:
            return_path = download_dataset_from_huggingface(path)
            new_dataset_dirs.append(return_path)
        else:
            new_dataset_dirs.append(path)
    return new_dataset_dirs

def write_video(
    file_name: str, 
    frames: Sequence[np.ndarray], 
    width: int = 640, 
    height: int = 360, 
    fps: int = 20
) -> None:
    """Write video frames to video files. """
    with av.open(file_name, mode="w", format='mp4') as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = width
        stream.height = height
        for frame in frames:
            assert frame.shape[1] == width and frame.shape[0] == height, f"frame shape {frame.shape} not match {width}x{height}"
            frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

def batchify(batch_in: Sequence[Dict[str, Any]]) -> Any:
    example = batch_in[0]
    if isinstance(example, Dict):
        batch_out = {
            k: batchify([item[k] for item in batch_in]) \
                for k in example.keys()
        }
    elif isinstance(example, torch.Tensor):
        batch_out = torch.stack(batch_in, dim=0)
    elif isinstance(example, int):
        batch_out = torch.tensor(batch_in, dtype=torch.int32)
    elif isinstance(example, float):
        batch_out = torch.tensor(batch_in, dtype=torch.float32)
    else:
        batch_out = batch_in
    return batch_out

class MineDistributedBatchSampler(Sampler):

    def __init__(
        self, 
        dataset, 
        batch_size, 
        num_replicas=None, # num_replicas is the number of processes participating in the training
        rank=None,         # rank is the rank of the current process within num_replicas
        shuffle=False, 
        drop_last=True,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            try:
                num_replicas = dist.get_world_size()
            except:
                num_replicas = 1
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            try:
                rank = dist.get_rank()
            except:
                rank = 0
        assert shuffle is False, "shuffle must be False in sampler."
        assert drop_last is True, "drop_last must be True in sampler."
        # print(f"{rank = }, {num_replicas = }")
        self.batch_size = batch_size
        self.dataset = dataset
        self.num_total_samples = len(self.dataset)
        self.num_samples_per_replica = self.num_total_samples // num_replicas
        replica_range = (self.num_samples_per_replica * rank, self.num_samples_per_replica * (rank + 1)) # [start, end)
        
        num_past_samples = 0
        episodes_within_replica = [] # (episode, epsode_start_idx, episode_end_idx, item_bias)
        self.episodes_with_items = self.dataset.episodes_with_items
        for episode, length, item_bias in self.episodes_with_items:
            if num_past_samples + length > replica_range[0] and num_past_samples < replica_range[1]:
                episode_start_idx = max(0, replica_range[0] - num_past_samples)
                episode_end_idx = min(length, replica_range[1] - num_past_samples)
                episodes_within_replica.append((episode, episode_start_idx, episode_end_idx, item_bias))
            num_past_samples += length
        self.episodes_within_replica = episodes_within_replica

    def __iter__(self):
        """
        Build batch of episodes, each batch is consisted of `self.batch_size` episodes.
        Only if one episodes runs out of samples, the batch is filled with the next episode.
        """
        next_episode_idx = 0
        reading_episodes = [ None for _ in range(self.batch_size) ]
        while True:
            batch = [ None for _ in range(self.batch_size) ]
            # feed `reading_episodes` with the next episode
            for i in range(self.batch_size):
                if reading_episodes[i] is None:
                    if next_episode_idx >= len(self.episodes_within_replica):
                        break
                    reading_episodes[i] = self.episodes_within_replica[next_episode_idx]
                    next_episode_idx += 1
            # use while loop to build batch
            while any([x is None for x in batch]):
                record_batch_length = sum([x is not None for x in batch])
                # get the position that needs to be filled
                for cur in range(self.batch_size):
                    if batch[cur] is None:
                        break
                # get the episode that has the next sample
                if reading_episodes[cur] is not None:
                    use_eps_idx = cur
                else:
                    for use_eps_idx in range(self.batch_size):
                        if reading_episodes[use_eps_idx] is not None:
                            break
                # if all episodes are None, then stop iteration
                if reading_episodes[use_eps_idx] is None:
                    return None
                # fill the batch with the next sample
                episode, start_idx, end_idx, item_bias = reading_episodes[use_eps_idx]
                batch[cur] = item_bias + start_idx
                if start_idx+1 < end_idx:
                    reading_episodes[use_eps_idx] = (episode, start_idx + 1, end_idx, item_bias)
                else:
                    reading_episodes[use_eps_idx] = None
            yield batch

    def __len__(self):
        return self.num_samples_per_replica // self.batch_size

def visualize_dataloader(
    dataloader, 
    draw_frame_callbacks: List[DrawFrameCallback],
    num_samples: int=1, 
    save_fps: int=20, 
    output_dir: str = "./",
) -> None:

    video_frames = []
    for batch_idx, data in enumerate(tqdm(dataloader)):
        if batch_idx > num_samples:
            break
        for sample_idx in range(data['image'].shape[0]):
            sample_frames = data['image'][sample_idx].numpy()
            for callback in draw_frame_callbacks:
                sample_frames = callback.draw_frames(sample_frames, data, sample_idx)
            video_frames = video_frames + sample_frames

    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    file_name = f"save_{timestamp}.mp4"
    file_path = os.path.join(output_dir, file_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    width, height = video_frames[0].shape[1], video_frames[0].shape[0]
    write_video(file_path, video_frames, fps=save_fps, width=width, height=height)
    print(f"Video saved to {file_path}. ")
