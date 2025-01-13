'''
Date: 2024-11-10 12:27:01
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2025-01-10 14:24:40
FilePath: /MineStudio/minestudio/data/minecraft/tools/convert_lmdb.py
'''

from functools import partial
import ray
import os
import re
import time
import uuid
import random
import pickle
import multiprocessing as mp
import ray.experimental.tqdm_ray as tqdm_ray
from multiprocessing.pool import ThreadPool, Pool
from hashlib import md5

import io
import av
import cv2
import lmdb
import argparse
import torch
import numpy as np
import shutil
from tqdm import tqdm
from rich import print
from rich.console import Console
from pathlib import Path
from typing import Sequence, Union, Tuple, List, Dict, Any
from collections import OrderedDict

from minestudio.data.minecraft.callbacks import (
    ModalConvertionCallback, 
    ImageConvertionCallback,
    ActionConvertionCallback,
    MetaInfoConvertionCallback,
    SegmentationConvertionCallback,
)

CONTRACTOR_PATTERN = r"^(.*?)-(\d+)$"
NUM_CPUS_PER_ACTOR = 1

@ray.remote(num_cpus=NUM_CPUS_PER_ACTOR)
class ConvertWorker:
    
    def __init__(
        self, 
        write_path: Union[str, Path], 
        source_type: str, 
        tasks: Dict, 
        chunk_size: int,
        remote_tqdm: Any, 
        thread_pool: int = 8
    ):
        self.tasks = tasks
        self.write_path = write_path
        self.source_type = source_type
        self.chunk_size = chunk_size
        self.remote_tqdm = remote_tqdm
        self.thread_pool = thread_pool
        if isinstance(write_path, str):
            write_path = Path(write_path) 
        if write_path.is_dir():
            print(f"Write path {write_path} exists, delete it. ")
            shutil.rmtree(write_path)
        write_path.mkdir(parents=True)
        self.lmdb_handler = lmdb.open(str(write_path), map_size=1<<40)

    def run(self):
        chunk_infos = []
        num_total_frames = 0
        eps_idx = 0
        for eps, segments in self.tasks.items():
            # if eps_idx > 3: #! debug !!!
            #     break
            eps_keys, eps_vals = [], []
            eps_keys, eps_vals, cost = self.dispatch(eps=eps, segments=segments)
            num_eps_frames = len(eps_keys) * self.chunk_size
            if num_eps_frames == 0:
                # empty video, skip it
                continue
            for key, val in zip(eps_keys, eps_vals):
                with self.lmdb_handler.begin(write=True) as txn:
                    lmdb_key = str((eps_idx, key))
                    txn.put(str(lmdb_key).encode(), val)
            chunk_info = {
                "episode": eps, 
                "episode_idx": eps_idx,
                "num_frames": num_eps_frames,
            }
            chunk_infos.append(chunk_info)
            num_total_frames += num_eps_frames
            eps_idx += 1

            self.remote_tqdm.update.remote(1)

        meta_info = {
            "__chunk_size__": self.chunk_size,
            "__chunk_infos__": chunk_infos,
            "__num_episodes__": eps_idx,
            "__num_total_frames__": num_total_frames,
        }
        
        with self.lmdb_handler.begin(write=True) as txn:
            for key, val in meta_info.items():
                txn.put(key.encode(), pickle.dumps(val))
        
        print(f"Worker finish: {self.write_path}. ")
        
        return meta_info


    def dispatch(self, **kwargs):
        if self.source_type == 'video':
            return self.process_video(**kwargs)
        elif self.source_type == 'action':
            return self.process_action(**kwargs)
        elif self.source_type == 'accomplishment':
            return self.process_accomplishment(**kwargs)
        elif self.source_type == 'contractor_info':
            return self.process_contractor_info(**kwargs)
        elif self.source_type == 'segment':
            return self.process_segment(**kwargs)

    def process_segment(self, eps: str, segments: List[Tuple[int, Path, Path]]) -> Tuple[List, List, float]:
        time_start = time.time()
        segmentation_convertion = SegmentationConvertionCallback(chunk_size=self.chunk_size)
        skip_frames = []
        modal_file_path = []
        for i in range(len(segments)):
            modal_file_path.append(segments[i][1])
            skip_frames.append(self._generate_frame_indicator(segments[i][2]))
        keys, vals = segmentation_convertion.do_convert(eps, skip_frames, modal_file_path)
        cost = time.time() - time_start
        print(f"episode: {eps}, chunks: {len(keys)}, frames: {len(keys) * self.chunk_size},"
                f"size: {sum(len(x) for x in vals) / (1024*1024):.2f} MB, cost: {cost:.2f} sec")
        return keys, vals, cost

    def _generate_frame_indicator(self, action_path: Path) -> Sequence[bool]:
        
        def _check_no_op(action: Dict):
            _sum = 0
            for key, val in action.items():
                _sum += val.sum()
            return _sum != np.array(0.)
        
        indicators = []
        with action_path.open("rb") as f:
            action_pkl = pickle.load(f)
            traj_len = len(action_pkl['attack'])
            for fid in range(traj_len):
                f_action = {key: val[fid:fid+1] for key, val in action_pkl.items()}
                no_op_flag = _check_no_op(f_action)
                indicators.append(no_op_flag)
        
        return indicators

class TaskManager:
    
    def __init__(
        self, 
        input_dir: List[str], 
        action_dir: List[str], # used to skip no-action frames
        output_dir: str,
        modal_convertion_kernel: ModalConvertionCallback,
        num_workers: int=16,
    ) -> None:
        self.input_dir  = input_dir
        self.action_dir = action_dir
        self.output_dir = output_dir
        self.modal_convertion_kernel = modal_convertion_kernel
        self.num_workers = num_workers
        
    def prepare_tasks(self):
        suffix_mapping = {'contractor_info': 'pkl', 'action': 'pkl', 'video': 'mp4', 'segment': 'rle'}
        self.resource = {}
        self.resource['action'] = self.load_source_files(self.action_dir, 'action', suffix_mapping['action'])
        if self.source_type != 'action':
            self.resource[self.source_type] = self.load_source_files(self.input_dir, self.source_type, suffix_mapping[self.source_type])
        
        episode_with_action = OrderedDict()
        num_removed_segs = 0
        for eps, source_segs in self.resource[self.source_type].items():
            if eps not in self.resource['action']:
                num_removed_segs += len(source_segs)
                continue
            action_segs = self.resource['action'][eps]
            for ord, source_path in source_segs:
                action_seg = [seg for seg in action_segs if seg[0] == ord]
                if len(action_seg) > 0:
                    action_seg = action_seg[0]
                else:
                    num_removed_segs += 1
                    continue
                if eps not in episode_with_action:
                    episode_with_action[eps] = []
                action_path = action_seg[1]
                episode_with_action[eps].append( (ord, source_path, action_path) )
        self.episode_with_action = episode_with_action
        print(f"num of removed segments: {num_removed_segs}")

    def load_source_files(self, source_dir: str, modal_name: str, suffix: str):
        episodes = OrderedDict()
        num_segments = 0
        for sub_dir in source_dir:
            print(f'input {modal_name} directory: {sub_dir}')
            for source_file in tqdm(Path(sub_dir).rglob(f"*.{suffix}"), desc="Looking for source files"):
                file_name = str(source_file.with_suffix('').relative_to(sub_dir))
                match = re.match(CONTRACTOR_PATTERN, file_name)
                if match:
                    eps, ord = match.groups()
                else:
                    eps, ord = file_name, "0"
                if eps not in episodes:
                    episodes[eps] = []
                file_path = Path(sub_dir) / f"{file_name}.{suffix}"
                episodes[eps].append( (ord, file_path) )
                num_segments += 1
        # rank the segments in an accending order
        for key, value in episodes.items():
            episodes[key] = sorted(value, key=lambda x: int(x[0]))
        # re-split episodes according to time
        new_episodes = OrderedDict()
        MAX_TIME = 1000
        for eps, segs in episodes.items():
            start_time = -MAX_TIME
            working_ord = -1
            for ord, file_path in segs:
                if int(ord) - start_time >= MAX_TIME:
                    working_ord = ord
                    new_episodes[f"{eps}-{working_ord}"] = []
                start_time = int(ord)
                new_episodes[f"{eps}-{working_ord}"].append( (ord, file_path) )
        episodes = new_episodes
        print(f'source: {modal_name}, num of episodes: {len(episodes)}, num of segments: {num_segments}') 
        return episodes
    
    def dispatch(self):
        sub_tasks = OrderedDict()
        workers = []
        remote_tqdm = ray.remote(tqdm_ray.tqdm).remote(total=len(self.episode_with_action))
        num_episodes_per_file = (len(self.episode_with_action) + self.num_workers - 1) // self.num_workers
        for idx, (eps, seg) in enumerate(self.episode_with_action.items()):
            sub_tasks[eps] = seg
            if (idx + 1) % num_episodes_per_file == 0 or (idx + 1) == len(self.episode_with_action):
                write_path = Path(self.output_dir) / self.source_type / f"{self.source_type}-{idx+1}"
                worker = ConvertWorker.remote(write_path, self.source_type, sub_tasks, self.lmdb_chunk_size, remote_tqdm)
                # worker = ConvertWorker(write_path, self.source_type, sub_tasks, self.lmdb_chunk_size)
                workers.append(worker)
                sub_tasks = OrderedDict()

        results = ray.get([worker.run.remote() for worker in workers])
        # results = [worker.run() for worker in workers]
        
        num_frames   = sum([result['__num_total_frames__'] for result in results])
        num_episodes = sum([result['__num_episodes__'] for result in results])
        
        ray.kill(remote_tqdm)
        print(f"Total frames: {num_frames}, Total episodes: {num_episodes}")

def main(args):
    
    ray.init()

    task_manager = TaskManager(
        input_dir=args.input_dir, 
        action_dir=args.action_dir,
        output_dir=args.output_dir,
        convertion_callback=args.source_type,
        lmdb_chunk_size=args.lmdb_chunk_size,
        num_workers=args.num_workers,
    )
    
    task_manager.prepare_tasks()
    task_manager.dispatch()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb-chunk-size", type=int, default=32,
                        help="lmdb chunk size")
    parser.add_argument("--input-dir", type=str, required=False, nargs='+', 
                        help="directory containing source files (videos, actions, contractor info, segment)")
    parser.add_argument("--action-dir", type=str, required=True, default="actions", nargs='+',
                        help="directory containing IDM action files")
    parser.add_argument("--output-dir", type=str, required=True, default="datasets",
                        help="directory saving lmdb files")
    parser.add_argument("--convertion-callback", type=str, required=True,
                        help="name of the modal convertion callback")
    parser.add_argument("--num-workers", default=16, type=int,  
                        help="the number of workers")
    
    args = parser.parse_args()
    main(args)