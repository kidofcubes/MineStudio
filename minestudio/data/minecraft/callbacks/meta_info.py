'''
Date: 2025-01-09 05:36:19
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-01-21 22:28:43
FilePath: /MineStudio/minestudio/data/minecraft/callbacks/meta_info.py
'''
import cv2
import pickle
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List, Dict, Callable, Any, Optional, Literal

from minestudio.data.minecraft.callbacks.callback import ModalKernelCallback, DrawFrameCallback, ModalConvertCallback
from minestudio.utils.register import Registers

@Registers.modal_kernel_callback.register
class MetaInfoKernelCallback(ModalKernelCallback):

    def create_from_config(config: Dict) -> 'MetaInfoKernelCallback':
        return MetaInfoKernelCallback(**config.get('meta_info', {}))

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return 'meta_info'

    def filter_dataset_paths(self, dataset_paths: List[str]) -> List[str]:
        action_paths = [path for path in dataset_paths if Path(path).stem in ['contractor_info', 'meta_info']]
        return action_paths

    def do_decode(self, chunk: bytes, **kwargs) -> Dict:
        return pickle.loads(chunk)

    def do_merge(self, chunk_list: List[bytes], **kwargs) -> Dict:
        chunks = [self.do_decode(chunk) for chunk in chunk_list]
        cache_chunks = {}
        for chunk in chunks:
            for frame_info in chunk:
                for key, value in frame_info.items():
                    if key not in cache_chunks:
                        cache_chunks[key] = []
                    cache_chunks[key].append(value)
        return cache_chunks

    def do_slice(self, data: Dict, start: int, end: int, skip_frame: int, **kwargs) -> Dict:
        sliced_data = {key: value[start:end:skip_frame] for key, value in data.items()}
        return sliced_data

    def do_pad(self, data: Dict, pad_len: int, pad_pos: Literal["left", "right"], **kwargs) -> Tuple[Dict, np.array]:
        pad_data = dict()
        for key, value in data.items():
            traj_len = len(value)
            if isinstance(value, np.ndarray):
                if pad_pos == "left":
                    pad_data[key] = np.concatenate([np.zeros(pad_len, dtype=value.dtype), value], axis=0)
                elif pad_pos == "right":
                    pad_data[key] = np.concatenate([value, np.zeros(pad_len, dtype=value.dtype)], axis=0)
            else:
                if pad_pos == "left":
                    pad_data[key] = [None] * pad_len + value
                elif pad_pos == "right":
                    pad_data[key] = value + [None] * pad_len
        if pad_pos == "left":
            pad_mask = np.concatenate([np.zeros(pad_len, dtype=np.uint8), np.ones(traj_len, dtype=np.uint8)], axis=0)
        elif pad_pos == "right":
            pad_mask = np.concatenate([np.ones(traj_len, dtype=np.uint8), np.zeros(pad_len, dtype=np.uint8)], axis=0)
        return pad_data, pad_mask

class MetaInfoDrawFrameCallback(DrawFrameCallback):

    def __init__(self, start_point: Tuple[int, int]=(150, 10)):
        super().__init__()
        self.x, self.y = start_point

    def draw_frames(self, frames: List, infos: Dict, sample_idx: int) -> np.ndarray:
        cache_frames = []
        for frame_idx, frame in enumerate(frames):
            frame = frame.copy()
            meta_info = infos['meta_info']
            try:
                pitch = meta_info['pitch'][sample_idx][frame_idx]
                yaw = meta_info['yaw'][sample_idx][frame_idx]
                cursor_x = meta_info['cursor_x'][sample_idx][frame_idx]
                cursor_y = meta_info['cursor_y'][sample_idx][frame_idx]
                isGuiInventory = meta_info['isGuiInventory'][sample_idx][frame_idx]
                isGuiOpen = meta_info['isGuiOpen'][sample_idx][frame_idx]
                cv2.putText(frame, f"Pitch: {pitch:.2f}", (self.x+10, self.y+35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Yaw: {yaw:.2f}", (self.x+10, self.y+55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"isGuiOpen: {isGuiOpen}", (self.x+10, self.y+75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"isGuiInventory: {isGuiInventory}", (self.x+10, self.y+95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"CursorX: {int(cursor_x)}", (self.x+10, self.y+115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"CursorY: {int(cursor_y)}", (self.x+10, self.y+135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            except:
                pass
            cache_frames.append(frame)
        return cache_frames

import re
from rich import print
from tqdm import tqdm
from collections import OrderedDict

class MetaInfoConvertCallback(ModalConvertCallback):

    def load_episodes(self):
        
        CONTRACTOR_PATTERN = r"^(.*?)-(\d+)$"
        
        episodes = OrderedDict()
        num_segments = 0
        for source_dir in self.input_dirs:
            print("Current input directory: ", source_dir) # action file ends with `.pkl`
            for file_path in tqdm(Path(source_dir).rglob("*.pkl"), desc="Looking for source files"):
                file_name = file_path.stem
                match = re.match(CONTRACTOR_PATTERN, file_name)
                if match:
                    eps, part_id = match.groups()
                else:
                    eps, part_id = file_name, "0"
                if eps not in episodes:
                    episodes[eps] = []
                episodes[eps].append( (part_id, file_path) )
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
            for part_id, file_path in segs:
                if int(part_id) - start_time >= MAX_TIME:
                    working_ord = part_id
                    new_episodes[f"{eps}-{working_ord}"] = []
                start_time = int(part_id)
                new_episodes[f"{eps}-{working_ord}"].append( (part_id, file_path) )
        episodes = new_episodes
        print(f'[Meta Info] - num of episodes: {len(episodes)}, num of segments: {num_segments}') 
        return episodes

    def do_convert(self, 
                   eps_id: str, 
                   skip_frames: List[List[bool]], 
                   modal_file_path: List[Union[str, Path]]) -> Tuple[List, List]:
        cache, keys, vals = [], [], []
        for _skip_frames, _modal_file_path in zip(skip_frames, modal_file_path):
            data = pickle.load(open(str(_modal_file_path), 'rb'))
            if _skip_frames is not None:
                cache += [ info for info, flag in zip(data, _skip_frames) if flag ]
            else:
                cache += data

        for chunk_start in range(0, len(cache), self.chunk_size):
            chunk_end = chunk_start + self.chunk_size
            if chunk_end > len(cache):
                break
            val = cache[chunk_start:chunk_end]
            keys.append(chunk_start)
            vals.append(pickle.dumps(val)) 

        return keys, vals

if __name__ == '__main__':
    """
    for debugging purpose
    """
    meta_info_convert = MetaInfoConvertCallback(
        input_dirs=[
            "/nfs-shared/data/contractors/all_9xx_Jun_29/privileged_infos"
        ], 
        chunk_size=32
    )
    episodes = meta_info_convert.load_episodes()
    for idx, (key, val) in enumerate(episodes.items()):
        print(key, val)
        if idx > 5:
            break
    import ipdb; ipdb.set_trace()
    