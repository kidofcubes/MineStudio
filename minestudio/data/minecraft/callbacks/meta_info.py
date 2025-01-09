'''
Date: 2025-01-09 05:36:19
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2025-01-09 15:33:35
FilePath: /MineStudio/minestudio/data/minecraft/callbacks/meta_info.py
'''
import cv2
import pickle
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List, Dict, Callable, Any, Optional, Literal

from minestudio.data.minecraft.callbacks.callback import ModalKernelCallback, DrawFrameCallback
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
        action_paths = [path for path in dataset_paths if Path(path).stem == 'contractor_info']
        return action_paths

    def do_decode(self, chunk: bytes) -> Dict:
        return pickle.loads(chunk)

    def do_merge(self, chunk_list: List[bytes]) -> Dict:
        chunks = [self.do_decode(chunk) for chunk in chunk_list]
        cache_chunks = {}
        for chunk in chunks:
            for frame_info in chunk:
                for key, value in frame_info.items():
                    if key not in cache_chunks:
                        cache_chunks[key] = []
                    cache_chunks[key].append(value)
        return cache_chunks

    def do_slice(self, data: Dict, start: int, end: int, skip_frame: int) -> Dict:
        sliced_data = {key: value[start:end:skip_frame] for key, value in data.items()}
        return sliced_data

    def do_pad(self, data: Dict, win_len: int) -> Tuple[Dict, np.array]:
        pad_data = dict()
        for key, value in data.items():
            traj_len = len(value)
            if isinstance(value, np.ndarray):
                pad_data[key] = np.concatenate([np.array(value), np.zeros(win_len-traj_len, dtype=value.dtype)], axis=0)
            else:
                pad_data[key] = value + [None] * (win_len - len(value))
        pad_mask = np.concatenate([np.ones(traj_len, dtype=np.uint8), np.zeros(win_len-traj_len, dtype=np.uint8)], axis=0)
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