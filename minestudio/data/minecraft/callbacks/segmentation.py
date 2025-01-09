'''
Date: 2025-01-09 05:42:00
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2025-01-09 15:45:06
FilePath: /MineStudio/minestudio/data/minecraft/callbacks/segmentation.py
'''
import cv2
import random
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List, Dict, Callable, Any, Optional, Literal

from minestudio.data.minecraft.callbacks.callback import ModalKernelCallback, DrawFrameCallback
from minestudio.utils.register import Registers

SEG_RE_MAP = {
    0: 0, 1: 3, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6
}

@Registers.modal_kernel_callback.register
class SegmentationKernelCallback(ModalKernelCallback):

    def create_from_config(config: Dict) -> 'SegmentationKernelCallback':
        return SegmentationKernelCallback(**config.get('segmentation', {}))

    def __init__(self, frame_width: int=224, frame_height: int=224):
        super().__init__()
        self.width = frame_width
        self.height = frame_height

    @property
    def name(self) -> str:
        return 'segmentation'

    def rle_encode(self, binary_mask):
        '''
        binary_mask: numpy array, 1 - mask, 0 - background
        Returns run length as string formated
        '''
        pixels = binary_mask.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

    def rle_decode(self, mask_rle, shape):
        '''
        mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return
        Returns numpy array, 1 - mask, 0 - background
        '''
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        binary_mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            binary_mask[lo:hi] = 1
        return binary_mask.reshape(shape)

    def filter_dataset_paths(self, dataset_paths: List[Union[str, Path]]) -> List[Path]:
        if isinstance(dataset_paths[0], str):
            dataset_paths = [Path(path) for path in dataset_paths]
        action_paths = [path for path in dataset_paths if Path(path).stem == 'segment']
        return action_paths

    def do_decode(self, chunk: bytes) -> Dict:
        return pickle.loads(chunk)

    def do_merge(self, chunk_list: List[bytes]) -> Dict:
        raw_content = []
        for chunk_bytes in chunk_list:
            raw_content += self.do_decode(chunk_bytes)

        nb_frames = len(raw_content)
        res_content = {
            "obj_id": [-1 for _ in range(nb_frames)],
            "obj_mask": [np.zeros((self.height, self.width), dtype=np.uint8) for _ in range(nb_frames)], 
            "event": ["" for _ in range(nb_frames)],
            "point": [np.array((-1, -1)) for _ in range(nb_frames)],
            "frame_id": [-1 for _ in range(nb_frames)],
            "frame_range": [np.array((-1, -1)) for _ in range(nb_frames)],
            "third_view_frame_id": [-1 for _ in range(nb_frames)],
        }
        
        last_key = None
        candidate_third_view_frame_ids = []
        for frame_idx in range(len(raw_content)-1, -2, -1):

            if frame_idx == -1 or last_key is None or last_key not in raw_content[frame_idx]:

                if len(candidate_third_view_frame_ids) > 0:
                    # the end of an interaction, do backward search
                    third_view_frame_id = random.choice(candidate_third_view_frame_ids)
                    for backward_frame_id in candidate_third_view_frame_ids:
                        res_content["third_view_frame_id"][backward_frame_id] = third_view_frame_id
                    candidate_third_view_frame_ids = []

                if frame_idx >= 0 and len(raw_content[frame_idx]) > 0:
                    # the start of a new interaction
                    last_key = random.choice(list(raw_content[frame_idx].keys()))
                    last_event = raw_content[frame_idx][last_key]["event"]
            
            if frame_idx == -1 or len(raw_content[frame_idx]) == 0:
                continue
            
            # during an interaction, `last_key` denotes the selected interaction
            frame_content = raw_content[frame_idx][last_key]
            res_content["obj_id"][frame_idx] = SEG_RE_MAP[ frame_content["obj_id"] ]
            obj_mask = self.rle_decode(frame_content["rle_mask"], (360, 640))
            res_content["obj_mask"][frame_idx] = cv2.resize(obj_mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            res_content["event"][frame_idx] = frame_content["event"]
            if frame_content["point"] is not None:
                res_content["point"][frame_idx] = np.array(frame_content["point"])
            res_content["frame_id"][frame_idx] = frame_content["frame_id"]
            res_content["frame_range"][frame_idx] = np.array(frame_content["frame_range"])
            candidate_third_view_frame_ids.append(frame_idx)

        for key in res_content:
            if key == 'event':
                continue
            res_content[key] = np.array(res_content[key])

        return res_content

    def do_slice(self, data: Dict, start: int, end: int, skip_frame: int) -> Dict:
        sliced_data = {key: value[start:end:skip_frame] for key, value in data.items()}
        return sliced_data

    def do_pad(self, data: Dict, win_len: int) -> Tuple[Dict, np.ndarray]:
        traj_len = len(data['obj_id'])
        pad_data = dict()
        pad_data['obj_id'] = np.concatenate([data['obj_id'], np.zeros(win_len-traj_len, dtype=np.int32)], axis=0)
        pad_data['obj_mask'] = np.concatenate([data['obj_mask'], np.zeros((win_len-traj_len, *data['obj_mask'].shape[1:]), dtype=np.uint8)], axis=0)
        pad_data['event'] = data['event'] + [''] * (win_len - traj_len)
        pad_data['point'] = np.concatenate([data['point'], np.zeros((win_len-traj_len, 2), dtype=np.int32)-1], axis=0)
        pad_data['frame_id'] = np.concatenate([data['frame_id'], np.zeros(win_len-traj_len, dtype=np.int32)-1], axis=0)
        pad_data['frame_range'] = np.concatenate([data['frame_range'], np.zeros((win_len-traj_len, 2), dtype=np.int32)-1], axis=0)
        pad_data['third_view_frame_id'] = np.concatenate([data['third_view_frame_id'], np.zeros(win_len-traj_len, dtype=np.int32)-1], axis=0)
        pad_mask = np.concatenate([np.ones(traj_len, dtype=np.uint8), np.zeros(win_len-traj_len, dtype=np.uint8)], axis=0)
        return pad_data, pad_mask

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), 
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 255, 255), (0, 0, 0), (128, 128, 128),
    (128, 0, 0), (128, 128, 0), (0, 128, 0),
    (128, 0, 128), (0, 128, 128), (0, 0, 128),
]

class SegmentationDrawFrameCallback(DrawFrameCallback):

    def __init__(self, start_point: Tuple[int, int]=(300, 10)):
        super().__init__()
        self.x, self.y = start_point

    def draw_frames(self, frames: Union[np.ndarray, List], infos: Dict, sample_idx: int) -> np.ndarray:
        cache_frames = []
        for frame_idx, frame in enumerate(frames):
            frame = frame.copy()
            frame_info = infos['segmentation']
            obj_id = frame_info['obj_id'][sample_idx][frame_idx].item()
            if obj_id != -1:
                obj_mask = frame_info['obj_mask'][sample_idx][frame_idx]
                if isinstance(obj_mask, torch.Tensor):
                    obj_mask = obj_mask.numpy()
                colors = np.array(COLORS[obj_id]).reshape(1, 1, 3)
                obj_mask = (obj_mask[..., None] * colors).astype(np.uint8)
                obj_mask = obj_mask[:, :, ::-1] # bgr -> rgb
                if frame_info['point'][sample_idx][frame_idx][0] != -1:
                    y, x = frame_info['point'][sample_idx][frame_idx]
                    x, y = x.item(), y.item()
                    cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                frame = cv2.addWeighted(frame, 1.0, obj_mask, 0.5, 0.0)
            cv2.putText(frame, f"Event: {frame_info['event'][sample_idx][frame_idx]}", (self.x+10, self.y+35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cache_frames.append(frame)
        return cache_frames

