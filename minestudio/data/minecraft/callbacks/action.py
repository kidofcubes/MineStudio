'''
Date: 2025-01-09 05:27:25
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2025-01-10 08:56:33
FilePath: /MineStudio/minestudio/data/minecraft/callbacks/action.py
'''
import cv2
import pickle
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List, Dict, Callable, Any, Optional, Literal

from minestudio.utils.vpt_lib.actions import ActionTransformer
from minestudio.utils.vpt_lib.action_mapping import CameraHierarchicalMapping
from minestudio.data.minecraft.callbacks.callback import ModalKernelCallback, DrawFrameCallback, ModalConvertionCallback
from minestudio.utils.register import Registers

@Registers.modal_kernel_callback.register
class ActionKernelCallback(ModalKernelCallback):

    def create_from_config(config: Dict) -> 'ActionKernelCallback':
        return ActionKernelCallback(**config.get('action', {}))

    def __init__(self, 
                 n_camera_bins: int=11,
                 camera_binsize: int=2,
                 camera_maxval: int=10,
                 camera_mu: int=10,
                 camera_quantization_scheme="mu_law"):
        super().__init__()
        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=n_camera_bins)
        self.action_transformer = ActionTransformer(
            camera_binsize=camera_binsize,
            camera_maxval=camera_maxval,
            camera_mu=camera_mu,
            camera_quantization_scheme=camera_quantization_scheme
        )

    @property
    def name(self) -> str:
        return 'action'

    def filter_dataset_paths(self, dataset_paths: List[Union[str, Path]]) -> List[Path]:
        if isinstance(dataset_paths[0], str):
            dataset_paths = [Path(path) for path in dataset_paths]
        action_paths = [path for path in dataset_paths if Path(path).stem == 'action']
        return action_paths

    def do_decode(self, chunk: bytes) -> Dict:
        return pickle.loads(chunk)

    def do_merge(self, chunk_list: List[bytes]) -> Dict:
        chunks = [self.do_decode(chunk) for chunk in chunk_list]
        cache_chunks = {}
        for chunk in chunks:
            for key, value in chunk.items():
                if key not in cache_chunks:
                    cache_chunks[key] = []
                cache_chunks[key].append(value)
        merged_chunks = {key: np.concatenate(value, axis=0) for key, value in cache_chunks.items()}
        return merged_chunks

    def do_slice(self, data: Dict, start: int, end: int, skip_frame: int) -> Dict:
        sliced_data = {key: value[start:end:skip_frame] for key, value in data.items()}
        return sliced_data

    def do_pad(self, data: Dict, win_len: int) -> Tuple[Dict, np.ndarray]:
        pad_data = dict()
        for key, value in data.items():
            traj_len = value.shape[0]
            dims = value.shape[1:]
            pad_value = np.concatenate([value, np.zeros((win_len-traj_len, *dims), dtype=np.uint8)], axis=0)
            pad_data[key] = pad_value
        pad_mask = np.concatenate([np.ones(traj_len, dtype=np.uint8), np.zeros(win_len-traj_len, dtype=np.uint8)], axis=0)
        return pad_data, pad_mask

    def do_postprocess(self, data: Dict) -> Dict:
        action = data.pop('action')
        data['env_action'] = action
        data['agent_action'] = self.action_mapper.from_factored(
            self.action_transformer.env2policy(action)
        )
        return data

class ActionDrawFrameCallback(DrawFrameCallback):

    def __init__(self, start_point: Tuple[int, int]=(10, 10)):
        super().__init__()
        self.x, self.y = start_point

    def draw_frames(self, frames: Union[np.ndarray, List], infos: Dict, sample_idx: int) -> np.ndarray:
        cache_frames = []
        env_action = infos['env_action']
        prev_env_action = infos.get('prev_env_action', env_action)
        for frame_idx, frame in enumerate(frames):
            frame = frame.copy()
            act = {k: v[sample_idx][frame_idx].numpy() for k, v in env_action.items()}
            prev_act = {k: v[sample_idx][frame_idx].numpy() for k, v in prev_env_action.items()}
            current_row = 0
            for (k, v), (_, pv) in zip(act.items(), prev_act.items()):
                if 'hotbar' in k:
                    continue
                if k != 'camera':
                    v = int(v.item())
                    pv = int(pv.item())
                else:
                    v = f"[{v[0].item():.3f}, {v[1].item():.3f}]"
                    pv = f"[{pv[0].item():.3f}, {pv[1].item():.3f}]"
                cv2.putText(frame, f"{k}: {v}({pv})", (self.x, self.y+35+current_row*15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                current_row += 1
            cache_frames.append(frame)
        return cache_frames

class ActionConvertionCallback(ModalConvertionCallback):

    def do_convert(self, 
                   eps_id: str, 
                   skip_frames: List[List[bool]], 
                   modal_file_path: List[Union[str, Path]]) -> Tuple[List, List]:
        cache, keys, vals = [], [], []
        for _skip_frames, _modal_file_path in zip(skip_frames, modal_file_path):
            data = pickle.load(open(str(_modal_file_path), 'rb'))
            if len(cache) == 0:
                cache = {k: v[_skip_frames] for k, v in data.items()}
            else:
                for k, v in data.items():
                    cache[k] = np.concatenate((cache[k], v[_skip_frames]), axis=0)

        for chunk_start in range(0, len(cache['attack']), self.chunk_size):
            chunk_end = chunk_start + self.chunk_size
            if chunk_end > len(cache['attack']):
                break
            val = {k: v[chunk_start:chunk_end] for k, v in cache.items()}
            keys.append(chunk_start)
            vals.append(pickle.dumps(val))
        
        return keys, vals