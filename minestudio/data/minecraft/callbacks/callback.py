'''
Date: 2025-01-09 05:08:19
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2025-01-09 12:07:53
FilePath: /MineStudio/minestudio/data/minecraft/callbacks/callback.py
'''
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List, Dict, Callable, Any, Optional, Literal

class ModalKernelCallback:

    def create_from_config(config: Dict) -> 'ModalKernelCallback':
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError
    
    def filter_dataset_paths(self, dataset_paths: List[Union[str, Path]]) -> List[Path]:
        """
        Given a list of directories, return the list of directories that should be processed. 
        """
        raise NotImplementedError
    
    def do_decode(self, chunk: bytes, **kwargs) -> Any:
        raise NotImplementedError
    
    def do_merge(self, chunk_list: List[bytes], **kwargs) -> Union[List, Dict]:
        """
        First call do_decode to decode the chunk_list, then merge the decoded data.
        """
        raise NotImplementedError

    def do_slice(self, data: Union[List, Dict], start: int, end: int, skip_frame: int, **kwargs) -> Union[List, Dict]:
        raise NotImplementedError

    def do_pad(self, data: Union[List, Dict], win_len: int, **kwargs) -> Tuple[Union[List, Dict], np.ndarray]:
        """
        return the padded data and the padding mask
        """
        raise NotImplementedError

    def do_postprocess(self, data: Dict, **kwargs) -> Dict:
        return data

class DrawFrameCallback:
    
    def draw_frames(self, frames: Union[np.ndarray, List], infos: Dict, sample_idx: int, **kwargs) -> np.ndarray:
        raise NotImplementedError