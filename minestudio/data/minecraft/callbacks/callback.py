'''
Date: 2025-01-09 05:08:19
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-01-21 22:28:09
FilePath: /MineStudio/minestudio/data/minecraft/callbacks/callback.py
'''
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List, Dict, Callable, Any, Optional, Literal

class ModalConvertCallback:

    """
    When the user attempts to convert their own trajectory data into the built-in format of MineStudio, 
    they need to implement the methods of this class to complete the data conversion.
    """

    def __init__(self, input_dirs: List[str], chunk_size: int):
        self.input_dirs = input_dirs
        self.chunk_size = chunk_size

    def load_episodes(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        The user needs to implement this method to identify and read raw data from the folder.
        The return value is a dictionary, where the keys are the names of complete trajectories, 
        and the values are lists of tuples, each representing which part of the episode it is 
        and the file path for that part. 
        """
        raise NotImplementedError

    def do_convert(self, eps_id: str, skip_frames: List[List[bool]], modal_file_path: List[Union[str, Path]]) -> Tuple[List, List]:
        """
        Note that, `skip_frames` are aligned with `modal_file_path`, they represent a list of files. 
        Given the modal file and frame skip flags (skip no action frames), return the converted frames. 
        return: (chunk_keys, chunk_vals)
        """
        raise NotImplementedError

    def gen_frame_skip_flags(self, file_name: str) -> List[bool]:
        """
        If the user wants to filter out certain frames based on the information of this modal, this method should be implemented.
        """
        raise NotImplementedError


class ModalKernelCallback:

    """
    Users must implement this callback for their customized modal data to 
    handle operations such as decoding, merging, slicing, and padding of the modal data. 
    """

    def create_from_config(config: Dict) -> 'ModalKernelCallback':
        raise NotImplementedError

    def __init__(self, read_bias: int=0, win_bias: int=0):
        self.read_bias = read_bias
        self.win_bias = win_bias

    @property
    def name(self) -> str:
        raise NotImplementedError
    
    def filter_dataset_paths(self, dataset_paths: List[Union[str, Path]]) -> List[Path]:
        """
        `dataset_paths` contains all possible paths that point to different lmdb folders.
        The user needs to implement this method to filter out the paths they need, 
        so that the pipeline knows which lmdb files to read data from. 
        """
        raise NotImplementedError
    
    def do_decode(self, chunk: bytes, **kwargs) -> Any:
        """
        The data is stored in lmdb files in the form of bytes, chunk by chunk, 
        and the decoding methods for different modalities of data are different. 
        Therefore, users need to implement this method to decode the data. 
        """
        raise NotImplementedError
    
    def do_merge(self, chunk_list: List[bytes], **kwargs) -> Union[List, Dict]:
        """
        When the user reads a long segment of trajectory, the pipeline will 
        automatically read out, decode, and stitch together continuous chunks 
        into a complete sequence. Therefore, the user needs to specify how each 
        modality's chunks are merged into a complete sequence.
        """
        raise NotImplementedError

    def do_slice(self, data: Union[List, Dict], start: int, end: int, skip_frame: int, **kwargs) -> Union[List, Dict]:
        """
        Due to the possibility of completely different data formats for different data,
        users need to implement slicing methods so that they can perform slicing operations on the data.
        """
        raise NotImplementedError

    def do_pad(self, data: Union[List, Dict], pad_len: int, pad_pos: Literal["left", "right"], **kwargs) -> Tuple[Union[List, Dict], np.ndarray]:
        """
        Users need to implement padding operations to handle cases where the data length is insufficient.
        """
        raise NotImplementedError

    def do_postprocess(self, data: Dict, **kwargs) -> Dict:
        """
        This is an optional operation, where users can add some additional actions, 
        such as performing data augmentation on the sampled data.
        """
        return data


class DrawFrameCallback:
    
    def draw_frames(self, frames: Union[np.ndarray, List], infos: Dict, sample_idx: int, **kwargs) -> np.ndarray:
        """
        When users need to visualize a dataset, this method needs to be implemented for drawing frame images.
        """
        raise NotImplementedError

