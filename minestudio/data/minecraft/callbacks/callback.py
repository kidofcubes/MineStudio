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
        """
        Initializes the ModalConvertCallback.

        :param input_dirs: A list of input directories containing the raw data.
        :type input_dirs: List[str]
        :param chunk_size: The size of chunks to process data in.
        :type chunk_size: int
        """
        self.input_dirs = input_dirs
        self.chunk_size = chunk_size

    def load_episodes(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        The user needs to implement this method to identify and read raw data from the folder.
        The return value is a dictionary, where the keys are the names of complete trajectories, 
        and the values are lists of tuples, each representing which part of the episode it is 
        and the file path for that part. 

        :returns: A dictionary of episodes.
        :rtype: Dict[str, List[Tuple[str, str]]]
        :raises NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    def do_convert(self, eps_id: str, skip_frames: List[List[bool]], modal_file_path: List[Union[str, Path]]) -> Tuple[List, List]:
        """
        Note that, `skip_frames` are aligned with `modal_file_path`, they represent a list of files. 
        Given the modal file and frame skip flags (skip no action frames), return the converted frames. 
        
        :param eps_id: The ID of the episode being converted.
        :type eps_id: str
        :param skip_frames: A list of lists, where each inner list contains boolean flags indicating whether to skip a frame in the corresponding modal file part.
        :type skip_frames: List[List[bool]]
        :param modal_file_path: A list of paths to the modal data files (or parts of files).
        :type modal_file_path: List[Union[str, Path]]
        :returns: A tuple containing chunk keys and chunk values.
        :rtype: Tuple[List, List]
        :raises NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    def gen_frame_skip_flags(self, file_name: str) -> List[bool]:
        """
        If the user wants to filter out certain frames based on the information of this modal, this method should be implemented.

        :param file_name: The name of the file to generate skip flags for.
        :type file_name: str
        :returns: A list of boolean flags indicating which frames to skip.
        :rtype: List[bool]
        :raises NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError


class ModalKernelCallback:

    """
    Users must implement this callback for their customized modal data to 
    handle operations such as decoding, merging, slicing, and padding of the modal data. 
    """

    def create_from_config(config: Dict) -> 'ModalKernelCallback':
        """
        Creates a ModalKernelCallback instance from a configuration dictionary.

        :param config: Configuration dictionary.
        :type config: Dict
        :returns: An instance of ModalKernelCallback.
        :rtype: ModalKernelCallback
        :raises NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    def __init__(self, read_bias: int=0, win_bias: int=0):
        """
        Initializes the ModalKernelCallback.

        :param read_bias: Bias for reading data.
        :type read_bias: int
        :param win_bias: Bias for windowing data.
        :type win_bias: int
        """
        self.read_bias = read_bias
        self.win_bias = win_bias

    @property
    def name(self) -> str:
        """
        Returns the name of the modal.

        :returns: The name of the modal.
        :rtype: str
        :raises NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError
    
    def filter_dataset_paths(self, dataset_paths: List[Union[str, Path]]) -> List[Path]:
        """
        `dataset_paths` contains all possible paths that point to different lmdb folders.
        The user needs to implement this method to filter out the paths they need, 
        so that the pipeline knows which lmdb files to read data from. 

        :param dataset_paths: A list of dataset paths.
        :type dataset_paths: List[Union[str, Path]]
        :returns: A filtered list of dataset paths.
        :rtype: List[Path]
        :raises NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError
    
    def do_decode(self, chunk: bytes, **kwargs) -> Any:
        """
        The data is stored in lmdb files in the form of bytes, chunk by chunk, 
        and the decoding methods for different modalities of data are different. 
        Therefore, users need to implement this method to decode the data. 

        :param chunk: Bytes to decode.
        :type chunk: bytes
        :param kwargs: Additional keyword arguments.
        :returns: Decoded data.
        :rtype: Any
        :raises NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError
    
    def do_merge(self, chunk_list: List[bytes], **kwargs) -> Union[List, Dict]:
        """
        When the user reads a long segment of trajectory, the pipeline will 
        automatically read out, decode, and stitch together continuous chunks 
        into a complete sequence. Therefore, the user needs to specify how each 
        modality's chunks are merged into a complete sequence.

        :param chunk_list: List of byte chunks.
        :type chunk_list: List[bytes]
        :param kwargs: Additional keyword arguments.
        :returns: Merged data.
        :rtype: Union[List, Dict]
        :raises NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    def do_slice(self, data: Union[List, Dict], start: int, end: int, skip_frame: int, **kwargs) -> Union[List, Dict]:
        """
        Due to the possibility of completely different data formats for different data,
        users need to implement slicing methods so that they can perform slicing operations on the data.

        :param data: Data to slice.
        :type data: Union[List, Dict]
        :param start: Start index for slicing.
        :type start: int
        :param end: End index for slicing.
        :type end: int
        :param skip_frame: Frame skipping interval.
        :type skip_frame: int
        :param kwargs: Additional keyword arguments.
        :returns: Sliced data.
        :rtype: Union[List, Dict]
        :raises NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    def do_pad(self, data: Union[List, Dict], pad_len: int, pad_pos: Literal["left", "right"], **kwargs) -> Tuple[Union[List, Dict], np.ndarray]:
        """
        Users need to implement padding operations to handle cases where the data length is insufficient.

        :param data: Data to pad.
        :type data: Union[List, Dict]
        :param pad_len: Length of padding to add.
        :type pad_len: int
        :param pad_pos: Position to add padding ("left" or "right").
        :type pad_pos: Literal["left", "right"]
        :param kwargs: Additional keyword arguments.
        :returns: A tuple containing the padded data and the padding mask.
        :rtype: Tuple[Union[List, Dict], np.ndarray]
        :raises NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    def do_postprocess(self, data: Dict, **kwargs) -> Dict:
        """
        This is an optional operation, where users can add some additional actions, 
        such as performing data augmentation on the sampled data.

        :param data: Data dictionary to postprocess.
        :type data: Dict
        :param kwargs: Additional keyword arguments.
        :returns: Postprocessed data dictionary.
        :rtype: Dict
        """
        return data


class DrawFrameCallback:
    """
    Callback for drawing overlay information on video frames during visualization.
    """
    def draw_frames(self, frames: Union[np.ndarray, List], infos: Dict, sample_idx: int, **kwargs) -> np.ndarray:
        """
        When users need to visualize a dataset, this method needs to be implemented for drawing frame images.

        :param frames: A list of frames or a numpy array of frames.
        :type frames: Union[np.ndarray, List]
        :param infos: Dictionary containing information to be drawn on frames.
        :type infos: Dict
        :param sample_idx: Index of the sample to process.
        :type sample_idx: int
        :param kwargs: Additional keyword arguments.
        :returns: A numpy array of frames with information drawn on them.
        :rtype: np.ndarray
        :raises NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError

