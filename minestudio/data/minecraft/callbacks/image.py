'''
Date: 2025-01-09 05:07:59
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-01-21 22:28:22
FilePath: /MineStudio/minestudio/data/minecraft/callbacks/image.py
'''
import re
import io
import av
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from rich import print
from tqdm import tqdm
from functools import partial
from multiprocessing.pool import ThreadPool, Pool
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from typing import Union, Tuple, List, Dict, Callable, Any, Optional, Literal

from minestudio.data.minecraft.callbacks.callback import ModalKernelCallback, ModalConvertCallback
from minestudio.utils.register import Registers

class VideoAugmentation:
    """
    Applies a sequence of augmentations to video frames.
    """
    
    def __init__(self, frame_width: int = 224, frame_height: int = 224):
        """
        Initializes the VideoAugmentation class.

        :param frame_width: The width of the frames to augment.
        :type frame_width: int
        :param frame_height: The height of the frames to augment.
        :type frame_height: int
        """
        self.transform = A.ReplayCompose([
            A.Sequential([
                A.ColorJitter(hue=(-0.1, 0.1), saturation=(0.8, 1.2), brightness=(0.8, 1.2), contrast=(0.8, 1.2), p=1.0), 
                A.Affine(rotate=(-4, 2), scale=(0.98, 1.02), shear=2, p=1.0),
            ], p=1.0), 
        ])

    def __call__(self, video: np.ndarray) -> np.ndarray:
        """
        Applies the defined augmentations to a video.

        :param video: A numpy array representing the video (sequence of frames).
        :type video: np.ndarray
        :returns: The augmented video as a numpy array.
        :rtype: np.ndarray
        """
        data = self.transform(image=video[0])
        future_images = []
        with ThreadPoolExecutor() as executor:
            for image in video:
                future_images += [executor.submit(partial(A.ReplayCompose.replay, data['replay'], image=image))]
        video = [future.result()['image'] for future in future_images]
        aug_video = np.array(video).astype(np.uint8)
        return aug_video

@Registers.modal_kernel_callback.register
class ImageKernelCallback(ModalKernelCallback):
    """
    Callback for processing image (video frame) data.

    Handles decoding, merging, slicing, padding, and optional augmentation of video frames.
    """

    def create_from_config(config: Dict) -> 'ImageKernelCallback':
        """
        Creates an ImageKernelCallback instance from a configuration dictionary.

        :param config: Configuration dictionary.
        :type config: Dict
        :returns: An instance of ImageKernelCallback.
        :rtype: ImageKernelCallback
        """
        return ImageKernelCallback(**config.get('image', {}))

    def __init__(
        self, 
        frame_width: int=128, 
        frame_height: int=128, 
        num_workers: int=4, 
        enable_video_aug: bool=False,
    ) -> None:
        """
        Initializes the ImageKernelCallback.

        :param frame_width: Target width for frames.
        :type frame_width: int
        :param frame_height: Target height for frames.
        :type frame_height: int
        :param num_workers: Number of worker threads for decoding.
        :type num_workers: int
        :param enable_video_aug: Whether to enable video augmentation.
        :type enable_video_aug: bool
        """
        super().__init__()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.num_workers = num_workers
        self.enable_video_aug = enable_video_aug
        if enable_video_aug:
            self.video_augmentor = self.augmentor = VideoAugmentation(frame_width,frame_height)

    @property
    def name(self) -> str:
        """
        Returns the name of the callback.

        :returns: The name 'image'.
        :rtype: str
        """
        return 'image'

    def filter_dataset_paths(self, dataset_paths: List[str]) -> List[str]:
        """
        Filters dataset paths to select only image/video related paths.

        :param dataset_paths: A list of dataset paths.
        :type dataset_paths: List[str]
        :returns: A list of paths pointing to image/video data.
        :rtype: List[str]
        """
        action_paths = [path for path in dataset_paths if Path(path).stem in ['video', 'image']]
        return action_paths

    def do_decode(self, chunk: bytes, **kwargs) -> np.ndarray:
        """
        Decodes a chunk of bytes (video data) into a numpy array of frames.

        :param chunk: Bytes representing video data.
        :type chunk: bytes
        :param kwargs: Additional keyword arguments.
        :returns: A numpy array of decoded and resized video frames (T, H, W, C).
        :rtype: np.ndarray
        """
        def convert_and_resize(frame, width, height):
            frame = frame.to_ndarray(format="rgb24")
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            return frame
        
        future_frames = []
        with io.BytesIO(chunk) as input:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                container = av.open(input, "r")
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                packet_generator = container.demux(stream)
                for packet in packet_generator:
                    for av_frame in packet.decode():
                        future = executor.submit(convert_and_resize, av_frame, self.frame_width, self.frame_height)
                        future_frames.append(future)
                frames = [future.result() for future in future_frames]
                stream.close()
                container.close()

        frames = np.array(frames)
        return frames
    
    def do_merge(self, chunk_list: List[bytes], **kwargs) -> np.ndarray:
        """
        Merges a list of decoded video chunks (byte strings) into a single numpy array of frames.

        :param chunk_list: List of byte chunks representing video segments.
        :type chunk_list: List[bytes]
        :param kwargs: Additional keyword arguments.
        :returns: A numpy array of merged video frames.
        :rtype: np.ndarray
        """
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            frames = list(executor.map(self.do_decode, chunk_list))
        merged_chunks = np.concatenate(frames, axis=0)
        return merged_chunks
    
    def do_slice(self, data: np.ndarray, start: int, end: int, skip_frame: int, **kwargs) -> np.ndarray:
        """
        Slices the video frame data.

        :param data: Numpy array of video frames.
        :type data: np.ndarray
        :param start: Start index for slicing.
        :type start: int
        :param end: End index for slicing.
        :type end: int
        :param skip_frame: Frame skipping interval.
        :type skip_frame: int
        :param kwargs: Additional keyword arguments.
        :returns: Sliced video frame data.
        :rtype: np.ndarray
        """
        return data[start:end:skip_frame]
    
    def do_pad(self, data: np.ndarray, pad_len: int, pad_pos: Literal["left", "right"], **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pads the video frame data.

        :param data: Numpy array of video frames.
        :type data: np.ndarray
        :param pad_len: Length of padding to add.
        :type pad_len: int
        :param pad_pos: Position to add padding ("left" or "right").
        :type pad_pos: Literal["left", "right"]
        :param kwargs: Additional keyword arguments.
        :returns: A tuple containing the padded video data and the padding mask.
        :rtype: Tuple[np.ndarray, np.ndarray]
        :raises ValueError: if pad_pos is not "left" or "right".
        """
        dims = data.shape[1:]
        if pad_pos == "left":
            pad_data = np.concatenate([np.zeros((pad_len, *dims), dtype=np.uint8), data], axis=0)
            pad_mask = np.concatenate([np.zeros(pad_len, dtype=np.uint8), np.ones(data.shape[0], dtype=np.uint8)], axis=0)
        elif pad_pos == "right":
            pad_data = np.concatenate([data, np.zeros((pad_len, *dims), dtype=np.uint8)], axis=0)
            pad_mask = np.concatenate([np.ones(data.shape[0], dtype=np.uint8), np.zeros(pad_len, dtype=np.uint8)], axis=0)
        else:
            raise ValueError(f"Invalid pad position: {pad_pos}")
        return pad_data, pad_mask

    def do_postprocess(self, data: Dict) -> Dict:
        """
        Postprocesses the image data, applying video augmentation if enabled.

        :param data: Data dictionary containing image frames.
        :type data: Dict
        :returns: Postprocessed data dictionary.
        :rtype: Dict
        """
        if self.enable_video_aug:
            data["image"] = self.video_augmentor(data["image"])
        return data

class ImageConvertCallback(ModalConvertCallback):
    """
    Callback for converting raw video data into the MineStudio format.
    """

    def __init__(self, *args, thread_pool: int=8, **kwargs):
        """
        Initializes the ImageConvertCallback.

        :param args: Positional arguments for the parent class.
        :param thread_pool: Number of threads to use for video writing.
        :type thread_pool: int
        :param kwargs: Keyword arguments for the parent class.
        """
        super().__init__(*args, **kwargs)
        self.thread_pool = thread_pool

    def load_episodes(self):
        """
        Loads and organizes video episode data from input directories.

        It identifies video files (ending with .mp4), groups them by episode,
        sorts segments within episodes, and re-splits episodes based on a maximum time interval.

        :returns: An OrderedDict of episodes, where keys are episode IDs and values are lists of (part_id, file_path) tuples.
        :rtype: OrderedDict
        """
        CONTRACTOR_PATTERN = r"^(.*?)-(\d+)$"
        episodes = OrderedDict()
        num_segments = 0
        for source_dir in self.input_dirs:
            print("Current input directory: ", source_dir) # action file ends with `.pkl`
            for file_path in tqdm(Path(source_dir).rglob("*.mp4"), desc="Looking for source files"):
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
        print(f'[Image] - num of episodes: {len(episodes)}, num of segments: {num_segments}') 
        return episodes

    def _write_video_chunk(self, args: Tuple) -> Tuple[int, bytes]:
        """
        Converts a chunk of video frames into a byte sequence (mp4 format).

        :param args: A tuple containing (frames, chunk_start, fps, width, height).
        :type args: Tuple
        :returns: A tuple containing (chunk_start_index, video_bytes).
        :rtype: Tuple[int, bytes]
        """
        frames, chunk_start, fps, width, height = args
        outStream = io.BytesIO()
        container = av.open(outStream, mode="w", format='mp4')
        stream = container.add_stream("h264", rate=fps)
        stream.width = width
        stream.height = height
        for frame in frames:
            frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        bytes = outStream.getvalue()
        outStream.close()
        return chunk_start, bytes

    def do_convert(self, 
                   eps_id: str, 
                   skip_frames: List[List[bool]], 
                   modal_file_path: List[Union[str, Path]]) -> Tuple[List, List]:
        """
        Converts video data for a given episode.

        It reads video frames from specified files, applies frame skipping,
        resizes frames, and encodes them into chunks.

        :param eps_id: Episode ID.
        :type eps_id: str
        :param skip_frames: A list of lists of boolean flags indicating whether to skip each frame for each segment file.
        :type skip_frames: List[List[bool]]
        :param modal_file_path: A list of file paths for the video data segments.
        :type modal_file_path: List[Union[str, Path]]
        :returns: A tuple containing a list of chunk start indices and a list of serialized chunk values (video bytes).
        :rtype: Tuple[List, List]
        """
        chunk_start = 0
        cache_frames, keys, vals = [], [], []
        if isinstance(modal_file_path, str):
            modal_file_path = Path(modal_file_path)
        
        for _skip_frames, _modal_file_path in zip(skip_frames, modal_file_path):
            # Get video meta-information
            cap = cv2.VideoCapture(str(_modal_file_path.absolute()))
            cv_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cv_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cv_fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = 0

            if cv_width != 640 or cv_height != 360:
                return [], []

            # Decode and encode frames
            container = av.open(str(_modal_file_path.absolute()), "r")
            for fid, frame in enumerate(container.decode(video=0)):
                total_frames += 1
                if _skip_frames is not None:
                    if fid >= len(_skip_frames) or (not _skip_frames[fid]):
                        continue
                frame = frame.to_ndarray(format="rgb24")
                # #! reszie to 224, 224
                cv_width, cv_height = 224, 224
                cv2.resize(frame, (cv_width, cv_height), interpolation=cv2.INTER_LINEAR)
                # #! reszie to 224, 224
                cache_frames.append(frame)
                if len(cache_frames) == self.chunk_size * self.thread_pool:
                    with ThreadPool(self.thread_pool) as pool:
                        args_list = []
                        while len(cache_frames) >= self.chunk_size:
                            chunk_end = chunk_start + self.chunk_size
                            args_list.append((cache_frames[:self.chunk_size], chunk_start, cv_fps, cv_width, cv_height))
                            
                            chunk_start += self.chunk_size
                            cache_frames = cache_frames[self.chunk_size:]
                        
                        for idx, bytes in pool.map(self._write_video_chunk, args_list):
                            keys.append(idx)
                            vals.append(bytes)

            if _skip_frames is None or len(_skip_frames) <= total_frames <= len(_skip_frames) + 1:  
                pass
            else:
                print(f"Warning: Expected frame numbers: {len(_skip_frames)}, actual frame numbers: {total_frames}. Source: {source_path}")
            
            print(f"episode: {eps_id}, segment: {ord}, frames: {total_frames}")
            
            # Close segment container
            container.close()

        # Encode remaining frames
        while len(cache_frames) >= self.chunk_size:
            idx, bytes = self._write_video_chunk((cache_frames[:self.chunk_size], chunk_start, cv_fps, cv_width, cv_height))
            keys.append(idx)
            vals.append(bytes)
            chunk_start += self.chunk_size
            cache_frames = cache_frames[self.chunk_size:]
        return keys, vals

if __name__ == '__main__':
    """
    for debugging purpose
    """
    image_convert = ImageConvertCallback(
        input_dirs=[
            "/nfs-shared/data/contractors/all_9xx_Jun_29/videos"
        ], 
        chunk_size=32
    )
    episodes = image_convert.load_episodes()
    for idx, (key, val) in enumerate(episodes.items()):
        print(key, val)
        if idx > 5:
            break
    import ipdb; ipdb.set_trace()
