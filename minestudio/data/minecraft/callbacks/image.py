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
    
    def __init__(self, frame_width: int = 224, frame_height: int = 224):
        self.transform = A.ReplayCompose([
            A.Sequential([
                A.ColorJitter(hue=(-0.1, 0.1), saturation=(0.8, 1.2), brightness=(0.8, 1.2), contrast=(0.8, 1.2), p=1.0), 
                A.Affine(rotate=(-4, 2), scale=(0.98, 1.02), shear=2, p=1.0),
                # A.OneOf([
                #     A.CropAndPad(px=(0, 30), keep_size=True, p=1.0),
                #     A.RandomResizedCrop(scale=(0.9, 0.9), ratio=(1.0, 1.0), width=frame_width, height=frame_height, p=1.0),
                # ], p=1.0),  
            ], p=1.0), 
        ])

    def __call__(self, video: np.ndarray) -> np.ndarray:
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

    def create_from_config(config: Dict) -> 'ImageKernelCallback':
        return ImageKernelCallback(**config.get('image', {}))

    def __init__(
        self, 
        frame_width: int=128, 
        frame_height: int=128, 
        num_workers: int=4, 
        enable_video_aug: bool=False,
    ) -> None:
        super().__init__()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.num_workers = num_workers
        self.enable_video_aug = enable_video_aug
        if enable_video_aug:
            self.video_augmentor = self.augmentor = VideoAugmentation(frame_width,frame_height)

    @property
    def name(self) -> str:
        return 'image'

    def filter_dataset_paths(self, dataset_paths: List[str]) -> List[str]:
        action_paths = [path for path in dataset_paths if Path(path).stem in ['video', 'image']]
        return action_paths

    def do_decode(self, chunk: bytes, **kwargs) -> np.ndarray:
        """Decode bytes to video frames."""

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
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            frames = list(executor.map(self.do_decode, chunk_list))
        merged_chunks = np.concatenate(frames, axis=0)
        return merged_chunks
    
    def do_slice(self, data: np.ndarray, start: int, end: int, skip_frame: int, **kwargs) -> np.ndarray:
        return data[start:end:skip_frame]
    
    def do_pad(self, data: np.ndarray, pad_len: int, pad_pos: Literal["left", "right"], **kwargs) -> Tuple[np.ndarray, np.ndarray]:
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
        if self.enable_video_aug:
            data["image"] = self.video_augmentor(data["image"])
        return data

class ImageConvertCallback(ModalConvertCallback):

    def __init__(self, *args, thread_pool: int=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.thread_pool = thread_pool

    def load_episodes(self):
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

    def _write_video_chunk(self, args: Tuple) -> Tuple[bool, int, bytes]:
        '''Convert frame sequence into bytes sequence. '''
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
    