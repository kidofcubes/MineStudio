'''
Date: 2024-11-10 10:06:28
LastEditors: Muyao 2350076251@qq.com
LastEditTime: 2025-03-04 00:29:58
FilePath: /MineStudio/var/minestudio/data/minecraft/utils.py
'''
import os
import av
import cv2
import requests
import numpy as np
from datetime import datetime
import torch
import torch.distributed as dist
import shutil
from torch.utils.data import Sampler
from tqdm import tqdm
from rich import print
import rich
import string
import random
from typing import Union, Tuple, List, Dict, Callable, Sequence, Mapping, Any, Optional, Literal
from huggingface_hub import hf_api, snapshot_download
from pathlib import Path
import pathlib
import uuid
from datetime import datetime
import pickle
import json



def get_repo_total_size(repo_id, repo_type="dataset", branch="main"):

    def fetch_file_list(path=""):
        url = f"https://huggingface.co/api/{repo_type}s/{repo_id}/tree/{branch}/{path}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def calculate_size(path=""):
        files = fetch_file_list(path)
        total_size = 0
        for file in files:
            if file["type"] == "file":
                total_size += file["size"]
            elif file["type"] == "directory":
                total_size += calculate_size(file["path"])
        return total_size
    total_size_bytes = calculate_size()
    total_size_gb = total_size_bytes / (1024 ** 3)
    return total_size_bytes, total_size_gb

def download_dataset_from_huggingface(name: Literal["6xx", "7xx", "8xx", "9xx", "10xx"], base_dir: Optional[str]=None):

    if base_dir is None:
        from minestudio.utils import get_mine_studio_dir
        base_dir = get_mine_studio_dir()
    
    total, used, free = shutil.disk_usage(base_dir)
    repo_id = f"CraftJarvis/minestudio-data-{name}"
    total_size, _ = get_repo_total_size(repo_id)
    print(
        f"""
        [bold]Download Dataset[/bold]
        Dataset: {name}
        Base Dir: {base_dir}
        Total Size: {total_size / 1024 / 1024 / 1024:.2f} GB
        Free Space: {free / 1024 / 1024 / 1024:.2f} GB
        """
    )
    if total_size > free:
        raise ValueError(f"Insufficient space for downloading {name}. ")
    dataset_dir = os.path.join(get_mine_studio_dir(), 'contractors', f'dataset_{name}')
    local_dataset_dir = snapshot_download(repo_id, repo_type="dataset", local_dir=dataset_dir)
    return local_dataset_dir

def pull_datasets_from_remote(dataset_dirs: List[str]) -> List[str]:
    new_dataset_dirs = []
    for path in dataset_dirs:
        if path in ['6xx', '7xx', '8xx', '9xx', '10xx']:
            return_path = download_dataset_from_huggingface(path)
            new_dataset_dirs.append(return_path)
        else:
            new_dataset_dirs.append(path)
    return new_dataset_dirs

import cv2

def read_video(file_path: str, start_idx: int = 0, width: int = None):
    # 打开视频文件
    cap = cv2.VideoCapture(file_path)
    
    if not cap.isOpened():
        raise ValueError("Error: Could not open video.")

    # 跳转到开始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    frames = []
    count = 0

    # 读取视频帧，直到达到指定的宽度或视频结束
    while True:
        ret, frame = cap.read()
        if not ret or (width is not None and count >= width):
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)  # 将每一帧添加到 frames 列表中
        count += 1
    
    # 释放视频文件
    cap.release()

    return frames

def write_video(
    file_name: str, 
    frames: Sequence[np.ndarray], 
    width: int = 640, 
    height: int = 360, 
    fps: int = 20
) -> None:
    """Write video frames to video files. """
    with av.open(file_name, mode="w", format='mp4') as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = width
        stream.height = height
        for frame in frames:
            assert frame.shape[1] == width and frame.shape[0] == height, f"frame shape {frame.shape} not match {width}x{height}"
            frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

def batchify(batch_in: Sequence[Dict[str, Any]]) -> Any:
    example = batch_in[0]
    if isinstance(example, Dict):
        batch_out = {
            k: batchify([item[k] for item in batch_in]) \
                for k in example.keys()
        }
    elif isinstance(example, torch.Tensor):
        batch_out = torch.stack(batch_in, dim=0)
    elif isinstance(example, int):
        batch_out = torch.tensor(batch_in, dtype=torch.int32)
    elif isinstance(example, float):
        batch_out = torch.tensor(batch_in, dtype=torch.float32)
    else:
        batch_out = batch_in
    return batch_out

class MineDistributedBatchSampler(Sampler):

    def __init__(
        self, 
        dataset, 
        batch_size, 
        num_replicas=None, # num_replicas is the number of processes participating in the training
        rank=None,         # rank is the rank of the current process within num_replicas
        shuffle=False, 
        drop_last=True,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            try:
                num_replicas = dist.get_world_size()
            except:
                num_replicas = 1
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            try:
                rank = dist.get_rank()
            except:
                rank = 0
        assert shuffle is False, "shuffle must be False in sampler."
        assert drop_last is True, "drop_last must be True in sampler."
        # print(f"{rank = }, {num_replicas = }")
        self.batch_size = batch_size
        self.dataset = dataset
        self.num_total_samples = len(self.dataset)
        self.num_samples_per_replica = self.num_total_samples // num_replicas
        replica_range = (self.num_samples_per_replica * rank, self.num_samples_per_replica * (rank + 1)) # [start, end)
        
        num_past_samples = 0
        episodes_within_replica = [] # (episode, epsode_start_idx, episode_end_idx, item_bias)
        self.episodes_with_items = self.dataset.episodes_with_items
        for episode, length, item_bias in self.episodes_with_items:
            if num_past_samples + length > replica_range[0] and num_past_samples < replica_range[1]:
                episode_start_idx = max(0, replica_range[0] - num_past_samples)
                episode_end_idx = min(length, replica_range[1] - num_past_samples)
                episodes_within_replica.append((episode, episode_start_idx, episode_end_idx, item_bias))
            num_past_samples += length
        self.episodes_within_replica = episodes_within_replica

    def __iter__(self):
        """
        Build batch of episodes, each batch is consisted of `self.batch_size` episodes.
        Only if one episodes runs out of samples, the batch is filled with the next episode.
        """
        next_episode_idx = 0
        reading_episodes = [ None for _ in range(self.batch_size) ]
        while True:
            batch = [ None for _ in range(self.batch_size) ]
            # feed `reading_episodes` with the next episode
            for i in range(self.batch_size):
                if reading_episodes[i] is None:
                    if next_episode_idx >= len(self.episodes_within_replica):
                        break
                    reading_episodes[i] = self.episodes_within_replica[next_episode_idx]
                    next_episode_idx += 1
            # use while loop to build batch
            while any([x is None for x in batch]):
                record_batch_length = sum([x is not None for x in batch])
                # get the position that needs to be filled
                for cur in range(self.batch_size):
                    if batch[cur] is None:
                        break
                # get the episode that has the next sample
                if reading_episodes[cur] is not None:
                    use_eps_idx = cur
                else:
                    for use_eps_idx in range(self.batch_size):
                        if reading_episodes[use_eps_idx] is not None:
                            break
                # if all episodes are None, then stop iteration
                if reading_episodes[use_eps_idx] is None:
                    return None
                # fill the batch with the next sample
                episode, start_idx, end_idx, item_bias = reading_episodes[use_eps_idx]
                batch[cur] = item_bias + start_idx
                if start_idx+1 < end_idx:
                    reading_episodes[use_eps_idx] = (episode, start_idx + 1, end_idx, item_bias)
                else:
                    reading_episodes[use_eps_idx] = None
            yield batch

    def __len__(self):
        return self.num_samples_per_replica // self.batch_size

def trans_dict_to_list(messages:dict,max_len:int):
    new_messages = []
    for i in range(max_len):
        new_messages.append({})
    for key,values in messages.items():
        if isinstance(values, torch.Tensor):
            values = values
        for idx,value in enumerate(values):
            if idx < len(new_messages):
                if isinstance(value, torch.Tensor):
                    value = value.tolist()
                new_messages[idx][key] = value
                
    return new_messages

def store_data(
    dataloader, 
    num_samples: int = 1, 
    resolution: Tuple[int, int] = (320, 180), 
    save_fps: int = 20, 
    save_dir:str=None, #结尾要包含是否为train/val
    source_video_dir:Path=None,
    **kwargs,
) -> None:
    input("注意，source_video_dir 有问题，如果已经知晓，输入yes: ")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True,exist_ok=True)
    
    for idx, data in enumerate(tqdm(dataloader)): # worker_id 组
        fid = str(uuid.uuid4())[:12]
        
        if idx > num_samples:  # 超出num_samples，则停止
            break

        text = data['text']
        tframess = data["image"]
        actionss = trans_dict_to_list(data['env_action'],len(text))
        infoss = trans_dict_to_list(data["contractor_info"],len(text))
        
        
        frames = []
        num = 0
        event_text = text[0] if len(text) else ""
        file_name = save_dir / f"{event_text}+{fid}.mp4"
        jp = JsonlProcessor(save_dir/f"{event_text}+{fid}.jsonl",if_backup=False)
        for bidx, (tframes, actions, infos, txt) in enumerate(zip(tframess,actionss,infoss,text)):  #序号，image，text,这是一段视频的全部内容

            #raw_video_start_idx = data['raw_start_frame_idx'][bidx]
            raw_video_name = data["raw_video_name"][bidx]
            win_len = data['win_len'][bidx]
            #raw_video_path = source_video_dir / f"{raw_video_name}.mp4"
            #tframes = read_video(raw_video_path,raw_video_start_idx,win_len)
            #if len(tframes)!=win_len:
            #    print(raw_video_name,"need frames",win_len,"actual",len(tframes))

            frames.extend(tframes)
            messages = {
                "actions":actions,
                "infos":infos,
                "text":txt,
                "begin_num":num,
                "id":fid,
            }
            jp.dump_line(messages)
            num+=len(tframes)
        frames = [frame.numpy() for frame in frames]
        write_video(file_name, frames, fps=save_fps, width=resolution[0], height=resolution[1])
        jp.close()


def visualize_dataloader(
    dataloader, 
    num_samples: int = 1, 
    resolution: Tuple[int, int] = (320, 180), 
    legend: bool = False,
    save_fps: int = 20, 
    output_dir: str = "./",
    **kwargs,
) -> None:
    frames = []
    for idx, data in enumerate(tqdm(dataloader)):
        # continue
        if idx > num_samples: # 超出num_samples，则停止
            break
        action = data['env_action']
        prev_action = data.get("env_prev_action", None)
        image = data['image'].numpy()
        text = data['text']

        color = (255, 0, 0)
        for bidx, (tframes, txt) in enumerate(zip(image, text)):
            cache_frames = []
            for tidx, frame in enumerate(tframes):
                frame = cv2.resize(frame, resolution, interpolation=cv2.INTER_LINEAR)
                if 'segment' in data:
                    COLORS = [
                        (255, 0, 0), (0, 255, 0), (0, 0, 255), 
                        (255, 255, 0), (255, 0, 255), (0, 255, 255),
                        (255, 255, 255), (0, 0, 0), (128, 128, 128),
                        (128, 0, 0), (128, 128, 0), (0, 128, 0),
                        (128, 0, 128), (0, 128, 128), (0, 0, 128),
                    ]
                    obj_id = data['segment']['obj_id'][bidx][tidx].item()
                    if obj_id != -1:
                        segment_mask = data['segment']['obj_mask'][bidx][tidx]
                        if isinstance(segment_mask, torch.Tensor):
                            segment_mask = segment_mask.numpy()
                        colors = np.array(COLORS[obj_id]).reshape(1, 1, 3)
                        segment_mask = (segment_mask[..., None] * colors).astype(np.uint8)
                        segment_mask = segment_mask[:, :, ::-1] # bgr -> rgb
                        frame = cv2.addWeighted(frame, 1.0, segment_mask, 0.5, 0.0)

                if 'timestamp' in data:
                    timestamp = data['timestamp'][bidx][tidx]
                    cv2.putText(frame, f"timestamp: {timestamp}", (150, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 55), 2)

                if legend:
                    cv2.putText(frame, f"frame: {tidx}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                    cv2.putText(frame, txt, (200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                    
                    if 'contractor_info' in data:
                        try:
                            pitch = data['contractor_info']['pitch'][bidx][tidx]
                            yaw = data['contractor_info']['yaw'][bidx][tidx]
                            cursor_x = data['contractor_info']['cursor_x'][bidx][tidx]
                            cursor_y = data['contractor_info']['cursor_y'][bidx][tidx]
                            isGuiInventory = data['contractor_info']['isGuiInventory'][bidx][tidx]
                            isGuiOpen = data['contractor_info']['isGuiOpen'][bidx][tidx]
                            cv2.putText(frame, f"Pitch: {pitch:.2f}", (150, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(frame, f"Yaw: {yaw:.2f}", (150, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(frame, f"isGuiOpen: {isGuiOpen}", (150, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(frame, f"isGuiInventory: {isGuiInventory}", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(frame, f"CursorX: {cursor_x:.2f}", (150, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(frame, f"CursorY: {cursor_y:.2f}", (150, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        except:
                            cv2.putText(frame, f"No Contractor Info", (150, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    act = {k: v[bidx][tidx].numpy() for k, v in action.items()}
                    if prev_action is not None:
                        pre_act = {k: v[bidx][tidx].numpy() for k, v in prev_action.items()}
                    for row, ((k, v), (_, pv)) in enumerate(zip(act.items(), pre_act.items())):
                        if k != 'camera':
                            v = int(v.item())
                            pv = int(pv.item())
                        cv2.putText(frame, f"{k}: {v}({pv})", (10, 45 + row*15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cache_frames.append(frame.astype(np.uint8))
            
            frames = frames + cache_frames
    
    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    file_name = f"save_{timestamp}.mp4"
    file_path = os.path.join(output_dir, file_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    write_video(file_path, frames, fps=save_fps, width=resolution[0], height=resolution[1])

def dump_trajectories(
    dataloader, 
    num_samples: int = 1, 
    save_fps: int = 20, 
    **kwargs
) -> None:
    
    def un_batchify_actions(actions_in: Dict[str, torch.Tensor]) -> List[Dict]:
        actions_out = []
        for bidx in range(len(actions_in['attack'])):
            action = {}
            for k, v in actions_in.items():
                action[k] = v[bidx].numpy()
            actions_out.append(action)
        return actions_out
    
    traj_dir = Path("./traj_dir")
    video_dir = traj_dir / "videos"
    action_dir = traj_dir / "actions"
    video_dir.mkdir(parents=True, exist_ok=True)
    action_dir.mkdir(parents=True, exist_ok=True)
    for idx, data in enumerate(tqdm(dataloader)):
        if idx > num_samples: break
        image = data['img']
        action = data['action']
        action = un_batchify_actions(action)
        B, T = image.shape[:2]
        for i in range(B):
            vid = ''.join(random.choices(string.ascii_letters + string.digits, k=11))
            write_video(
                file_name=str(video_dir / f"{vid}.mp4"),
                frames=image[i].numpy().astype(np.uint8),
            )
            with open(action_dir / f"{vid}.pkl", 'wb') as f:
                pickle.dump(action[i], f)
                
                


def generate_uuid():
    return str(uuid.uuid4())

def generate_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def load_json_file(file_path:Union[str , pathlib.Path], data_type="dict"):
    if isinstance(file_path,pathlib.Path):
        file_path = str(file_path)
    if data_type == "dict":
        json_file = dict()
    elif data_type == "list":
        json_file = list()
    else:
        raise ValueError("数据类型不对")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                json_file = json.load(f)
        except IOError as e:
            rich.print(f"[red]无法打开文件{file_path}：{e}")
        except json.JSONDecodeError as e:
            rich.print(f"[red]解析 JSON 文件时出错{file_path}：{e}")
    else:
        rich.print(f"[yellow]{file_path}文件不存在，正在传入空文件...[/yellow]")
    return json_file

def dump_json_file(json_file, file_path:Union[str , pathlib.Path],indent=4,if_print = True,if_backup = True,if_backup_delete=False):
    if isinstance(file_path,pathlib.Path):
        file_path = str(file_path)
    backup_path = file_path + ".bak"  # 定义备份文件的路径
    if os.path.exists(file_path) and if_backup:
        shutil.copy(file_path, backup_path)
    before_exist = os.path.exists(file_path)
    try:
        with open(file_path, 'w',encoding="utf-8") as f:
            json.dump(json_file, f, indent=indent,ensure_ascii=False)
        if before_exist and if_print:
            rich.print(f"[yellow]更新{file_path}[/yellow]")
        elif if_print:
            rich.print(f"[green]创建{file_path}[/green]")
    except IOError as e:
        if os.path.exists(backup_path) and if_backup:
            shutil.copy(backup_path, file_path)
            if if_print:
                rich.print(f"[red]文件{file_path}写入失败，已从备份恢复原文件: {e}[/red]")
        else:
            if if_print:
                rich.print(f"[red]文件{file_path}写入失败，且无备份可用：{e}[/red]")
    finally:
        # 清理，删除备份文件
        if if_backup:
            if os.path.exists(backup_path) and if_backup_delete:
                os.remove(backup_path)
            if not os.path.exists(backup_path) and not if_backup_delete : #如果一开始是空的
                shutil.copy(file_path, backup_path)

def dump_jsonl(jsonl_file:list,file_path:Union[str , pathlib.Path],if_print=True):
    if isinstance(file_path,pathlib.Path):
        file_path = str(file_path)
    before_exist = os.path.exists(file_path)
    try:
        with open(file_path, 'w',encoding="utf-8") as f:
            for entry in jsonl_file:
                json_str = json.dumps(entry,ensure_ascii=False)
                f.write(json_str + '\n') 
        if before_exist and if_print:
            rich.print(f"[yellow]更新{file_path}[/yellow]")
        elif if_print:
            rich.print(f"[green]创建{file_path}[/green]")
    except IOError as e:
        print(f"[red]文件{file_path}写入失败，{e}[/red]") 

def split_dump_jsonl(jsonl_file:list,file_path:Union[str , pathlib.Path],split_num = 1, if_print=True):
    # 确保 file_path 是字符串类型
    if isinstance(file_path, pathlib.Path):
        file_path = str(file_path)
    
    # 检查原文件是否存在
    before_exist = os.path.exists(file_path)
    
    # 计算每份数据的大小
    chunk_size = len(jsonl_file) // split_num
    remainder = len(jsonl_file) % split_num  # 计算余数，用来均衡每份的大小

    # 将数据切分成 5 份
    chunks = []
    start_idx = 0
    for i in range(split_num):
        end_idx = start_idx + chunk_size + (1 if i < remainder else 0)  # 如果有余数，前几个切片多分一个
        chunks.append(jsonl_file[start_idx:end_idx])
        start_idx = end_idx
    
    # 写入每份数据到文件
    for i, chunk in enumerate(chunks):
        try:
            # 构造文件路径
            chunk_file_path = file_path[:-6] + f"{i}.jsonl"
            with open(chunk_file_path, 'w', encoding="utf-8") as f:
                for entry in chunk:
                    json_str = json.dumps(entry, ensure_ascii=False)
                    f.write(json_str + '\n')

            # 打印文件创建或更新信息
            if before_exist and if_print:
                rich.print(f"[yellow]更新{chunk_file_path}[/yellow]")
            elif if_print:
                rich.print(f"[green]创建{chunk_file_path}[/green]")
        except IOError as e:
            print(f"[red]文件{chunk_file_path}写入失败，{e}[/red]")

def load_jsonl(file_path:Union[str , pathlib.Path]):
    if isinstance(file_path,pathlib.Path):
        file_path = str(file_path)
    jsonl_file = []
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    jsonl_file.append(json.loads(line))
        except IOError as e:
            rich.print(f"[red]无法打开文件：{e}")
        except json.JSONDecodeError as e:
            rich.print(f"[red]解析 JSON 文件时出错：{e}")
    else:
        rich.print(f"[yellow]{file_path}文件不存在，正在传入空文件...[/yellow]")
    return jsonl_file 
                
class JsonlProcessor:
    def __init__(self, file_path:Union[str , pathlib.Path],
                 if_backup = True,
                 if_print=True
                 ):
        
        self.file_path = file_path if not isinstance(file_path,pathlib.Path) else str(file_path)
        
        self.if_print = if_print
        self.if_backup = if_backup

        self._mode = ""

        self._read_file = None
        self._write_file = None
        self._read_position = 0
        self.lines = 0

    @property
    def bak_file_path(self):
        return str(self.file_path) + ".bak"
    
    def exists(self):
        return os.path.exists(self.file_path)

    def len(self):
        file_length = 0
        if not self.exists():
            return file_length
        if self.lines == 0:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                while file.readline():
                    file_length+=1
            self.lines = file_length
        return self.lines

    def close(self,mode = "rw"):
        # 关闭文件资源
        if "r" in mode:
            if self._write_file:
                self._write_file.close()
                self._write_file = None
        if "w" in mode:
            if self._read_file:
                self._read_file.close()
                self._read_file = None
            self.lines = 0
        

    def reset(self, file_path:Union[str , pathlib.Path]):
        self.close()
        self.file_path = file_path if not isinstance(file_path,pathlib.Path) else str(file_path)


    def load_line(self,fast:bool=False):
        if not fast:
            if not self.exists():
                rich.print(f"[yellow]{self.file_path}文件不存在,返回{None}")
                return None
            if self._mode != "r":
                self.close("r")
                
        if not self._read_file:
            self._read_file = open(self.file_path, 'r', encoding='utf-8')
            
        if not fast:
            self._read_file.seek(self._read_position)
            self._mode = "r"
       
        try:
            line = self._read_file.readline()
            self._read_position = self._read_file.tell()
            if not line:
                self.close()
                return None
            return json.loads(line.strip())
        except json.JSONDecodeError as e:
            self.close()
            rich.print(f"[red]文件{self.file_path}解析出现错误：{e}")
            return None
        except IOError as e:
            self.close()
            rich.print(f"[red]无法打开文件{self.file_path}：{e}")
            return None
    
    def load_lines(self):
        """获取jsonl中的line，直到结尾"""
        lines = []
        while True:
            line = self.load_line()
            if line ==None:
                break
            lines.append(line)
        return lines
        

    def load_restart(self):
        self.close(mode="r")
        self._read_position = 0
         
    def dump_line(self, data,fast:bool=False):
        if not isinstance(data,dict) and not isinstance(data,list):
            raise ValueError("数据类型不对")
        if not fast:
            # 备份
            if self.len() % 50 == 1 and self.if_backup:
                shutil.copy(self.file_path, self.bak_file_path)
            self._mode = "a"
            # 如果模型尚未打开
        if not self._write_file:
            self._write_file = open(self.file_path, 'a', encoding='utf-8')
        try:
            json_line = json.dumps(data,ensure_ascii=False)
            self._write_file.write(json_line + '\n')
            self._write_file.flush()
            self.lines += 1  
            return True
        except Exception as e:
            
            if os.path.exists(self.bak_file_path) and self.if_backup:
                shutil.copy(self.bak_file_path, self.file_path)
                if self.if_print:
                    rich.print(f"[red]文件{self.file_path}写入失败，已从备份恢复原文件: {e}[/red]")
            else:
                if self.if_print:
                    rich.print(f"[red]文件{self.file_path}写入失败，且无备份可用：{e}[/red]") 
            return False

    def dump_lines(self,datas):
        if not isinstance(datas,list):
            raise ValueError("数据类型不对")
        if self.if_backup and os.path.exists(self.file_path):
            shutil.copy(self.file_path, self.bak_file_path)
        self._mode = "a"
        if not self._write_file:
            self._write_file = open(self.file_path, 'a', encoding='utf-8')
        try:
            self.len()
            for data in datas:
                json_line = json.dumps(data,ensure_ascii=False)
                self._write_file.write(json_line + '\n')
                self.lines += 1  
            self._write_file.flush()
            return True
        except Exception as e:
            if os.path.exists(self.bak_file_path) and self.if_backup:
                shutil.copy(self.bak_file_path, self.file_path)
                if self.if_print:
                    rich.print(f"[red]文件{self.file_path}写入失败，已从备份恢复原文件: {e}[/red]")
            else:
                if self.if_print:
                    rich.print(f"[red]文件{self.file_path}写入失败，且无备份可用：{e}[/red]") 
                return False
            
    def dump_restart(self):
        self.close()
        self._mode= "w"
        with open(self.file_path, 'w', encoding='utf-8') as file:
            pass 
          
    def load(self):
        jsonl_file = []
        if self.exists():
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        jsonl_file.append(json.loads(line))
            except IOError as e:
                rich.print(f"[red]无法打开文件：{e}")
            except json.JSONDecodeError as e:
                rich.print(f"[red]解析 JSON 文件时出错：{e}")
        else:
            rich.print(f"[yellow]{self.file_path}文件不存在，正在传入空文件...[/yellow]")
        return jsonl_file

    def dump(self,jsonl_file:list):
        before_exist = self.exists()
        if self.if_backup and before_exist:
            shutil.copy(self.file_path, self.bak_file_path)
        try:
            self.close()
            self._mode = "w"
            with open(self.file_path, 'w', encoding='utf-8') as f:
                for entry in jsonl_file:
                    json_str = json.dumps(entry,ensure_ascii=False)
                    f.write(json_str + '\n') 
                    self.lines += 1
            if before_exist and self.if_print:
                rich.print(f"[yellow]更新{self.file_path}[/yellow]")
            elif self.if_print:
                rich.print(f"[green]创建{self.file_path}[/green]")
            return True
        except Exception as e:
            if os.path.exists(self.bak_file_path) and self.if_backup:
                shutil.copy(self.bak_file_path, self.file_path)
                if self.if_print:
                    rich.print(f"[red]文件{self.file_path}写入失败，已从备份恢复原文件: {e}[/red]")
            else:
                if self.if_print:
                    rich.print(f"[red]文件{self.file_path}写入失败，且无备份可用：{e}[/red]") 
            return False  
