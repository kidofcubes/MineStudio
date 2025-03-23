'''
Date: 2024-11-11 16:40:57
LastEditors: Muyao 2350076251@qq.com
LastEditTime: 2025-03-22 22:59:26
FilePath: /MineStudio/minestudio/simulator/callbacks/record.py
'''
import av
from pathlib import Path
from minestudio.simulator.callbacks.callback import MinecraftCallback
from minestudio.simulator.callbacks.record import  RecordCallback
from typing import Literal
from rich import print
from copy import deepcopy
import numpy as np
from gymnasium import spaces
from collections import defaultdict
import json
import cv2

class RecordCallback2(RecordCallback):
    
    def forget(self):
        self.frames = []
        self.actions = []
        self.infos = []
        self.texts = []
    
    def _save_episode(self):
        if len(self.frames) == 0:
            return 
        output_path = self.record_path / f'episode_{self.episode_id}.mp4'
        
        font = cv2.FONT_HERSHEY_SIMPLEX # cv2.FONT_HERSHEY_PLAIN # cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)
        thickness = 1
        line_type = cv2.LINE_AA
    
        with av.open(output_path, mode="w", format='mp4') as container:
            stream = container.add_stream("h264", rate=self.fps)
            stream.width = self.frames[0].shape[1]
            stream.height = self.frames[0].shape[0]
            for idx,frame in enumerate(self.frames):
                # print actions on the frames
                if self.show_actions and idx < len(self.texts):
                        frame = frame.copy()
                        cv2.putText(frame, self.texts[idx][:50], (10, 20), font, font_scale, font_color, thickness,line_type)
                frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        if self.record_origin_observation:
            output_origin_path = self.record_path / f'episode_{self.episode_id}.npy'
            all_frames = np.array(self.frames)
            np.save(output_origin_path, all_frames)
        print(f'[green]Episode {self.episode_id} saved at {output_path}[/green], alen: {len(self.actions)}; ilen: {len(self.infos)}')
        self.frames = []
        
        if self.record_actions: # assert self.actions>0 sense self.frame > 0
            output_action_path = self.record_path / f'episode_{self.episode_id}_action.json'
            record_actions = [self._process_action(action) for action in self.actions]
            with open(output_action_path, 'w', encoding="utf-8") as file:
                json.dump(record_actions, file)
            self.actions = []
        
        if self.record_infos: # assert self.actions>0 sense self.frame > 0
            output_info_path = self.record_path / f'episode_{self.episode_id}_info.json'
            record_infos = [self._process_info(info) for info in self.infos]
            with open(output_info_path, 'w', encoding="utf-8") as file:
                json.dump(record_infos, file)
            self.infos = []
            
        
    