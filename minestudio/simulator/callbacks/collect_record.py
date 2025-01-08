from minestudio.simulator.callbacks.callback import RecordCallback
from typing import Literal


class CollectCallback(RecordCallback):
    def __init__(self, record_path: str, fps: int = 20, frame_type: Literal['pov', 'obs'] = 'pov', recording: bool = True,
                    show_actions=False,record_actions=False,record_infos=False,record_origin_observation=False,save_episode:bool=True,
                 **kwargs):
        super().__init(record_path,fps,frame_type,recording,show_actions,record_actions,record_infos,record_origin_observation)
        
        self.save_episode = save_episode #是否每个episode保存一次

        
    
    def before_reset(self, sim, reset_flag: bool) -> bool:
        if self.recording and self.save_episode:
            self._save_episode()
            self.episode_id += 1
        return reset_flag
    
    