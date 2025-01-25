'''
Date: 2024-11-11 15:59:37
LastEditors: muzhancun muzhancun@126.com
LastEditTime: 2025-01-18 20:13:19
FilePath: /MineStudio/minestudio/models/base_policy.py
'''
from abc import ABC, abstractmethod
import numpy as np
import torch
import typing
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import OrderedDict
from omegaconf import DictConfig, OmegaConf
import gymnasium
from einops import rearrange
from minestudio.utils.vpt_lib.action_head import make_action_head
from minestudio.utils.vpt_lib.normalize_ewma import NormalizeEwma
from minestudio.utils.vpt_lib.scaled_mse_head import ScaledMSEHead

def dict_map(fn, d):
    if isinstance(d, Dict) or isinstance(d, DictConfig):
        return {k: dict_map(fn, v) for k, v in d.items()}
    else:
        return fn(d)

T = typing.TypeVar("T")
def recursive_tensor_op(fn, d: T) -> T:
    if isinstance(d, torch.Tensor):
        return fn(d)
    elif isinstance(d, list):
        return [recursive_tensor_op(fn, elem) for elem in d] # type: ignore
    elif isinstance(d, tuple):
        return tuple(recursive_tensor_op(fn, elem) for elem in d) # type: ignore
    elif isinstance(d, dict):
        return {k: recursive_tensor_op(fn, v) for k, v in d.items()} # type: ignore
    elif d is None:
        return None # type: ignore
    else:
        raise ValueError(f"Unexpected type {type(d)}")

class MinePolicy(torch.nn.Module, ABC):
    def __init__(self, hiddim, action_space=None, temperature=1.0, nucleus_prob=None) -> None:
        torch.nn.Module.__init__(self)
        if action_space is None:
            action_space = gymnasium.spaces.Dict({
                "camera": gymnasium.spaces.MultiDiscrete([121]), 
                "buttons": gymnasium.spaces.MultiDiscrete([8641]),
            })
        self.value_head = ScaledMSEHead(hiddim, 1, norm_type="ewma", norm_kwargs=None)
        self.pi_head = make_action_head(action_space, hiddim, temperature=temperature, nucleus_prob=nucleus_prob)

    def reset_parameters(self):
        self.pi_head.reset_parameters()
        self.value_head.reset_parameters()

    @abstractmethod
    def forward(self, 
                input: Dict[str, Any], 
                state_in: Optional[List[torch.Tensor]] = None,
                **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        """
        Returns:
            latents: containing `pi_logits` and `vpred` latent tensors.
            state_out: containing the updated state tensors.
        """
        pass

    @abstractmethod
    def initial_state(self, batch_size: Optional[int] = None) -> List[torch.Tensor]:
        pass

    @torch.inference_mode()
    def get_action(self,
                   input: Dict[str, Any],
                   state_in: Optional[List[torch.Tensor]],
                   deterministic: bool = False,
                   input_shape: str = "BT*",
                   **kwargs, 
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        if input_shape == "*":
            input = dict_map(self._batchify, input)
            if state_in is not None:
                state_in = recursive_tensor_op(lambda x: x.unsqueeze(0), state_in)
        elif input_shape != "BT*":
            raise NotImplementedError
        latents, state_out = self.forward(input, state_in, **kwargs)
        action = self.pi_head.sample(latents['pi_logits'], deterministic)
        self.vpred = latents['vpred']
        if input_shape == "BT*":
            self.cache_latents = latents
            return action, state_out
        elif input_shape == "*":
            self.cache_latents = dict_map(lambda tensor: tensor[0][0], latents)
            return dict_map(lambda tensor: tensor[0][0], action), recursive_tensor_op(lambda x: x[0], state_out)
        else:
            raise NotImplementedError

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def _batchify(self, elem):
        if isinstance(elem, (int, float)):
            elem = torch.tensor(elem, device=self.device)
        if isinstance(elem, np.ndarray):
            return torch.from_numpy(elem).unsqueeze(0).unsqueeze(0).to(self.device)
        elif isinstance(elem, torch.Tensor):
            return elem.unsqueeze(0).unsqueeze(0).to(self.device)
        elif isinstance(elem, str):
            return [[elem]]
        else:
            raise NotImplementedError

    # For online
    def merge_input(self, inputs) -> torch.tensor:
        raise NotImplementedError
    
    def merge_state(self, states) -> Optional[List[torch.Tensor]]:
        raise NotImplementedError

    def split_state(self, state, split_num) -> Optional[List[List[torch.Tensor]]]:
        raise NotImplementedError
    
    def split_action(self, action, split_num) -> Optional[List[Dict[str, torch.Tensor]]]:
        if isinstance(action, dict):
            # for k, v in action.items():
            #     action[k] = v.view(-1,1)
            result_actions = [{k: v[i].cpu().numpy() for k, v in action.items()} for i in range(0, split_num)]
            return result_actions
        elif isinstance(action, torch.Tensor):
            action_np = action.cpu().numpy()
            result_actions = [action_np[i] for i in range(0, split_num)]
            return result_actions
        elif isinstance(action, list):
            return action
        else:
            raise NotImplementedError

class MineGenerativePolicy(torch.nn.Module, ABC):
    
    ACTION_KEYS = OrderedDict({
        'camera': 2, 'attack': 1, 'forward': 1, 'back': 1, 'left': 1, 'right': 1, 'jump': 1, 'sneak': 1, 'sprint': 1, 'use': 1, 'drop': 1, 'inventory': 1, 
        'hotbar.1': 1, 'hotbar.2': 1, 'hotbar.3': 1, 'hotbar.4': 1, 'hotbar.5': 1, 'hotbar.6': 1, 'hotbar.7': 1, 'hotbar.8': 1, 'hotbar.9': 1,
    })

    def __init__(self, horizon = 1, action_chunk_size = 10) -> None:
        torch.nn.Module.__init__(self)
        self.horizon = horizon # horizon for sampling actions
        self.current_timestep = 0
        self.actions = None
        self.action_chunk_size = action_chunk_size

    @abstractmethod
    def forward(self, 
                input: Dict[str, Any], 
                state_in: Optional[List[torch.Tensor]] = None,
                **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        """
        Returns:
            latents: latent tensors.
            state_out: containing the updated state tensors.
        """
        pass

    @abstractmethod
    def sample(self, input: Dict[str, Any], state_in: Optional[List[torch.Tensor]] = None, **kwargs) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def get_state_out(self, input: Dict[str, Any], state_in: Optional[List[torch.Tensor]] = None, **kwargs) -> torch.Tensor:
        pass

    @torch.inference_mode()
    def get_action(self,
                   input: Dict[str, Any],
                   state_in: Optional[List[torch.Tensor]],
                   input_shape: str = "BT*",
                   **kwargs, 
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        if input_shape == "*":
            input = dict_map(self._batchify, input)
            if state_in is not None:
                state_in = recursive_tensor_op(lambda x: x.unsqueeze(0), state_in)
        elif input_shape != "BT*":
            raise NotImplementedError

        if self.actions is not None and self.current_timestep < self.horizon:
            action = self.actions[self.current_timestep]
            self.current_timestep += 1
            state_out = self.get_state_out(input, state_in, **kwargs)
            if input_shape == "*":
                state_out = recursive_tensor_op(lambda x: x[0], state_out)
            return action, state_out
        else:
            actions, state_out = self.sample(input, state_in, **kwargs) # actions is a b, t, d tensor
            b, t, d = actions.shape
            assert d % self.action_chunk_size == 0, "Action dimension must be divisible by action_chunk_size"
            if input_shape == "*":
                actions = actions[0][0]
                state_out = recursive_tensor_op(lambda x: x[0], state_out)
            else:
                raise NotImplementedError
            self.actions = self.convert_action(actions)

            self.current_timestep = 1
            return self.actions[0], state_out
            
    def convert_action(self, vector: torch.Tensor) -> Union[List[Dict], Dict]:
        vector = vector.cpu().numpy()
        length = len(vector.shape)
        if len(vector.shape) == 1:
            vector = vector[np.newaxis, ...]
        actions = []
        for i in range(vector.shape[0]):
            action = [{key: [] for key in self.ACTION_KEYS} for _ in range(self.action_chunk_size)]
            offset = 0
            for t in range(self.action_chunk_size):
                for idx, (key, dim) in enumerate(self.ACTION_KEYS.items()):
                    action[t][key] = vector[i, offset: offset+dim]
                    if key == 'camera':
                        action[t][key] = action[t][key] * 180.
                    else:
                        action[t][key] = (action[t][key] >= 0).astype(np.uint8)
                    offset += dim
            actions.append(action)
        if length == 1:
            return actions[0]
        return actions

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def _batchify(self, elem):
        if isinstance(elem, (int, float)):
            elem = torch.tensor(elem, device=self.device)
        if isinstance(elem, np.ndarray):
            return torch.from_numpy(elem).unsqueeze(0).unsqueeze(0).to(self.device)
        elif isinstance(elem, torch.Tensor):
            return elem.unsqueeze(0).unsqueeze(0).to(self.device)
        elif isinstance(elem, str):
            return [[elem]]
        else:
            return elem
            # raise NotImplementedError
