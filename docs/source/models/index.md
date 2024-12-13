<!--
 * @Date: 2024-12-03 04:47:37
 * @LastEditors: caishaofei caishaofei@stu.pku.edu.cn
 * @LastEditTime: 2024-12-13 08:48:33
 * @FilePath: /MineStudio/docs/source/models/index.md
-->
# Models

We provided a template for the Minecraft Policy and based on this template, we created various different baseline models, currently suppoerting VPT, STEVE-1, GROOT, ROCKET-1. 


```{toctree}
:caption: MineStudio Models

baseline-vpt
baseline-steve1
baseline-groot
baseline-rocket1
```

## Quick Start
```{include} quick-models.md
```

## Policy Template

```{warning}
One must implement their own policies based on the template to be compatible with our training and inference pipelines. 
```

The policy template lies in `minestudio.models.base_policy.MinePolicy`. It consists of the following methods:

````{dropdown} __init__(self, hiddim, action_space=None)

The constructor of the policy. It initializes the policy head and value head. The `hiddim` is the hidden dimension of the policy. The `action_space` is the action space of the environment. If it is not provided, the default action space of the Minecraft environment will be used. 

```python
def __init__(self, hiddim, action_space=None) -> None:
    torch.nn.Module.__init__(self)
    if action_space is None:
        action_space = gymnasium.spaces.Dict({
            "camera": gymnasium.spaces.MultiDiscrete([121]), 
            "buttons": gymnasium.spaces.MultiDiscrete([8641]),
        })
    self.pi_head = make_action_head(action_space, hiddim, temperature=2.0)
    self.value_head = ScaledMSEHead(hiddim, 1, norm_type="ewma", norm_kwargs=None)
```

```{hint}
If users want to customize the `pi_head` and `value_head` modules, they can override them after calling the `super().__init__` method. 
```

````

````{dropdown} forward(self, input, state_in, **kwargs)

The forward method of the policy. It takes the input and the state tensors and returns the latent tensors and the updated state tensors. 

```python
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
```

```{note}
This method should be implemented by the derived classes. 
```

````

````{dropdown} initial_state(self, batch_size=None)

This is an important method that returns the initial state of the policy. 

```python
@abstractmethod
def initial_state(self, batch_size: Optional[int] = None) -> List[torch.Tensor]:
    pass
```

````

````{dropdown} get_action(self, input, state_in, deterministic, input_shape)

This is the method that returns the action of the policy. It takes the input, the state tensors, and the deterministic flag, and returns the action tensor and the updated state tensors. This method is usually called during the inference process. 

```python
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
        return action, state_out
    elif input_shape == "*":
        return dict_map(lambda tensor: tensor[0][0], action), recursive_tensor_op(lambda x: x[0], state_out)
    else:
        raise NotImplementedError
```

```{note}
`deterministic` is a flag that indicates whether the action is generates with `argmax` or `stochastic sampling`. 
We emperically find that setting `deterministic=False` can improve the performance of the policy. 
```

```{note}
`input_shape` is a string that indicates the shape of the input. It can be `"BT*"` or `"*"`. `"BT*"` means the input is a batch of time sequences, and `"*"` means the input is a single sample. Generally speaking, if you are in inference mode, you feed an observation once a time. So you should set `input_shape="*"`. 
```

````

````{dropdown} device(self)

This is a property method that returns the device of the policy.

```python
@property
def device(self) -> torch.device:
    return next(self.parameters()).device
```

````

```{hint}
The minimal set you need to care about is `forward` and `initial_state`. 
```

## Your First Policy

Load the necessary modules:

```python
import torch
import torch.nn as nn
from minestudio.models.base_policy import MinePolicy
```

To customize a condition-free policy, you can follow this example:

```python
class MyConditionFreePolicy(MinePolicy):
    def __init__(self, hiddim, action_space=None) -> None:
        super().__init__(hiddim, action_space)
        # we use the original pi_head and value_head here.
        self.net = nn.Sequential(
            nn.Linear(128*128*3, hiddim), 
            nn.ReLU(),
            nn.Linear(hiddim, hiddim),
            nn.ReLU()
        )
        # we implement a simple mlp network here as the backbone. 

    def forward(self, input, state_in, **kwargs):
        x = rearrange(input['image'] / 255., 'b t h w c -> b t (h w c)')
        x = self.net(x)
        result = {
            'pi_logits': self.pi_head(x), 
            'vpred': self.value_head(x), 
        }
        return result, state_in

    def initial_state(self, batch_size=None):
        # we implement a simple markov policy here, so the state is always None.
        None
```

To customize a condition-based policy, you can follow this example:

```python
class MyConditionBasedPolicy(MinePolicy):
    def __init__(self, hiddim, action_space=None) -> None:
        super().__init__(hiddim, action_space)
        # we use the original pi_head and value_head here.
        self.net = nn.Sequential(
            nn.Linear(128*128*3, hiddim), 
            nn.ReLU(),
            nn.Linear(hiddim, hiddim),
            nn.ReLU()
        )
        self.condition_net = nn.Embedding(10, hiddim)
        # we implement a simple mlp network here as the backbone. 

    def forward(self, input, state_in, **kwargs):
        x = rearrange(input['image'] / 255., 'b t h w c -> b t (h w c)')
        x = self.net(x) # b t c
        y = self.condition_net(input['condition']) # b t -> b t c
        z = x + y # simple addition fusion
        result = {
            'pi_logits': self.pi_head(z), 
            'vpred': self.value_head(z), 
        }
        return result, state_in

    def initial_state(self, batch_size=None):
        # we implement a simple markov policy here, so the state is always None.
        None
```

```{warning}
These examples are just for demonstration purposes and may perform poorly in practice. 
```


