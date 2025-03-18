<!--
 * @Date: 2024-12-01 08:30:33
 * @LastEditors: caishaofei caishaofei@stu.pku.edu.cn
 * @LastEditTime: 2024-12-30 14:20:20
 * @FilePath: /MineStudio/docs/source/data/quick-data.md
-->
Here is a minimal example to show how we load a trajectory from the dataset. 

```python
from minestudio.data import load_dataset

dataset = load_dataset(
    mode='raw', 
    dataset_dirs=['6xx', '7xx', '8xx', '9xx', '10xx'], 
    enable_video=True,
    enable_action=True,
    frame_width=224, 
    frame_height=224,
    win_len=128, 
    split='train', 
    split_ratio=0.9, 
    verbose=True
)
item = dataset[0]
print(item.keys())
```

You may see the output like this: 
```
[08:14:15] [Kernel] Driver video load 15738 episodes.  
[08:14:15] [Kernel] Driver action load 15823 episodes. 
[08:14:15] [Kernel] episodes: 15655, frames: 160495936. 
dict_keys(['text', 'timestamp', 'episode', 'progress', 'env_action', 'agent_action', 'env_prev_action', 'agent_prev_action', 'image', 'mask'])
```

```{hint}
Please note that the `dataset_dirs` parameter here is a list that can contain multiple dataset directories. In this example, we have loaded five dataset directories. 

If an element in the list is one of `6xx`, `7xx`, `8xx`, `9xx`, or `10xx`, the program will automatically download it from [Hugging Face](https://huggingface.co/CraftJarvis), so please ensure your network connection is stable and you have enough storage space. 

If an element in the list is a directory like `/nfs-shared/data/contractors/dataset_6xx`, the program will load data directly from that directory.

**You can also mix the two types of elements in the list.**
```


```{button-ref}  ./dataset-raw
:color: primary
:outline:
:expand:

Learn more about Raw Dataset
```

Alternatively, you can also load trajectories that have specific events, for example, loading all trajectories that contain the ``kill entity`` event. 

```python
from minestudio.data import load_dataset

dataset = load_dataset(
    mode='event', 
    dataset_dirs=['7xx'], 
    enable_video=True,
    enable_action=True,
    frame_width=224, 
    frame_height=224,
    win_len=128, 
    split='train', 
    split_ratio=0.9, 
    verbose=True,
    event_regex='minecraft.kill_entity:.*'
)
item = dataset[0]
print(item.keys())
```

You may see the output like this: 
```
[08:19:14] [Kernel] Driver video load 4617 episodes.
[08:19:14] [Kernel] Driver action load 4681 episodes. 
[08:19:14] [Kernel] episodes: 4568, frames: 65291168. 
[08:19:14] [Event Kernel] Number of loaded events: 58. 
[08:19:14] [Event Dataset] Regex: minecraft.kill_entity:.*, Number of events: 58, number of items: 19652
dict_keys(['text', 'env_action', 'agent_action', 'env_prev_action', 'agent_prev_action', 'image', 'mask'])
```

```{button-ref}  ./dataset-event
:color: primary
:outline:
:expand:

Learn more about Event Dataset
```