<!--
 * @Date: 2024-12-01 08:37:10
 * @LastEditors: caishaofei caishaofei@stu.pku.edu.cn
 * @LastEditTime: 2024-12-12 09:07:39
 * @FilePath: /MineStudio/docs/source/data/dataset-raw.md
-->

# Raw Dataset

The Raw Dataset refers to a simple way of reading the original data, which stores the raw trajectory segments in chronological order. 
```{hint}
Users can choose to read random segments from it or opt to read segments continuously in chronological order. 
```

## Basic Information

Here are the special arguments of the `RawDataset` class:

| Arguments | Description |
| --- | --- |
| `dataset_dirs` | the directories of the dataset |
| `enable_video` | whether to load video data |
| `enable_action` | whether to load action data |
| `enable_contractor_info` | whether to load contractor meta information |
| `enable_segment` | whether to load segmentation data |
| `frame_width` | the width of the frame |
| `frame_height` | the height of the frame |
| `win_len` | segment length of an item | 
| `skip_frame` | the number of frames to skip when building the segment item |
| `split` | the split mode of the dataset, including `train`, `val` |
| `split_ratio` | the ratio of the split mode, e.g., `0.8` means `train`:`val`=8:2 |
| `shuffle` | whether to shuffle trajectory order, this could affect train-val split |
| `verbose` | whether to print the information of the dataset |


## Loading Segment-level Data

When the user does not have a need to process long trajectories, segments from the same trajectory are independent and can be read randomly. This reading method is suitable for some simple tasks, such as training a policy that can perform short-range tasks, like GROOT-1. At this point, the user only needs to wrap `RawDataset` with PyTorch's built-in dataloader to achieve data reading.

Here is an example of how to load the segment-level data:

```python
from torch.utils.data import DataLoader
from minestudio.data import RawDataset
from minestudio.data.minecraft.utils import batchify

kernel_kwargs = dict(
    dataset_dirs=[
        '/nfs-shared-2/data/contractors/dataset_6xx', 
        '/nfs-shared-2/data/contractors/dataset_7xx', 
    ], 
    enable_video=True,
    enable_action=True,
    enable_segment=True,
    enable_contractor_info=True,
)

dataset = RawDataset(
    frame_width=224,
    frame_height=224,
    win_len=128, 
    skip_frame=1,
    split='train',
    split_ratio=0.8,
    verbose=True,
    **kernel_kwargs, 
)

loader = DataLoader(dataset, collate_fn=batchify)

for item in loader:
    print(
        f"{item.keys() = }\n", 
        f"{item['image'].shape = }\n", 
        f"{item['contractor_info'].keys() = }\n", 
        f"{item['env_action'].keys() = }\n", 
        f"{item['agent_action'].keys() = }\n", 
        f"{item['timestamp'].shape = }\n"
    )
    break
```

Now, you can see the following output:

```python
item.keys() = dict_keys(['contractor_info', 'segment', 'text', 'timestamp', 'episode', 'progress', 'env_action', 'agent_action', 'env_prev_action', 'agent_prev_action', 'image', 'mask'])
item['image'].shape = torch.Size([1, 128, 224, 224, 3])
item['contractor_info'].keys() = dict_keys(['yaw', 'pitch', 'xpos', 'ypos', 'zpos', 'hotbar', 'inventory', 'isGuiOpen', 'isGuiInventory', 'delta_yaw', 'delta_pitch', 'events', 'cursor_x', 'cursor_y'])
item['env_action'].keys() = dict_keys(['back', 'drop', 'forward', 'hotbar.1', 'hotbar.2', 'hotbar.3', 'hotbar.4', 'hotbar.5', 'hotbar.6', 'hotbar.7', 'hotbar.8', 'hotbar.9', 'inventory', 'jump', 'left', 'right', 'sneak', 'sprint', 'camera', 'attack', 'use'])
item['agent_action'].keys() = dict_keys(['buttons', 'camera'])
item['timestamp'].shape = torch.Size([1, 128])
```

## Loading Episode-level Data

When the user needs to process long trajectories, the segments from the same trajectory are related and need to be read in order. This reading method is suitable for some complex tasks, such as training a policy that demands long-range dependencies, like VPT. At this point, the user needs to combine `RawDataset` with the `MineDistributedBatchSampler` to achieve data reading. 

```{note}
`MineDistributedBatchSampler` ensures that each batch slot mantains the order of the trajectory, only if the trajectory runs out of segments, the slot will be filled with a new trajectory. 
```

```{note}
When using distributed training strategy, `MineDistributedBatchSampler` will automatically separate the dataset into different parts according to the number of GPUs. Most episodes will only belongs to one part. If an episode is seprated into two parts, each part will be treated as a new episode. Don't worry about that. 
```

Here is an example:
```python
...

from minestudio.data.minecraft.utils import MineDistributedBatchSampler
from torch.utils.data import DataLoader

sampler = MineDistributedBatchSampler(
    dataset, 
    batch_size=4, 
    num_replicas=1, 
    rank=0, 
    shuffle=False, 
    drop_last=True
)

loader = DataLoader(dataset, batch_sampler=sampler, num_workers=4)
for idx, batch in enumerate(loader):
    print(
        "\t".join(
            [f"{a} {b}" for a, b in zip(batch['episode'], batch['progress'])]
        )
    )
```
Now, you can see the following outputs:
```
Luka-0f43d5f87f94-20220408-204320 0/79  Luka-1c017eee3612-20220407-175134 0/82  Luka-1d592f9d17a8-20220407-185701 0/43  Luka-1d592f9d17a8-20220407-190201 0/42
Luka-0f43d5f87f94-20220408-204320 1/79  Luka-1c017eee3612-20220407-175134 1/82  Luka-1d592f9d17a8-20220407-185701 1/43  Luka-1d592f9d17a8-20220407-190201 1/42
Luka-0f43d5f87f94-20220408-204320 2/79  Luka-1c017eee3612-20220407-175134 2/82  Luka-1d592f9d17a8-20220407-185701 2/43  Luka-1d592f9d17a8-20220407-190201 2/42
Luka-0f43d5f87f94-20220408-204320 3/79  Luka-1c017eee3612-20220407-175134 3/82  Luka-1d592f9d17a8-20220407-185701 3/43  Luka-1d592f9d17a8-20220407-190201 3/42
Luka-0f43d5f87f94-20220408-204320 4/79  Luka-1c017eee3612-20220407-175134 4/82  Luka-1d592f9d17a8-20220407-185701 4/43  Luka-1d592f9d17a8-20220407-190201 4/42
Luka-0f43d5f87f94-20220408-204320 5/79  Luka-1c017eee3612-20220407-175134 5/82  Luka-1d592f9d17a8-20220407-185701 5/43  Luka-1d592f9d17a8-20220407-190201 5/42
... ... ... ...
Luka-0f43d5f87f94-20220408-204320 37/79 Luka-1c017eee3612-20220407-175134 37/82 Luka-1d592f9d17a8-20220407-185701 37/43 Luka-1d592f9d17a8-20220407-190201 37/42
Luka-0f43d5f87f94-20220408-204320 38/79 Luka-1c017eee3612-20220407-175134 38/82 Luka-1d592f9d17a8-20220407-185701 38/43 Luka-1d592f9d17a8-20220407-190201 38/42
Luka-0f43d5f87f94-20220408-204320 39/79 Luka-1c017eee3612-20220407-175134 39/82 Luka-1d592f9d17a8-20220407-185701 39/43 Luka-1d592f9d17a8-20220407-190201 39/42
Luka-0f43d5f87f94-20220408-204320 40/79 Luka-1c017eee3612-20220407-175134 40/82 Luka-1d592f9d17a8-20220407-185701 40/43 Luka-1d592f9d17a8-20220407-190201 40/42
Luka-0f43d5f87f94-20220408-204320 41/79 Luka-1c017eee3612-20220407-175134 41/82 Luka-1d592f9d17a8-20220407-185701 41/43 Luka-1d592f9d17a8-20220407-190201 41/42
Luka-0f43d5f87f94-20220408-204320 42/79 Luka-1c017eee3612-20220407-175134 42/82 Luka-1d592f9d17a8-20220407-185701 42/43 Luka-1f36a7a0dcdf-20220408-200958 0/78
Luka-0f43d5f87f94-20220408-204320 43/79 Luka-1c017eee3612-20220407-175134 43/82 Luka-23ba24fd20b7-20220407-172356 0/42  Luka-1f36a7a0dcdf-20220408-200958 1/78
Luka-0f43d5f87f94-20220408-204320 44/79 Luka-1c017eee3612-20220407-175134 44/82 Luka-23ba24fd20b7-20220407-172356 1/42  Luka-1f36a7a0dcdf-20220408-200958 2/78
Luka-0f43d5f87f94-20220408-204320 45/79 Luka-1c017eee3612-20220407-175134 45/82 Luka-23ba24fd20b7-20220407-172356 2/42  Luka-1f36a7a0dcdf-20220408-200958 3/78
Luka-0f43d5f87f94-20220408-204320 46/79 Luka-1c017eee3612-20220407-175134 46/82 Luka-23ba24fd20b7-20220407-172356 3/42  Luka-1f36a7a0dcdf-20220408-200958 4/78
... ... ... ...
```

```{note}
We can see from the output, the `MineDistributedBatchSampler` ensures that each batch slot mantains the order of the trajectory. 
```

If you want to know more about `MineDistributedBatchSampler`, here are the arguments: 

| Arguments | Description |
| --- | --- |
| dataset | the dataset to sample from |
| batch_size | how many samples per batch to load |
| num_replicas | the number of processes participating in the training; lightning will set this for you |
| rank | the rank of the current process within num_replicas; lightning will set this for you |
| shuffle | must be `false`, you can do shuffle operation in the `RawDataset` |
| drop_last | must be `true` |

## Using Lightning to Simplify the Data Loading Process

We can use lightning fabric to simplify the distributed data loading (using built-in distributed data parallel strategy, ddp). Here is an example:

```python
import lightning as L
from minestudio.data import MineDataModule


fabric = L.Fabric(accelerator="cuda", devices=2, strategy="ddp")
fabric.launch()
data_module = MineDataModule(
    data_params=dict(
        mode='raw',
        dataset_dirs=[
            '/nfs-shared-2/data/contractors/dataset_6xx',
            '/nfs-shared-2/data/contractors/dataset_7xx',
        ],
        frame_width=224,
        frame_height=224,
        win_len=128,
        split_ratio=0.8,
    ),
    batch_size=2,
    num_workers=2,
    prefetch_factor=4,
    shuffle_episodes=True,
    episode_continuous_batch=True, 
)
data_module.setup()
train_loader = data_module.train_dataloader()
train_loader = fabric.setup_dataloaders(train_loader, use_distributed_sampler=False)
rank = fabric.local_rank
for idx, batch in enumerate(train_loader):
    print(
        f"{rank = } \t" + "\t".join(
            [f"{a[-20:]} {b}" for a, b in zip(batch['episode'], batch['progress'])]
        )
    )
```
Here is the output:
```
rank = 1        3f61-20220105-200026 233/505    f1e3-20220209-191720 0/43       3f61-20220119-194043 0/35
rank = 1        3f61-20220105-200026 234/505    f1e3-20220209-191720 1/43       3f61-20220119-194043 1/35
rank = 0        3f61-20220211-054736 0/90       2777-20220212-055512 0/44       3f61-20211017-215056 0/28
rank = 0        3f61-20220211-054736 1/90       2777-20220212-055512 1/44       3f61-20211017-215056 1/28
rank = 1        3f61-20220105-200026 235/505    f1e3-20220209-191720 2/43       3f61-20220119-194043 2/35
rank = 1        3f61-20220105-200026 236/505    f1e3-20220209-191720 3/43       3f61-20220119-194043 3/35
rank = 0        3f61-20220211-054736 2/90       2777-20220212-055512 2/44       3f61-20211017-215056 2/28
rank = 0        3f61-20220211-054736 3/90       2777-20220212-055512 3/44       3f61-20211017-215056 3/28
rank = 1        3f61-20220105-200026 237/505    f1e3-20220209-191720 4/43       3f61-20220119-194043 4/35
rank = 1        3f61-20220105-200026 238/505    f1e3-20220209-191720 5/43       3f61-20220119-194043 5/35
rank = 0        3f61-20220211-054736 4/90       2777-20220212-055512 4/44       3f61-20211017-215056 4/28
rank = 0        3f61-20220211-054736 5/90       2777-20220212-055512 5/44       3f61-20211017-215056 5/28
rank = 0        3f61-20220211-054736 6/90       2777-20220212-055512 6/44       3f61-20211017-215056 6/28
rank = 0        3f61-20220211-054736 7/90       2777-20220212-055512 7/44       3f61-20211017-215056 7/28
rank = 1        3f61-20220105-200026 239/505    f1e3-20220209-191720 6/43       3f61-20220119-194043 6/35
rank = 1        3f61-20220105-200026 240/505    f1e3-20220209-191720 7/43       3f61-20220119-194043 7/35
rank = 0        3f61-20220211-054736 8/90       2777-20220212-055512 8/44       3f61-20211017-215056 8/28
rank = 0        3f61-20220211-054736 9/90       2777-20220212-055512 9/44       3f61-20211017-215056 9/28
rank = 1        3f61-20220105-200026 241/505    f1e3-20220209-191720 8/43       3f61-20220119-194043 8/35
rank = 1        3f61-20220105-200026 242/505    f1e3-20220209-191720 9/43       3f61-20220119-194043 9/35
rank = 0        3f61-20220211-054736 10/90      2777-20220212-055512 10/44      3f61-20211017-215056 10/28
rank = 0        3f61-20220211-054736 11/90      2777-20220212-055512 11/44      3f61-20211017-215056 11/28
rank = 1        3f61-20220105-200026 243/505    f1e3-20220209-191720 10/43      3f61-20220119-194043 10/35
rank = 1        3f61-20220105-200026 244/505    f1e3-20220209-191720 11/43      3f61-20220119-194043 11/35
rank = 0        3f61-20220211-054736 12/90      2777-20220212-055512 12/44      3f61-20211017-215056 12/28
rank = 0        3f61-20220211-054736 13/90      2777-20220212-055512 13/44      3f61-20211017-215056 13/28
rank = 1        3f61-20220105-200026 245/505    f1e3-20220209-191720 12/43      3f61-20220119-194043 12/35
rank = 1        3f61-20220105-200026 246/505    f1e3-20220209-191720 13/43      3f61-20220119-194043 13/35
rank = 0        3f61-20220211-054736 14/90      2777-20220212-055512 14/44      3f61-20211017-215056 14/28
rank = 0        3f61-20220211-054736 15/90      2777-20220212-055512 15/44      3f61-20211017-215056 15/28
rank = 1        3f61-20220105-200026 247/505    f1e3-20220209-191720 14/43      3f61-20220119-194043 14/35
rank = 1        3f61-20220105-200026 248/505    f1e3-20220209-191720 15/43      3f61-20220119-194043 15/35
rank = 0        3f61-20220211-054736 16/90      2777-20220212-055512 16/44      3f61-20211017-215056 16/28
rank = 0        3f61-20220211-054736 17/90      2777-20220212-055512 17/44      3f61-20211017-215056 17/28
rank = 1        3f61-20220105-200026 249/505    f1e3-20220209-191720 16/43      3f61-20220119-194043 16/35
rank = 1        3f61-20220105-200026 250/505    f1e3-20220209-191720 17/43      3f61-20220119-194043 17/35
rank = 0        3f61-20220211-054736 18/90      2777-20220212-055512 18/44      3f61-20211017-215056 18/28
rank = 0        3f61-20220211-054736 19/90      2777-20220212-055512 19/44      3f61-20211017-215056 19/28
rank = 1        3f61-20220105-200026 251/505    f1e3-20220209-191720 18/43      3f61-20220119-194043 18/35
rank = 1        3f61-20220105-200026 252/505    f1e3-20220209-191720 19/43      3f61-20220119-194043 19/35
rank = 0        3f61-20220211-054736 20/90      2777-20220212-055512 20/44      3f61-20211017-215056 20/28
rank = 0        3f61-20220211-054736 21/90      2777-20220212-055512 21/44      3f61-20211017-215056 21/28
rank = 1        3f61-20220105-200026 253/505    f1e3-20220209-191720 20/43      3f61-20220119-194043 20/35
rank = 1        3f61-20220105-200026 254/505    f1e3-20220209-191720 21/43      3f61-20220119-194043 21/35
rank = 0        3f61-20220211-054736 22/90      2777-20220212-055512 22/44      3f61-20211017-215056 22/28
... ... ... ...
```

```{note}
We can see from the output, for each distributed process, the batch slot also mantains the order of the trajectory. 
```