<!--
 * @Date: 2024-12-12 09:18:35
 * @LastEditors: caishaofei caishaofei@stu.pku.edu.cn
 * @LastEditTime: 2024-12-30 14:27:43
 * @FilePath: /MineStudio/docs/source/data/visualization.md
-->

# Visualization

We provide a visual script that allows users to observe whether the configured Dataloader meets expectations. It is useful for debugging and verifying the correctness of the data. 

## Visualize Dataloader

Here is the arguments of the `visualize_dataloader` function:

| Arguments | Description |
| --- | --- |
| `dataloader` | PyTorch dataloader |
| `num_samples` | Number of batches to visualize |
| `resolution` | Resolution of the video |
| `legend` | Print action, contractor info, and segment info in the video |
| `save_fps` | FPS of the saved video |
| `output_dir` | Output directory for the saved video |

## Visualize Continuous Batches

When visualizing continuous video frames, set `episode_continuous_batch=True`, `batch_size=1` in the `MineDataModule` configuration. 

```python
import lightning as L
from tqdm import tqdm
from minestudio.data import MineDataModule
from minestudio.data.minecraft.utils import visualize_dataloader

data_module = MineDataModule(
    data_params=dict(
        mode='raw',
        dataset_dirs=[
            '/nfs-shared-2/data/contractors/dataset_10xx',
        ],
        frame_width=224,
        frame_height=224,
        win_len=128,
        split_ratio=0.8,
    ),
    batch_size=1, # set to 1 for visualizing continuous video frames
    num_workers=2,
    prefetch_factor=4,
    shuffle_episodes=True,
    episode_continuous_batch=True,  # `True` for visualizing continuous video frames
)
data_module.setup()
dataloader = data_module.val_dataloader()

visualize_dataloader(
    dataloader, 
    num_samples=5, 
    resolution=(640, 360), 
    legend=True,  # print action, contractor info, and segment info ... in the video
    save_fps=30, 
    output_dir="./"
)
```

Here is the example video:
```{youtube} JvlFptYjOm0
```



## Visualize Batches with Special Events

When visualizing video frames with special events, set `event_regex` in the `MineDataModule` configuration. 

```python
import lightning as L
from tqdm import tqdm
from minestudio.data import MineDataModule
from minestudio.data.minecraft.utils import visualize_dataloader

data_module = MineDataModule(
    data_params=dict(
        mode='event',
        dataset_dirs=[
            '/nfs-shared-2/data/contractors/dataset_10xx',
        ],
        frame_width=224,
        frame_height=224,
        win_len=128,
        split_ratio=0.8,
        shuffle_episodes=True,
        event_regex='minecraft.mine_block:.*diamond.*',
    ),
    batch_size=2,
)
data_module.setup()
dataloader = data_module.val_dataloader()

visualize_dataloader(
    dataloader, 
    num_samples=5, 
    resolution=(640, 360), 
    legend=True,  # print action, contractor info, and segment info ... in the video
    save_fps=30, 
    output_dir="./"
)
```

Here is the example video:

```{youtube} 9YU3y0ZWh8Y
```
