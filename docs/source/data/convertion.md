<!--
 * @Date: 2024-12-12 09:18:35
 * @LastEditors: caishaofei caishaofei@stu.pku.edu.cn
 * @LastEditTime: 2024-12-30 16:06:47
 * @FilePath: /MineStudio/docs/source/data/convertion.md
-->

# Convertion

We provide a convertion script that allows users to convert the raw data to the MineStudio format. 

```{warning}
It is essential to perform the conversion to ensure that our engineering efforts on the data can be effectively utilized. 
```

## Prepare Raw Trajectories

The raw data should contain `video` and `action` directories. The `video` contains a list of `mp4-format` video files, and the `action` contains a list of `dict-format` action files. The `video` and `action` files should have the same length and the same name. 

- `/path/to/raw_episodes/videos` directory:
    ```python
    video = [
        'episode_0001.mp4',
        'episode_0002.mp4',
        'episode_0003.mp4',
        ...
    ]
    ```

- `/path/to/raw_episodes/actions` directory:
    ```python
    action = [
        'episode_0001.pkl',
        'episode_0002.pkl',
        'episode_0003.pkl',
        ...
    ]
    ```

    ````{note}
    Each action file is a `dict` object that contains the following keys:
    ```python
    dict_keys(['back', 'drop', 'forward', 'hotbar.1', 'hotbar.2', 'hotbar.3', 'hotbar.4', 'hotbar.5', 'hotbar.6', 'hotbar.7', 'hotbar.8', 'hotbar.9', 'inventory', 'jump', 'left', 'right', 'sneak', 'sprint', 'camera', 'attack', 'use'])
    ```
    The shape of `attack`: ```(4376,)```

    The shape of `camera`: ```(4376, 2)```
    ````

## Convert Raw Trajectories to MineStudio format

- convert `action` to MineStudio format:
    ```console
    python -m minestudio.data.minecraft.tools.convert_lmdb \
           --num-workers 4 \
           --input-dir '/path/to/raw_episodes/actions' \
           --action-dir '/path/to/raw_episodes/actions' \
           --output-dir '/path/to/output/dataset' \
           --source-type 'action'
    ```

- convert `video` to MineStudio format:
    ```console
    python -m minestudio.data.minecraft.tools.convert_lmdb \
           --num-workers 4 \
           --input-dir '/path/to/raw_episodes/videos' \
           --action-dir '/path/to/raw_episodes/actions' \
           --output-dir '/path/to/output/dataset' \
           --source-type 'video'
    ```

```{note}
`num-workers` arguments specify the number of convertion workers. **It also determines the number of the resulting MineStudio dataset files.** 
```
The resulting MineStudio dataset files will be stored in the `/path/to/output/dataset` directory.
```console
tree /path/to/output/dataset
├── action
│   ├── action-1000 
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── action-1500
│   │   ├── data.mdb
│   │   └── lock.mdb
│   ├── action-1904
│   │   ├── data.mdb
│   │   └── lock.mdb
│   └── action-500
│       ├── data.mdb
│       └── lock.mdb
└── video
    ├── video-1428  
    │   ├── data.mdb
    │   └── lock.mdb
    ├── video-1903
    │   ├── data.mdb
    │   └── lock.mdb
    ├── video-476
    │   ├── data.mdb
    │   └── lock.mdb
    └── video-952
        ├── data.mdb
        └── lock.mdb
```

## Check the MineStudio-Format Dataset

You can check the generated MineStudio dataset files using the following command:
```python
from minestudio.data import load_dataset

dataset = load_dataset(
    mode='raw', 
    dataset_dirs=['/path/to/output/dataset'], 
    frame_width=224, 
    frame_height=224,
    win_len=128, 
    split='train', 
    split_ratio=0.9, 
    verbose=True,
)
item = dataset[0]
print(item.keys())
```