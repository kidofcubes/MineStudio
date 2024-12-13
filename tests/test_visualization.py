'''
Date: 2024-12-12 11:00:06
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-12-12 11:31:04
FilePath: /MineStudio/tests/test_visualization.py
'''

import lightning as L
from tqdm import tqdm
from minestudio.data import MineDataModule
from minestudio.data.minecraft.utils import visualize_dataloader

# data_module = MineDataModule(
#     data_params=dict(
#         mode='raw',
#         dataset_dirs=[
#             '/nfs-shared-2/data/contractors/dataset_10xx',
#         ],
#         frame_width=224,
#         frame_height=224,
#         win_len=128,
#         split_ratio=0.8,
#     ),
#     batch_size=1, # set to 1 for visualizing continuous video frames
#     num_workers=2,
#     prefetch_factor=4,
#     shuffle_episodes=True,
#     episode_continuous_batch=True,  # `True` for visualizing continuous video frames
# )
# data_module.setup()
# dataloader = data_module.val_dataloader()

# visualize_dataloader(
#     dataloader, 
#     num_samples=5, 
#     resolution=(640, 360), 
#     legend=True,  # print action, contractor info, and segment info ... in the video
#     save_fps=30, 
#     output_dir="./"
# )

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