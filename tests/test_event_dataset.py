'''
Date: 2024-12-12 08:10:07
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-12-30 14:06:01
FilePath: /MineStudio/tests/test_event_dataset.py
'''
from tqdm import tqdm
from torch.utils.data import DataLoader
from minestudio.data import load_dataset
from minestudio.data.minecraft.utils import batchify


kernel_kwargs = dict(
    dataset_dirs=[
        '/nfs-shared-2/data/contractors/dataset_6xx', 
        '/nfs-shared-2/data/contractors/dataset_7xx', 
        '/nfs-shared-2/data/contractors/dataset_8xx', 
        '/nfs-shared-2/data/contractors/dataset_9xx', 
        '/nfs-shared-2/data/contractors/dataset_10xx', 
    ], 
)

event_dataset = load_dataset(
    mode='event', 
    win_len=128, 
    skip_frame=1, 
    split='train', 
    split_ratio=1.0, 
    verbose=True, 
    event_regex='minecraft.kill_entity:.*', 
    **kernel_kwargs, 
)

dataloader = DataLoader(
    event_dataset, 
    batch_size=4, 
    num_workers=2, 
    shuffle=True, 
    collate_fn=batchify,
)

# for idx, item in enumerate(dataloader):
#     if idx >= 5:
#         break
#     print(
#         f"{idx = }\n"
#         f"{item.keys() = }\n", 
#         f"{item['image'].shape = }\n", 
#         f"{item['text'] = }\n"
#     )
