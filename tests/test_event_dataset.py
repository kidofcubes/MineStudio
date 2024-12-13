'''
Date: 2024-12-12 08:10:07
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-12-12 08:21:34
FilePath: /MineStudio/tests/test_event.py
'''
from torch.utils.data import DataLoader
from minestudio.data import EventDataset
from minestudio.data.minecraft.utils import batchify


kernel_kwargs = dict(
    dataset_dirs=[
        '/nfs-shared-2/data/contractors/dataset_6xx', 
        '/nfs-shared-2/data/contractors/dataset_7xx', 
    ], 
)

event_dataset = EventDataset(
    win_len=128, 
    skip_frame=1, 
    split='train', 
    split_ratio=0.8, 
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

for item in dataloader:
    print(
        f"{item.keys() = }\n", 
        f"{item['image'].shape = }\n", 
        f"{item['text'] = }\n"
    )
    

