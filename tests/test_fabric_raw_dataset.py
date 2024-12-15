import lightning as L
from tqdm import tqdm
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
    batch_size=3,
    num_workers=2,
    prefetch_factor=4,
    shuffle_episodes=True,
    episode_continuous_batch=True, 
)
data_module.setup()
train_loader = data_module.train_dataloader()
train_loader = fabric.setup_dataloaders(train_loader, use_distributed_sampler=False)
rank = fabric.local_rank
for idx, batch in enumerate(tqdm(train_loader, disable=True)):
    if idx > 50:
        break
    print(
        f"{rank = } \t" + "\t".join(
            [f"{a[-20:]} {b}" for a, b in zip(batch['episode'], batch['progress'])]
        )
    )
