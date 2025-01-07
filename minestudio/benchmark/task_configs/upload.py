'''
Date: 2025-01-06 20:08:00
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2025-01-07 10:24:24
FilePath: /MineStudio/minestudio/benchmark/task_configs/upload.py
'''
from huggingface_hub import HfApi

if __name__ == '__main__':
    local_dir = "./simple"
    repo_id = "CraftJarvis/MineStudio_task_group.simple"
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type='dataset')
    api.upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type="dataset")