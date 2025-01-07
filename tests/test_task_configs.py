'''
Date: 2025-01-07 10:23:19
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2025-01-07 03:32:38
FilePath: /MineStudio/tests/test_task_configs.py
'''
from rich import print
from minestudio.benchmark import prepare_task_configs
from minestudio.simulator.callbacks import load_callbacks_from_config

if __name__ == '__main__':
    file_list = prepare_task_configs("simple", path="CraftJarvis/MineStudio_task_group.simple")
    name, file_path = file_list.popitem()
    callbacks = load_callbacks_from_config(file_path)
    print(file_list)