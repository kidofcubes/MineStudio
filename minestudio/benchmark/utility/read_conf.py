'''
Date: 2024-12-06 16:42:49
LastEditors: zhengxinyue
LastEditTime: 2024-12-11 17:44:10
FilePath: /MineStudio/minestudio/benchmark/utility/read_conf.py
'''
import os
import yaml

def convert_yaml_to_callbacks(yaml_file):
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    commands = data.get('custom_init_commands', [])
    commands_callback = commands

    text = data.get('text', '')
    task_name = os.path.splitext(os.path.basename(yaml_file))[0]
    task_callback = {}
    task_callback['name'] = task_name
    task_callback['text'] = text

    return commands_callback, task_callback
