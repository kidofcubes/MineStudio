from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RecordCallback, SpeedTestCallback, RewardsCallback
from minestudio.models import VPTPolicy, load_vpt_policy
import argparse

import ray
from tqdm import tqdm
import random 
import os 
import json 
from datetime import datetime
import logging
import torch
import uuid 

# Set the logging level to INFO
logging.basicConfig(level=logging.INFO)
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s')

def post_info(info:dict) -> dict:
    processed_info = {'stats': {}}
    for k, v in info.items():
        if k == 'isGuiOpen' or k == 'is_gui_open':
            processed_info['is_gui_open'] = v
        elif k == 'location_stats':
            processed_info['location_stats'] = {
                'biome_id': int(v['biome_id']), 
                'biome_rainfall': float(v['biome_rainfall']), 
                'biome_temperature': float(v['biome_temperature']), 
                'can_see_sky': bool(v['can_see_sky']), 
                'is_raining': bool(v['is_raining']), 
                'light_level': int(v['light_level']),
                'sea_level': int(v['sea_level']),
                'sky_light_level': float(v['sky_light_level']),
                'sun_brightness': float(v['sun_brightness']),
                'pitch': float(v['pitch']), # array(0.), 
                'yaw': float(v['yaw']), # array(0.), 
                'xpos': float(v['xpos']), # array(978.5),
                'ypos': float(v['ypos']), # array(64.), 
                'zpos': float(v['zpos']) # array(922.5)
            }
        elif k == 'health':
            processed_info['health'] = float(v)
        elif k == 'food_level':
            processed_info['food_level'] = float(v)
        elif k == 'pov':
            continue 
        elif k == 'voxels':
            continue
        elif k == 'inventory': 
            processed_info['inventory'] = v # keep same to avoid other errors
        elif k == 'equipped_items':
            equipped_items = {
                'chest': v['chest']['type'],
                'feet': v['feet']['type'],
                'head': v['head']['type'], # {'damage': array(0), 'maxDamage': array(0), 'type': 'air'}, 
                'legs': v['legs']['type'], # {'damage': array(0), 'maxDamage': array(0), 'type': 'air'}, 
                'mainhand': v['mainhand']['type'], # {'damage': array(0), 'maxDamage': array(0), 'type': 'air'}, 
                'offhand': v['offhand']['type'] # {'damage': array(0), 'maxDamage': array(0), 'type': 'air'}
            }
            processed_info['equipped_item'] = equipped_items
        elif k in ['pickup', 'break_item', 'craft_item', 'mine_block', 'kill_entity', 'use_item', 'drop', 'entity_killed_by']:
            # break_item and entity_killed_by is special log, which is caused from outside environment
            # mine_block, kill_entity, craft_item is the important info   
            # use_item, pickup, drop is usually can be omitted
            for item_k, item_v in v.items(): # 'acacia_boat': array(0.),
                if int(item_v) > 0:
                    processed_info['stats'][f'{k}:{item_k}'] = int(item_v)
        elif k == 'damage_dealt':
            continue
        elif k == 'player_pos':
            continue
        elif k == 'mobs':
            continue
        elif k == 'message':
            continue
        else:
            raise NotImplementedError(f"Key {k} not implemented yet in info processing function.")
    return processed_info

def post_action(action:dict) -> dict:
    processed_action = {}
    for k,v in action.items():
        if k == 'camera':
            processed_action['camera'] = list(v)
        else:
            processed_action[k] = int(v)
    return processed_action

@ray.remote(num_gpus=0.5)
def rollout(args):
    try:
        policy = load_vpt_policy(
            model_path=args.model_path,
            weights_path=args.weights_path
        ).to("cuda")

        rollout_path = os.path.join(
            args.record_path, 
            datetime.now().strftime('%Y-%m-%d'), 
            datetime.now().strftime('%y-%m-%d-%H-%M-%S')+'-'+str(uuid.uuid4()).replace('-', '')[:8])
        os.makedirs(rollout_path, exist_ok=True)

        reward_cfg = [{
                "event": "mine_block", 
                "identity": "mine diamond ore blocks", 
                "objects": ["diamond_ore"], 
                "reward": 1.0, 
                "max_reward_times": 1, 
            }]

        env = MinecraftSim(
            obs_size=(128, 128), 
            preferred_spawn_biome=random.choice(["forest", "plains"]), 
            callbacks=[
                RecordCallback(record_path=rollout_path, fps=30, frame_type="pov"),
                SpeedTestCallback(500),
                RewardsCallback(reward_cfg)

            ]
        )

        info_file_path = os.path.join(rollout_path, "info.jsonl")
        action_file_path = os.path.join(rollout_path, "action.jsonl")

        memory = None
        reward = 0
        obs, info = env.reset()
        for i in tqdm(range(args.num_steps)):
            action, memory = policy.get_action(obs, memory, input_shape='*')
            # import ipdb; ipdb.set_trace()
            processed_info = post_info(info)
            with open(info_file_path, 'a', encoding='utf-8') as f:
                # 将字典对象序列化为 JSON 字符串，并写入文件
                f.write(json.dumps(processed_info, ensure_ascii=False) + '\n')
            processed_action = post_action(env.agent_action_to_env_action(action))
            with open(action_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(processed_action, ensure_ascii=False) + '\n')
            if reward > 0:
                env.close()
                return
            obs, reward, terminated, truncated, info = env.step(action) # reward: 
            # import ipdb; ipdb.set_trace()
        env.close()
    except Exception as e:
        print(e)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/2x.model")
    parser.add_argument("--weights_path", type=str, default="weights/rl-from-early-game-2x.weights")
    parser.add_argument("--record_path", type=str, default="./output")
    parser.add_argument("--num_steps", type=int, default=20000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_rollouts", type=int, default=100)
    args = parser.parse_args()
    
    # 检测可用 GPU
    num_gpus = torch.cuda.device_count()
    logging.info(f"Available GPUs: {num_gpus}")

    # 初始化 Ray
    ray.init(num_gpus=num_gpus, log_to_driver=True)
    # ray.init(address='auto')

    # 任务分批
    futures = [rollout.remote(args) for _ in range(args.num_rollouts)]
    print(f"Rollout futures: {len(futures)}")

    # 获取任务结果
    results = ray.get(futures)
    for result in results:
        logging.info(result)

    # 关闭 Ray
    ray.shutdown()
    
    # rollout(args)



    