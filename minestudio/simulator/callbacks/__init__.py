'''
Date: 2024-11-11 07:53:19
LastEditors: Muyao 2350076251@qq.com
LastEditTime: 2025-03-18 22:58:34
FilePath: /MineStudio/minestudio/simulator/callbacks/__init__.py
'''
from minestudio.simulator.callbacks.callback import MinecraftCallback
from minestudio.simulator.callbacks.speed_test import SpeedTestCallback
from minestudio.simulator.callbacks.record import RecordCallback
from minestudio.simulator.callbacks.record_demo import RecordCallback2
from minestudio.simulator.callbacks.summon_mobs import SummonMobsCallback
from minestudio.simulator.callbacks.mask_actions import MaskActionsCallback
from minestudio.simulator.callbacks.rewards import RewardsCallback
from minestudio.simulator.callbacks.fast_reset import FastResetCallback,FastResetCallback2
from minestudio.simulator.callbacks.commands import CommandsCallback
from minestudio.simulator.callbacks.task import TaskCallback
from minestudio.simulator.callbacks.play import PlayCallback
from minestudio.simulator.callbacks.point import PointCallback, PlaySegmentCallback
from minestudio.simulator.callbacks.demonstration import DemonstrationCallback
from minestudio.simulator.callbacks.judgereset import JudgeResetCallback
from minestudio.simulator.callbacks.reward_gate import GateRewardsCallback
from minestudio.simulator.callbacks.voxels import VoxelsCallback
from minestudio.simulator.callbacks.init_inventory import InitInventoryCallback
from minestudio.simulator.callbacks.teleport import TeleportCallback