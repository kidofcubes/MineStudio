# Rollout

In `online/rollout`, we have defined Rollout Process for collecting the online trajectories.

The top-level of this structure is `RolloutManager`, it's parameters can be seen in rollout_config in [Config](config). You can also create and launch an Trainer with: 

Here are informations about some key elements from low level to high level.

## EnvWorker

`EnvWorker` is defined in `minestudio.online.rollout.env_worker`. This module is in charge of directly interacting with the environment and collecting data necessary for subsequent processing and analysis.

## RolloutWorker

The `RolloutWorker` is defined in `minestudio.online.rollout.env_worker`, which is responsible for compressing the observations, states, and other information from multiple `EnvWorker`s into a batches, passing them to the agent for efficient computation of actions, and then distributing the actions to the corresponding `EnvWorker`s. At the same time, it sends all the information to the `ReplayManager`.

## RolloutManager

The `RolloutManager` is defined in `minestudio.online.rollout.env_worker`, which receives the information from the `RolloutWorker` and internally maintains the work progress of all `EnvWorker`s. When a worker has been working for a certain number of consecutive frames, it clips the work into fragments and sends them, along with the information, to the `ReplayBuffer`.

You can launch `RolloutManager` directly be `minestudio.online.rollout.start_manager.start_rolloutmanager`.