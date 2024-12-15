# Overview

The code in the Online Training Module is mainly divided into four parts: `run`, `rollout`, and `train`. Each part is contained in its corresponding subfolder under the `online` folder.

## Rollout
The `rollout` section is located under `online/rollout` and mainly consists of several parts:

The `EnvWorker`  can obtain control information such as actions from the `RolloutWorker`. interact directly with the environment and pass the obtained information to the `RolloutWorker`.

The `RolloutWorker` is responsible for compressing the observations, states, and other information from multiple `EnvWorker`s into a batches, passing them to the agent for efficient computation of actions, and then distributing the actions to the corresponding `EnvWorker`s. At the same time, it sends all the information to the `ReplayManager`.

The `RolloutManager` receives the information from the `RolloutWorker` and internally maintains the work progress of all `EnvWorker`s. When a worker has been working for a certain number of consecutive frames, it clips the work into fragments and sends them, along with the information, to the `ReplayBuffer`.

## Trainer
This folder contains our defined Online Trainer, including the parent class `BaseTrainer` and subclasses such as `PPOTrainer`. These trainers accept `config` and the computation results of `GaeWorker` on the `ReplayBuffer` as parameters.

By inheriting from `BaseTrainer`, you can customize the trainer according to your needs. Refer to [Customization](quick-online) for details.

