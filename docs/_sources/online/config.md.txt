# Config

To launch our online code, we need to prepare a configuration (`config`) and pass it along with serializable functions `env_generator` and `policy_generator` into `online.rollout.start_manager.start_rolloutmanager` and `minestudio.online.trainer.start_trainer.start_trainer`. 

The `online_dict` configuration is a dictionary that specifies parameters for training, rollout management, and logging in the online system. Below is the format and explanation for each element.

These are some of the more important elements in the settings:

## Important Config

### `trainer_name`


# All Keys' Descriptioin



## Top-Level Keys

### `trainer_name`
- **Type**: String  
- **Description**: Specifies the trainer to use.  
- **Example**: `"PPOTrainer"`

### `detach_rollout_manager`
- **Type**: Boolean  
- **Description**: Indicates whether to detach the rollout manager process.  
- **Example**: `True`

---

## `rollout_config`
Configuration related to the rollout manager.

### `num_rollout_workers`
- **Type**: Integer  
- **Description**: Number of rollout worker processes.  
- **Example**: `2`

### `num_gpus_per_worker`
- **Type**: Float  
- **Description**: Number of GPUs allocated per rollout worker.  
- **Example**: `1.0`

### `num_cpus_per_worker`
- **Type**: Integer  
- **Description**: Number of CPUs allocated per rollout worker.  
- **Example**: `1`

### `fragment_length`
- **Type**: Integer  
- **Description**: Number of steps per rollout fragment.  
- **Example**: `256`

### `to_send_queue_size`
- **Type**: Integer  
- **Description**: Size of the queue for sending rollout data to the trainer.  
- **Example**: `4`

#### `worker_config`
- **Description**: Configuration for individual rollout workers.
  - `num_envs`: Number of environments per worker. **Example**: `16`
  - `batch_size`: Batch size for each worker. **Example**: `2`
  - `restart_interval`: Restart interval for workers (in seconds). **Example**: `3600`
  - `video_fps`: Frames per second for video output. **Example**: `20`
  - `video_output_dir`: Directory for video outputs. **Example**: `"output/videos"`

#### `replay_buffer_config`
- **Description**: Configuration for the replay buffer.
  - `max_chunks`: Maximum number of chunks in the buffer. **Example**: `4800`
  - `max_reuse`: Maximum reuse count for data chunks. **Example**: `2`
  - `max_staleness`: Maximum staleness of data chunks. **Example**: `2`
  - `fragments_per_report`: Fragments to report per iteration. **Example**: `40`
  - `fragments_per_chunk`: Fragments stored per chunk. **Example**: `1`
  - `database_config`: Configuration for the database.
    - `path`: Path to database files. **Example**: `"output/replay_buffer_cache"`
    - `num_shards`: Number of shards in the database. **Example**: `8`

---

## `train_config`
Configuration related to training.

### `num_workers`
- **Type**: Integer  
- **Description**: Number of training worker processes.  
- **Example**: `2`

### `num_gpus_per_worker`
- **Type**: Float  
- **Description**: Number of GPUs allocated per training worker.  
- **Example**: `1.0`

### `num_iterations`
- **Type**: Integer  
- **Description**: Number of training iterations.  
- **Example**: `4000`

### Other Parameters
- `learning_rate`: Learning rate for the optimizer. **Example**: `0.00002`
- `batch_size_per_gpu`: Batch size per GPU. **Example**: `1`
- `ppo_clip`: PPO clip range. **Example**: `0.2`
- `save_interval`: Interval for saving models. **Example**: `10`
- `save_path`: Directory for saving models. The default saving path is in ray's default working path: **~/ray_results**, **Example**: `output` would save checkpoints in **~/ray_results/output**, you can also pass absolute path in it.

---

## `logger_config`
Configuration for wandb logging.

### `project`
- **Type**: String  
- **Description**: Name of the logging project.  
- **Example**: `"minestudio_online"`

### `name`
- **Type**: String  
- **Description**: Name of the logging instance.  
- **Example**: `"cow"`