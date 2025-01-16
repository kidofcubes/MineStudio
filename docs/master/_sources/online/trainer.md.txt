# Trainer

In `online/trainer`, we have defined `PPOTrainer` and other online trainers. These trainers need to inherit from `minestudio.online.trainer.basetrainer.BaseTrainer`.

`BaseTrainer` is a foundational class in the MineStudio framework, designed to facilitate online reinforcement learning training. It manages the interaction between various components such as rollout workers, environment generators, and the policy model. Developers can extend this class to implement custom trainers for specific algorithms like PPO.

You can launch `Trainer` directly by `minestudio.online.trainer.start_trainer`.

---

## Constructor

### `__init__`

The constructor initializes the trainer with various configurations and components.

### Parameters
- **`rollout_manager: ActorHandle`**  
  Handles rollout workers.
- **`policy_generator: Callable[[], MinePolicy]`**  
  Function to generate the policy.
- **`env_generator: Callable[[], MinecraftSim]`**  
  Function to generate the environment.
- **`num_workers: int`**  
  Number of workers for parallel training.
- **`num_readers: int`**  
  Number of data readers.
- **`num_cpus_per_reader: int`**  
  Number of CPUs allocated per reader.
- **`num_gpus_per_worker: int`**  
  Number of GPUs allocated per worker.
- **`prefetch_batches: int`**  
  Number of batches to prefetch.
- **`discount: float`**  
  Discount factor for rewards.
- **`gae_lambda: float`**  
  Lambda for Generalized Advantage Estimation (GAE).
- **`context_length: int`**  
  Maximum context length for model input.
- **`use_normalized_vf: bool`**  
  Whether to normalize value function outputs.
- **`inference_batch_size_per_gpu: int`**  
  Batch size per GPU during inference.
- **`resume: Optional[str]`**  
  Path to checkpoint for resuming training.
- **`resume_optimizer: bool`**  
  Whether to resume optimizer state from the checkpoint.
- **`kwargs`**  
  Additional arguments.

---

## Methods

### `broadcast_model_to_rollout_workers`
Broadcasts the updated model to rollout workers.

#### Parameters
- **`new_version: bool`**  
  Whether to increment the model version.

---

### `fetch_fragments_and_estimate_advantages`
Fetches fragments from the replay buffer, calculates advantages, and prepares data for training.

#### Parameters
- **`num_fragments: int`**  
  Number of fragments to fetch.

#### Returns
- **`Dict[str, Any]`**: Processed data including records, TD targets, advantages, and old policy information.

---

### `setup_model_and_optimizer`
Abstract method to define the model and optimizer.

#### Returns
- **`Tuple[MinecraftSim, torch.optim.Optimizer]`**: Model and optimizer instances.

---

### `_train_func`
Main training loop function, executed by `TorchTrainer`.

---

### `train`
Abstract method to implement custom training logic.

---

### `fit`
Executes the training process using `TorchTrainer`.

---

## Attributes

### `rollout_manager`
Manages rollout workers.

### `policy_generator`
Function to create the policy model.

### `env_generator`
Function to generate environments.

### `gae_actor`
Actor for calculating GAE and reward targets.

---

## Usage
To use `BaseTrainer`, extend it and implement the abstract methods `setup_model_and_optimizer` and `train`.

```python
class PPOTrainer(BaseTrainer):
    def setup_model_and_optimizer(self):
        # Define model and optimizer
        pass

    def train(self):
        # Custom training logic
        pass
```

Refer to `minestudio.online.trainer.ppotrainer.PPOTrainer` for an example.