```markdown
# MineStudio Benchmark Automation Tutorial

This tutorial introduces the codebase for automating and batch-testing tasks in the MineStudio benchmark. The framework is designed to evaluate agents' performance on Minecraft-based tasks using automated pipelines. It provides a modular, scalable structure to handle a variety of evaluation scenarios.

---

## Overview

The MineStudio benchmark framework enables:
- **Automated Evaluation**: Analyze agent performance on individual tasks or batches of tasks.
- **Video-Based Scoring**: Evaluate task success using recorded videos.
- **Batch Processing**: Execute tasks in parallel to improve testing efficiency.
- **Customizable Configurations**: Use YAML-based configurations for flexible task definitions.

The structure is divided into several key modules, ensuring clarity and extensibility for researchers and developers.

---

## Code Structure

### Directory Layout

```plaintext
├─ auto_eval/
│  ├─ batch_video_rating.py      # Batch evaluation of videos for task performance.
│  ├─ eval_video/                # Example videos for evaluation.
│  │  ├─ build_pillar/           # Task: Build a pillar (videos for evaluation).
│  │  ├─ combat_spider/          # Task: Combat a spider (videos for evaluation).
│  ├─ individual_video_rating.py # Evaluate videos individually for detailed analysis.
│  ├─ video_comparison.py        # Compare videos to measure performance differences.
├─ output/                       # Directory for task-generated videos and outputs.
│  ├─ make_obsidian_by_wate/
│  ├─ find_forest/
│  ├─ smelt_food/
├─ task_configs/                 # Task configuration files (YAML-based).
│  ├─ simple/                    # Simple task definitions.
│  ├─ hard/                      # More complex task definitions.
├─ test_pipeline.py              # Script for batch testing and parallelization.
├─ test.py                       # Script for individual or small-batch task testing.
├─ utility/                      # Utility scripts for callbacks and configuration.
│  ├─ record_call.py             # Handles recording of task-related data.
│  ├─ read_conf.py               # Reads YAML configs and generates callbacks.
│  ├─ task_call.py               # Implements task-specific logic.
```

---

## Key Functionalities

### 1. **Automated Evaluation**

- The framework uses YAML files in `task_configs/` to define tasks.
- Video evaluations are stored under `auto_eval/eval_video/` and can be processed using scripts like `batch_video_rating.py`.

### 2. **Video-Based Scoring**

- Videos recorded during tasks are saved in the `output/` directory.
- Evaluation can be performed on individual videos (`individual_video_rating.py`) or in batches (`batch_video_rating.py`).
- `video_comparison.py` allows comparing agent behaviors across different videos.

### 3. **Batch Processing and Parallelization**

- The `test_pipeline.py` script leverages [Ray](https://www.ray.io/) for parallel execution of tasks, significantly speeding up evaluations.
- The `test.py` script demonstrates running small-scale batch tests for quick validations.

---

## Workflow Overview

### Task Configuration

Tasks are defined in YAML files located in the `task_configs/` directory. Each YAML file includes:
- **Initial Setup**: Commands to configure the environment.
- **Task Goals**: Objectives for the agent to accomplish.

Example YAML:
```yaml
name: find_forest
init_commands:
  - /give @p compass
  - /time set day
task_goal:
  description: Navigate to the nearest forest biome.
```

---

### Running Tests

1. **Individual or Small-Scale Tests**:
   - Use `test.py` for running specific tasks or testing new configurations.
   - Example:
     ```bash
     python test.py
     ```

2. **Batch Testing with Parallelization**:
   - Use `test_pipeline.py` for executing tasks in parallel.
   - Example:
     ```bash
     python test_pipeline.py
     ```

3. **Video Evaluation**:
   - Evaluate recorded videos in `output/` using scripts in `auto_eval/`.
   - Batch evaluation:
     ```bash
     python auto_eval/batch_video_rating.py
     ```

---

### Code Explanation: `test.py`

This script demonstrates how to evaluate tasks using YAML-based configurations. Below is an outline of its workflow:

1. **Task Setup**:
   - Load configuration files from `task_configs/simple`.
   - Parse YAML files into callbacks using `convert_yaml_to_callbacks`.

2. **Environment Initialization**:
   - Use `MinecraftSim` to create a simulation environment.
   - Add callbacks:
     - `RecordCallback`: Saves video frames for evaluation.
     - `CommandsCallback`: Initializes the environment.
     - `TaskCallback`: Implements task-specific behavior.

3. **Policy Loading**:
   - Load a pretrained Vision-Language Model (VLM) for decision-making.
   - Utilize GPU acceleration for enhanced performance.

4. **Task Execution**:
   - Reset the environment and run the task for 12 steps.
   - Save observations, actions, and outputs for analysis.

5. **Result Storage**:
   - Videos and logs are saved in the `output/` directory.

### Example Code
```python
for file_name in os.listdir(conf_path):
    if file_name.endswith('.yaml'):
        file_path = os.path.join(conf_path, file_name)
        commands_callback, task_callback = convert_yaml_to_callbacks(file_path)

        env = MinecraftSim(
            obs_size=(128, 128), 
            callbacks=[
                RecordCallback(record_path=f"./output/{file_name[:-5]}/", fps=30, frame_type="pov"),
                CommandsCallback(commands_callback),
                TaskCallback(task_callback),
            ]
        )
        policy = load_openai_policy(
            model_path="/nfs-shared/jarvisbase/pretrained/foundation-model-2x.model",
            weights_path="/nfs-shared/jarvisbase/pretrained/foundation-model-2x.weights"
        ).to("cuda")

        obs, info = env.reset()
        for i in range(12):
            action, memory = policy.get_action(obs, memory, input_shape='*')
            obs, reward, terminated, truncated, info = env.step(action)
        env.close()
```

---

## Conclusion

The MineStudio benchmark framework is a powerful tool for evaluating agent performance in Minecraft-based tasks. With modular design, video-based evaluation, and batch processing capabilities, it caters to a wide range of research scenarios. Customize tasks using YAML configurations and use the provided scripts for efficient testing and evaluation.
```