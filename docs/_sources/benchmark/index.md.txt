# Benchmark

This tutorial provides an overview of the codebase for automating and batch-testing tasks in the MineStudio benchmark. It includes the structure, purpose, and main functionalities of the framework.

---

## Overview

The MineStudio benchmark is designed to evaluate agent performance on various Minecraft-based tasks using automated pipelines. This framework supports the following key functionalities:

- **Task Criteria Management**: Store and apply specific evaluation criteria for tasks.
- **Video-Based Evaluation**: Analyze videos generated during tasks for automated scoring.
- **Batch Task Execution**: Run multiple tasks in parallel to streamline testing.
- **Customizable Callbacks**: Leverage task-specific callbacks for seamless integration with simulations.

The provided structure ensures modularity, reusability, and ease of scalability for diverse testing scenarios.

---

## Code Structure

### Directory Layout

```plaintext
criteria files/
    └── Contains criteria files for evaluating videos for each task.

eval_video/
    └── Stores example videos and provides a structure for saving task-specific evaluation videos.

benchmark/
    ├── test_pipeline.py
        └── Example script for parallelized and batched task execution.
    ├── test.py
        └── Example script for running batch tests.
    ├── utility/
        ├── record_call.py
            └── Callback for recording task-related data.
        ├── read_conf.py
            └── Parses task configuration files and converts them into callbacks.
        ├── task_call.py
            └── Task-specific callback implementation.
```



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

---

## Purpose of Each Component

### 1. **Criteria Files**
   - Stores YAML configuration files defining task-specific evaluation criteria.
   - Ensures consistency in assessing agent performance.

### 2. **Eval Video Directory**
   - Holds videos generated during task execution.
   - Acts as a repository for manual or automated evaluations.

### 3. **Utility Scripts**
   - **`record_call.py`**: Records key metrics, observations, or outputs during task execution.
   - **`read_conf.py`**: Reads YAML configuration files and generates corresponding callbacks.
   - **`task_call.py`**: Defines task-specific behaviors and integrates them into the simulator.

---

## Main Functionalities

### 1. **Automated Task Evaluation**
The `test.py` script demonstrates how to automatically evaluate tasks using YAML-defined configurations. It employs modular callbacks for task initialization, reward tracking, and recording outputs.

Key Features:
- **Episode Pipeline**: Automatically resets and runs tasks.
- **Callback Integration**: Supports various callbacks to customize task interactions.

### 2. **Batch Testing with Parallelization**
The `test_pipeline.py` script enables parallel execution of multiple tasks using [Ray](https://www.ray.io/). This significantly accelerates the testing process by distributing workloads across multiple cores or nodes.

---

## Detailed Explanation of `test.py`

Below is a breakdown of the script's functionality:

1. **Initialization**:
   - Uses `ray.init()` to initialize distributed computing.
   - Loads task configurations from `./task_configs/simple`.

2. **Callback Creation**:
   - Reads each YAML file in the configuration directory.
   - Converts the YAML file into task-specific callbacks using `convert_yaml_to_callbacks`.

3. **Environment Setup**:
   - Creates a Minecraft simulation environment using `MinecraftSim`.
   - Integrates three callbacks:
     - `RecordCallback`: Saves POV frames as videos.
     - `CommandsCallback`: Executes initial setup commands.
     - `TaskCallback`: Implements the specific task logic.

4. **Policy Loading**:
   - Loads a pretrained Vision-Language Model (VLM) for decision-making.
   - Utilizes GPU acceleration (`.to("cuda")`).

5. **Task Execution**:
   - Resets the environment and begins a new rollout.
   - Executes 12 steps in the simulation, generating observations, rewards, and other metrics.
   - Closes the environment upon completion.

### Example Code Snippet
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

## How to Use

1. **Prepare Task Configurations**:
   - Write YAML files in the `criteria files` folder to define task settings.

2. **Run Batch Tests**:
   - Use `test_pipeline.py` or `test.py` to execute multiple tasks.
   - Ensure your environment supports GPU acceleration for optimal performance.

3. **Analyze Results**:
   - Review generated videos and metrics in the `eval_video` folder.
   - Use criteria files to score and validate task completion.

---

## Conclusion

This benchmark provides a robust framework for automating the evaluation of agent performance in Minecraft-based tasks. Its modular design and batch testing capabilities make it highly adaptable for diverse research and development needs. Customize task configurations and callbacks to explore new experiments or refine existing setups.
```




