# Automatic Evaluation Pipeline

The pipeline automates evaluation tasks in the MineStudio framework, enabling the generation of criteria and video evaluation for agent performance analysis.

---


## Code Structure

```plaintext

auto_eval/
    ├── criteria_files/
    │   └── Contains criteria files for evaluating videos for each task.
    ├── eval_video/
    │   └── Stores example videos and provides a structure for saving task-specific evaluation videos.
    ├── batch_video_rating.py
    │   └── Batch evaluation of videos for task performance.
    ├── individual_video_rating.py
    │   └── Evaluate videos individually for detailed analysis.
    ├── video_comparison.py
        └── Compare videos to measure performance differences.
```


## Evaluating Videos with Vision-Language Models (VLM)

The following commands evaluate task performance using pre-recorded videos and predefined criteria.

### 1. Compare Two Videos

```bash
python video_comparison.py \
  --video_path_a='./eval_video/build_gate/build_gate_5.mp4' \
  --video_path_b='./eval_video/build_gate/build_gate_7.mp4' \
  --criteria_path='./auto_eval/criteria_files/build_gate.txt'
```

### 2. Individual Video Evaluation

```bash
python individual_video_rating.py \
  --video_path='./eval_video/build_gate/build_gate_5.mp4' \
  --criteria_path='./auto_eval/criteria_files/build_gate.txt'
```

### 3. Batch Video Evaluation

```bash
python batch_video_rating.py \
  --videos_path='./eval_video/' \
  --criteria_files_path='./auto_eval/criteria_files/'
```

---

## Organizing Files for Batch Evaluation

### Video Directory Structure

Organize your task-specific videos under the `videos_path` directory:

```
videos_path     
├── build_waterfall     # task_name_1     
│     ├── episode_1.mp4
│     ├── episode_2.mp4
├── build_house         # task_name_2
│     ├── episode_1.mp4
│     ├── episode_2.mp4
├── task_name_3
│     ├── episode_1.mp4
│     ├── episode_2.mp4
```

### Criteria Files Directory Structure

Store criteria files under the `criteria_files_path` directory, matching the task names:

```
criteria_files_path     
├── build_waterfall.txt # task_name_1     
├── build_house.txt     # task_name_2
├── task_name_3.txt
```


By adhering to the outlined file structures and using the provided scripts, you can efficiently evaluate agent performance across multiple tasks in the MineStudio framework.

---

### Example Output

The following is an example output of the **Automatic Evaluation Pipeline**, showcasing how two videos are compared across several evaluation criteria. Each criterion is explained with observations, and an overall assessment is provided. This structured format ensures transparency and consistency in evaluating agent performance.

---

#### Example Output Format

```json
[
    {
        "Task Progress": "B is better",
        "Action Control": "B is better",
        "Error Recognition and Correction": "B is better",
        "Creative Attempts": "tie",
        "Task Completion Efficiency": "B is better",
        "Material Selection and Usage": "tie",
        "video_1_path": "./eval_video/build_gate_5.mp4",
        "video_2_path": "./eval_video/build_gate_7.mp4"
    },
    "Task Progress:\n- Video B constructs two pillars and an arch; A does not complete the arch.\nresult: B is better\n\nAction Control:\n- Video A shows more wandering and redundant actions.\nresult: B is better\n\nError Recognition and Correction:\n- Video B corrects structure misalignments.\nresult: B is better\n\nCreative Attempts:\n- Neither video shows creative elements like decorations.\nresult: tie\n\nTask Completion Efficiency:\n- Video B completes the task faster and more efficiently.\nresult: B is better\n\nMaterial Selection and Usage:\n- Both use oak planks appropriately.\nresult: tie\n"
]
```

---

#### Key Features

1. **Assessment Dimensions**:
  - **Task Progress**: Measures how much of the task is completed.
  - **Action Control**: Assesses movement precision and avoidance of redundant actions.
  - **Error Recognition and Correction**: Evaluates the agent’s ability to detect and fix mistakes.
  - **Creative Attempts**: Considers innovative or decorative efforts beyond task requirements.
  - **Task Completion Efficiency**: Tracks speed and resourcefulness in completing the task.
  - **Material Selection and Usage**: Ensures appropriate materials are used.

2. **Structured Results**:
   - The first section provides a concise summary of the evaluation for each criterion.
   - Example:
     - `"Task Progress": "B is better"`
     - `"Creative Attempts": "tie"`

3. **Detailed Observations**:
   - The second section explains the reasoning behind each result.
   - Example:
     - **Task Progress**: "Video B constructs two pillars and an arch; A does not complete the arch."
     - **Creative Attempts**: "Neither video shows creative elements like decorations."

---