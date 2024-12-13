### Automatic Evaluation Pipeline

The pipeline automates evaluation tasks in the MineStudio framework, enabling the generation of criteria and video evaluation for agent performance analysis.

---

#### Generating Criteria Files

To create criteria files for new tasks:
```bash
cd auto_eval/
python rule_generation.py
```

---

### Evaluating Videos with Vision-Language Models (VLM)

The following commands evaluate task performance using pre-recorded videos and predefined criteria.

#### 1. Compare Two Videos

```bash
python video_comparison.py \
  --video_path_a='./eval_video/build_gate/build_gate_5.mp4' \
  --video_path_b='./eval_video/build_gate/build_gate_7.mp4' \
  --criteria_path='./auto_eval/criteria_files/build_gate.txt'
```

#### 2. Individual Video Evaluation

```bash
python individual_video_rating.py \
  --video_path='./eval_video/build_gate/build_gate_5.mp4' \
  --criteria_path='./auto_eval/criteria_files/build_gate.txt'
```

#### 3. Batch Video Evaluation

```bash
python batch_video_rating.py \
  --videos_path='./eval_video/' \
  --criteria_files_path='./auto_eval/criteria_files/'
```

---

### Organizing Files for Batch Evaluation

#### Video Directory Structure

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

#### Criteria Files Directory Structure

Store criteria files under the `criteria_files_path` directory, matching the task names:

```
criteria_files_path     
├── build_waterfall.txt # task_name_1     
├── build_house.txt     # task_name_2
├── task_name_3.txt
```

---

By adhering to the outlined file structures and using the provided scripts, you can efficiently evaluate agent performance across multiple tasks in the MineStudio framework.