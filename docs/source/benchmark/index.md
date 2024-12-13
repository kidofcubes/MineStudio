# Benchmark

This tutorial provides an overview of the codebase for automating and batch-testing tasks in the MineStudio benchmark. It includes the structure, purpose, and main functionalities of the framework.


```{toctree}
:caption: MineStudio Benchmark

quick-benchmark
automatic-evaluation
```

---

## Overview


The MineStudio benchmark is a comprehensive framework for evaluating agent performance across a wide range of Minecraft-based tasks. It offers the following key features:  

- **Diverse Task Support**: Evaluate agents on tasks such as building, mining, crafting, collecting, tool usage, and more.  
- **Game Mode Variability**: Includes both simple and challenging game modes to test agents under varying levels of difficulty.  
- **Batch Task Execution**: Run multiple tasks simultaneously and record task completion videos for analysis.  
- **VLM-Based Evaluation**: Leverage Vision-Language Models to analyze task videos and provide automated scoring.  

---

## How to Use

1. **Run Batch Tests**:
   - Use `test_pipeline.py` or `test.py` to execute tasks.
   - Ensure your environment supports GPU acceleration for optimal performance.

2. **Analyze Results**:
   - Review generated videos and metrics in the `eval_video` folder.
   - Use criteria files to score and validate task completion.

