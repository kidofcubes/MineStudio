<!--
 * @Date: 2024-11-30 13:20:04
 * @LastEditors: muzhancun muzhancun@126.com
 * @LastEditTime: 2025-01-02 19:55:34
 * @FilePath: /MineStudio/README.md
-->

<div align="center">
<img src="./docs/source/_static/banner.png" width="60%" alt="MineStudio" />
</div>

<hr>

<div align="center">
	<a href="https://craftjarvis.github.io/"><img alt="Homepage" src="https://img.shields.io/badge/%20CraftJarvis-HomePage-ffc107?color=blue&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
	<a href="https://huggingface.co/CraftJarvis""><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CraftJarvis-ffc107?color=3b65ab&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
	<a href="https://github.com/CraftJarvis/MineStudio/blob/master/LICENSE"><img src="https://img.shields.io/badge/Code License-MIT-blue"/></a>
</div>
<div align="center">	
	<a href="https://arxiv.org/abs/2412.18293"><img src="https://img.shields.io/badge/arXiv-2412.18293-b31b1b.svg"></a>
	<a href="https://craftjarvis.github.io/MineStudio/"><img src="https://img.shields.io/badge/Doc-Sphinx-yellow"/></a>
    	<a href="https://pypi.org/project/minestudio/"><img src="https://img.shields.io/pypi/v/minestudio.svg"/></a>
	<a href="https://huggingface.co/CraftJarvis"><img src="https://img.shields.io/badge/Dataset-Released-orange"/></a>
	<a href="https://github.com/CraftJarvis/MineStudio"><img src="https://visitor-badge.laobi.icu/badge?page_id=CraftJarvis.MineStudio"/></a>
	<a href="https://github.com/CraftJarvis/MineStudio"><img src="https://img.shields.io/github/stars/CraftJarvis/MineStudio"/></a>
</div>

<p align="center">
  <img src="https://github.com/phython96/Images/blob/master/minestudio_gallary.gif" />
  <div align="center" style="color: grey; font-size: 12px;">
    Demos of agents trained with MineStudio <b>online</b> module: building a nether portal and shooting animals.
  </div>
</p>



## Overview

<div align="center">
<img src="./docs/source/_static/workflow.png" width="" alt="Workflow" />
</div>

MineStudio contains a series of tools and APIs that can help you quickly develop Minecraft AI agents:
- [Simulator](https://craftjarvis.github.io/MineStudio/simulator/index.html): Easily customizable Minecraft simulator based on [MineRL](https://github.com/minerllabs/minerl).
- [Data](https://craftjarvis.github.io/MineStudio/data/index.html): A trajectory data structure for efficiently storing and retrieving arbitray trajectory segment.
- [Models](https://craftjarvis.github.io/MineStudio/models/index.html): A template for Minecraft policy model and a gallery of baseline models.
- [Offline Training](https://craftjarvis.github.io/MineStudio/offline/index.html): A straightforward pipeline for pre-training Minecraft agents with offline data.
- [Online Training](https://craftjarvis.github.io/MineStudio/online/index.html): Efficient RL implementation supporting memory-based policies and simulator crash recovery.
- [Inference](https://craftjarvis.github.io/MineStudio/inference/index.html): Pallarelized and distributed inference framework based on [Ray](https://docs.ray.io/en/latest/index.html).
- [Benchmark](https://craftjarvis.github.io/MineStudio/benchmark/index.html): Automating and batch-testing of diverse Minecraft tasks.

**This repository is under development.** We welcome any contributions and suggestions.

## Installation

For a more detailed installation guide, please refer to the [documentation](https://craftjarvis.github.io/MineStudio/overview/installation.html).

MineStudio requires Python 3.10 or later. We recommend using conda to maintain an environment on Linux systems. JDK 8 is also required for running the Minecraft simulator.

```bash
conda create -n minestudio python=3.10 -y
conda activate minestudio
conda install --channel=conda-forge openjdk=8 -y
```

MineStudio is available on PyPI. You can install it via pip.
```bash
pip install MineStudio
```
To install MineStudio from source, you can run the following command:
```bash
pip install git+https://github.com/CraftJarvis/MineStudio.git
```

Minecraft simulator requires rendering tools. For users with nvidia graphics cards, we recommend installing **VirtualGL**. For other users, we recommend using **Xvfb**, which supports CPU rendering but is relatively slower. Refer to the [documentation](https://craftjarvis.github.io/MineStudio/overview/installation.html#install-the-rendering-tool) for installation commands.

After the installation, you can run the following command to check if the installation is successful:
```bash
python -m minestudio.simulator.entry # using Xvfb
MINESTUDIO_GPU_RENDER=1 python -m minestudio.simulator.entry # using VirtualGL
```

## Why MineStudio

## Acknowledgement

The simulation environment is built upon [MineRL](https://github.com/minerllabs/minerl) and [Project Malmo](https://github.com/microsoft/malmo).
We also refer to [Ray](https://docs.ray.io/en/latest/index.html), [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) for distributed training and inference.
Thanks for their great work.

## Citation

```bibtex
@inproceedings{MineStudio,
  title={MineStudio: A Streamlined Package for Minecraft AI Agent Development},
  author={Shaofei Cai and Zhancun Mu and Kaichen He and Bowei Zhang and Xinyue Zheng and Anji Liu and Yitao Liang},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:274992448}
}
```
