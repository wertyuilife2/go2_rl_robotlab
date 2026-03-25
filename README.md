# go2_rl_robotlab

## Overview

Train Unitree Go2 with MoE-CTS.

This is a reproduction version of [go2_rl_gym](https://github.com/wty-yy/go2_rl_gym) on RobotLab/IsaacLab.

## Installation Guide

### 1. Install IsaacLab
Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

Notice that we use certain version of IsaacLab packages, make sure:
```
isaacsim <= 5.1.0.0 # tested on 5.1.0.0
isaaclab <= 0.53.1 # tested on 0.53.1
isaaclab-rl <= 0.4.7 # tested on 0.4.7
```

### 2. Install customized RSL-RL and RobotLab
We uses a customized version of `rsl_rl` and `robot_lab`. To install it, run the following commands:

```bash
python -m pip install -e source/robot_lab
python -m pip install -e source/rsl_rl
```

## Try examples

Use the following commands to train and play:

```bash
# Train
python scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Go2-v0 --headless

# Play
python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Go2-v0
```

## Configuration

1. Modify `source/robot_lab/robot_lab/tasks/go2/env_cfg.py` for environment config.

2. Modify `source/robot_lab/robot_lab/tasks/go2/rsl_rl_cfg.py` for algorithm config.

3. Modify `source/robot_lab/robot_lab/tasks/go2/__init__.py` to add your own task with new config.

4. Add args in commands to override above configs, for example:

    ```
    --experiment_name=moe_cts
    --run_name=v1
    --num_envs=16384
    --resume
    --checkpoint=path/to/your/checkpoint
    ```
    for more usage, see [robot_lab](https://github.com/fan-ziqi/robot_lab.git).

## Differences with `go2_rl_gym`

- Terrain's composition are different(see code).
- tracking reward are different (fixed sigma vs. dynamic sigma).

## Acknowledgements
This repository would not exist without the following open-source projects:

- [isaac_lab](https://github.com/isaac-sim/IsaacLab): Unified framework for robot learning built on NVIDIA Isaac Sim.
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git): Reinforcement learning algorithms.
- [robot_lab](https://github.com/fan-ziqi/robot_lab.git): RL Extension Library for Robots, Based on IsaacLab.

Related publications implemented in this repo:
- [CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion](https://arxiv.org/pdf/2405.10830)
