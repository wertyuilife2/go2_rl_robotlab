# go2_rl_robotlab

## Overview

Train Unitree Go2 with MoE-CTS on IsaacLab and deploy it to MuJoCo.

This is a reproduction version of [go2_rl_gym](https://github.com/wty-yy/go2_rl_gym) on RobotLab/IsaacLab.

## Installation Guide

### 1. Install IsaacLab
Install IsaacLab 2.3.0 release by following the [installation guide](https://isaac-sim.github.io/IsaacLab/release/2.3.0/source/setup/installation/pip_installation.html#).

After installation, your environment should satisfy the following requirements:
```
isaacsim <= 5.1.0.0 # tested on 5.1.0.0
isaaclab <= 0.53.1 # tested on 0.53.1
isaaclab-rl <= 0.4.7 # tested on 0.4.7
```

Higher version may cause conflicts with our customized `rsl_rl==3.3.0` and `robot_lab==2.3.0`.

### 2. Install customized RSL-RL and RobotLab
We uses a customized version of `rsl_rl` and `robot_lab`. To install it, run the following commands:

```bash
python -m pip install -e source/robot_lab
python -m pip install -e source/rsl_rl
```

### 3. Install MuJoCo for Sim2Sim (optional)
If you want to use `mujoco` for Sim2Sim, install it by running the following command:
```bash
pip install mujoco # tested on mujoco 3.4.0 & 3.6.0
```

## Train and Play

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

## MuJoCo Sim2Sim

Use the following command to run the Sim2Sim with MuJoCo:

```bash
python deploy/deploy_mujoco/deploy_go2.py
```

- **Automatic detection**: When connecting the handle, the script automatically activates the handle control mode.
- **Controller not available**: The script will use the default commands in the configuration file.

Handle axis mapping:

- `LX/LY`: Forward/Lateral Speed Command
- `RX`: Angular velocity (steering) command

Modify the `xml_path` parameter in `deploy/deploy_mujoco/config/go2.yaml` to switch simulation scenarios:

```yaml
# Flat
xml_path: "{ROOT_DIR}/resources/go2/flat.xml"

# Stairs
xml_path: "{ROOT_DIR}/resources/go2/stairs.xml"

# Boxes
xml_path: "{ROOT_DIR}/resources/go2/boxes.xml"

# Custom
xml_path: "{ROOT_DIR}/resource/go2/your-custom-scene.xml"
```

## Differences with `go2_rl_gym`

- Terrain's composition are different(see code).
- tracking reward are different (fixed sigma vs. dynamic sigma).

## Acknowledgements
This repository would not exist without the following open-source projects:

- [isaac_lab](https://github.com/isaac-sim/IsaacLab): Unified framework for robot learning built on NVIDIA Isaac Sim.
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git): Reinforcement learning algorithms.
- [robot_lab](https://github.com/fan-ziqi/robot_lab.git): RL Extension Library for Robots, Based on IsaacLab.
- [mujoco](https://github.com/google-deepmind/mujoco.git): High-performance CPU physics simulator.

Related publications implemented in this repo:
- [CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion](https://arxiv.org/pdf/2405.10830)
