# go2_rl_robotlab

## Overview

Train Unitree Go2 with MoE-CTS.

This is a reproduction version of [go2_rl_gym](https://github.com/wty-yy/go2_rl_gym) on robotlab/isaaclab.

## Installation Guide

### 1. Base Installation
Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

Notice that we use certain version of IsaacLab packages, make sure:
```
isaacsim <= 5.1.0.0 # tested on 5.1.0.0
isaaclab <= 0.53.1 # tested on 0.53.1
isaaclab-rl <= 0.4.7 # tested on 0.4.7
```

### 2. Modified Library Setup
This branch uses a customized version of `rsl_rl` and `robot_lab`. To install it, run the following commands in your terminal:

```bash
python -m pip install -e source/robot_lab
python -m pip install -e source/rsl_rl
```

## Try examples

You can use the following commands to run all environments:

RSL-RL:

```bash
# Train
python scripts/reinforcement_learning/rsl_rl/train.py --task=<ENV_NAME> --headless

# Play
python scripts/reinforcement_learning/rsl_rl/play.py --task=<ENV_NAME>
```
