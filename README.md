# go2_rl_robotlab

## Overview


Trains the Unitree Go2 robot using MoE-CTS in IsaacLab and deploys the trained policy to MuJoCo for simulation transfer (Sim2Sim).

It is a reproduction of [go2_rl_gym](https://github.com/wty-yy/go2_rl_gym), adapted to the RobotLab / IsaacLab ecosystem.

---

<p align="center">
  <img src="resources/go2/isaaclab_scene.png" width="70%"/>
</p>

---

## Installation Guide

### 1. Install IsaacLab

Install IsaacLab 2.3.0 release by following the [installation guide](https://isaac-sim.github.io/IsaacLab/release/2.3.0/source/setup/installation/pip_installation.html#). Please make sure to clone the IsaacLab repository and perform the installation using the release/2.3.0 branch.

After installation, your environment should satisfies:

```bash
isaacsim <= 5.1.0.0   # tested on 5.1.0.0
isaaclab <= 0.54.3    # tested on 0.54.3, 0.53.1
isaaclab-rl <= 0.4.7  # tested on 0.4.7, 0.4.4
```

Higher version may cause conflicts with our customized `rsl_rl==3.3.0` and `robot_lab==2.3.0`.

---

### 2. Install Customized RSL-RL and RobotLab

We uses a customized version of `rsl_rl` and `robot_lab`. Install them in editable mode:

```bash
python -m pip install -e source/robot_lab
python -m pip install -e source/rsl_rl
```

---

### 3. Install MuJoCo (Optional, for Sim2Sim)

To enable MuJoCo-based simulation:

```bash
pip install mujoco  # tested on 3.4.0 and 3.6.0
```

---

## Training and Evaluation

Run the following commands:

```bash
# Train
python scripts/rsl_rl/train.py --task=RobotLab-Go2-v0 --headless

# Evaluate
python scripts/rsl_rl/play.py --task=RobotLab-Go2-v0
```

---

## Configuration

The training pipeline can be configured at two levels: task-level Python configuration files and runtime arguments passed to the training script.

### Task-Level Configuration

The default task settings are defined in the following files:

1. **Environment configuration**
   ```
   source/robot_lab/robot_lab/tasks/go2/env_cfg.py
   ```

2. **RL algorithm configuration**
   ```
   source/robot_lab/robot_lab/tasks/go2/rsl_rl_cfg.py
   ```

3. **Task registration**
   ```
   source/robot_lab/robot_lab/tasks/go2/__init__.py
   ```

### Runtime Overrides

In addition to the default configuration files, `train.py` and `play.py` supports several command-line arguments for runtime overrides:

```bash
python scripts/rsl_rl/train.py \
   --task=RobotLab-Go2-v0 \
   --headless \
   --experiment_name <YOUR_EXP_NAME> \
   --run_name <YOUR_RUN_NAME> \
   --num_envs <NUM_ENVS> \
   --checkpoint <PATH_TO_CHECKPOINT>
```

For more details, refer to the [robot_lab repo](https://github.com/fan-ziqi/robot_lab.git).

---

## MuJoCo Sim2Sim

Set the `policy_path` in `deploy/deploy_mujoco/configs/go2.yaml`:

```yaml
policy_path: "{ROOT_DIR}/deploy/pre_train/go2/xxx.pt" # policy.pt exported by running scripts/rsl_rl/play.py
```

Run the deployment script:

```bash
python deploy/deploy_mujoco/deploy_go2.py
```

### Controller Behavior

- **Automatic detection**: If a controller is connected, control mode is enabled automatically.
- **Fallback mode**: If no controller is detected, default commands from the config file are used.

### Controller Mapping

| Input | Function |
|------|--------|
| `LX / LY` | Forward / lateral velocity |
| `RX` | Angular velocity (steering) |

---

### Switching Simulation Scenarios

Modify `xml_path` in `deploy/deploy_mujoco/config/go2.yaml`:

```yaml
# Flat terrain
xml_path: "{ROOT_DIR}/resources/go2/flat.xml"

# Stairs
xml_path: "{ROOT_DIR}/resources/go2/stairs.xml"

# Boxes
xml_path: "{ROOT_DIR}/resources/go2/boxes.xml"

# Custom
xml_path: "{ROOT_DIR}/resources/go2/your-custom-scene.xml"
```

---

## Differences from `go2_rl_gym`

- Different tracking reward formulation (fixed sigma vs. dynamic sigma)
- Different reward weights (e.g., lower dof_acc_l2 weight in Lab due to physics-step level implementation and sensitivity to outliers)
- Lack domain_rand: randomize_motor_strength

---

## TODO

- Try replacing ActionManager-level delay with `DelayedPDActuatorCfg` or `UnitreeActuatorCfg_Go2HV`, and make sure the randomization of motor parameters  works correctly.

---

## Acknowledgements
This repository would not exist without the following open-source projects:

- [isaac_lab](https://github.com/isaac-sim/IsaacLab): Unified framework for robot learning built on NVIDIA Isaac Sim.
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git): Reinforcement learning algorithms.
- [robot_lab](https://github.com/fan-ziqi/robot_lab.git): RL Extension Library for Robots, Based on IsaacLab.
- [mujoco](https://github.com/google-deepmind/mujoco.git): High-performance CPU physics simulator.

Related publications implemented in this repo:
- [CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion](https://arxiv.org/pdf/2405.10830)
