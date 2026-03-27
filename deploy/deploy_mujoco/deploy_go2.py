"""CTS Policy deployment for Unitree Go2 in MuJoCo."""

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import imageio
import mujoco
import mujoco.viewer
import numpy as np
import pygame
import torch
import yaml
from argparse import ArgumentParser


# ============================================================================
# Types & Constants
# ============================================================================

ROOT_DIR = str(Path(__file__).parent.parent.parent)
CONFIG_DIR = f"{ROOT_DIR}/deploy/deploy_mujoco/configs"
VIDEO_DIR = Path(__file__).parent / "videos"

class CTSPolicyInputs(NamedTuple):
    """Input format for CTS policy."""
    policy: torch.Tensor
    single_obs: torch.Tensor


@dataclass
class ObsBlockCfg:
    """Configuration for observation block."""
    name: str
    dim: int  # Single frame dimension


ACTUATOR_GROUPS = {
    "hip": [0, 3, 6, 9],
    "thigh": [1, 4, 7, 10],
    "calf": [2, 5, 8, 11],
}


# ============================================================================
# Helper Functions
# ============================================================================

def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    """Compute gravity vector in body frame from quaternion."""
    qw, qx, qy, qz = quaternion

    gravity = np.zeros(3)
    gravity[0] = 2 * (-qz * qx + qw * qy)
    gravity[1] = -2 * (qz * qy + qw * qx)
    gravity[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity


def pd_control(target_q: np.ndarray, q: np.ndarray, kp: np.ndarray,
               target_dq: np.ndarray, dq: np.ndarray, kd: np.ndarray) -> np.ndarray:
    """Compute PD control torques."""
    return (target_q - q) * kp + (target_dq - dq) * kd


def get_joystick_command(joystick, max_cmd: np.ndarray) -> np.ndarray:
    """Read command from Xbox controller."""
    pygame.event.pump()

    dead_zone = 0.1
    axes = [joystick.get_axis(i) for i in [0, 1, 3]]  # LX, LY, RX
    axes = [0 if abs(a) < dead_zone else a for a in axes]

    cmd = np.array([-axes[1] * max_cmd[0], -axes[0] * max_cmd[1], -axes[2] * max_cmd[2]], dtype=np.float32)
    return cmd


def load_config(config_file: str) -> dict:
    """Load and parse YAML configuration."""
    with open(f"{CONFIG_DIR}/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Replace path placeholders
    config["policy_path"] = config["policy_path"].replace("{ROOT_DIR}", ROOT_DIR)
    config["xml_path"] = config["xml_path"].replace("{ROOT_DIR}", ROOT_DIR)

    return config


def build_observation(obs: np.ndarray, features: dict, obs_cfg: list, history_len: int) -> None:
    """Update stacked observation buffer with new frame features (in-place)."""
    ptr = 0
    for cfg in obs_cfg:
        dim = cfg.dim
        start, end = ptr, ptr + dim * history_len

        # Roll history and insert new frame
        obs[start:end] = np.roll(obs[start:end], shift=-dim, axis=0)
        obs[end - dim:end] = features[cfg.name]

        ptr = end


def extract_single_obs(obs: np.ndarray, obs_cfg: list, history_len: int) -> np.ndarray:
    """Extract most recent single-frame observation from stacked buffer."""
    single = []
    ptr = 0
    for cfg in obs_cfg:
        dim = cfg.dim
        block = obs[ptr:ptr + dim * history_len]
        single.append(block[-dim:])  # Get last dim elements (most recent frame)
        ptr += dim * history_len
    return np.concatenate(single, axis=0)


def apply_action(action: np.ndarray, default_angles: np.ndarray,
                 action_pos_scale: float) -> np.ndarray:
    """Transform policy action to position targets."""
    return action * action_pos_scale + default_angles


def set_initial_state(data: mujoco.MjData, base_pos: np.ndarray, base_quat: np.ndarray,
                      joint_pos: np.ndarray) -> None:
    """Apply the Isaac-style initial pose before the first mj_forward."""
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[:3] = base_pos
    data.qpos[3:7] = base_quat
    data.qpos[7:] = joint_pos


def build_delay_buffers(default_pos: np.ndarray, delay_max: int) -> deque:
    """Create history buffers for delayed control targets."""
    history_len = delay_max + 1
    pos_history = deque((default_pos.copy() for _ in range(history_len)), maxlen=history_len)
    return pos_history


def sample_delayed_targets(pos_history: deque, delay_min: int,
                           delay_max: int, rng: np.random.Generator) -> np.ndarray:
    """Apply per-actuator-group discrete control delays."""
    delayed_pos = pos_history[-1].copy()

    for joint_ids in ACTUATOR_GROUPS.values():
        delay_steps = int(rng.integers(delay_min, delay_max + 1)) if delay_max > 0 else 0
        pos_source = pos_history[-1 - delay_steps]
        delayed_pos[joint_ids] = pos_source[joint_ids]

    return delayed_pos


def init_joystick() -> tuple:
    """Initialize pygame and joystick if available."""
    pygame.init()

    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"Detected Joystick: {joystick.get_name()}")
        return joystick, True

    print("No Joystick detected. Using default commands from config.")
    return None, False


def setup_viewer_camera(viewer) -> None:
    """Configure tracking camera."""
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = 1
    viewer.cam.distance = 3.0
    viewer.cam.elevation = -30.0
    viewer.cam.azimuth = 0.0

def display_current_command(cmd: np.ndarray) -> None:
    """Refresh the current command on the same terminal line."""
    cmd_text = f"\rCurrent command | vx: {cmd[0]: .3f}  vy: {cmd[1]: .3f}  wz: {cmd[2]: .3f}"
    print(cmd_text, end="", flush=True)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = ArgumentParser()
    parser.add_argument("--save-video", action="store_true", help="Save video of simulation.")
    args = parser.parse_args()

    # Configuration
    config_file = "go2.yaml"
    render_fps = 240

    config = load_config(config_file)

    # Extract config parameters
    sim_cfg = {
        "duration": config["simulation_duration"],
        "dt": config["simulation_dt"],
        "decimation": config["control_decimation"],
    }

    # PD controller gains
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    base_init_pos = np.array(config["base_init_pos"], dtype=np.float32)
    base_init_quat = np.array(config["base_init_quat"], dtype=np.float32)
    delay_min = int(config.get("actuator_delay_min", 0))
    delay_max = int(config.get("actuator_delay_max", 0))
    delay_seed = int(config.get("actuator_delay_seed", 0))
    if delay_min < 0 or delay_max < delay_min:
        raise ValueError(
            f"Invalid actuator delay range: min={delay_min}, max={delay_max}."
        )
    delay_rng = np.random.default_rng(delay_seed)

    # Scaling factors
    scales = {
        "lin_vel": config["lin_vel_scale"],
        "ang_vel": config["ang_vel_scale"],
        "dof_pos": config["dof_pos_scale"],
        "dof_vel": config["dof_vel_scale"],
        "action_pos": config["action_pos_scale"],
        "cmd": np.array(config["cmd_scale"], dtype=np.float32),
    }

    # Policy dimensions
    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    history_len = config.get("history_len", 1)
    max_cmd = np.array(config["max_cmd"], dtype=np.float32)
    cmd = np.array(config["cmd_init"], dtype=np.float32)

    # Joint name mapping
    idx_model2mj = idx_mj2model = list(range(num_actions))
    if "mujoco_joint_names" in config and "model_joint_names" in config:
        mj_names = config["mujoco_joint_names"]
        model_names = config["model_joint_names"]
        idx_model2mj = [model_names.index(j) for j in mj_names]
        idx_mj2model = [mj_names.index(j) for j in model_names]

    # Initialize joystick
    joystick, use_joystick = init_joystick()
    display_current_command(cmd)


    # Prepare video output
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    model_name = Path(config["policy_path"]).stem
    cmd_str = f"cmd_{cmd[0]}_{cmd[1]}_{cmd[2]}"

    # Initialize state
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    target_dof_vel = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs * history_len, dtype=np.float32)
    pos_history = build_delay_buffers(target_dof_pos, delay_max)

    # Build observation config dynamically
    num_joints = num_actions
    obs_cfg = [
        ObsBlockCfg("ang_vel", 3),
        ObsBlockCfg("gravity", 3),
        ObsBlockCfg("cmd", 3),
        ObsBlockCfg("joint_pos", num_joints),
        ObsBlockCfg("joint_vel", num_actions),
        ObsBlockCfg("last_action", num_actions),
    ]

    # Load MuJoCo model
    m = mujoco.MjModel.from_xml_path(config["xml_path"])
    d = mujoco.MjData(m)
    m.opt.timestep = sim_cfg["dt"]
    set_initial_state(d, base_init_pos, base_init_quat, default_angles)
    mujoco.mj_forward(m, d)

    renderer = mujoco.Renderer(m, height=360, width=640)

    # Load policy
    policy = torch.jit.load(config["policy_path"])

    # Setup video recording
    writer = None
    if args.save_video:
        video_path = VIDEO_DIR / f"{model_name}_{cmd_str}.mp4"
        video_fps = 50
        sim_fps = 1.0 / sim_cfg["dt"]
        frame_skip = max(1, int(sim_fps / video_fps))
        writer = imageio.get_writer(video_path, fps=video_fps)
        print(f"Recording: {video_path} (Sim FPS: {sim_fps:.1f}, Video FPS: {video_fps})")

    render_substeps = int((1.0 / render_fps) / sim_cfg["dt"])

    # Run simulation
    with mujoco.viewer.launch_passive(m, d) as viewer:
        setup_viewer_camera(viewer)

        start_time = time.time()
        counter = 0

        while viewer.is_running() and time.time() - start_time < sim_cfg["duration"]:
            step_start = time.time()

            # Update command from joystick
            if use_joystick and counter % sim_cfg["decimation"] == 0:
                cmd = get_joystick_command(joystick, max_cmd)

            # Compute and apply control
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, target_dof_vel, d.qvel[6:], kds)
            d.ctrl[:] = tau

            mujoco.mj_step(m, d)

            # Record frame
            if writer and counter % frame_skip == 0:
                try:
                    renderer.update_scene(d, camera=viewer.cam)
                    writer.append_data(renderer.render())
                except Exception as e:
                    print(f"Render error: {e}")

            counter += 1

            # Policy update at control frequency
            if counter % sim_cfg["decimation"] == 0:
                # Extract sensor data
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                ang_vel = d.qvel[3:6]

                # Scale observations
                qj = (qj - default_angles) * scales["dof_pos"]
                dqj = dqj * scales["dof_vel"]
                ang_vel = ang_vel * scales["ang_vel"]

                gravity = get_gravity_orientation(quat)

                # Build observation features
                features = {
                    "ang_vel": ang_vel,
                    "gravity": gravity,
                    "cmd": cmd * scales["cmd"],
                    "joint_pos": qj[idx_mj2model],
                    "joint_vel": dqj[idx_mj2model],
                    "last_action": action[idx_mj2model],
                }

                # Update stacked observation
                build_observation(obs, features, obs_cfg, history_len)

                # Extract single-frame observation
                single_obs = extract_single_obs(obs, obs_cfg, history_len)

                # Run policy
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                single_tensor = torch.from_numpy(single_obs).unsqueeze(0)

                result = policy(CTSPolicyInputs(policy=obs_tensor, single_obs=single_tensor))
                action = result.detach().cpu().numpy().squeeze()[idx_model2mj]

                # Apply action
                latest_target_pos = apply_action(action, default_angles, scales["action_pos"])
                pos_history.append(latest_target_pos.copy())
                target_dof_pos = sample_delayed_targets(pos_history, delay_min, delay_max, delay_rng)
                display_current_command(cmd)
                
            # Sync viewer
            if counter % render_substeps == 0:
                viewer.sync()

            # Time management
            sleep_time = sim_cfg["dt"] - (time.time() - step_start) - 0.1
            if sleep_time > 0:
                time.sleep(sleep_time)
    print()

    # Cleanup
    if writer:
        print(f"Video saved: {video_path}")
        writer.close()


if __name__ == "__main__":
    main()
