"""Run the Go2 D435i depth camera demo with the exported MoE-CTS policy.

Example:
    python scripts/tests/test_go2_camera.py --headless
"""

from __future__ import annotations

import argparse
import sys
import time
import math
from pathlib import Path

import h5py  # noqa: F401
import tensordict  # noqa: F401
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
for path in (ROOT_DIR / "source" / "robot_lab", ROOT_DIR / "source" / "rsl_rl"):
    sys.path.insert(0, str(path))

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo Go2 D435i depth camera reset with a TorchScript policy.")
    parser.add_argument("--task", type=str, default="RobotLab-Go2-D435i-v0", help="Gym task name.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT_DIR / "deploy/pre_train/go2/go2_moe_cts_185k_0.6828.pt"),
        help="Path to the exported TorchScript checkpoint.",
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. The exported policy supports 1.")
    parser.add_argument("--steps", type=int, default=-1, help="Number of policy steps to run, -1 for infinite.")
    parser.add_argument("--reset_interval", type=int, default=-1, help="Reset all envs and refresh depth every N steps.")
    parser.add_argument("--save_depth", action="store_true", help="Save depth tensors after each reset.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT_DIR / "logs/camera_tests"),
        help="Directory for optional saved depth tensors.",
    )
    parser.add_argument("--real-time", action="store_true", default=False, help="Throttle to the environment step time.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    return args_cli


def depth_stats(depth: torch.Tensor) -> str:
    valid = torch.isfinite(depth) & (depth > 0.0)
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        return f"shape={tuple(depth.shape)} valid=0"
    valid_depth = depth[valid]
    return (
        f"shape={tuple(depth.shape)} valid={valid_count} "
        f"min={valid_depth.min().item():.3f}m "
        f"mean={valid_depth.mean().item():.3f}m "
        f"max={valid_depth.max().item():.3f}m"
    )


def normalized_depth_obs_stats(depth_obs: torch.Tensor) -> str:
    valid = torch.isfinite(depth_obs)
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        return f"shape={tuple(depth_obs.shape)} valid=0"
    valid_depth = depth_obs[valid]
    return (
        f"shape={tuple(depth_obs.shape)} valid={valid_count} "
        f"min={valid_depth.min().item():.3f} "
        f"mean={valid_depth.mean().item():.3f} "
        f"max={valid_depth.max().item():.3f}"
    )


def refresh_depth(env, reset_index: int, output_dir: Path | None = None) -> torch.Tensor:
    from isaaclab.managers import SceneEntityCfg

    from robot_lab.sensors import depth_image

    camera = env.unwrapped.scene["front_depth_camera"]
    depth = camera.data.output["distance_to_image_plane"].detach().clone()
    depth_obs = depth_image(
        env.unwrapped,
        sensor_cfg=SceneEntityCfg("front_depth_camera"),
        use_delay=True,
        enable_noise=False,
    ).detach().clone()
    print(f"[depth reset {reset_index:03d}] {depth_stats(depth)}")
    print(f"[depth obs reset {reset_index:03d}] {normalized_depth_obs_stats(depth_obs)}")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(depth.cpu(), output_dir / f"depth_reset_{reset_index:03d}.pt")
        torch.save(depth_obs.cpu(), output_dir / f"depth_obs_reset_{reset_index:03d}.pt")
    return depth


def run(args_cli: argparse.Namespace) -> None:
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    import robot_lab.tasks  # noqa: F401

    def main() -> None:
        if args_cli.num_envs != 1:
            raise ValueError("The exported TorchScript CTS policy in deploy/pre_train/go2 supports batch size 1 only.")

        env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.observations.policy.enable_corruption = False
        if hasattr(env_cfg.observations, "depth"):
            env_cfg.observations.depth.enable_corruption = False
        if hasattr(env_cfg.scene, "front_depth_camera") and env_cfg.scene.front_depth_camera is not None:
            env_cfg.scene.front_depth_camera.debug_vis = True
        print(f"[INFO] Creating environment: {args_cli.task} ({args_cli.num_envs} env)")

        if hasattr(env_cfg.events, "randomize_push_robot"):
            env_cfg.events.randomize_push_robot = None

        env = gym.make(args_cli.task, cfg=env_cfg)
        print("[INFO] Environment created.")
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
        policy = torch.jit.load(str(checkpoint), map_location=env.unwrapped.device).eval()
        policy.reset()
        print(f"[INFO] Loaded TorchScript policy: {checkpoint}")
        camera = env.unwrapped.scene["front_depth_camera"]
        camera.set_debug_vis(True)
        print(f"[INFO] Camera prim: {camera.cfg.prim_path}")
        print("[INFO] Ray-caster camera debug visualization enabled.")

        output_dir = Path(args_cli.output_dir).expanduser().resolve() if args_cli.save_depth else None
        obs, _ = env.reset()
        reset_index = 0
        refresh_depth(env, reset_index, output_dir)

        dt = env.unwrapped.step_dt
        try:
            step = 1
            while step < (math.inf if args_cli.steps < 0 else args_cli.steps):
                start_time = time.time()
                with torch.inference_mode():
                    actions = policy(obs["single_obs"].to(env.unwrapped.device))
                actions = actions.detach().clone()
                obs, _, terminated, truncated, _ = env.step(actions)
                dones = terminated | truncated
                if bool(dones.any()):
                    policy.reset()

                if args_cli.reset_interval > 0 and step % args_cli.reset_interval == 0:
                    obs, _ = env.reset()
                    policy.reset()
                    reset_index += 1
                    refresh_depth(env, reset_index, output_dir)

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)
                step += 1
        finally:
            env.close()

    try:
        main()
    except Exception as exc:
        print(f"[ERROR] Demo failed: {type(exc).__name__}: {exc}")
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    args_cli = parse_args()
    run(args_cli)
