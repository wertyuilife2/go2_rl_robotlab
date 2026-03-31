"""Shared helpers for MuJoCo deployment scripts."""

from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import imageio
import numpy as np
import pygame
import torch
import yaml

from typing import Union, Literal
import mujoco
import mujoco.viewer

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).with_name("configs")


class CTSPolicyInputs(NamedTuple):
    policy: torch.Tensor
    single_obs: torch.Tensor


def load_config(config_name: str):
    with (CONFIG_DIR / config_name).open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    def path_value(name: str) -> Path:
        return Path(raw[name].replace("{ROOT_DIR}", str(ROOT_DIR)))

    data = {
        "policy_path": path_value("policy_path"),
        "xml_path": path_value("xml_path"),
        "duration": float(raw["simulation_duration"]),
        "dt": float(raw["simulation_dt"]),
        "decimation": int(raw["control_decimation"]),
        "history_len": int(raw.get("history_len", 1)),
        "num_actions": int(raw["num_actions"]),
        "num_obs": int(raw["num_obs"]),
        "delay_min": int(raw.get("actuator_delay_min", 0)),
        "delay_max": int(raw.get("actuator_delay_max", 0)),
        "delay_seed": int(raw.get("actuator_delay_seed", 0)),
        "save_video": bool(raw.get("save_video", False)),
        "render_fps": int(raw.get("render_fps", 60)),
        "video_fps": int(raw.get("video_fps", 50)),
    }
    if data["delay_min"] < 0 or data["delay_max"] < data["delay_min"]:
        raise ValueError(f"Invalid actuator delay range: min={data['delay_min']}, max={data['delay_max']}.")

    for name in (
        "kps",
        "kds",
        "default_angles",
        "torque_limit",
        "base_init_pos",
        "base_init_quat",
        "max_cmd",
        "cmd_init",
        "cmd_scale",
    ):
        if name in raw:
            data[name] = np.asarray(raw[name], dtype=np.float32)

    for name in (
        "lin_vel_scale",
        "ang_vel_scale",
        "dof_pos_scale",
        "dof_vel_scale",
        "action_pos_scale",
        "action_vel_scale",
    ):
        if name in raw:
            data[name] = float(raw[name])

    idx_model2mj = idx_mj2model = np.arange(data["num_actions"], dtype=np.int64)
    if "mujoco_joint_names" in raw and "model_joint_names" in raw:
        mj_names = raw["mujoco_joint_names"]
        model_names = raw["model_joint_names"]
        idx_model2mj = np.asarray([model_names.index(name) for name in mj_names], dtype=np.int64)
        idx_mj2model = np.asarray([mj_names.index(name) for name in model_names], dtype=np.int64)
    data["idx_model2mj"] = idx_model2mj
    data["idx_mj2model"] = idx_mj2model
    return SimpleNamespace(**data)


def gravity_from_quat(quaternion: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quaternion
    return np.array(
        [
            2 * (-qz * qx + qw * qy),
            -2 * (qz * qy + qw * qx),
            1 - 2 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def pd_control(target_q: np.ndarray, q: np.ndarray, kp: np.ndarray,
               target_dq: np.ndarray, dq: np.ndarray, kd: np.ndarray) -> np.ndarray:
    return (target_q - q) * kp + (target_dq - dq) * kd


def init_joystick():
    pygame.init()
    if pygame.joystick.get_count() == 0:
        print("No Joystick detected. Using default commands from config.")
        return None
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Detected Joystick: {joystick.get_name()}")
    return joystick


def read_joystick_command(joystick, max_cmd: np.ndarray) -> np.ndarray:
    pygame.event.pump()
    axes = np.array([joystick.get_axis(i) for i in (0, 1, 3)], dtype=np.float32)
    axes[np.abs(axes) < 0.1] = 0.0
    return np.array(
        [-axes[1] * max_cmd[0], -axes[0] * max_cmd[1], -axes[2] * max_cmd[2]],
        dtype=np.float32,
    )


def display_current_command(cmd: np.ndarray) -> None:
    print(f"\rCurrent command | vx: {cmd[0]: .3f}  vy: {cmd[1]: .3f}  wz: {cmd[2]: .3f}", end="", flush=True)


def set_initial_state(data, base_pos: np.ndarray, base_quat: np.ndarray, joint_pos: np.ndarray) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[:3] = base_pos
    data.qpos[3:7] = base_quat
    data.qpos[7:] = joint_pos


def setup_tracking_camera(viewer, *, trackbodyid: int = 1, distance: float = 3.0,
                          elevation: float = -30.0, azimuth: float = 0.0) -> None:

    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = trackbodyid
    viewer.cam.distance = distance
    viewer.cam.elevation = elevation
    viewer.cam.azimuth = azimuth


def open_video_writer(enabled: bool, *, policy_path: Path, cmd: np.ndarray,
                      dt: float, video_dir: Path, video_fps: int):
    if not enabled:
        return None, 1, None
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"{policy_path.stem}_cmd_{cmd[0]}_{cmd[1]}_{cmd[2]}.mp4"
    frame_skip = max(1, int((1.0 / dt) / video_fps))
    writer = imageio.get_writer(video_path, fps=video_fps)
    print(f"Recording: {video_path} (Sim FPS: {1.0 / dt:.1f}, Video FPS: {video_fps})")
    return writer, frame_skip, video_path


def push_obs_history(obs: np.ndarray, features: dict, layout: list[tuple[str, int]], history_len: int) -> None:
    offset = 0
    for name, dim in layout:
        block = obs[offset:offset + dim * history_len]
        block[:-dim] = block[dim:]
        block[-dim:] = features[name]
        offset += dim * history_len


def latest_obs_frame(obs: np.ndarray, layout: list[tuple[str, int]], history_len: int) -> np.ndarray:
    frames = []
    offset = 0
    for _, dim in layout:
        end = offset + dim * history_len
        frames.append(obs[end - dim:end])
        offset = end
    return np.concatenate(frames, axis=0)


def infer_action(policy, obs: np.ndarray, single_obs: np.ndarray, idx_model2mj: np.ndarray) -> np.ndarray:
    result = policy(
        CTSPolicyInputs(
            policy=torch.from_numpy(obs).unsqueeze(0),
            single_obs=torch.from_numpy(single_obs).unsqueeze(0),
        )
    )
    return result.detach().cpu().numpy().squeeze()[idx_model2mj]


def build_delay_buffers(*defaults: np.ndarray, delay_max: int):
    size = delay_max + 1
    buffers = tuple(deque((value.copy() for _ in range(size)), maxlen=size) for value in defaults)
    return buffers[0] if len(buffers) == 1 else buffers


def sample_delayed_targets(histories, actuator_groups, delay_min: int, delay_max: int, rng: np.random.Generator):
    if isinstance(histories, deque):
        histories = (histories,)
    delayed = [history[-1].copy() for history in histories]
    for joint_ids in actuator_groups:
        delay_steps = int(rng.integers(delay_min, delay_max + 1)) if delay_max > 0 else 0
        for i, history in enumerate(histories):
            delayed[i][joint_ids] = history[-1 - delay_steps][joint_ids]
    return delayed[0] if len(delayed) == 1 else tuple(delayed)


class MujocoRenderUtils:
    def __init__(self):
        self.target_velocity = None

        self.vis_smooth_factor = 1.0
        self.ren_smooth_factor = 1.0

        self.vis_cur_vel = np.zeros(3)
        self.ren_cur_vel = np.zeros(3)

        self.mj_data = None
        self._renderer_capacity_warned = False
        self._viewer_capacity_warned = False
    
    def update(self, target_velocity, mj_data):
        self.target_velocity = target_velocity
        self.mj_data = mj_data
        
    def update_external_rendering(self,
            handle: Union[mujoco.viewer.Handle, mujoco.Renderer],
            ctype: Literal['viewer', 'renderer'],
        ):
        """ Update external rendering handle (viewer or renderer). """

        def has_geom_slot(container, index: int, warned_attr: str) -> bool:
            capacity = len(container.geoms)
            if index < capacity:
                return True
            if not getattr(self, warned_attr):
                print(f"Warning: {ctype} geom capacity exceeded ({capacity}); skipping velocity arrows.")
                setattr(self, warned_attr, True)
            return False

        def add_thick_arrow(geom_elem, pos, vec, rgba, scale=0.7):
            vel_norm = np.linalg.norm(vec)
            display_norm = min(vel_norm * scale, 1.0)

            if display_norm < 0.10:
                mujoco.mjv_initGeom(
                    geom_elem,
                    type=mujoco.mjtGeom.mjGEOM_NONE,
                    size=[0,0,0], pos=pos, mat=np.eye(3).flatten(), rgba=[0,0,0,0]
                )
                return

            mat = np.zeros(9)
            target_quat = np.zeros(4)
            vec_normalized = vec / vel_norm
            mujoco.mju_quatZ2Vec(target_quat, vec_normalized)
            mujoco.mju_quat2Mat(mat, target_quat)
            
            mat = mat.reshape(3, 3)
            mat[:, 2] *= display_norm 
            
            mujoco.mjv_initGeom(
                geom_elem,
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=[0.02, 0.02, display_norm], # [height, width, length]
                pos=pos,
                mat=mat.flatten(),
                rgba=rgba
            )

        viewer_geom_idx = 0
        if ctype == 'viewer':
            handle.user_scn.ngeom = 0  # reset user scene geometry
        
        if self.target_velocity is not None:
            base_pos_world = self.mj_data.qpos[:3]
            base_quat = self.mj_data.qpos[3:7]
            
            # rendering arrows start position
            offset_body = np.array([0.0, 0.0, 0.2])
            offset_world = np.zeros(3)
            mujoco.mju_rotVecQuat(offset_world, offset_body, base_quat)
            start_pos = base_pos_world + offset_world

            tgt_vel_body = np.array([self.target_velocity[0], self.target_velocity[1], 0.0])
            
            raw_cur_vel_world = self.mj_data.qvel[:3]
            raw_cur_vel = np.zeros(3)
            neg_quat = np.zeros(4)
            mujoco.mju_negQuat(neg_quat, base_quat)
            mujoco.mju_rotVecQuat(raw_cur_vel, raw_cur_vel_world, neg_quat)
            cur_vel_body = np.array([raw_cur_vel[0], raw_cur_vel[1], 0.0])

            # EMA: v_smooth = alpha * v_new + (1 - alpha) * v_old
            # alpha = self.vis_smooth_factor if ctype == 'viewer' else self.ren_smooth_factor
            self.vis_cur_vel = cur_vel_body
            self.ren_cur_vel = cur_vel_body

            tgt_vel_world = np.zeros(3)
            cur_vel_world = np.zeros(3)
            mujoco.mju_rotVecQuat(tgt_vel_world, tgt_vel_body, base_quat)
            if ctype == 'viewer':
                mujoco.mju_rotVecQuat(cur_vel_world, self.vis_cur_vel, base_quat)
            else:
                mujoco.mju_rotVecQuat(cur_vel_world, self.ren_cur_vel, base_quat)

            COLOR_CMD = [0, 1, 0, 1]   # Green 0x00ff00
            COLOR_REAL = [0, 0, 1, 1]  # Blue  0x0000ff

            if ctype == 'viewer':
                # Cmd Arrow
                if has_geom_slot(handle.user_scn, viewer_geom_idx, "_viewer_capacity_warned"):
                    add_thick_arrow(handle.user_scn.geoms[viewer_geom_idx], start_pos, tgt_vel_world, COLOR_CMD)
                    viewer_geom_idx += 1
                # Real Arrow
                if has_geom_slot(handle.user_scn, viewer_geom_idx, "_viewer_capacity_warned"):
                    add_thick_arrow(handle.user_scn.geoms[viewer_geom_idx], start_pos, cur_vel_world, COLOR_REAL)
                    viewer_geom_idx += 1
            else:
                scene = handle.scene
                if has_geom_slot(scene, scene.ngeom, "_renderer_capacity_warned"):
                    add_thick_arrow(scene.geoms[scene.ngeom], start_pos, tgt_vel_world, COLOR_CMD)
                    scene.ngeom += 1
                if has_geom_slot(scene, scene.ngeom, "_renderer_capacity_warned"):
                    add_thick_arrow(scene.geoms[scene.ngeom], start_pos, cur_vel_world, COLOR_REAL)
                    scene.ngeom += 1

        if ctype == 'viewer':
            handle.user_scn.ngeom = viewer_geom_idx
