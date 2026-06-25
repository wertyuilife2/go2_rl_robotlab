"""Symmetry observation and action augmentation for Go2."""

from __future__ import annotations

import math
from collections.abc import Generator

import torch
from tensordict import TensorDict


JOINT_PERM = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
JOINT_SIGN = [-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1]
FOOT_PERM = [1, 0, 3, 2]
VECTOR_SIGNS = {
    "base_lin_vel": [1, -1, 1],
    "base_ang_vel": [-1, 1, -1],
    "projected_gravity": [1, -1, 1],
    "velocity_commands": [1, -1, -1],
}
JOINT_TERMS = {"joint_pos", "joint_vel", "joint_acc", "joint_torque", "actions"}


def reverse_vector(value: torch.Tensor, signs: list[int]) -> torch.Tensor:
    return value * value.new_tensor(signs)


def reverse_joints(value: torch.Tensor) -> torch.Tensor:
    perm = torch.tensor(JOINT_PERM, device=value.device)
    signs = value.new_tensor(JOINT_SIGN)
    return value.index_select(-1, perm) * signs


def _base_env(env):
    return env.unwrapped if hasattr(env, "unwrapped") else env


def _restore_history(value: torch.Tensor, cfg) -> torch.Tensor:
    history_length = getattr(cfg, "history_length", 0)
    if history_length > 0 and getattr(cfg, "flatten_history_dim", False):
        term_dim = value.shape[-1] // history_length
        return value.reshape(*value.shape[:-1], history_length, term_dim)
    return value


def _flatten_history(value: torch.Tensor, cfg) -> torch.Tensor:
    history_length = getattr(cfg, "history_length", 0)
    if history_length > 0 and getattr(cfg, "flatten_history_dim", False):
        return value.reshape(*value.shape[:-2], -1)
    return value


def _reverse_feet(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-1] % len(FOOT_PERM) != 0:
        return value
    perm = torch.tensor(FOOT_PERM, device=value.device)
    return value.reshape(*value.shape[:-1], -1, len(FOOT_PERM)).index_select(-1, perm).reshape_as(value)


def _height_scan_shape(env, cfg) -> tuple[int, int, str] | None:
    sensor_cfg = cfg.params.get("sensor_cfg")
    scene_cfg = getattr(_base_env(env).cfg, "scene", None)
    sensor_scene_cfg = getattr(scene_cfg, sensor_cfg.name, None) if sensor_cfg is not None else None
    pattern_cfg = getattr(sensor_scene_cfg, "pattern_cfg", None)
    if pattern_cfg is None:
        return None

    nx = int(round(pattern_cfg.size[0] / pattern_cfg.resolution)) + 1
    ny = int(round(pattern_cfg.size[1] / pattern_cfg.resolution)) + 1
    return nx, ny, pattern_cfg.ordering


def _reverse_height_scan(env, cfg, value: torch.Tensor) -> torch.Tensor:
    shape = _height_scan_shape(env, cfg)
    if shape is None:
        return value

    nx, ny, ordering = shape
    if value.shape[-1] != nx * ny:
        return value

    if ordering == "xy":
        return value.reshape(*value.shape[:-1], ny, nx).flip(-2).reshape_as(value)
    return value.reshape(*value.shape[:-1], nx, ny).flip(-1).reshape_as(value)


def _reverse_term(env, name: str, cfg, value: torch.Tensor) -> torch.Tensor:
    if name in VECTOR_SIGNS:
        return reverse_vector(value, VECTOR_SIGNS[name])
    if name in JOINT_TERMS:
        return reverse_joints(value)
    if name == "contact_force":
        return _reverse_feet(value)
    if name == "height_scan":
        return _reverse_height_scan(env, cfg, value)
    return value


def _reverse_obs_group(env, group_name: str, value: torch.Tensor) -> torch.Tensor:
    manager = _base_env(env).observation_manager
    names = manager._group_obs_term_names[group_name]
    dims = manager._group_obs_term_dim[group_name]
    cfgs = manager._group_obs_term_cfgs[group_name]
    widths = [int(math.prod(dim)) for dim in dims]
    if sum(widths) != value.shape[-1]:
        return value

    chunks = torch.split(value, widths, dim=-1)
    reversed_chunks = []
    for name, cfg, chunk in zip(names, cfgs, chunks):
        term = _restore_history(chunk, cfg)
        term = _reverse_term(env, name, cfg, term)
        reversed_chunks.append(_flatten_history(term, cfg))
    return torch.cat(reversed_chunks, dim=-1)


def _reverse_obs(env, obs: TensorDict) -> TensorDict:
    manager = _base_env(env).observation_manager
    reversed_obs = obs.clone()
    for key, value in obs.items():
        if key in manager._group_obs_term_names:
            reversed_obs[key] = _reverse_obs_group(env, key, value)
    return reversed_obs


def _choose(mask: torch.Tensor, original, reversed_value):
    if isinstance(original, TensorDict):
        out = original.clone()
        for key, value in original.items():
            out[key] = _choose(mask, value, reversed_value[key])
        return out

    view_mask = mask.to(device=original.device).reshape(-1, *([1] * (original.ndim - 1)))
    return torch.where(view_mask, reversed_value, original)


def data_augmentation_generator(env, obs: TensorDict, actions: torch.Tensor) -> Generator:
    """Yield two symmetry-mixed batches with complementary random masks."""
    reversed_obs = _reverse_obs(env, obs)
    reversed_actions = reverse_joints(actions)

    mask = torch.rand(obs.batch_size[0], device=actions.device) < 0.5
    yield _choose(mask, obs, reversed_obs), _choose(mask, actions, reversed_actions)
    yield _choose(~mask, obs, reversed_obs), _choose(~mask, actions, reversed_actions)
