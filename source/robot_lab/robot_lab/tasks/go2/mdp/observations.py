# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor

def joint_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]

def foot_contact_force_norm(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history # [B, T_hist, num_bodies, 3]
    
    contact_force_norm = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1) # [B, T_hist, num_legs]
    max_contact_force_norm, _ = torch.max(contact_force_norm, dim=1)  # [B, num_legs]
    contact_force_norm = torch.concat([max_contact_force_norm.unsqueeze(1), contact_force_norm], dim=1)  # [B, T_hist+1, num_legs]
    
    return contact_force_norm.flatten(start_dim=-2)  # [B, (T_hist+1)*num_legs]


def depth_image(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    data_type: str = "distance_to_image_plane",
    decimation: int = 8,
    crop_size: tuple[int, int] = (60, 60),
    max_depth: float = 10.0,
    normalize: bool = True,
    ignore_zero: bool = False,
    enable_augmentation: bool = False,
    augmentation_cfg_group: str | None = None,
    noise_std: float = 0.0,
    dropout_prob: float = 0.0,
    scale_range: tuple[float, float] = (1.0, 1.0),
) -> torch.Tensor:
    """Downsample a depth image by block decimation and return a flat center crop.

    For the D435i 640x480 stream, decimation=8 gives 80x60. The returned
    observation is the center 60x60 crop flattened to [num_envs, 3600].
    """
    camera = env.scene.sensors[sensor_cfg.name]
    depth = camera.data.output[data_type].float()

    if depth.ndim == 4:
        if depth.shape[-1] == 1:
            depth = depth.squeeze(-1)
        elif depth.shape[1] == 1:
            depth = depth.squeeze(1)
    if depth.ndim != 3:
        raise ValueError(f"Expected depth image with shape [B,H,W] or [B,H,W,1], got {tuple(depth.shape)}")

    depth = torch.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=0.0).clamp_(0.0, max_depth)
    num_envs, height, width = depth.shape
    decimated_height = (height // decimation) * decimation
    decimated_width = (width // decimation) * decimation
    crop_height, crop_width = crop_size

    if decimated_height // decimation < crop_height or decimated_width // decimation < crop_width:
        raise ValueError(
            f"Decimated depth image {(decimated_height // decimation, decimated_width // decimation)} "
            f"is smaller than requested crop {crop_size}."
        )

    depth = depth[:, :decimated_height, :decimated_width]
    if ignore_zero:
        valid = depth > 0.0
        depth_blocks = depth.masked_fill(~valid, 0.0).reshape(
            num_envs,
            decimated_height // decimation,
            decimation,
            decimated_width // decimation,
            decimation,
        )
        valid_blocks = valid.reshape(
            num_envs,
            decimated_height // decimation,
            decimation,
            decimated_width // decimation,
            decimation,
        )
        valid_counts = valid_blocks.sum(dim=(2, 4))
        depth = depth_blocks.sum(dim=(2, 4)) / valid_counts.clamp_min(1)
        depth = torch.where(valid_counts > 0, depth, depth.new_full((), max_depth))
    else:
        depth = depth.reshape(
            num_envs,
            decimated_height // decimation,
            decimation,
            decimated_width // decimation,
            decimation,
        ).mean(dim=(2, 4))

    crop_top = (depth.shape[1] - crop_height) // 2
    crop_left = (depth.shape[2] - crop_width) // 2
    depth = depth[:, crop_top : crop_top + crop_height, crop_left : crop_left + crop_width]

    should_augment = enable_augmentation
    if augmentation_cfg_group is not None:
        observations_cfg = getattr(env.cfg, "observations", None)
        group_cfg = getattr(observations_cfg, augmentation_cfg_group, None)
        should_augment = should_augment and bool(getattr(group_cfg, "enable_corruption", False))

    if should_augment:
        if scale_range[0] != scale_range[1]:
            scale = torch.empty((num_envs, 1, 1), device=depth.device, dtype=depth.dtype).uniform_(*scale_range)
            depth = depth * scale
        if noise_std > 0.0:
            depth = depth + torch.randn_like(depth) * noise_std
        if dropout_prob > 0.0:
            depth = depth.masked_fill(torch.rand_like(depth) < dropout_prob, max_depth)
        depth = depth.clamp_(0.0, max_depth)

    if normalize:
        depth = depth / max_depth

    return depth.flatten(1)
