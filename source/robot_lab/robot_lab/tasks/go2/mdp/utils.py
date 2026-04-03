# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for terrain-aware operations."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _get_terrain_column_range(terrain_cfg, terrain_name: str, device) -> tuple[int, int] | None:
    """Helper function to calculate column range for a terrain type.

    Args:
        terrain_cfg: The terrain generator configuration.
        terrain_name: Name of the terrain.
        device: Torch device.

    Returns:
        Tuple of (col_start, col_end) or None if terrain not found.
    """
    if terrain_cfg.sub_terrains is None or terrain_name not in terrain_cfg.sub_terrains:
        return None

    sub_terrain_names = list(terrain_cfg.sub_terrains.keys())
    proportions = torch.tensor([sub_cfg.proportion for sub_cfg in terrain_cfg.sub_terrains.values()], device=device)
    proportions = proportions / proportions.sum()
    cumsum_props = torch.cumsum(proportions, dim=0)

    terrain_idx = sub_terrain_names.index(terrain_name)
    # Use round() instead of int() to properly allocate columns
    col_start = round((0.0 if terrain_idx == 0 else cumsum_props[terrain_idx - 1].item()) * terrain_cfg.num_cols)
    col_end = round(cumsum_props[terrain_idx].item() * terrain_cfg.num_cols)

    return (col_start, col_end)


def is_env_assigned_to_terrain(env: ManagerBasedEnv, terrain_name: str) -> torch.Tensor:
    """Check which environments are initially assigned to the specified terrain type.

    Each environment is assigned to a specific terrain cell at initialization.
    This function returns a mask indicating which environments were assigned to the given terrain type.

    Args:
        env: The environment instance.
        terrain_name: Name of the terrain to check (e.g., "pits", "stairs").

    Returns:
        Boolean tensor of shape (num_envs,) where True means the environment is assigned to this terrain.
    """
    # Check if terrain and terrain generator are available
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or not hasattr(terrain, "terrain_types"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if terrain.cfg.terrain_type != "generator" or terrain.cfg.terrain_generator is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    terrain_cfg = terrain.cfg.terrain_generator
    col_range = _get_terrain_column_range(terrain_cfg, terrain_name, env.device)
    if col_range is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    col_start, col_end = col_range
    # terrain_types directly stores column indices, so just check if they're in range
    return (terrain.terrain_types >= col_start) & (terrain.terrain_types < col_end)


def is_robot_on_terrain(env: ManagerBasedEnv, terrain_name: str, asset_name: str = "robot") -> torch.Tensor:
    """Check which environments are currently assigned to the specified terrain type.

    The terrain importer tracks the active terrain column for every environment.
    This helper uses that assignment directly instead of inferring terrain membership
    from robot world positions.

    Args:
        env: The environment instance.
        terrain_name: Name of the terrain to check (e.g., "pits", "stairs").
        asset_name: Name of the robot asset. Defaults to "robot".

    Returns:
        Boolean tensor of shape (num_envs,) where True means the robot is currently on this terrain.
    """
    # Check if terrain and terrain generator are available
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or not hasattr(terrain, "terrain_types"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if terrain.cfg.terrain_type != "generator" or terrain.cfg.terrain_generator is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    terrain_cfg = terrain.cfg.terrain_generator
    col_range = _get_terrain_column_range(terrain_cfg, terrain_name, env.device)
    if col_range is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    col_start, col_end = col_range

    # The terrain importer already tracks the active terrain column for each environment.
    # Using that source of truth keeps this aligned with curriculum updates and avoids
    # misclassifying robots by searching the nearest tile in world coordinates.
    del asset_name
    return (terrain.terrain_types >= col_start) & (terrain.terrain_types < col_end)


"""Commands Utilities"""

def sample_disjoint_intervals(env_ids, limit_bound, cfg_min, cfg_max, device):
    """Sample uniform distribution from [cfg_min, -limit_bound] U [limit_bound, cfg_max]"""
    width_neg = torch.nn.functional.relu(-limit_bound - cfg_min)
    width_pos = torch.nn.functional.relu(cfg_max - limit_bound)
    
    total_width = width_neg + width_pos + 1e-6 # 加极小值防除零
    u = torch.rand(len(env_ids), device=device) * total_width
    
    samples = torch.where(
        u < width_neg, 
        cfg_min + u, 
        cfg_max - width_pos + (u - width_neg)
    )
    return samples

def sample_single_interval(env_ids, cfg_min, cfg_max, device):
    """Sample uniform distribution from [cfg_min, cfg_max]"""
    r = torch.rand(len(env_ids), device=device)
    samples = cfg_min + r * (cfg_max - cfg_min)
    return samples
