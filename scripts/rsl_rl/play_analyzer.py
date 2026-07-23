# -*- coding: utf-8 -*-
"""
@File    : play_analyzer.py
@Time    : 2026/07/23 16:51:34
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.github.io/
@Desc    : Used to analyze the play data for the RL rewards.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardTermCfg

from robot_lab.tasks.go2.mdp.utils import is_robot_on_terrain


class PlayAnalyzer:
    """Analyze reward terms by terrain type during play."""

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        log_dir: str | Path,
        interval: int = 1,
        enable: bool = True,
        config: dict[str, Any] | None = None,
    ):
        self.env = env
        self.log_dir = Path(log_dir) / time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        self.interval = max(1, int(interval))
        self.enable = enable
        self.config = config
        self.cnt = 0
        self.num_envs = self.env.num_envs
        self.output_path = self.log_dir / "reward_analysis.csv"

        self.reward_manager = None
        self.reward_keys: list[str] = []
        self.reward_indices: list[int] = []
        self.terrain_env_ids: dict[str, torch.Tensor] = {}
        self.row_types: list[str] = []
        self.reward_sums: dict[str, torch.Tensor] = {}
        self.reward_counts: dict[str, int] = {}

        if self.enable:
            self.__post_init__()

    def __post_init__(self):
        print(f"[INFO] PlayAnalyzer: Logging reward analysis to {self.log_dir}")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.reward_manager = self.env.reward_manager
        self.reward_keys, self.reward_indices = self._get_active_reward_terms()
        self.terrain_env_ids = self._get_terrain_env_ids()
        self.row_types = ["avg", *self.terrain_env_ids.keys()]

        for terrain_type in self.row_types:
            self.reward_sums[terrain_type] = torch.zeros(len(self.reward_keys), dtype=torch.float64)
            self.reward_counts[terrain_type] = 0

        if self.config is not None:
            with open(self.log_dir / "config.yaml", "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, default_flow_style=False)

    def update(self):
        """Accumulate current-step reward values.

        Call this once after each ``env.step(...)``. IsaacLab's RewardManager updates
        ``_step_reward`` inside ``env.step(...)``, so this method reads that cached result.
        """
        if not self.enable:
            return

        self.cnt += 1
        if self.cnt % self.interval != 0 or len(self.reward_indices) == 0:
            return

        step_rewards = self.reward_manager._step_reward[:, self.reward_indices].detach()
        self._accumulate("avg", step_rewards)

        for terrain_type, env_ids in self.terrain_env_ids.items():
            if env_ids.numel() == 0:
                continue
            self._accumulate(terrain_type, step_rewards.index_select(0, env_ids))

        self.save()

    def save(self) -> Path | None:
        """Write the reward summary table to csv."""
        if not self.enable:
            return None

        df = self.to_dataframe()
        df.to_csv(self.output_path, index=False)
        return self.output_path

    def to_dataframe(self) -> pd.DataFrame:
        """Return the current summary table.

        Rows are ``avg`` followed by every terrain name. Columns are ``type``
        followed by reward names.
        """
        rows = []
        for terrain_type in self.row_types:
            count = self.reward_counts[terrain_type]
            if count > 0:
                reward_avg = self.reward_sums[terrain_type] / count
            else:
                reward_avg = torch.full((len(self.reward_keys),), float("nan"), dtype=torch.float64)
            row = {"type": terrain_type}
            row.update({name: reward_avg[idx].item() for idx, name in enumerate(self.reward_keys)})
            rows.append(row)

        return pd.DataFrame(rows, columns=["type", *self.reward_keys])

    def _accumulate(self, terrain_type: str, values: torch.Tensor):
        """Accumulate reward sums for one row of the output table."""
        if values.numel() == 0:
            return
        self.reward_sums[terrain_type] += values.sum(dim=0).cpu().to(torch.float64)
        self.reward_counts[terrain_type] += values.shape[0]

    def _get_active_reward_terms(self) -> tuple[list[str], list[int]]:
        """Get active reward names and their column indices in RewardManager."""
        active_terms = list(self.reward_manager.active_terms)
        rewards_cfg = getattr(self.env.cfg, "rewards", None)

        reward_keys = []
        reward_indices = []
        if rewards_cfg is not None:
            for name, term_cfg in vars(rewards_cfg).items():
                if isinstance(term_cfg, RewardTermCfg) and name in active_terms:
                    reward_keys.append(name)
                    reward_indices.append(active_terms.index(name))

        # Keep any active manager terms that are not visible from env.cfg.rewards.
        for name in active_terms:
            if name not in reward_keys:
                reward_keys.append(name)
                reward_indices.append(active_terms.index(name))

        return reward_keys, reward_indices

    def _get_terrain_env_ids(self) -> dict[str, torch.Tensor]:
        """Get terrain names and the env indices assigned to each terrain."""
        terrain_cfg = getattr(getattr(self.env.cfg.scene, "terrain", None), "terrain_generator", None)
        if terrain_cfg is None or terrain_cfg.sub_terrains is None:
            return {}

        terrain_env_ids = {}
        for terrain_name in terrain_cfg.sub_terrains.keys():
            mask = is_robot_on_terrain(self.env, terrain_name)
            terrain_env_ids[terrain_name] = mask.nonzero(as_tuple=False).flatten()

        return terrain_env_ids
