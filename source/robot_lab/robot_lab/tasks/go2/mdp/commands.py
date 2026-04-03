# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
import torch
import copy
from collections.abc import Sequence
from typing import TYPE_CHECKING
from dataclasses import MISSING
from itertools import product

from isaaclab.managers import CommandTerm
from isaaclab.utils import configclass
from isaaclab.assets import Articulation
from isaaclab.envs.mdp import UniformVelocityCommandCfg
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers

# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2025, The Isaac Lab Project Developers

from .utils import is_robot_on_terrain

if TYPE_CHECKING:
    from robot_lab.tasks.go2.env.go2_env import ManagerBasedRLEnv


class UniformVelTerrainCmd(CommandTerm):
    cfg: UniformVelTerrainCmdCfg
    
    def __init__(self, cfg: UniformVelTerrainCmdCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel, heading
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)

        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        if self.cfg.use_phase_cmd:
            self.phase = torch.zeros(self.num_envs, 1, device=self.device)
        self.cycle_time = cfg.cycle_time
        self.max_angular_envs = cfg.max_angular_envs
        self.limit_vel_envs = cfg.limit_vel_envs
        self.stop_heading = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 获取当前环境的地形类型索引  
        self.terrain_types = list(env.scene.terrain.cfg.terrain_generator.sub_terrains.keys())
        if not isinstance(self.cfg.ranges, dict):
            single_range = cfg.ranges
            expanded_ranges = {}
            for t_name in self.terrain_types:
                expanded_ranges[t_name] = copy.deepcopy(single_range)
            self.cfg.ranges = expanded_ranges
        assert set(self.terrain_types) == set(list(self.cfg.ranges.keys())), \
            "Terrain types in cfg.ranges do not match those in terrain generator config."
        self.terrain_type_to_id = {name: i for i, name in enumerate(self.terrain_types)}
        self.global_env_terrain_idx = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        for t_type in self.terrain_types:
            ids = is_robot_on_terrain(self._env, t_type).nonzero(as_tuple=True)[0]
            if len(ids) > 0:
                self.global_env_terrain_idx[ids] = self.terrain_type_to_id[t_type]
        # 初始化累计指令和最大位移
        self.commands_xy_accumulation = torch.zeros(self.num_envs, 2, device=self.device)
        self.dt = self._env.step_dt
        self.max_episode_length = self._env.max_episode_length
        self.target_dist = env.scene.terrain.cfg.terrain_generator.size[0] * 0.625
        self.max_move_distance = torch.zeros(self.num_envs, device=self.device)
        self.env_origins = env.scene.env_origins
        
        # 极限指令组合
        limit_options_x = [-1, 1]
        limit_options_y = [-1, 1]
        limit_options_z = [-1, 0, 1] # 允许 Yaw 为 0，即直线冲刺
        self.limit_vel_combinations = torch.tensor(
            list(product(limit_options_x, limit_options_y, limit_options_z)),
            dtype=torch.long, 
            device=self.device
        )
        
        self.last_is_limit_vel = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}\n"
        msg += f"\tMax angular vel envs: {self.max_angular_envs}\n"
        msg += f"\tLimit linear vel envs: {self.limit_vel_envs}\n"
        for t_type in self.terrain_types:
            msg += f"\tTerrain type '{t_type}' command ranges: {self.cfg.ranges[t_type]}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        if self.cfg.use_phase_cmd:
            phase = self.phase * 2 * torch.pi
            return torch.cat([self.vel_command_b, torch.sin(phase), torch.cos(phase)], dim=-1)
        return self.vel_command_b

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )
    
    def reset(self, env_ids: Sequence[int] | None = None):
        self.commands_xy_accumulation[env_ids] = 0.0
        self.max_move_distance[env_ids] = 0.0
        extra = super().reset(env_ids)
        return extra

    def _terrain_cmd_sample(self, dim: str, env_ids: torch.Tensor, min_abs_vel: torch.Tensor | None = None):
        """根据不同地形sample不同指令"""
        cmd = torch.zeros(len(env_ids), device=self.device)
        batch_terrain_idxs = self.global_env_terrain_idx[env_ids]
        
        for t_type in self.terrain_types:
            t_id = self.terrain_type_to_id[t_type]
            mask = (batch_terrain_idxs == t_id)
            
            if mask.any():
                count = mask.sum().item()
                if dim == "x":
                    r = self.cfg.ranges[t_type].lin_vel_x
                elif dim == "y":
                    r = self.cfg.ranges[t_type].lin_vel_y
                elif dim == "z":
                    r = self.cfg.ranges[t_type].ang_vel_z
                elif dim == "heading":
                    r = self.cfg.ranges[t_type].heading
                
                sampled_vals = torch.empty(count, device=self.device).uniform_(*r) # type: ignore

                if min_abs_vel is not None and (dim == "x" or dim == "y"):
                    lower_bound = min_abs_vel[mask]
                    
                    max_abs = max(abs(r[0]), abs(r[1]))
                    lower_bound = torch.clamp(lower_bound, max=max_abs)

                    # 重新采样 Magnitude: [lower_bound, max_abs]
                    mag = torch.empty(count, device=self.device).uniform_(0, 1) * (max_abs - lower_bound) + lower_bound
                    
                    if r[0] >= 0:
                        sampled_vals = mag
                    elif r[1] <= 0:
                        sampled_vals = -mag
                    else: 
                        sign = torch.sign(torch.empty(count, device=self.device).uniform_(-1, 1))
                        sign = torch.where(sign == 0, torch.ones_like(sign), sign) 
                        sampled_vals = mag * sign
                
                cmd[mask] = sampled_vals
        
        return cmd
        
    def _get_cmd_board(self, dim: str, env_ids: torch.Tensor):
        """根据预计算的地形类型获取速度指令边界"""
        cmd_min = torch.zeros(len(env_ids), device=self.device)
        cmd_max = torch.zeros(len(env_ids), device=self.device)
        
        batch_terrain_idxs = self.global_env_terrain_idx[env_ids]

        for t_type in self.terrain_types:
            t_id = self.terrain_type_to_id[t_type]
            mask = (batch_terrain_idxs == t_id)
            
            if mask.any():
                if dim == "x":
                    r = self.cfg.ranges[t_type].lin_vel_x
                elif dim == "y":
                    r = self.cfg.ranges[t_type].lin_vel_y
                elif dim == "z":
                    r = self.cfg.ranges[t_type].ang_vel_z
                else:
                    continue
                
                cmd_min[mask] = r[0]
                cmd_max[mask] = r[1]
                
        return cmd_min, cmd_max
    
    def _resample_command(self, env_ids: Sequence[int]):
        _env_ids = torch.tensor(env_ids, device=self.device)
        
        # 计算剩余距离: 目标距离 - 已经指令累积走过的距离 * 上一次重采样时间
        dist_covered = torch.norm(self.commands_xy_accumulation[env_ids], dim=1) * self.cfg.resampling_time_range[0]
        remaining_dist = torch.clamp(self.target_dist - dist_covered, min=0.0)
        
        # 计算剩余时间: (最大步数 - 当前步数) * dt
        time_left = (self.max_episode_length - self._env.episode_length_buf[env_ids]) * self.dt
        # 计算下限速度: 距离 / 时间
        vel_low_bound = torch.zeros(len(env_ids), device=self.device)
        # 避免除以零或负数时间
        valid_time_mask = time_left > 1e-4
        if valid_time_mask.any():
            vel_low_bound[valid_time_mask] = remaining_dist[valid_time_mask] / time_left[valid_time_mask]

        # 根据不同地形类型采样速度指令
        self.vel_command_b[env_ids, 0] = self._terrain_cmd_sample("x", _env_ids, min_abs_vel=vel_low_bound)
        self.vel_command_b[env_ids, 0] = torch.where(
                    self.vel_command_b[env_ids, 0].abs() < 0.1,
                   torch.zeros_like(self.vel_command_b[env_ids, 0]),
                    self.vel_command_b[env_ids, 0]
                )
        self.vel_command_b[env_ids, 1] = self._terrain_cmd_sample("y", _env_ids, min_abs_vel=vel_low_bound)
        self.vel_command_b[env_ids, 1] = torch.where(
                    self.vel_command_b[env_ids, 1].abs() < 0.1,
                   torch.zeros_like(self.vel_command_b[env_ids, 1]),
                    self.vel_command_b[env_ids, 1]
                )
        self.vel_command_b[env_ids, 2] = self._terrain_cmd_sample("z", _env_ids)
        self.vel_command_b[env_ids, 2] = torch.where(
                    self.vel_command_b[env_ids, 2].abs() < 0.1,
                   torch.zeros_like(self.vel_command_b[env_ids, 2]),
                    self.vel_command_b[env_ids, 2]
                )
        
        # 计算heading和stand
        r = torch.empty(len(env_ids), device=self.device)
        if self.cfg.heading_command:
            self.heading_target[env_ids] = self._terrain_cmd_sample("heading", _env_ids)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
        
        # 静止环境sample最大转向
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        if len(standing_env_ids) > 0:
            self.vel_command_b[standing_env_ids, :] = 0.0
            ang_vel_rand = torch.rand(len(standing_env_ids), device=self.device)
            add_ang_mask = ang_vel_rand < self.max_angular_envs
            add_ang_env_ids = standing_env_ids[add_ang_mask]
            if self.max_angular_envs > 0 and len(add_ang_env_ids) > 0:
                direction_rand = torch.rand(len(add_ang_env_ids), device=self.device)
                min_z, max_z = self._get_cmd_board("z", add_ang_env_ids)
                self.vel_command_b[add_ang_env_ids, 2] = torch.where(direction_rand < 0.5, min_z, max_z)
                self.stop_heading[add_ang_env_ids] = True
        
        limit_vel_env_ids = (self.is_standing_env == 0).nonzero(as_tuple=False).flatten()
        
        # 非静止环境sample极限速度
        current_ids = env_ids
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[current_ids] = True
        target_limit_ids = limit_vel_env_ids[mask[limit_vel_env_ids]]

        if self.limit_vel_envs > 0 and len(target_limit_ids) > 0:
            limit_prob = torch.rand(len(target_limit_ids), device=self.device)
            execute_limit_mask = limit_prob < self.limit_vel_envs
            execute_ids = target_limit_ids[execute_limit_mask]

            if len(execute_ids) > 0:
                
                num_combs = self.limit_vel_combinations.shape[0]
                comb_indices = torch.randint(0, num_combs, (len(execute_ids),), device=self.device)
                selected_combs = self.limit_vel_combinations[comb_indices] # (N, 3)
                
                min_x, max_x = self._get_cmd_board("x", execute_ids)
                min_y, max_y = self._get_cmd_board("y", execute_ids)
                min_z, max_z = self._get_cmd_board("z", execute_ids)
                
                vals_x = torch.zeros_like(min_x)
                vals_x = torch.where(selected_combs[:, 0] == -1, min_x, vals_x)
                vals_x = torch.where(selected_combs[:, 0] == 1, max_x, vals_x)
                
                vals_y = torch.zeros_like(min_y)
                vals_y = torch.where(selected_combs[:, 1] == -1, min_y, vals_y)
                vals_y = torch.where(selected_combs[:, 1] == 1, max_y, vals_y)
                
                vals_z = torch.zeros_like(min_z)
                vals_z = torch.where(selected_combs[:, 2] == -1, min_z, vals_z)
                vals_z = torch.where(selected_combs[:, 2] == 1, max_z, vals_z)
                
                self.vel_command_b[execute_ids, 0] = vals_x
                self.vel_command_b[execute_ids, 1] = vals_y
                self.vel_command_b[execute_ids, 2] = vals_z
                
                self.stop_heading[execute_ids] = True
        
        if self.cfg.use_phase_cmd:
            self.phase[env_ids] = torch.rand((len(env_ids), 1), device=self.device)
        self.commands_xy_accumulation[env_ids] += self.vel_command_b[env_ids, :2]
        
    def _update_command(self):
        # 停止heading更新
        stop_heading_env_ids = self.stop_heading.nonzero(as_tuple=False).flatten()
        if len(stop_heading_env_ids) > 0:
            self.is_heading_env[stop_heading_env_ids] = False
            
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute angular velocity
            if len(env_ids) > 0:
                heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
                min_z, max_z = self._get_cmd_board("z", env_ids)
                
                self.vel_command_b[env_ids, 2] = torch.clip(
                    self.cfg.heading_control_stiffness * heading_error,
                    min=min_z,
                    max=max_z,
                )
        if self.cfg.use_phase_cmd:
            self.phase = self._env.episode_length_buf[:, None] * self._env.step_dt / self.cycle_time
        current_dist = torch.norm(self.robot.data.root_pos_w[:, :2] - self.env_origins[:, :2], dim=1)
        self.max_move_distance = torch.max(self.max_move_distance, current_dist)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat


@configclass
class UniformVelTerrainCmdCfg(UniformVelocityCommandCfg):

    class_type: type = UniformVelTerrainCmd
    cycle_time:float = 0.5
    max_angular_envs: float = 0.2
    limit_vel_envs: float = 0.2
    ranges: dict[str, UniformVelocityCommandCfg.Ranges] | UniformVelocityCommandCfg.Ranges = MISSING # 当前指令范围
    terrain_max_ranges: dict[str, UniformVelocityCommandCfg.Ranges] = MISSING # type:ignore 地形最大指令范围
    curriculum_schedule: list[dict] | None = None 
    use_phase_cmd: bool = True
    