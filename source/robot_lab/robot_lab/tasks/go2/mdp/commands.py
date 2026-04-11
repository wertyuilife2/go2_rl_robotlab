# -*- coding: utf-8 -*-
'''
@File    : commands_go2_rl_gym.py
@Time    : 2026/04/01 17:16:44
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.github.io/
@Desc    : CommandTerm for go2_rl_gym style command generation, reference to https://github.com/wty-yy/go2_rl_gym

IsaacLab CommandTerm working flow:
Env: after compute reward call command.compute(dt)
1. self._update_metrics(): update self.metrics dict for logging
2. self.time_left -= dt
3. self._resample(self.time_left <= 0)
4. self._update_command(): update command if needed

Get command from self.command property, return command, shape=(num_envs, command_dim)

Note:
1. We don't use original self._resample(env_ids) and self._resample_command(env_ids), because it will randomize time_left
2. Remove heading command
3. We don't use curriculum item to update curriculum, inplace update
'''

from __future__ import annotations  # For forward reference of type hints

from typing import TYPE_CHECKING, Sequence
if TYPE_CHECKING:  # Avoid circular import for type checking
    from robot_lab.tasks.go2.env.go2_env import ActionDelayGo2Env

from itertools import product

import torch

from isaaclab.utils import configclass
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.assets import Articulation
import isaaclab.utils.math as math_utils

from robot_lab.tasks.go2.mdp.utils import is_robot_on_terrain, sample_disjoint_intervals, sample_single_interval


class Go2RLGymCommand(CommandTerm):
    cfg: Go2RLGymCommandCfg
    _env: ActionDelayGo2Env
    
    def __init__(self, cfg: Go2RLGymCommandCfg, env: ActionDelayGo2Env):
        """Reference: https://github.com/wty-yy/go2_rl_gym/blob/master/legged_gym/envs/base/legged_robot.py
        LeggedRobot._resample_command() and LeggedRobot._post_physics_step_callback()
        """
        super().__init__(cfg, env)
        self.commands_xy_accumulation = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.max_move_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.last_is_limit_vel = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)  # [lin_vel_x, lin_vel_y, ang_vel_yaw]
        self.command_ranges = self.cfg.ranges.to_dict()
        self.env_command_ranges = {
            'lin_vel_x': torch.tensor(self.command_ranges['lin_vel_x'], device=self.device).repeat(self.num_envs, 1),
            'lin_vel_y': torch.tensor(self.command_ranges['lin_vel_y'], device=self.device).repeat(self.num_envs, 1),
            'ang_vel_yaw': torch.tensor(self.command_ranges['ang_vel_yaw'], device=self.device).repeat(self.num_envs, 1),
        }
        self.max_lin_vel = max(abs(self.command_ranges["lin_vel_x"][0]), abs(self.command_ranges["lin_vel_x"][1]),
                               abs(self.command_ranges["lin_vel_y"][0]), abs(self.command_ranges["lin_vel_y"][1]))
        self.limit_vel_comb = torch.tensor(list(product(
            self.cfg.limit_vel["lin_vel_x"],
            self.cfg.limit_vel["lin_vel_y"],
            self.cfg.limit_vel["ang_vel_yaw"]
        )), device=self.device)
        self._init_terrain_infos()
        self._update_env_command_ranges()
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.zero_command_prob = 0
        self.max_command_x = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.cfg.command_range_curriculum = sorted(self.cfg.command_range_curriculum, key=lambda x: x['iter'], reverse=True)

    def __str__(self) -> str:
        """Return a string representation of the command term."""
        msg = (f"""Go2RLGymCommand:\n"""
               f"""Command shape: {self.commands.shape}""")
        return msg

    def _init_terrain_infos(self):
        """Initialize terrain types and indices for each environment."""
        self.terrain_types = list(self._env.scene.terrain.cfg.terrain_generator.sub_terrains.keys())
        for terrain_type in self.terrain_types:
            if terrain_type not in self.cfg.terrain_max_command_ranges:
                raise ValueError(f"Terrain type '{terrain_type}' is not defined in cfg.terrain_max_command_ranges.")
        self.terrain_type2idx = {terrain_type: idx for idx, terrain_type in enumerate(self.terrain_types)}
        self.terrain_idxs = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        for terrain_type in self.terrain_types:
            idxs = is_robot_on_terrain(self._env, terrain_type).nonzero().flatten()
            if len(idxs) > 0:
                self.terrain_idxs[idxs] = self.terrain_type2idx[terrain_type]
        self.terrain_length = self._env.scene.terrain.cfg.terrain_generator.size[0]

    @property
    def command(self) -> torch.Tensor:
        return self.commands

    def _update_metrics(self):
        self.max_command_x[:] = self.command_ranges["lin_vel_x"][1]
        self.metrics["max_command_x"] = self.max_command_x

    def reset(self, env_ids: Sequence[int] | None = None):
        self.time_left[env_ids] = self.cfg.resampling_time
        self.commands_xy_accumulation[env_ids] = 0.0
        self.max_move_distance[env_ids] = 0.0
        return super().reset(env_ids)
        
    def _resample(self, env_ids: Sequence[int]):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        env = self._env
        if len(env_ids) == 0:
            return
        # update command curriculum with train steps
        if len(self.cfg.command_range_curriculum):
            current_iter = env.common_step_counter // self.cfg.num_steps_per_iter
            for i in range(len(self.cfg.command_range_curriculum)-1, -1, -1):  # iterate backwards to be able to pop entries
                cfg = self.cfg.command_range_curriculum[i]
                if current_iter >= cfg["iter"]:
                    self.command_ranges["lin_vel_x"] = cfg["lin_vel_x"]
                    self.command_ranges["lin_vel_y"] = cfg["lin_vel_y"]
                    self.command_ranges["ang_vel_yaw"] = cfg["ang_vel_yaw"]
                    self.max_lin_vel = max(abs(self.command_ranges["lin_vel_x"][0]), abs(self.command_ranges["lin_vel_x"][1]),
                                           abs(self.command_ranges["lin_vel_y"][0]), abs(self.command_ranges["lin_vel_y"][1]))
                    self.cfg.command_range_curriculum.pop(i)
                    self._update_env_command_ranges()
                    print(f"Command range updated at iter {current_iter}: {self.command_ranges}")
        remaining_dist = torch.clip(0.625 * self.terrain_length - torch.norm(self.commands_xy_accumulation[env_ids], dim=1) * self.cfg.resampling_time, 0.0)
        self.time_left[env_ids] = self.cfg.resampling_time
        if self.cfg.dynamic_resample_commands:
            # arrive at boundary 0.625 times the width of the remaining distance
            if ((env.max_episode_length - env.episode_length_buf[env_ids]) + 1 == 0).any():
                raise ValueError("Some envs have zero remaining episode length during command resampling")
            vel_low_bound = torch.clip(remaining_dist / ((env.max_episode_length - env.episode_length_buf[env_ids] + 1 + 1e-9) * env.step_dt), 0.0)
            self.commands[env_ids, 0] = sample_disjoint_intervals(
                env_ids,
                vel_low_bound,
                self.env_command_ranges["lin_vel_x"][env_ids, 0],
                self.env_command_ranges["lin_vel_x"][env_ids, 1],
                self.device
            )
            self.commands[env_ids, 1] = sample_disjoint_intervals(
                env_ids,
                vel_low_bound,
                self.env_command_ranges["lin_vel_y"][env_ids, 0],
                self.env_command_ranges["lin_vel_y"][env_ids, 1],
                self.device
            )
            r = torch.rand(len(env_ids), device=self.device)
            lower = self.env_command_ranges["ang_vel_yaw"][env_ids, 0]
            upper = self.env_command_ranges["ang_vel_yaw"][env_ids, 1]
            self.commands[env_ids, 2] = (upper - lower) * r + lower
        else:
            self.commands[env_ids, 0] = sample_single_interval(
                env_ids,
                self.env_command_ranges["lin_vel_x"][env_ids, 0],
                self.env_command_ranges["lin_vel_x"][env_ids, 1],
                self.device
            )
            self.commands[env_ids, 1] = sample_single_interval(
                env_ids,
                self.env_command_ranges["lin_vel_y"][env_ids, 0],
                self.env_command_ranges["lin_vel_y"][env_ids, 1],
                self.device
            )
            self.commands[env_ids, 2] = sample_single_interval(
                env_ids,
                self.env_command_ranges["ang_vel_yaw"][env_ids, 0],
                self.env_command_ranges["ang_vel_yaw"][env_ids, 1],
                self.device
            )

            # set small commands to zero
            self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        rand_prob = torch.rand(len(env_ids), device=self.device)
        min_prob, max_prob = 0.0, 0.0
        # set limitation lin vel
        if self.cfg.limit_vel_prob > 0.0:
            max_prob += self.cfg.limit_vel_prob
            lim_mask = (rand_prob >= min_prob) * (rand_prob < max_prob)
            lim_env_ids = env_ids[lim_mask]
            if len(lim_env_ids) > 0:
                change_lim_env_ids = lim_env_ids
                if self.cfg.limit_vel_invert_when_continuous:
                    was_limited = self.last_is_limit_vel[lim_env_ids]
                    invert_env_ids = lim_env_ids[was_limited]
                    self.commands[invert_env_ids, 0] *= -1.0
                    self.commands[invert_env_ids, 1] *= -1.0
                    self.commands[invert_env_ids, 2] *= -1.0
                    change_lim_env_ids = lim_env_ids[~was_limited]
                vel_idx = torch.randint(0, self.limit_vel_comb.shape[0], (len(change_lim_env_ids),), device=self.device)
                lin_vel_x_lim = torch.where(
                    self.limit_vel_comb[vel_idx, 0] == -1,
                    self.env_command_ranges["lin_vel_x"][change_lim_env_ids, 0],
                    self.env_command_ranges["lin_vel_x"][change_lim_env_ids, 1],
                )
                lin_vel_x_lim[self.limit_vel_comb[vel_idx, 0] == 0] = 0.0
                lin_vel_y_lim = torch.where(
                    self.limit_vel_comb[vel_idx, 1] == -1,
                    self.env_command_ranges["lin_vel_y"][change_lim_env_ids, 0],
                    self.env_command_ranges["lin_vel_y"][change_lim_env_ids, 1]
                )
                lin_vel_y_lim[self.limit_vel_comb[vel_idx, 1] == 0] = 0.0
                ang_vel_z_lim = torch.where(
                    self.limit_vel_comb[vel_idx, 2] == -1,
                    self.env_command_ranges["ang_vel_yaw"][change_lim_env_ids, 0],
                    self.env_command_ranges["ang_vel_yaw"][change_lim_env_ids, 1]
                )
                ang_vel_z_lim[self.limit_vel_comb[vel_idx, 2] == 0] = 0.0
                self.commands[change_lim_env_ids, 0] = lin_vel_x_lim
                self.commands[change_lim_env_ids, 1] = lin_vel_y_lim
                self.commands[change_lim_env_ids, 2] = ang_vel_z_lim
                self.last_is_limit_vel[env_ids] = False
                self.last_is_limit_vel[lim_env_ids] = True
            else:
                self.last_is_limit_vel[env_ids] = False
            min_prob += self.cfg.limit_vel_prob

        # set all commands to zero with some probability
        if self.cfg.zero_command_curriculum is not None:
            self.zero_command_prob = self.get_current_scale(self.cfg.zero_command_curriculum)
        if self.zero_command_prob > 0.0:
            max_prob += self.zero_command_prob
            next_time_left = torch.clip(
                env.max_episode_length_s - env.episode_length_buf[env_ids] * env.step_dt - (remaining_dist / (0.8 * self.max_lin_vel + 1e-9)),
                min=0.0,
                max=self.cfg.resampling_time,
            )
            zero_mask = (rand_prob >= min_prob) * (rand_prob < max_prob) * (next_time_left > 0.0)
            zero_env_ids = env_ids[zero_mask]
            if len(zero_env_ids) > 0:
                self.commands[zero_env_ids, :2] = 0.0
                self.time_left[zero_env_ids] = next_time_left[zero_mask]
                if self.cfg.limit_ang_vel_at_zero_command_prob > 0.0:
                    ang_vel_rand = torch.rand(len(zero_env_ids), device=self.device)  # independent distribution
                    add_ang_mask = ang_vel_rand < self.cfg.limit_ang_vel_at_zero_command_prob
                    add_ang_env_ids = zero_env_ids[add_ang_mask]
                    if len(add_ang_env_ids) > 0:
                        direction_rand = torch.rand(len(add_ang_env_ids), device=self.device)
                        self.commands[add_ang_env_ids, 2] = torch.where(
                            direction_rand < 0.5,
                            self.env_command_ranges["ang_vel_yaw"][add_ang_env_ids, 0],
                            self.env_command_ranges["ang_vel_yaw"][add_ang_env_ids, 1]
                        )
            min_prob += self.zero_command_prob

        self.commands_xy_accumulation[env_ids] += self.commands[env_ids, :2]

    def _update_command(self):
        current_dist = torch.norm(self.robot.data.root_pos_w[:, :2] - self._env.scene.env_origins[:, :2], dim=1)
        self.max_move_distance = torch.max(self.max_move_distance, current_dist)

    def _update_env_command_ranges(self):
        """ Update environment-wise command ranges based on current command ranges and terrain type """
        for terrain_type, terrain_command_ranges in self.cfg.terrain_max_command_ranges.items():
            if terrain_type not in self.terrain_type2idx:
                continue
            terrain_idx = self.terrain_type2idx[terrain_type]
            env_ids = (self.terrain_idxs == terrain_idx).nonzero().flatten()
            self.env_command_ranges['lin_vel_x'][env_ids, 0] = max(
                terrain_command_ranges['lin_vel_x'][0],
                self.command_ranges['lin_vel_x'][0],
            )
            self.env_command_ranges['lin_vel_x'][env_ids, 1] = min(
                terrain_command_ranges['lin_vel_x'][1],
                self.command_ranges['lin_vel_x'][1]
            )
            self.env_command_ranges['lin_vel_y'][env_ids, 0] = max(
                terrain_command_ranges['lin_vel_y'][0],
                self.command_ranges['lin_vel_y'][0]
            )
            self.env_command_ranges['lin_vel_y'][env_ids, 1] = min(
                terrain_command_ranges['lin_vel_y'][1],
                self.command_ranges['lin_vel_y'][1]
            )
            self.env_command_ranges['ang_vel_yaw'][env_ids, 0] = max(
                terrain_command_ranges['ang_vel_yaw'][0],
                self.command_ranges['ang_vel_yaw'][0]
            )
            self.env_command_ranges['ang_vel_yaw'][env_ids, 1] = min(
                terrain_command_ranges['ang_vel_yaw'][1],
                self.command_ranges['ang_vel_yaw'][1]
            )
    
    def get_current_scale(self, config: dict):
        """config: {'start_iter': 0, 'end_iter': 1500, 'start_value': 1.0, 'end_value': 0.0}"""
        current_iter = self._env.common_step_counter // self.cfg.num_steps_per_iter
        cfg_start_iter = config['start_iter']
        cfg_end_iter = config['end_iter']
        cfg_start_val = config['start_value']
        cfg_end_val = config['end_value']

        percentage = (current_iter - cfg_start_iter) / (cfg_end_iter - cfg_start_iter)
        percentage = max(min(percentage, 1.0), 0.0)
        
        current_scale = (1.0 - percentage) * cfg_start_val + percentage * cfg_end_val
        return current_scale

    """Debug Visualization"""

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

    def _resample_command(self):
        ...

@configclass
class Go2RLGymCommandCfg(CommandTermCfg):
    class_type: type = Go2RLGymCommand

    asset_name: str = "robot"
    """Name of the asset in the environment for which the commands are generated."""

    dynamic_resample_commands: bool = True
    """Sample commands with low bounds"""
    limit_vel_invert_when_continuous: bool = True
    """Invert the limit logic when using continuous sample limit velocity commands"""

    zero_command_curriculum: dict = {'start_iter': 0, 'end_iter': 1500, 'start_value': 0.0, 'end_value': 0.1}
    """Start training with zero commands and then gradually increase zero command probability"""
    limit_vel: dict = {"lin_vel_x": [-1, 1], "lin_vel_y": [-1, 1], "ang_vel_yaw": [-1, 0, 1]}
    """Sample vel commands from min [-1] or zero [0] or max [1] range only"""
    command_range_curriculum: list[dict] = [{
        'iter': 20000, # training iteration at which the command ranges are updated
        'lin_vel_x': [-1.0, 1.0], # min max [m/s]
        'lin_vel_y': [-1.0, 1.0], # min max [m/s]
        'ang_vel_yaw': [-1.5, 1.5], # min max [rad/s]
    }, {
        'iter': 50000, # training iteration at which the command ranges are updated
        'lin_vel_x': [-2.0, 2.0], # min max [m/s]
        'lin_vel_y': [-1.0, 1.0], # min max [m/s]
        'ang_vel_yaw': [-2.0, 2.0], # min max [rad/s]
    }]
    """List for command range curriculums at specific training iterations"""
    # terrain_max_command_ranges: dict[str, dict] = {
    #     'random_rough':
    #         {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5]},
    #     'hf_pyramid_slope':
    #         {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5]},
    #     'hf_pyramid_slope_inv':
    #         {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5]},
    #     'pyramid_stairs':
    #         {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5]},
    #     'pyramid_stairs_inv':
    #         {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5]},
    #     'boxes':
    #         {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5]},
    #     'flat':
    #         {'lin_vel_x': [-2.0, 2.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.0, 2.0]},
    # }
    terrain_max_command_ranges: dict[str, dict] = {
        'wave':
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5]},
        'slope':
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5]},
        'rough_slope':
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5]},
        'stairs_up':
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5]},
        'stairs_down':
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5]},
        'obstacles':
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5]},
        # 'stepping_stones':
        #     {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5]},
        # 'gap':
        #     {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5]},
        'flat':
            {'lin_vel_x': [-2.0, 2.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.0, 2.0]},
    }
    resampling_time: float = 5.0
    resampling_time_range: tuple[float, float] = (5.0, 5.0)
    """Time before command are changed [s]"""
    limit_ang_vel_at_zero_command_prob: float = 0.2
    """Probability of add limiting angular velocity commands when zero command is sampled"""
    limit_vel_prob: float = 0.2
    """Probability of limiting linear velocity command"""
    num_steps_per_iter: int = 24
    """Number of envs steps for each training iteration"""

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = [-0.5, 0.5]
        """Range for the linear-x velocity command [m/s]"""
        lin_vel_y: tuple[float, float] = [-0.5, 0.5]
        """Range for the linear-y velocity command [m/s]"""
        ang_vel_yaw: tuple[float, float] = [-1.0, 1.0]
        """Range for the angular-z velocity command [rad/s]"""

    ranges: Ranges = Ranges()

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""
