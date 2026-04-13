# Unitree actuator class that implements a torque-speed curve for the actuators.
# source: https://github.com/unitreerobotics/unitree_rl_lab

from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class UnitreeActuator(DelayedPDActuator):
    """Unitree actuator class that implements a torque-speed curve for the actuators.

    The torque-speed curve is defined as follows:

            Torque Limit, N·m
                ^
    Y2──────────|
                |──────────────Y1
                |              │\
                |              │ \
                |              │  \
                |              |   \
    ------------+--------------|------> velocity: rad/s
                              X1   X2

    - Y1: Peak Torque Test (Torque and Speed in the Same Direction)
    - Y2: Peak Torque Test (Torque and Speed in the Opposite Direction)
    - X1: Maximum Speed at Full Torque (T-N Curve Knee Point)
    - X2: No-Load Speed Test

    - Fs: Static friction coefficient
    - Fd: Dynamic friction coefficient
    - Va: Velocity at which the friction is fully activated
    """

    cfg: UnitreeActuatorCfg

    armature: torch.Tensor
    """The armature of the actuator joints. Shape is (num_envs, num_joints).
        armature = J2 + J1 * i2 ^ 2 + Jr * (i1 * i2) ^ 2
    """

    def __init__(self, cfg: UnitreeActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._effort_y1 = self._parse_joint_parameter(cfg.Y1, 1e9)
        self._effort_y2 = self._parse_joint_parameter(cfg.Y2, cfg.Y1)
        self._velocity_x1 = self._parse_joint_parameter(cfg.X1, 1e9)
        self._velocity_x2 = self._parse_joint_parameter(cfg.X2, 1e9)
        self._friction_static = self._parse_joint_parameter(cfg.Fs, 0.0)
        self._friction_dynamic = self._parse_joint_parameter(cfg.Fd, 0.0)
        self._activation_vel = self._parse_joint_parameter(cfg.Va, 0.01)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # save current joint vel
        self._joint_vel[:] = joint_vel
        # calculate the desired joint torques
        control_action = super().compute(control_action, joint_pos, joint_vel)

        # apply friction model on the torque
        self.applied_effort -= (
            self._friction_static * torch.tanh(joint_vel / self._activation_vel) + self._friction_dynamic * joint_vel
        )

        control_action.joint_positions = None
        control_action.joint_velocities = None
        control_action.joint_efforts = self.applied_effort

        return control_action

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        # check if the effort is the same direction as the joint velocity
        same_direction = (self._joint_vel * effort) > 0
        max_effort = torch.where(same_direction, self._effort_y1, self._effort_y2)
        # check if the joint velocity is less than the max speed at full torque
        max_effort = torch.where(
            self._joint_vel.abs() < self._velocity_x1, max_effort, self._compute_effort_limit(max_effort)
        )
        return torch.clip(effort, -max_effort, max_effort)

    def _compute_effort_limit(self, max_effort):
        k = -max_effort / (self._velocity_x2 - self._velocity_x1)
        limit = k * (self._joint_vel.abs() - self._velocity_x1) + max_effort
        return limit.clip(min=0.0)


@configclass
class UnitreeActuatorCfg(DelayedPDActuatorCfg):
    """
    Configuration for Unitree actuators.
    """

    class_type: type = UnitreeActuator

    X1: float = 1e9
    """Maximum Speed at Full Torque(T-N Curve Knee Point) Unit: rad/s"""

    X2: float = 1e9
    """No-Load Speed Test Unit: rad/s"""

    Y1: float = MISSING
    """Peak Torque Test(Torque and Speed in the Same Direction) Unit: N*m"""

    Y2: float | None = None
    """Peak Torque Test(Torque and Speed in the Opposite Direction) Unit: N*m"""

    Fs: float = 0.0
    """ Static friction coefficient """

    Fd: float = 0.0
    """ Dynamic friction coefficient """

    Va: float = 0.01
    """ Velocity at which the friction is fully activated """


@configclass
class UnitreeActuatorCfg_Go2HV(UnitreeActuatorCfg):
    X1 = 13.5
    X2 = 30
    Y1 = 20.2
    Y2 = 23.4

