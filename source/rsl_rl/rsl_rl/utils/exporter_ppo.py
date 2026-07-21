# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import os

import torch


def export_ppo_policy_as_jit(
    policy: object,
    actor_obs_normalizer: object | None,
    num_single_obs: int,
    num_actions: int,
    path: str,
    filename: str = "policy.pt",
) -> None:
    """Export a PPO policy with internal observation history as TorchScript.

    Args:
        policy: PPO actor-critic module to export.
        actor_obs_normalizer: Normalizer applied to stacked actor observations.
        num_single_obs: Number of features in one current observation frame.
        num_actions: Number of robot actions and joint features.
        path: Directory in which to save the exported model.
        filename: Exported TorchScript filename.
    """
    policy_exporter = _TorchPPOPolicyExporter(
        policy,
        actor_obs_normalizer=actor_obs_normalizer,
        num_single_obs=num_single_obs,
        num_actions=num_actions,
    )
    policy_exporter.export(path, filename)


def export_ppo_policy_as_onnx(
    policy: object,
    actor_obs_normalizer: object | None,
    num_single_obs: int,
    num_actions: int,
    path: str,
    filename: str = "policy.onnx",
    verbose: bool = False,
) -> None:
    """Export a PPO policy with explicit observation history as ONNX.

    Args:
        policy: PPO actor-critic module to export.
        actor_obs_normalizer: Normalizer applied to stacked actor observations.
        num_single_obs: Number of features in one current observation frame.
        num_actions: Number of robot actions and joint features.
        path: Directory in which to save the exported model.
        filename: Exported ONNX filename.
        verbose: Whether to print ONNX export details.
    """
    policy_exporter = _OnnxPPOPolicyExporter(
        policy,
        actor_obs_normalizer=actor_obs_normalizer,
        num_single_obs=num_single_obs,
        num_actions=num_actions,
        verbose=verbose,
    )
    policy_exporter.export(path, filename)


class _TorchPPOPolicyExporter(torch.nn.Module):
    """Export a PPO actor that consumes one RoboGauge observation frame."""

    def __init__(
        self,
        policy: object,
        actor_obs_normalizer: object | None,
        num_single_obs: int,
        num_actions: int,
    ) -> None:
        """Initialize the PPO history wrapper.

        The history layout matches Isaac Lab's term-wise flattened observation
        history and the existing CTS deployment exporter.

        Args:
            policy: PPO actor-critic module to copy.
            actor_obs_normalizer: Normalizer applied to stacked observations.
            num_single_obs: Number of features in one current observation frame.
            num_actions: Number of robot actions and joint features.
        """
        if policy.is_recurrent:
            raise ValueError("RoboGauge PPO history export does not support recurrent policies.")
        super().__init__()

        if not hasattr(policy, "actor"):
            raise ValueError("PPO policy does not have an actor module.")
        self.actor = copy.deepcopy(policy.actor)
        self.actor_obs_normalizer = (
            copy.deepcopy(actor_obs_normalizer) if actor_obs_normalizer is not None else torch.nn.Identity()
        )
        self.state_dependent_std = bool(policy.state_dependent_std)
        self.num_single_obs = int(num_single_obs)
        self.num_actor_obs = int(self.actor[0].in_features)
        if self.num_actor_obs % self.num_single_obs != 0:
            raise ValueError(
                f"num_actor_obs ({self.num_actor_obs}) must be divisible by num_single_obs ({self.num_single_obs})."
            )

        self.history_len = self.num_actor_obs // self.num_single_obs
        self.feature_dims = [3, 3, 3, int(num_actions), int(num_actions), int(num_actions)]
        if sum(self.feature_dims) != self.num_single_obs:
            raise ValueError(
                "Unsupported single_obs layout: expected 3+3+3+3*num_actions to match num_single_obs."
            )
        self.register_buffer("obs_history", torch.zeros(1, self.num_actor_obs, dtype=torch.float32))

    def forward(self, single_obs: torch.Tensor) -> torch.Tensor:
        """Update observation history and compute a deterministic PPO action.

        Args:
            single_obs: Current observation with shape ``[1, num_single_obs]``.

        Returns:
            Deterministic policy action tensor.
        """
        if single_obs.dim() == 1:
            single_obs = single_obs.unsqueeze(0)
        if single_obs.shape[-1] != self.num_single_obs:
            raise ValueError(
                f"Expected single_obs last dimension {self.num_single_obs}, got {single_obs.shape[-1]}."
            )
        if single_obs.shape[0] != 1:
            raise ValueError("TorchScript RoboGauge deployment supports batch size 1 only.")

        next_history = self.obs_history.clone()
        history_offset = 0
        single_offset = 0
        for dim in self.feature_dims:
            block_size = dim * self.history_len
            block_end = history_offset + block_size
            single_end = single_offset + dim
            block = self.obs_history[:, history_offset:block_end]
            shifted_block = torch.cat([block[:, dim:], single_obs[:, single_offset:single_end]], dim=-1)
            next_history[:, history_offset:block_end] = shifted_block
            history_offset = block_end
            single_offset = single_end
        self.obs_history.copy_(next_history)

        actor_obs = self.actor_obs_normalizer(self.obs_history)
        actor_output = self.actor(actor_obs)
        if self.state_dependent_std:
            return actor_output[..., 0, :]
        return actor_output

    @torch.jit.export
    def reset(self) -> None:
        """Clear the internal observation history."""
        self.obs_history.zero_()

    def export(self, path: str, filename: str) -> None:
        """Save this wrapper as a CPU TorchScript module.

        Args:
            path: Directory in which to save the model.
            filename: Exported TorchScript filename.
        """
        os.makedirs(path, exist_ok=True)
        export_path = os.path.join(path, filename)
        self.eval()
        self.to("cpu")
        torch.jit.script(self).save(export_path)


class _OnnxPPOPolicyExporter(torch.nn.Module):
    """Export a PPO actor with the same ONNX contract as MoE-CTS."""

    def __init__(
        self,
        policy: object,
        actor_obs_normalizer: object | None,
        num_single_obs: int,
        num_actions: int,
        verbose: bool = False,
    ) -> None:
        """Initialize the PPO ONNX wrapper.

        Args:
            policy: PPO actor-critic module to copy.
            actor_obs_normalizer: Normalizer applied to stacked observations.
            num_single_obs: Number of features in one current observation frame.
            num_actions: Number of robot actions and joint features.
            verbose: Whether to print ONNX export details.
        """
        if policy.is_recurrent:
            raise ValueError("PPO ONNX deployment export does not support recurrent policies.")
        super().__init__()

        if not hasattr(policy, "actor"):
            raise ValueError("PPO policy does not have an actor module.")
        self.actor = copy.deepcopy(policy.actor)
        self.actor_obs_normalizer = (
            copy.deepcopy(actor_obs_normalizer) if actor_obs_normalizer is not None else torch.nn.Identity()
        )
        self.state_dependent_std = bool(policy.state_dependent_std)
        self.num_single_obs = int(num_single_obs)
        self.num_actor_obs = int(self.actor[0].in_features)
        self.num_actions = int(num_actions)
        self.verbose = verbose

        if self.num_actor_obs % self.num_single_obs != 0:
            raise ValueError(
                f"num_actor_obs ({self.num_actor_obs}) must be divisible by num_single_obs ({self.num_single_obs})."
            )
        feature_dims = [3, 3, 3, self.num_actions, self.num_actions, self.num_actions]
        if sum(feature_dims) != self.num_single_obs:
            raise ValueError(
                "Unsupported single_obs layout: expected 3+3+3+3*num_actions to match num_single_obs."
            )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute deterministic actions from an explicit observation history.

        Args:
            obs: Term-wise flattened observation history with shape ``[1, num_actor_obs]``.

        Returns:
            Deterministic policy action tensor.
        """
        actor_output = self.actor(self.actor_obs_normalizer(obs))
        if self.state_dependent_std:
            return actor_output[..., 0, :]
        return actor_output

    def export(self, path: str, filename: str) -> None:
        """Save this wrapper as an ONNX model.

        Args:
            path: Directory in which to save the model.
            filename: Exported ONNX filename.
        """
        os.makedirs(path, exist_ok=True)
        export_path = os.path.join(path, filename)
        self.eval()
        self.to("cpu")
        torch.onnx.export(
            self,
            torch.zeros(1, self.num_actor_obs),
            export_path,
            export_params=True,
            opset_version=18,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )
