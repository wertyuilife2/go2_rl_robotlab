# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Generator
from tensordict import TensorDict

from rsl_rl.networks import HiddenState
from rsl_rl.utils import split_and_pad_trajectories
from functools import partial

class RolloutStorageCTS:
    """Storage for the data collected during a rollout.

    The rollout storage is populated by adding transitions during the rollout phase. It then returns a generator for
    learning, depending on the algorithm and the policy architecture.
    """

    class Transition:
        """Storage for a single state transition."""

        def __init__(self) -> None:
            self.observations: TensorDict | None = None
            self.actions: torch.Tensor | None = None
            self.privileged_actions: torch.Tensor | None = None
            self.rewards: torch.Tensor | None = None
            self.dones: torch.Tensor | None = None
            self.values: torch.Tensor | None = None
            self.actions_log_prob: torch.Tensor
            self.action_mean: torch.Tensor | None = None
            self.action_sigma: torch.Tensor | None = None
            self.hidden_states: tuple[HiddenState, HiddenState] = (None, None)

        def clear(self) -> None:
            self.__init__()

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        teacher_num_envs: int, 
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
        device: str = "cpu",
    ) -> None:
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.actions_shape = actions_shape
        self.teacher_num_envs = teacher_num_envs
        self.student_num_envs = num_envs - teacher_num_envs

        # Core
        self.observations = TensorDict(
            {key: torch.zeros(num_transitions_per_env, *value.shape, device=device) for key, value in obs.items()},
            batch_size=[num_transitions_per_env, num_envs],
            device=self.device,
        )

        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
        
        # For distillation
        if training_type == "distillation":
            self.privileged_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # For reinforcement learning
        if training_type == "rl":
            self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        # For RNN networks
        self.saved_hidden_state_a = None
        self.saved_hidden_state_c = None

        # Counter for the number of transitions stored
        self.step = 0

    def add_transition(self, transition: Transition) -> None:
        # Check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # For distillation
        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)

        # For reinforcement learning
        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)

        # For RNN networks
        self._save_hidden_states(transition.hidden_states)

        # Increment the counter
        self.step += 1

    def clear(self) -> None:
        self.step = 0

    # For distillation
    def generator(self) -> Generator:
        if self.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            yield self.observations[i], self.actions[i], self.privileged_actions[i], self.dones[i]

    # For reinforcement learning with feedforward networks
    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator:
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        
        # Prepare indices
        teacher_samples_num = self.teacher_num_envs * self.num_transitions_per_env
        student_samples_num = self.student_num_envs * self.num_transitions_per_env
        teacher_mini_batch_size = teacher_samples_num // num_mini_batches
        student_mini_batch_size = student_samples_num // num_mini_batches
        teacher_indices = torch.randperm(teacher_samples_num, requires_grad=False, device=self.device)
        student_indices = teacher_samples_num + torch.randperm(student_samples_num, requires_grad=False, device=self.device)
        
        # Core
        observations = self.observations.transpose(0, 1).flatten(0, 1)
        actions = self.actions.transpose(0, 1).flatten(0, 1)
        values = self.values.transpose(0, 1).flatten(0, 1)
        returns = self.returns.transpose(0, 1).flatten(0, 1)

        # For PPO
        old_actions_log_prob = self.actions_log_prob.transpose(0, 1).flatten(0, 1)
        advantages = self.advantages.transpose(0, 1).flatten(0, 1)
        old_mu = self.mu.transpose(0, 1).flatten(0, 1)
        old_sigma = self.sigma.transpose(0, 1).flatten(0, 1)
        
        def _get_teacher_student_samples(data, slice):
            (i1, i2), (j1, j2) = slice
            return torch.cat([data[teacher_indices[i1:i2]], data[student_indices[j1:j2]]], 0).detach()

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                slice = (
                    (i * teacher_mini_batch_size, (i+1) * teacher_mini_batch_size),
                    (i * student_mini_batch_size, (i+1) * student_mini_batch_size),
                )
                
                # Create the mini-batch
                get_batch = partial(_get_teacher_student_samples, slice=slice)
                obs_batch, actions_batch, target_values_batch, returns_batch, \
                old_actions_log_prob_batch, advantages_batch, old_mu_batch, \
                old_sigma_batch = map(get_batch, [
                    observations,
                    actions,
                    values,
                    returns,
                    old_actions_log_prob,
                    advantages,
                    old_mu,
                    old_sigma
                ])

                hidden_state_a_batch = None
                hidden_state_c_batch = None
                masks_batch = None

                # Yield the mini-batch
                yield (
                    obs_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    (
                        hidden_state_a_batch,
                        hidden_state_c_batch,
                    ),
                    masks_batch,
                )

    # For reinforcement learning with recurrent networks
    def recurrent_mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator:
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")

        teacher_mini_batch_size = self.teacher_num_envs // num_mini_batches
        student_mini_batch_size = self.student_num_envs // num_mini_batches
        if teacher_mini_batch_size == 0 or student_mini_batch_size == 0:
            raise ValueError(
                "CTS recurrent mini-batches require at least one teacher and one student environment per mini-batch."
            )

        def _cat_hidden_state(hidden_state_1, hidden_state_2):
            if hidden_state_1 is None and hidden_state_2 is None:
                return None
            if isinstance(hidden_state_1, tuple):
                return tuple(torch.cat([h1, h2], dim=1) for h1, h2 in zip(hidden_state_1, hidden_state_2))
            return torch.cat([hidden_state_1, hidden_state_2], dim=1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                teacher_start = i * teacher_mini_batch_size
                teacher_stop = (i + 1) * teacher_mini_batch_size
                student_start = self.teacher_num_envs + i * student_mini_batch_size
                student_stop = self.teacher_num_envs + (i + 1) * student_mini_batch_size

                if self.saved_hidden_state_c is None:
                    teacher_obs = self.observations[:, teacher_start:teacher_stop]
                    teacher_masks = torch.ones(
                        self.num_transitions_per_env,
                        teacher_mini_batch_size,
                        dtype=torch.bool,
                        device=self.device,
                    )
                    teacher_hidden_state_a = self._get_initial_hidden_state_segment(
                        self.saved_hidden_state_a, teacher_start, teacher_stop
                    )
                    teacher_hidden_state_c = None
                else:
                    teacher_obs, teacher_masks = split_and_pad_trajectories(
                        self.observations[:, teacher_start:teacher_stop],
                        self.dones[:, teacher_start:teacher_stop],
                    )
                    teacher_hidden_state_a = self._get_hidden_state_segment(
                        self.saved_hidden_state_a, teacher_start, teacher_stop
                    )
                    teacher_hidden_state_c = self._get_hidden_state_segment(
                        self.saved_hidden_state_c, teacher_start, teacher_stop
                    )
                student_obs, student_masks = split_and_pad_trajectories(
                    self.observations[:, student_start:student_stop],
                    self.dones[:, student_start:student_stop],
                )

                obs_batch = torch.cat([teacher_obs, student_obs], dim=1)
                masks_batch_tensor = torch.cat([teacher_masks, student_masks], dim=1)
                actions_batch = torch.cat(
                    [self.actions[:, teacher_start:teacher_stop], self.actions[:, student_start:student_stop]], dim=1
                )
                old_mu_batch = torch.cat(
                    [self.mu[:, teacher_start:teacher_stop], self.mu[:, student_start:student_stop]], dim=1
                )
                old_sigma_batch = torch.cat(
                    [self.sigma[:, teacher_start:teacher_stop], self.sigma[:, student_start:student_stop]], dim=1
                )
                returns_batch = torch.cat(
                    [self.returns[:, teacher_start:teacher_stop], self.returns[:, student_start:student_stop]], dim=1
                )
                advantages_batch = torch.cat(
                    [self.advantages[:, teacher_start:teacher_stop], self.advantages[:, student_start:student_stop]],
                    dim=1,
                )
                values_batch = torch.cat(
                    [self.values[:, teacher_start:teacher_stop], self.values[:, student_start:student_stop]], dim=1
                )
                old_actions_log_prob_batch = torch.cat(
                    [
                        self.actions_log_prob[:, teacher_start:teacher_stop],
                        self.actions_log_prob[:, student_start:student_stop],
                    ],
                    dim=1,
                )

                student_hidden_state_a = self._get_hidden_state_segment(
                    self.saved_hidden_state_a, student_start, student_stop
                )
                student_hidden_state_c = self._get_hidden_state_segment(
                    self.saved_hidden_state_c, student_start, student_stop
                )

                hidden_state_a_batch = _cat_hidden_state(teacher_hidden_state_a, student_hidden_state_a)
                hidden_state_c_batch = _cat_hidden_state(teacher_hidden_state_c, student_hidden_state_c)
                masks_batch = {
                    "masks": masks_batch_tensor,
                    "teacher_trajectories": teacher_masks.shape[1],
                    "teacher_envs": teacher_mini_batch_size,
                    "student_envs": student_mini_batch_size,
                }

                yield (
                    obs_batch,
                    actions_batch,
                    values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    (
                        hidden_state_a_batch,
                        hidden_state_c_batch,
                    ),
                    masks_batch,
                )

    def _get_hidden_state_segment(self, saved_hidden_state, env_start: int, env_stop: int):
        if saved_hidden_state is None:
            return None

        dones = self.dones[:, env_start:env_stop].squeeze(-1)
        last_was_done = torch.zeros_like(dones, dtype=torch.bool)
        last_was_done[1:] = dones[:-1].bool()
        last_was_done[0] = True
        last_was_done = last_was_done.permute(1, 0)

        hidden_state_batch = [
            saved_hidden_state_i[:, :, env_start:env_stop]
            .permute(2, 0, 1, 3)[last_was_done]
            .transpose(1, 0)
            .contiguous()
            for saved_hidden_state_i in saved_hidden_state
        ]
        return hidden_state_batch[0] if len(hidden_state_batch) == 1 else tuple(hidden_state_batch)

    def _get_initial_hidden_state_segment(self, saved_hidden_state, env_start: int, env_stop: int):
        if saved_hidden_state is None:
            return None

        hidden_state_batch = [
            saved_hidden_state_i[0, :, env_start:env_stop].contiguous() for saved_hidden_state_i in saved_hidden_state
        ]
        return hidden_state_batch[0] if len(hidden_state_batch) == 1 else tuple(hidden_state_batch)

    def _save_hidden_states(self, hidden_states: tuple[HiddenState, HiddenState]) -> None:
        if hidden_states == (None, None):
            return

        def as_tuple(hidden_state):
            if hidden_state is None:
                return None
            return hidden_state if isinstance(hidden_state, tuple) else (hidden_state,)

        hidden_state_a = as_tuple(hidden_states[0])
        hidden_state_c = as_tuple(hidden_states[1])
        if self.saved_hidden_state_a is None and hidden_state_a is not None:
            self.saved_hidden_state_a = [
                torch.zeros(self.observations.shape[0], *hidden_state_a[i].shape, device=self.device)
                for i in range(len(hidden_state_a))
            ]
        if self.saved_hidden_state_c is None and hidden_state_c is not None:
            self.saved_hidden_state_c = [
                torch.zeros(self.observations.shape[0], *hidden_state_c[i].shape, device=self.device)
                for i in range(len(hidden_state_c))
            ]
        if hidden_state_a is not None:
            for i in range(len(hidden_state_a)):
                self.saved_hidden_state_a[i][self.step].copy_(hidden_state_a[i])
        if hidden_state_c is not None:
            for i in range(len(hidden_state_c)):
                self.saved_hidden_state_c[i][self.step].copy_(hidden_state_c[i])
