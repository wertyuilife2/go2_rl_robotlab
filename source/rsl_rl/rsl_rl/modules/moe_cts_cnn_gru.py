# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, NoReturn

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.modules.actor_critic_moe_cts import ActorCriticMoECTS, StudentMoEEncoder
from rsl_rl.networks import EmpiricalNormalization, L2Norm, SimNorm
from rsl_rl.networks.moe import MLP
from rsl_rl.utils import unpad_trajectories


class DepthCNNGRUEncoder(nn.Module):
    """CNN + GRU encoder for flattened depth images."""

    def __init__(
        self,
        image_shape: tuple[int, int] = (60, 60),
        in_channels: int = 1,
        cnn_channels: tuple[int, int, int] = (16, 32, 64),
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        pooled_shape: tuple[int, int] = (15, 15),
        gru_hidden_dim: int = 225,
        gru_num_layers: int = 1,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.image_shape = tuple(image_shape)
        self.in_channels = in_channels
        self.pooled_shape = tuple(pooled_shape)
        self.cnn_feature_dim = self.pooled_shape[0] * self.pooled_shape[1]
        self.hidden_dim = gru_hidden_dim
        self.num_layers = gru_num_layers

        act = nn.ELU if activation == "elu" else nn.ReLU
        layers: list[nn.Module] = []
        last_channels = in_channels
        for channels in cnn_channels:
            layers.extend(
                [
                    nn.Conv2d(
                        last_channels,
                        channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    act(),
                ]
            )
            last_channels = channels
        self.cnn = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(self.pooled_shape)
        self.gru = nn.GRU(self.cnn_feature_dim, gru_hidden_dim, gru_num_layers)
        self.hidden_state: torch.Tensor | None = None

    def reset(self, dones: torch.Tensor | None = None) -> None:
        if dones is None:
            self.hidden_state = None
        elif self.hidden_state is not None:
            self.hidden_state[:, dones == 1, :] = 0.0

    def _format_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 2:
            image = image.reshape(image.shape[0], self.in_channels, *self.image_shape)
        elif image.ndim == 3:
            image = image.reshape(image.shape[0] * image.shape[1], self.in_channels, *self.image_shape)
        elif image.ndim == 4:
            pass
        elif image.ndim == 5:
            image = image.reshape(image.shape[0] * image.shape[1], *image.shape[2:])
        else:
            raise ValueError(f"Unsupported depth image shape: {tuple(image.shape)}")
        return image

    def _encode_cnn(self, image: torch.Tensor, leading_shape: tuple[int, ...]) -> torch.Tensor:
        x = self._format_image(image)
        x = self.cnn(x)
        x = self.avgpool(x)
        x = x.mean(dim=1).flatten(1)
        return x.reshape(*leading_shape, self.cnn_feature_dim)

    def forward(
        self,
        image: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
        update_hidden: bool = False,
    ) -> torch.Tensor:
        if masks is not None:
            leading_shape = image.shape[:2]
            features = self._encode_cnn(image, leading_shape)
            out, _ = self.gru(features, hidden_state)
            return unpad_trajectories(out, masks)

        if image.ndim in (3, 5):
            leading_shape = image.shape[:2]
            features = self._encode_cnn(image, leading_shape)
            out, next_hidden_state = self.gru(features, hidden_state if hidden_state is not None else self.hidden_state)
            if update_hidden:
                self.hidden_state = next_hidden_state.detach()
            return out

        leading_shape = (image.shape[0],)
        features = self._encode_cnn(image, leading_shape).unsqueeze(0)
        out, next_hidden_state = self.gru(features, hidden_state if hidden_state is not None else self.hidden_state)
        if update_hidden:
            self.hidden_state = next_hidden_state.detach()
        return out.squeeze(0)


class ActorCriticMoECTSCNNGRU(ActorCriticMoECTS):
    """MoE CTS actor-critic with a depth CNN-GRU encoder for the student."""

    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = (512, 256, 128),
        critic_hidden_dims: tuple[int] | list[int] = (512, 256, 128),
        teacher_encoder_hidden_dims: tuple[int] | list[int] = (512, 256),
        student_encoder_hidden_dims: tuple[int] | list[int] = (512, 256, 256),
        expert_num: int = 8,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        latent_dim: int = 32,
        norm_type: str = "l2norm",
        actor_image_obs_groups: Iterable[str] | None = None,
        image_shape: tuple[int, int] = (60, 60),
        cnn_channels: tuple[int, int, int] = (16, 32, 64),
        cnn_kernel_size: int = 3,
        cnn_stride: int = 2,
        cnn_padding: int = 1,
        cnn_pooled_shape: tuple[int, int] = (15, 15),
        gru_hidden_dim: int = 225,
        gru_num_layers: int = 1,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticMoECTSCNNGRU.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        assert norm_type in ["l2norm", "simnorm"], f"Normalization type {norm_type} not supported!"
        assert "policy" in obs.keys() and "critic" in obs.keys() and "single_obs" in obs.keys(), (
            "obs must contain 'policy', 'critic' and 'single_obs' keys for ActorCriticMoECTSCNNGRU."
        )
        nn.Module.__init__(self)

        self.num_actions = num_actions
        self.obs_groups = obs_groups
        self.actor_image_obs_groups = list(actor_image_obs_groups or self._infer_image_groups(obs, obs_groups["policy"]))
        if not self.actor_image_obs_groups:
            raise ValueError("ActorCriticMoECTSCNNGRU requires at least one actor image observation group.")

        self.actor_obs_groups_1d = [group for group in obs_groups["policy"] if group not in self.actor_image_obs_groups]
        self.critic_obs_groups_1d = list(obs_groups["critic"])
        self.num_actor_obs = sum(obs[group].shape[-1] for group in self.actor_obs_groups_1d)
        self.num_critic_obs = sum(obs[group].shape[-1] for group in self.critic_obs_groups_1d)
        self.num_single_obs = obs["single_obs"].shape[-1]
        self.image_shape = tuple(image_shape)

        self.student_cnn_gru = DepthCNNGRUEncoder(
            image_shape=self.image_shape,
            in_channels=self._image_channels(obs, self.actor_image_obs_groups),
            cnn_channels=tuple(cnn_channels),
            kernel_size=cnn_kernel_size,
            stride=cnn_stride,
            padding=cnn_padding,
            pooled_shape=tuple(cnn_pooled_shape),
            gru_hidden_dim=gru_hidden_dim,
            gru_num_layers=gru_num_layers,
            activation=activation,
        )

        self.teacher_encoder = nn.Sequential(
            MLP(self.num_critic_obs, latent_dim, list(teacher_encoder_hidden_dims), activation=activation),
            L2Norm() if norm_type == "l2norm" else SimNorm(),
        )
        self.student_moe_encoder = StudentMoEEncoder(
            expert_num=expert_num,
            input_dim=self.num_actor_obs + gru_hidden_dim,
            hidden_dims=list(student_encoder_hidden_dims),
            output_dim=latent_dim,
            activation=activation,
            norm_type=norm_type,
        )
        print(f"Student CNN-GRU: {self.student_cnn_gru}")
        print(f"Teacher Encoder: {self.teacher_encoder}")
        print(f"Student MoE Encoder: {self.student_moe_encoder}")

        self.state_dependent_std = state_dependent_std
        actor_input_dim = latent_dim + self.num_single_obs
        if self.state_dependent_std:
            self.actor = MLP(actor_input_dim, [2, num_actions], list(actor_hidden_dims), activation)
        else:
            self.actor = MLP(actor_input_dim, num_actions, list(actor_hidden_dims), activation)
        print(f"Actor MLP: {self.actor}")

        self.critic = MLP(latent_dim + self.num_critic_obs, 1, list(critic_hidden_dims), activation)
        print(f"Critic MLP: {self.critic}")

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(self.num_actor_obs)
            self.single_obs_normalizer = EmpiricalNormalization(self.num_single_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
            self.single_obs_normalizer = torch.nn.Identity()

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(self.num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def _infer_image_groups(obs: TensorDict, groups: list[str]) -> list[str]:
        inferred = []
        for group in groups:
            value = obs[group]
            if len(value.shape) == 4:
                inferred.append(group)
            elif len(value.shape) == 2:
                side = math.isqrt(value.shape[-1])
                if side * side == value.shape[-1] and ("depth" in group or "image" in group):
                    inferred.append(group)
        return inferred

    @staticmethod
    def _image_channels(obs: TensorDict, groups: list[str]) -> int:
        channels = 0
        for group in groups:
            value = obs[group]
            channels += value.shape[1] if len(value.shape) == 4 else 1
        return channels

    def _image_obs(self, obs: TensorDict, groups: list[str]) -> torch.Tensor:
        images = []
        for group in groups:
            image = obs[group]
            if image.ndim in (2, 3):
                images.append(image)
            elif image.ndim == 4:
                images.append(image.flatten(1))
            elif image.ndim == 5:
                images.append(image.flatten(2))
            else:
                raise ValueError(f"Unsupported image observation shape for {group}: {tuple(image.shape)}")
        return torch.cat(images, dim=-1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.student_cnn_gru.reset(dones)

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, latent_and_obs: torch.Tensor) -> None:
        if self.state_dependent_std:
            mean_and_std = self.actor(latent_and_obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            mean = self.actor(latent_and_obs)
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.actor_obs_groups_1d], dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.critic_obs_groups_1d], dim=-1)

    def _normalize_actor_obs(self, obs: TensorDict, masks: torch.Tensor | None = None) -> torch.Tensor:
        obs_a = self.actor_obs_normalizer(self.get_actor_obs(obs))
        if masks is not None:
            obs_a = unpad_trajectories(obs_a, masks)
        return obs_a

    def _normalize_critic_obs(self, obs: TensorDict, masks: torch.Tensor | None = None) -> torch.Tensor:
        obs_c = self.critic_obs_normalizer(self.get_critic_obs(obs))
        if masks is not None:
            obs_c = unpad_trajectories(obs_c, masks)
        return obs_c

    def student_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
        update_memory: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_a = self._normalize_actor_obs(obs, masks)
        image = self._image_obs(obs, self.actor_image_obs_groups)
        image_feature = self.student_cnn_gru(image, masks=masks, hidden_state=hidden_state, update_hidden=update_memory)
        moe_input = torch.cat([obs_a, image_feature], dim=-1)
        if moe_input.ndim > 2:
            leading_shape = moe_input.shape[:-1]
            latent, weights = self.student_moe_encoder(moe_input.reshape(-1, moe_input.shape[-1]))
            return latent.reshape(*leading_shape, latent.shape[-1]), weights.reshape(*leading_shape, weights.shape[-1])
        return self.student_moe_encoder(moe_input)

    def teacher_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
        update_memory: bool = False,
    ) -> torch.Tensor:
        obs_c = self._normalize_critic_obs(obs, masks)
        return self.teacher_encoder(obs_c)

    def act(
        self,
        obs: TensorDict,
        is_teacher: bool,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
        **kwargs: dict[str, Any],
    ) -> torch.Tensor:
        single_obs = self.single_obs_normalizer(obs["single_obs"])
        if masks is not None:
            single_obs = unpad_trajectories(single_obs, masks)
        if is_teacher:
            latent = self.teacher_latent(obs, masks=masks, hidden_state=hidden_state, update_memory=masks is None)
        else:
            with torch.no_grad():
                latent, _ = self.student_latent(obs, masks=masks, hidden_state=hidden_state, update_memory=masks is None)
        self._update_distribution(torch.cat([latent, single_obs], dim=-1))
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        single_obs = self.single_obs_normalizer(obs["single_obs"])
        latent, _ = self.student_latent(obs, update_memory=True)
        latent_and_obs = torch.cat([latent, single_obs], dim=-1)
        if self.state_dependent_std:
            return self.actor(latent_and_obs)[..., 0, :]
        return self.actor(latent_and_obs)

    def act_and_evaluate_cts(
        self,
        obs: TensorDict,
        teacher_env_idxs: torch.Tensor,
        student_env_idxs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        single_obs = self.single_obs_normalizer(obs["single_obs"])
        teacher_latent = self.teacher_latent(obs, update_memory=True)
        with torch.no_grad():
            student_latent, _ = self.student_latent(obs, update_memory=True)

        latent = torch.empty_like(teacher_latent)
        latent[teacher_env_idxs] = teacher_latent[teacher_env_idxs]
        latent[student_env_idxs] = student_latent[student_env_idxs]
        self._update_distribution(torch.cat([latent, single_obs], dim=-1))
        actions = self.distribution.sample()

        obs_c = self._normalize_critic_obs(obs)
        value_latent = latent.detach()
        values = self.critic(torch.cat([value_latent, obs_c], dim=-1))
        return actions, values, self.get_actions_log_prob(actions), self.action_mean, self.action_std

    def evaluate(
        self,
        obs: TensorDict,
        is_teacher: bool,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
        **kwargs: dict[str, Any],
    ) -> torch.Tensor:
        obs_c = self._normalize_critic_obs(obs, masks)
        if is_teacher:
            latent = self.teacher_latent(obs, masks=masks, hidden_state=hidden_state)
        else:
            latent, _ = self.student_latent(obs, masks=masks, hidden_state=hidden_state)
        return self.critic(torch.cat([latent.detach(), obs_c], dim=-1))

    def evaluate_cts(
        self,
        obs: TensorDict,
        teacher_env_idxs: torch.Tensor,
        student_env_idxs: torch.Tensor,
    ) -> torch.Tensor:
        teacher_latent = self.teacher_latent(obs)
        student_latent, _ = self.student_latent(obs)
        latent = torch.empty_like(teacher_latent)
        latent[teacher_env_idxs] = teacher_latent[teacher_env_idxs]
        latent[student_env_idxs] = student_latent[student_env_idxs]
        obs_c = self._normalize_critic_obs(obs)
        return self.critic(torch.cat([latent.detach(), obs_c], dim=-1))

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        student_hidden_state = (
            None if self.student_cnn_gru.hidden_state is None else self.student_cnn_gru.hidden_state.detach()
        )
        return student_hidden_state, None

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
            self.single_obs_normalizer.update(obs["single_obs"])
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def ppo_parameters(self) -> list[dict[str, Any]]:
        noise_params: list[nn.Parameter] = []
        if hasattr(self, "std"):
            noise_params.append(self.std)
        if hasattr(self, "log_std"):
            noise_params.append(self.log_std)
        return [
            {"params": self.teacher_encoder.parameters()},
            {"params": self.critic.parameters()},
            {"params": self.actor.parameters()},
            {"params": noise_params},
        ]

    def student_encoder_parameters(self):
        return list(self.student_cnn_gru.parameters()) + list(self.student_moe_encoder.parameters())

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True
