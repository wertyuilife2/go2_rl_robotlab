"""Symmetry observation and action augmentation for Go2."""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict


class Go2SymmetryMapper:
    """Mirror Go2 observations and actions across the robot sagittal plane."""

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

    def __init__(self, env) -> None:
        """Store the environment used for observation term metadata.

        Args:
            env: Vectorized IsaacLab environment or wrapper.
        """
        self.env = env

    @property
    def base_env(self):
        """Return the unwrapped environment that owns managers and configs.

        Returns:
            The underlying environment when a wrapper exposes ``unwrapped``.
        """
        return self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env

    def reverse_vector(self, value: torch.Tensor, signs: list[int]) -> torch.Tensor:
        """Flip coordinate axes for a vector observation term.

        The signs encode a left-right reflection: lateral components change sign,
        while forward or vertical components stay unchanged when appropriate.

        Args:
            value: Vector term to mirror.
            signs: Per-dimension sign multiplier.

        Returns:
            Mirrored vector term.
        """
        return value * value.new_tensor(signs)

    def reverse_joints(self, value: torch.Tensor) -> torch.Tensor:
        """Swap left and right legs and flip signed joint axes.

        The permutation exchanges FL with FR and RL with RR; the sign vector then
        flips abduction/adduction axes so joint states remain in mirrored action space.

        Args:
            value: Joint-ordered tensor with Go2's 12-DoF leg ordering.

        Returns:
            Mirrored joint tensor.
        """
        perm = torch.tensor(self.JOINT_PERM, device=value.device)
        signs = value.new_tensor(self.JOINT_SIGN)
        return value.index_select(-1, perm) * signs

    def permute_joints(self, value: torch.Tensor) -> torch.Tensor:
        """Swap left and right legs without changing signs.

        This is used for action standard deviations, where sign flips are invalid
        because distribution scales must remain non-negative.

        Args:
            value: Joint-ordered tensor with Go2's 12-DoF leg ordering.

        Returns:
            Joint tensor with mirrored leg order only.
        """
        perm = torch.tensor(self.JOINT_PERM, device=value.device)
        return value.index_select(-1, perm)

    def restore_history(self, value: torch.Tensor, cfg) -> torch.Tensor:
        """Restore a flattened history term before applying symmetry.

        Some observation terms are flattened from ``[history, dim]`` into one
        axis; symmetry must act on the term dimension rather than the flattened axis.

        Args:
            value: Observation term chunk.
            cfg: Observation term config with optional history metadata.

        Returns:
            Restored tensor when history is flattened, otherwise the input tensor.
        """
        history_length = getattr(cfg, "history_length", 0)
        if history_length > 0 and getattr(cfg, "flatten_history_dim", False):
            term_dim = value.shape[-1] // history_length
            return value.reshape(*value.shape[:-1], history_length, term_dim)
        return value

    def flatten_history(self, value: torch.Tensor, cfg) -> torch.Tensor:
        """Flatten a restored history term back to the manager layout.

        Args:
            value: Observation term after symmetry.
            cfg: Observation term config with optional history metadata.

        Returns:
            Flattened tensor when the term originally used flattened history.
        """
        history_length = getattr(cfg, "history_length", 0)
        if history_length > 0 and getattr(cfg, "flatten_history_dim", False):
            return value.reshape(*value.shape[:-2], -1)
        return value

    def reverse_feet(self, value: torch.Tensor) -> torch.Tensor:
        """Swap foot-ordered contact features between left and right legs.

        The foot order is mirrored as FL/FR/RL/RR to FR/FL/RR/RL while preserving
        any per-foot feature width packed before the final foot axis.

        Args:
            value: Tensor containing foot-ordered features.

        Returns:
            Foot tensor with left and right feet exchanged.
        """
        if value.shape[-1] % len(self.FOOT_PERM) != 0:
            return value
        perm = torch.tensor(self.FOOT_PERM, device=value.device)
        return value.reshape(*value.shape[:-1], -1, len(self.FOOT_PERM)).index_select(-1, perm).reshape_as(value)

    def height_scan_shape(self, cfg) -> tuple[int, int, str] | None:
        """Infer the 2D grid shape used by a height scan term.

        Args:
            cfg: Height scan observation term config.

        Returns:
            Grid width, height, and ordering when available.
        """
        sensor_cfg = cfg.params.get("sensor_cfg")
        scene_cfg = getattr(self.base_env.cfg, "scene", None)
        sensor_scene_cfg = getattr(scene_cfg, sensor_cfg.name, None) if sensor_cfg is not None else None
        pattern_cfg = getattr(sensor_scene_cfg, "pattern_cfg", None)
        if pattern_cfg is None:
            return None

        nx = int(round(pattern_cfg.size[0] / pattern_cfg.resolution)) + 1
        ny = int(round(pattern_cfg.size[1] / pattern_cfg.resolution)) + 1
        return nx, ny, pattern_cfg.ordering

    def reverse_height_scan(self, cfg, value: torch.Tensor) -> torch.Tensor:
        """Flip the height scan grid along the lateral axis.

        The flip dimension depends on the sensor pattern ordering, so the flat
        scan is restored to the configured grid before mirroring.

        Args:
            cfg: Height scan observation term config.
            value: Flattened height scan tensor.

        Returns:
            Mirrored height scan tensor.
        """
        shape = self.height_scan_shape(cfg)
        if shape is None:
            return value

        nx, ny, ordering = shape
        if value.shape[-1] != nx * ny:
            return value

        if ordering == "xy":
            return value.reshape(*value.shape[:-1], ny, nx).flip(-2).reshape_as(value)
        return value.reshape(*value.shape[:-1], nx, ny).flip(-1).reshape_as(value)

    def reverse_term(self, name: str, cfg, value: torch.Tensor) -> torch.Tensor:
        """Mirror one observation term according to its semantic name.

        Args:
            name: Observation term name from the observation manager.
            cfg: Observation term config.
            value: Observation term tensor.

        Returns:
            Mirrored term when a rule exists, otherwise the input tensor.
        """
        if name in self.VECTOR_SIGNS:
            return self.reverse_vector(value, self.VECTOR_SIGNS[name])
        if name in self.JOINT_TERMS:
            return self.reverse_joints(value)
        if name == "contact_force":
            return self.reverse_feet(value)
        if name == "height_scan":
            return self.reverse_height_scan(cfg, value)
        return value

    def reverse_obs_group(self, group_name: str, value: torch.Tensor) -> torch.Tensor:
        """Mirror a flattened observation group term by term.

        The observation manager supplies term names and widths, allowing symmetry
        to be applied to semantic chunks instead of raw concatenated vectors.

        Args:
            group_name: Observation group key, such as actor or critic.
            value: Flattened observation group tensor.

        Returns:
            Mirrored observation group tensor.
        """
        manager = self.base_env.observation_manager
        names = manager._group_obs_term_names[group_name]
        dims = manager._group_obs_term_dim[group_name]
        cfgs = manager._group_obs_term_cfgs[group_name]
        widths = [int(math.prod(dim)) for dim in dims]
        if sum(widths) != value.shape[-1]:
            return value

        chunks = torch.split(value, widths, dim=-1)
        reversed_chunks = []
        for name, cfg, chunk in zip(names, cfgs, chunks):
            term = self.restore_history(chunk, cfg)
            term = self.reverse_term(name, cfg, term)
            reversed_chunks.append(self.flatten_history(term, cfg))
        return torch.cat(reversed_chunks, dim=-1)

    def reverse_obs(self, obs: TensorDict) -> TensorDict:
        """Mirror every known observation group in a TensorDict.

        Args:
            obs: Observation TensorDict with group keys managed by IsaacLab.

        Returns:
            Cloned TensorDict with mirrored observation groups.
        """
        manager = self.base_env.observation_manager
        reversed_obs = obs.clone()
        for key, value in obs.items():
            if key in manager._group_obs_term_names:
                reversed_obs[key] = self.reverse_obs_group(key, value)
        return reversed_obs

    def data_augmentation(self, obs: TensorDict, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor]:
        """Return the original batch followed by its mirrored counterpart.

        Args:
            obs: Observation batch to mirror.
            actions: Action batch to mirror.

        Returns:
            Concatenated original and mirrored observations and actions.
        """
        return torch.cat([obs, self.reverse_obs(obs)], dim=0), torch.cat([actions, self.reverse_joints(actions)], dim=0)


class Go2Symmetry:
    """Go2 symmetry augmentation for PPO and MoE-CTS mini-batches."""

    def __init__(self, env) -> None:
        """Create the shared Go2 symmetry wrapper.

        Args:
            env: Vectorized IsaacLab environment or wrapper.
        """
        self.env = env
        self.mapper = Go2SymmetryMapper(env)
        self.num_aug = 2

    def repeat_batch(self, value: torch.Tensor) -> torch.Tensor:
        """Repeat a PPO tensor for its original and mirrored samples.

        Args:
            value: Tensor aligned with the original PPO mini-batch.

        Returns:
            Tensor ordered as ``[original, mirrored]``.
        """
        repeat_shape = (self.num_aug, *([1] * (value.ndim - 1)))
        return value.repeat(repeat_shape)

    def get_original_mask(self, teacher_samples: int, student_samples: int, device: torch.device) -> torch.Tensor:
        """Return a boolean mask for the original samples in an augmented batch.

        Args:
            teacher_samples: Number of teacher samples in the original mini-batch.
            student_samples: Number of student samples in the original mini-batch.
            device: Device for the returned mask.

        Returns:
            Boolean mask with ``True`` for original samples and ``False`` for mirrored samples.
        """
        total_samples = (teacher_samples + student_samples) * self.num_aug
        mask = torch.zeros(total_samples, dtype=torch.bool, device=device)
        mask[:teacher_samples] = True
        st = teacher_samples * self.num_aug
        mask[st : st + student_samples] = True
        return mask

    def repeat_segmented_batch(
        self,
        value: torch.Tensor,
        teacher_samples: int,
    ) -> torch.Tensor:
        """Repeat teacher and student segments without changing CTS ordering.

        Args:
            value: Tensor aligned with the original mini-batch.
            teacher_samples: Number of teacher samples in the original mini-batch.

        Returns:
            Tensor repeated as ``[teacher_aug, student_aug]``.
        """
        repeat_shape = (self.num_aug, *([1] * (value.ndim - 1)))
        teacher_batch = value[:teacher_samples].repeat(repeat_shape)
        student_batch = value[teacher_samples:].repeat(repeat_shape)
        return torch.cat([teacher_batch, student_batch], dim=0)

    def augment_segment(
        self,
        obs_batch: TensorDict,
        actions_batch: torch.Tensor,
        old_mu_batch: torch.Tensor,
        old_sigma_batch: torch.Tensor,
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Append mirrored samples to a contiguous action-data segment.

        Args:
            obs_batch: Observation segment.
            actions_batch: Action segment.
            old_mu_batch: Stored action mean for the segment.
            old_sigma_batch: Stored action standard deviation for the segment.

        Returns:
            Original and mirrored segment tensors.
        """
        sym_obs_batch, sym_actions_batch = self.mapper.data_augmentation(obs_batch, actions_batch)
        return (
            sym_obs_batch,
            sym_actions_batch,
            torch.cat([old_mu_batch, self.mapper.reverse_joints(old_mu_batch)], dim=0),
            torch.cat([old_sigma_batch, self.mapper.permute_joints(old_sigma_batch)], dim=0),
        )

    def augment_ppo_batch(
        self,
        obs_batch: TensorDict,
        actions_batch: torch.Tensor,
        target_values_batch: torch.Tensor,
        advantages_batch: torch.Tensor,
        returns_batch: torch.Tensor,
        old_actions_log_prob_batch: torch.Tensor,
        old_mu_batch: torch.Tensor,
        old_sigma_batch: torch.Tensor,
    ) -> tuple[
        TensorDict,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Append mirrored samples to one PPO mini-batch.

        Rollout targets are copied for the mirrored transitions. Stored action
        means are mirrored like actions, while standard deviations are only
        permuted to keep distribution scales non-negative.

        Args:
            obs_batch: Observation mini-batch.
            actions_batch: Action mini-batch.
            target_values_batch: Stored value targets.
            advantages_batch: Advantage estimates.
            returns_batch: Return targets.
            old_actions_log_prob_batch: Stored action log probabilities.
            old_mu_batch: Stored action mean from rollout collection.
            old_sigma_batch: Stored action standard deviation from rollout collection.

        Returns:
            PPO tensors ordered as original samples followed by mirrored samples.
        """
        obs_batch, actions_batch, old_mu_batch, old_sigma_batch = self.augment_segment(
            obs_batch,
            actions_batch,
            old_mu_batch,
            old_sigma_batch,
        )
        return (
            obs_batch,
            actions_batch,
            self.repeat_batch(target_values_batch),
            self.repeat_batch(advantages_batch),
            self.repeat_batch(returns_batch),
            self.repeat_batch(old_actions_log_prob_batch),
            old_mu_batch,
            old_sigma_batch,
        )

    def augment_moe_cts_batch(
        self,
        obs_batch: TensorDict,
        actions_batch: torch.Tensor,
        target_values_batch: torch.Tensor,
        advantages_batch: torch.Tensor,
        returns_batch: torch.Tensor,
        old_actions_log_prob_batch: torch.Tensor,
        old_mu_batch: torch.Tensor,
        old_sigma_batch: torch.Tensor,
        teacher_samples: int,
    ) -> tuple[
        TensorDict,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Append mirrored samples to one MoECTS mini-batch.

        Rollout targets are repeated for original and mirrored samples. Old
        action means are mirrored like actions, while old action standard
        deviations are only permuted so distribution scales stay non-negative.

        Args:
            obs_batch: Observation mini-batch.
            actions_batch: Action mini-batch.
            target_values_batch: Stored value targets.
            advantages_batch: Advantage estimates.
            returns_batch: Return targets.
            old_actions_log_prob_batch: Stored action log probabilities.
            old_mu_batch: Stored action mean from rollout collection.
            old_sigma_batch: Stored action standard deviation from rollout collection.
            teacher_samples: Number of teacher samples in the original mini-batch.

        Returns:
            Augmented batch tensors in segmented MoE-CTS order.
        """
        teacher_obs, teacher_actions, teacher_mu, teacher_sigma = self.augment_segment(
            obs_batch[:teacher_samples],
            actions_batch[:teacher_samples],
            old_mu_batch[:teacher_samples],
            old_sigma_batch[:teacher_samples],
        )
        student_obs, student_actions, student_mu, student_sigma = self.augment_segment(
            obs_batch[teacher_samples:],
            actions_batch[teacher_samples:],
            old_mu_batch[teacher_samples:],
            old_sigma_batch[teacher_samples:],
        )
        return (
            torch.cat([teacher_obs, student_obs], dim=0),
            torch.cat([teacher_actions, student_actions], dim=0),
            self.repeat_segmented_batch(target_values_batch, teacher_samples),
            self.repeat_segmented_batch(advantages_batch, teacher_samples),
            self.repeat_segmented_batch(returns_batch, teacher_samples),
            self.repeat_segmented_batch(old_actions_log_prob_batch, teacher_samples),
            torch.cat([teacher_mu, student_mu], dim=0),
            torch.cat([teacher_sigma, student_sigma], dim=0),
        )

    def augment_moe_cts_batch_generator(self, generator, teacher_samples: int):
        """Yield original-plus-mirrored MoECTS mini-batches from storage.

        Args:
            generator: Rollout mini-batch generator from storage.
            teacher_samples: Number of teacher samples in each original mini-batch.

        Returns:
            A generator over original or original-plus-mirrored MoECTS mini-batches.
        """
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            (
                obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
            ) = self.augment_moe_cts_batch(
                obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                teacher_samples,
            )
            yield (
                obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hidden_states_batch,
                masks_batch,
            )
