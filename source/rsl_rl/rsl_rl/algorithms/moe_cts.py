# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict
import itertools

from rsl_rl.modules import ActorCriticMoECTS
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorageCTS


class MoECTS:
    """Concurrent Teacher-Student algorithm (https://arxiv.org/abs/2405.10830) with MoE."""

    policy: ActorCriticMoECTS
    """The actor critic module."""

    def __init__(
        self,
        policy: ActorCriticMoECTS,
        storage: RolloutStorageCTS,
        num_envs: int,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        betas: tuple = (0.9, 0.999),
        weight_decay: float = 0.0,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        load_balance_coef: float = 0.01,
        learning_rate: float = 0.001,
        student_encoder_learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        teacher_env_ratio: float = 0.75,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        assert isinstance(policy, ActorCriticMoECTS), "Policy must be an instance of ActorCriticMoECTS."
        assert not policy.is_recurrent, "Recurrent policies are not supported yet for MoECTS."
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg:
            # Extract parameters used in ppo
            rnd_lr = rnd_cfg.pop("learning_rate", 1e-3)
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_lr)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            print("[WARNING] `symmetry_cfg` detected, but MoECTS does not currently support symmetry; the configuration will be ignored.")
        self.symmetry = None

        # PPO components
        self.policy = policy
        self.policy.to(self.device)

        # Create the optimizer
        params1 = [
            {"params": self.policy.teacher_encoder.parameters()},
            {"params": self.policy.critic.parameters()},
            {"params": self.policy.actor.parameters()},
            {"params": getattr(self.policy, 'std', getattr(self.policy, 'log_std', []))}
        ]
        self.optimizer = optim.Adam(params1, lr=learning_rate, betas=betas, weight_decay=weight_decay)
        self.optimizer_stu_enc = optim.Adam(self.policy.student_moe_encoder.parameters(), lr=student_encoder_learning_rate, betas=betas, weight_decay=weight_decay)

        # Add storage
        self.storage = storage
        self.transition = RolloutStorageCTS.Transition()

        # MoECTS & PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.load_balance_coef = load_balance_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        
        # Teacher-student environment split
        self.teacher_num_envs = max(int(num_envs * teacher_env_ratio), 1)
        self.student_num_envs = num_envs - self.teacher_num_envs
        student_env_ratio = 1 - teacher_env_ratio
        self.teacher_env_idxs = torch.tensor([i for i in range(num_envs) if i % int(1/student_env_ratio) != 0], device=self.device)
        self.student_env_idxs = torch.tensor([i for i in range(num_envs) if i % int(1/student_env_ratio) == 0], device=self.device)
        assert len(self.teacher_env_idxs) == self.teacher_num_envs, f"{len(self.teacher_env_idxs)=} != {self.teacher_num_envs=}"
        assert len(self.student_env_idxs) == self.student_num_envs, f"{len(self.student_env_idxs)=} != {self.student_num_envs=}"
        
    def act(self, obs: TensorDict) -> torch.Tensor:
        # Compute the actions and values
        def _get_results(obs, is_teacher):
            actions = self.policy.act(obs, is_teacher)
            return (
                actions.detach(),
                self.policy.evaluate(obs, is_teacher).detach(),
                self.policy.get_actions_log_prob(actions).detach(),
                self.policy.action_mean.detach(),
                self.policy.action_std.detach(),
            )
        ti, si = self.teacher_env_idxs, self.student_env_idxs
        teacher_results = _get_results(obs[ti], is_teacher=True)
        student_results = _get_results(obs[si], is_teacher=False)
        results = []
        for x1, x2 in zip(teacher_results, student_results):
            results.append(torch.cat([x1, x2], dim=0))
        self.transition.actions = results[0]
        self.transition.values = results[1]
        self.transition.actions_log_prob = results[2]
        self.transition.action_mean = results[3]
        self.transition.action_sigma = results[4]
                
        # Record observations before env.step()
        self.transition.observations = torch.cat([obs[ti], obs[si]], dim=0)
        
        # Reconstruct the actions in the original order
        reordered_actions = torch.zeros_like(self.transition.actions)
        reordered_actions[ti] = self.transition.actions[:self.teacher_num_envs]
        reordered_actions[si] = self.transition.actions[self.teacher_num_envs:]
        return reordered_actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        # Update the normalizers
        self.policy.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        ti, si = self.teacher_env_idxs, self.student_env_idxs
        rewards = rewards.clone()
        self.transition.rewards = torch.cat([rewards[ti], rewards[si]], dim=0)
        self.transition.dones = torch.cat([dones[ti], dones[si]], dim=0)

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            reordered_obs = torch.cat([obs[ti], obs[si]], dim=0)
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(reordered_obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            time_outs = extras["time_outs"].to(self.device)
            reordered_time_outs = torch.cat([time_outs[ti], time_outs[si]], dim=0)
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * reordered_time_outs.unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        st = self.storage
        # Compute value for the last step
        ti, si = self.teacher_env_idxs, self.student_env_idxs
        last_values = torch.cat([
            self.policy.evaluate(obs[ti], is_teacher=True).detach(),
            self.policy.evaluate(obs[si], is_teacher=False).detach(),
        ], dim=0)
        # Compute returns and advantages
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            # If we are at the last step, bootstrap the return value
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - st.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            st.returns[step] = advantage + st.values[step]
        # Compute the advantages
        st.advantages = st.returns - st.values
        # Normalize the advantages if per minibatch normalization is not used
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_latent_loss = 0
        mean_load_balance_loss = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None

        # Get mini batch generator
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        data = list(generator)

        # Iterate over batches
        teacher_samples = self.teacher_num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        student_samples = self.student_num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
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
        ) in data:
            original_batch_size = obs_batch.batch_size[0]

            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
 
            def _get_results(start, end, is_teacher):
                self.policy.act(obs_batch[start:end], is_teacher)
                actions_log_prob = self.policy.get_actions_log_prob(actions_batch[start:end])
                value = self.policy.evaluate(obs_batch[start:end], is_teacher)
                mu = self.policy.action_mean
                sigma = self.policy.action_std
                entropy = self.policy.entropy
                return actions_log_prob, value, mu, sigma, entropy
            teacher_results = _get_results(0, teacher_samples, is_teacher=True)
            student_results = _get_results(teacher_samples, teacher_samples + student_samples, is_teacher=False)
            results = []
            for x1, x2 in zip(teacher_results, student_results):
                results.append(torch.cat([x1, x2], dim=0))
            actions_log_prob_batch, value_batch, mu_batch, sigma_batch, entropy_batch = results

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_losses = torch.max(surrogate, surrogate_clipped)
            teacher_surrogate_loss = surrogate_losses[:teacher_samples].mean()
            student_surrogate_loss = surrogate_losses[teacher_samples:].mean()
            surrogate_loss = teacher_surrogate_loss + student_surrogate_loss

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # RND loss
            # TODO: Move this processing to inside RND module.
            if self.rnd:
                # Extract the rnd_state
                # TODO: Check if we still need torch no grad. It is just an affine transformation.
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                # Predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # Compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            params_to_clip = itertools.chain.from_iterable(g['params'] for g in self.optimizer.param_groups)
            nn.utils.clip_grad_norm_(params_to_clip, self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()

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
        ) in data:
            # Student encoder loss
            obs_a_batch = self.policy.get_actor_obs(obs_batch)
            obs_a_batch = self.policy.actor_obs_normalizer(obs_a_batch)
            student_latent, gating_weights = self.policy.student_moe_encoder(obs_a_batch[teacher_samples:])
            with torch.no_grad():
                obs_c_batch = self.policy.get_critic_obs(obs_batch)
                obs_c_batch = self.policy.critic_obs_normalizer(obs_c_batch)
                teacher_latent = self.policy.teacher_encoder(obs_c_batch[teacher_samples:])
            latent_loss = (teacher_latent - student_latent).pow(2).mean()

            # Load balance loss
            mean_usage = torch.mean(gating_weights, dim=0)
            target_usage = torch.full_like(mean_usage, 1.0 / gating_weights.shape[1])
            load_balance_loss = torch.mean((mean_usage - target_usage).pow(2))
            # load_balance_loss = torch.sum(mean_usage.pow(2)) * gating_weights.shape[1]  # Switch Transformer style
            student_loss = latent_loss + self.load_balance_coef * load_balance_loss
            
            self.optimizer_stu_enc.zero_grad()
            student_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.student_moe_encoder.parameters(), self.max_grad_norm)
            self.optimizer_stu_enc.step()

            mean_latent_loss += latent_loss.item()
            mean_load_balance_loss += load_balance_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_latent_loss /= num_updates
        mean_load_balance_loss /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates

        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "mean_latent_loss": mean_latent_loss,
            "mean_load_balance_loss": mean_load_balance_loss
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss

        return loss_dict

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel
