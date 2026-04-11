# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time
import torch
import warnings
import yaml
import numpy as np
from tensordict import TensorDict

from rsl_rl.algorithms import MoECTS
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCriticMoECTS,
    resolve_rnd_config,
    resolve_symmetry_config,
)
from rsl_rl.storage import RolloutStorageCTS
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from rsl_rl.utils.logger_cts import LoggerCTS
from rsl_rl.utils.exporter_cts import export_cts_policy_as_jit


def numpy_representer(dumper: yaml.SafeDumper, data: np.floating) -> yaml.Node:
    return dumper.represent_float(float(data))


def numpy_int_representer(dumper: yaml.SafeDumper, data: np.integer) -> yaml.Node:
    return dumper.represent_int(int(data))


yaml.add_representer(np.float32, numpy_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(np.float64, numpy_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(np.int32, numpy_int_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(np.int64, numpy_int_representer, Dumper=yaml.SafeDumper)


class OnPolicyRunnerCTS:
    """On-policy runner for training and evaluation of actor-critic methods."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.cfg = train_cfg
        self.policy_cfg = train_cfg["policy"]
        self.alg_cfg = train_cfg["algorithm"]
        self.device = device
        self.env = env

        # Setup multi-GPU training if enabled
        self._configure_multi_gpu()

        # Query observations from environment for algorithm construction
        obs = self.env.get_observations()
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], self._get_default_obs_sets())

        # Create the algorithm
        self.alg = self._construct_algorithm(obs)
        
        # Create the logger
        self.logger = LoggerCTS(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            teacher_env_idxs=self.alg.teacher_env_idxs,
            device=self.device,
        )

        self.current_learning_iteration = 0

        # robogauge client
        try:
            robogauge_cfg = train_cfg.get("robogauge", {})
            if not robogauge_cfg.get("enabled", False):
                raise ImportError("config disabled")
            from robogauge.scripts.client import RoboGaugeClient

            self.robogauge_client = RoboGaugeClient(f"http://127.0.0.1:{robogauge_cfg.get('port', 9973)}")
        except Exception as e:
            print(f"[INFO] RoboGauge client could not be initialized: {e}, disabling RoboGauge interface.")
            self.robogauge_client = None

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    # Sample actions
                    actions = self.alg.act(obs)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg_cfg["rnd_cfg"] else None
                    # Book keeping
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Log information
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.policy.action_std,
                rnd_weight=self.alg.rnd.weight if self.alg_cfg["rnd_cfg"] else None,
            )

            # Save model
            if it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"), it=it, last_model=False)  # type: ignore

        # Save the final model after training
        if self.logger.log_dir is not None and not self.logger.disable_logs:
            self.save(
                os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"),
                it=self.current_learning_iteration,
                last_model=True,
            )

    def save(self, path: str, it: int, last_model: bool, infos: dict | None = None) -> None:
        # Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "optimizer_stu_enc_state_dict": self.alg.optimizer_stu_enc.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # Save RND model if used
        if self.alg_cfg["rnd_cfg"]:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            if self.alg.rnd_optimizer:
                saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging services
        self.logger.save_model(path, self.current_learning_iteration)
        self.update_robogauge(it, last_model)

    def update_robogauge(self, it: int, last_model: bool) -> None:
        if self.robogauge_client is None or self.logger.log_dir is None or self.logger.disable_logs:
            return

        try:
            if it % 500 == 0 or last_model:
                # export jit model
                jit_dir = os.path.join(self.logger.log_dir, "jit_models")
                jit_path = os.path.join(jit_dir, f"policy_jit_{it}.pt")
                export_cts_policy_as_jit(
                    self.alg.policy,
                    actor_obs_normalizer=self.alg.policy.actor_obs_normalizer,
                    single_obs_normalizer=self.alg.policy.single_obs_normalizer,
                    path=jit_dir,
                    filename=f"policy_jit_{it}.pt",
                )
                # upload to robogauge
                self.robogauge_client.submit_task(
                    model_path=jit_path,
                    step=it,
                    task_name="go2_lab",
                    experiment_name=self.cfg["experiment_name"],
                )
        except Exception as e:
            print(f"[WARN] RoboGauge submit failed at step {it}: {e}")
            return

        check_times = 1
        if last_model:
            check_times = int(1e9)  # keep checking until manually stopped

        while check_times > 0:
            check_times -= 1
            try:
                self.robogauge_client.monitor_tasks()
            except Exception as e:
                print(f"[WARN] RoboGauge monitor failed at step {it}: {e}")
                break

            results_dir = os.path.join(self.logger.log_dir, "robogauge_results")
            os.makedirs(results_dir, exist_ok=True)
            result_received = False

            for task_id, resp in self.robogauge_client.response_data.items():
                if not isinstance(resp, dict):
                    print(f"[WARN] RoboGauge returned an invalid response for task {task_id}: {resp}")
                    continue
                results = resp.get("results")
                step = resp.get("step", it)
                if results is None:
                    print(f"[WARN] RoboGauge returned empty results for task {task_id} at step {step}.")
                    continue
                scores = results.get("scores")
                if scores is None:
                    print(f"[WARN] RoboGauge results for task {task_id} at step {step} do not contain 'scores'.")
                    continue
                if step == it:
                    result_received = True
                if self.logger.writer is not None:
                    for key, val in scores.items():
                        self.logger.writer.add_scalar(f"RoboGauge/{key}", val, step)
                results_path = os.path.join(results_dir, f"results_{step}.yaml")
                with open(results_path, "w", encoding="utf-8") as f:
                    yaml.dump(results, f, allow_unicode=True, sort_keys=False)

            if last_model and result_received:
                print(f"RoboGauge result for step {it} received. Exiting wait loop.")
                break

            if check_times > 0:
                print("Sleeping for 1 minute before checking RoboGauge results again...")
                time.sleep(60)  # wait for 1 minute before checking again

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        # Load model
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        # Load RND model if used
        if self.alg_cfg["rnd_cfg"]:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # Load optimizer if used
        if load_optimizer and resumed_training:
            # Algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # Student encoder optimizer
            self.alg.optimizer_stu_enc.load_state_dict(loaded_dict["optimizer_stu_enc_state_dict"])
            # RND optimizer if used
            if self.alg_cfg["rnd_cfg"]:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        # Load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device: str | None = None) -> callable:
        self.eval_mode()  # Switch to evaluation mode (e.g. for dropout)
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference

    def train_mode(self) -> None:
        # PPO
        self.alg.policy.train()
        # RND
        if self.alg_cfg["rnd_cfg"]:
            self.alg.rnd.train()

    def eval_mode(self) -> None:
        # PPO
        self.alg.policy.eval()
        # RND
        if self.alg_cfg["rnd_cfg"]:
            self.alg.rnd.eval()

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.logger.git_status_repos.append(repo_file_path)

    def _get_default_obs_sets(self) -> list[str]:
        """Get the the default observation sets required for the algorithm.

        .. note::
            See :func:`resolve_obs_groups` for more details on the handling of observation sets.
        """
        default_sets = ["critic"]
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        return default_sets

    def _configure_multi_gpu(self) -> None:
        """Configure multi-gpu training."""
        # Check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # If not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # Get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # Make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # Rank of the main process
            "local_rank": self.gpu_local_rank,  # Rank of the current process
            "world_size": self.gpu_world_size,  # Total number of processes
        }

        # Check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # Validate multi-GPU configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # Initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # Set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)

    def _construct_algorithm(self, obs: TensorDict) -> MoECTS:
        """Construct the actor-critic algorithm."""
        # Resolve RND config if used
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)

        # Resolve symmetry config if used
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # Resolve deprecated normalization config
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # Initialize the policy
        # actor_critic_class = resolve_callable(self.policy_cfg.pop("class_name"))
        actor_critic_class = eval(self.policy_cfg.pop("class_name")) # temporally use eval to avoid import bugs
        actor_critic: ActorCriticMoECTS = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # Initialize the storage
        storage = RolloutStorageCTS(
            "rl", self.env.num_envs, max(int(self.env.num_envs*self.alg_cfg["teacher_env_ratio"]), 1), self.cfg["num_steps_per_env"], obs, [self.env.num_actions], self.device
        )

        # Initialize the algorithm
        # alg_class = resolve_callable(self.alg_cfg.pop("class_name"))
        alg_class = eval(self.alg_cfg.pop("class_name")) # temporally use eval to avoid import bugs
        alg: MoECTS = alg_class(
            actor_critic, storage, self.env.num_envs, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        return alg
