"""Export CTS policies for deployment with TorchScript or ONNX."""

import copy
import os

import torch


def export_cts_policy_as_jit(
    policy: object,
    actor_obs_normalizer: object | None,
    single_obs_normalizer: object | None,
    path: str,
    filename: str = "policy.pt",
) -> None:
    """Export a CTS policy as a stateful TorchScript module.

    Args:
        policy: CTS policy module to export.
        actor_obs_normalizer: Normalizer for stacked actor observations.
        single_obs_normalizer: Normalizer for the current observation frame.
        path: Directory where the exported file is saved.
        filename: Name of the exported TorchScript file.
    """
    policy_exporter = _TorchPolicyExporter(policy, actor_obs_normalizer, single_obs_normalizer)
    policy_exporter.export(path, filename)


def export_cts_policy_as_onnx(
    policy: object,
    path: str,
    actor_obs_normalizer: object | None = None,
    single_obs_normalizer: object | None = None,
    filename: str = "policy.onnx",
    verbose: bool = False,
) -> None:
    """Export a CTS policy as an ONNX model using stacked observations.

    Args:
        policy: CTS policy module to export.
        path: Directory where the exported file is saved.
        actor_obs_normalizer: Normalizer for stacked actor observations.
        single_obs_normalizer: Normalizer for the current observation frame.
        filename: Name of the exported ONNX file.
        verbose: Whether to print the ONNX export graph.
    """
    os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(policy, actor_obs_normalizer, single_obs_normalizer, verbose)
    policy_exporter.export(path, filename)


class _TorchPolicyExporter(torch.nn.Module):
    """Wrap a CTS policy for stateful single-frame TorchScript inference."""

    def __init__(self, policy, actor_obs_normalizer=None, single_obs_normalizer=None):
        """Initialize the TorchScript exporter.

        Args:
            policy: Source CTS policy module.
            actor_obs_normalizer: Normalizer for stacked actor observations.
            single_obs_normalizer: Normalizer for the current observation frame.
        """
        assert not policy.is_recurrent, "CTS policy should not be recurrent"
        super().__init__()

        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        self.student_moe_encoder = copy.deepcopy(policy.student_moe_encoder)
        self.state_dependent_std = policy.state_dependent_std
        self.num_actions = int(policy.num_actions)
        self.num_single_obs = int(policy.num_single_obs)
        self.num_actor_obs = int(policy.num_actor_obs)
        if self.num_actor_obs % self.num_single_obs != 0:
            raise ValueError(
                f"num_actor_obs ({self.num_actor_obs}) must be divisible by num_single_obs ({self.num_single_obs})."
            )
        self.history_len = self.num_actor_obs // self.num_single_obs
        self.feature_dims = [3, 3, 3, self.num_actions, self.num_actions, self.num_actions]
        if sum(self.feature_dims) != self.num_single_obs:
            raise ValueError(
                "Unsupported single_obs layout: expected 3+3+3+3*num_actions to match num_single_obs."
            )
        self.register_buffer("obs_history", torch.zeros(1, self.num_actor_obs, dtype=torch.float32))

        if actor_obs_normalizer:
            self.actor_obs_normalizer = copy.deepcopy(actor_obs_normalizer)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        if single_obs_normalizer:
            self.single_obs_normalizer = copy.deepcopy(single_obs_normalizer)
        else:
            self.single_obs_normalizer = torch.nn.Identity()

    def forward(self, single_obs: torch.Tensor):
        """Compute an action and append one observation to internal history.

        Args:
            single_obs: Current observation with shape ``[1, num_single_obs]``.

        Returns:
            Deterministic policy actions.
        """
        if single_obs.dim() == 1:
            single_obs = single_obs.unsqueeze(0)
        if single_obs.shape[-1] != self.num_single_obs:
            raise ValueError(
                f"Expected single_obs last dimension {self.num_single_obs}, got {single_obs.shape[-1]}."
            )
        if single_obs.shape[0] != 1:
            raise ValueError("TorchScript CTS deployment currently supports batch size 1 only.")

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

        single_obs = self.single_obs_normalizer(single_obs)
        obs_a = self.actor_obs_normalizer(self.obs_history)
        latent, _ = self.student_moe_encoder(obs_a)
        latent_and_obs = torch.cat([latent, single_obs], dim=-1)
        if self.state_dependent_std:
            return self.actor(latent_and_obs)[..., 0, :]
        return self.actor(latent_and_obs)

    @torch.jit.export
    def reset(self):
        """Clear the internal observation history."""
        self.obs_history.zero_()

    def export(self, path, filename):
        """Compile and save the TorchScript module.

        Args:
            path: Directory where the exported file is saved.
            filename: Name of the exported TorchScript file.
        """
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    """Wrap a CTS policy for stateless stacked-observation ONNX inference."""

    def __init__(self, policy, actor_obs_normalizer=None, single_obs_normalizer=None, verbose=False):
        """Initialize the ONNX exporter.

        Args:
            policy: Source CTS policy module.
            actor_obs_normalizer: Normalizer for stacked actor observations.
            single_obs_normalizer: Normalizer for the current observation frame.
            verbose: Whether to print the ONNX export graph.
        """
        assert not policy.is_recurrent, "CTS policy should not be recurrent"
        super().__init__()
        self.verbose = verbose

        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        self.student_moe_encoder = copy.deepcopy(policy.student_moe_encoder)
        self.num_actions = int(policy.num_actions)
        self.num_single_obs = int(policy.num_single_obs)
        self.num_actor_obs = int(policy.num_actor_obs)
        if self.num_actor_obs % self.num_single_obs != 0:
            raise ValueError(
                f"num_actor_obs ({self.num_actor_obs}) must be divisible by num_single_obs ({self.num_single_obs})."
            )
        self.history_len = self.num_actor_obs // self.num_single_obs
        self.feature_dims = [3, 3, 3, self.num_actions, self.num_actions, self.num_actions]
        if sum(self.feature_dims) != self.num_single_obs:
            raise ValueError(
                "Unsupported single_obs layout: expected 3+3+3+3*num_actions to match num_single_obs."
            )
        self.state_dependent_std = policy.state_dependent_std

        if actor_obs_normalizer:
            self.actor_obs_normalizer = copy.deepcopy(actor_obs_normalizer)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        if single_obs_normalizer:
            self.single_obs_normalizer = copy.deepcopy(single_obs_normalizer)
        else:
            self.single_obs_normalizer = torch.nn.Identity()

    def _extract_single_obs_from_history(self, history: torch.Tensor) -> torch.Tensor:
        """Extract each observation term's newest frame from stacked history.

        Args:
            history: Term-wise stacked history with shape ``[B, num_actor_obs]``.

        Returns:
            Current observations with shape ``[B, num_single_obs]``.
        """
        if history.dim() == 1:
            history = history.unsqueeze(0)
        if history.shape[-1] != self.num_actor_obs:
            raise ValueError(f"Expected history last dimension {self.num_actor_obs}, got {history.shape[-1]}.")

        single_obs_terms = []
        offset = 0
        for dim in self.feature_dims:
            end = offset + dim * self.history_len
            single_obs_terms.append(history[:, end - dim:end])
            offset = end
        return torch.cat(single_obs_terms, dim=-1)

    def forward(self, history: torch.Tensor):
        """Compute deterministic actions from stacked observation history.

        Args:
            history: Term-wise stacked history with shape ``[B, num_actor_obs]``.

        Returns:
            Deterministic policy actions.
        """
        if history.dim() == 1:
            history = history.unsqueeze(0)
        single_obs = self._extract_single_obs_from_history(history)
        single_obs = self.single_obs_normalizer(single_obs)
        obs_a = self.actor_obs_normalizer(history)
        latent, _ = self.student_moe_encoder(obs_a)
        latent_and_obs = torch.cat([latent, single_obs], dim=-1)
        if self.state_dependent_std:
            return self.actor(latent_and_obs)[..., 0, :]
        return self.actor(latent_and_obs)

    def export(self, path, filename):
        """Export and save the ONNX graph.

        Args:
            path: Directory where the exported file is saved.
            filename: Name of the exported ONNX file.
        """
        self.to("cpu")
        self.eval()
        torch.onnx.export(
            self,
            torch.zeros(1, self.num_actor_obs),
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )
