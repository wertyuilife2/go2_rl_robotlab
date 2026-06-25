"""
References: https://github.com/leggedrobotics/rsl_rl/blob/v5.4.1/rsl_rl/extensions/symmetry.py
"""

from __future__ import annotations

import torch
from typing import Callable
from collections.abc import Generator

from rsl_rl.utils import resolve_callable
from rsl_rl.env.vec_env import VecEnv


class Symmetry:
    """Symmetry data augmentation and mirror loss.

    The extension supports two (optionally simultaneous) uses of a user-provided symmetry function:

    - :attr:`use_data_augmentation` appends mirrored observation/action pairs to every mini-batch, so that the policy
      and value loss are evaluated on both the original and the mirrored samples.
    - :attr:`use_mirror_loss` adds an auxiliary MSE term that penalizes the policy for disagreeing with itself when
      evaluated on mirrored observations.

    If both flags are disabled the symmetry loss is still computed for logging purposes but detached from the graph.

    References:
        - Mittal et al. "Symmetry Considerations for Learning Task Symmetric Robot Policies." ICRA (2024).
    """

    def __init__(
        self,
        env: VecEnv,
        data_augmentation_generator: str | Callable,
        use_data_augmentation: bool = False,
    ) -> None:
        """Initialize the symmetry extension.

        Args:
            data_augmentation_generator: Callable that generates mirrored observations / actions. Resolved using
                :func:`~rsl_rl.utils.utils.resolve_callable`.
            use_data_augmentation: Whether to append mirrored samples to every mini-batch.
        """
        # Environmenet
        self.env = env

        # Symmetry parameters
        self.use_data_augmentation = use_data_augmentation

        # Resolve the augmentation function
        self.data_augmentation_generator = resolve_callable(data_augmentation_generator)

    def augment_batch_generator(self, batch_generator: Generator) -> Generator:
        for obs_batch, actions_batch, *remain_batch in batch_generator:
            if not self.use_data_augmentation:
                yield obs_batch, actions_batch, *remain_batch
                continue

            for aug_obs_batch, aug_actions_batch in self.data_augmentation_generator(self.env, obs_batch, actions_batch):
                yield aug_obs_batch, aug_actions_batch, *remain_batch


def resolve_symmetry_config(alg_cfg: dict, env: VecEnv) -> dict:
    """Resolve the symmetry configuration.

    Args:
        alg_cfg: Algorithm configuration dictionary.
        env: Environment object.

    Returns:
        The resolved algorithm configuration dictionary.
    """
    # If using symmetry then pass the environment config object
    # Note: This is used by the symmetry function for handling different observation terms
    if "symmetry_cfg" in alg_cfg and alg_cfg["symmetry_cfg"] is not None:
        alg_cfg["symmetry_cfg"]["env"] = env
    else:
        alg_cfg["symmetry_cfg"] = None
    return alg_cfg
