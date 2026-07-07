"""Ray-caster camera with randomized output delay."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCameraCfg
from isaaclab.sensors.ray_caster.ray_caster_camera import RayCasterCamera
from isaaclab.utils import configclass


class DelayRayCasterCamera(RayCasterCamera):
    """RayCasterCamera that returns a delayed image from a per-env history buffer.

    The underlying ray-cast is still computed by IsaacLab's official
    :class:`RayCasterCamera`. This class only samples an integer sensor-frame
    delay in ``[0, max_delay]`` for each environment on reset and exposes the
    corresponding historical image through ``data.output``.
    """

    cfg: "DelayRayCasterCameraCfg"

    def _initialize_rays_impl(self):
        super()._initialize_rays_impl()
        if self._uses_random_intrinsics():
            self._randomize_intrinsics(self._ALL_INDICES)

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        if not hasattr(self, "_ALL_INDICES"):
            return

        env_ids = self._resolve_env_ids(env_ids)
        if self.cfg.randomize_intrinsics_on_reset and self._uses_random_intrinsics():
            self._randomize_intrinsics(env_ids)
        if not hasattr(self, "_delay_steps"):
            return

        self._delay_steps[env_ids] = self._sample_delay_steps(len(env_ids))
        self._delay_write_index[env_ids] = 0
        self._delay_history_initialized[env_ids] = False

    def _create_buffers(self):
        super()._create_buffers()

        self._raw_output = {name: torch.zeros_like(value) for name, value in self._data.output.items()}
        sensor_dt = self.cfg.update_period if self.cfg.update_period > 0.0 else self._sim_physics_dt
        self._delay_sensor_dt = sensor_dt
        self._max_delay_steps = int(math.floor(self.cfg.max_delay / sensor_dt + 1e-9))
        self._delay_history_length = self._max_delay_steps + 1
        self._delay_write_index = torch.zeros(self._view.count, dtype=torch.long, device=self._device)
        self._delay_steps = self._sample_delay_steps(self._view.count)
        self._delay_history_initialized = torch.zeros(self._view.count, dtype=torch.bool, device=self._device)
        self._delay_history = {
            name: torch.zeros(
                self._delay_history_length,
                self._view.count,
                *value.shape[1:],
                device=self._device,
                dtype=value.dtype,
            )
            for name, value in self._data.output.items()
        }

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        super()._update_buffers_impl(env_ids)
        env_ids = self._resolve_env_ids(env_ids)
        write_index = self._delay_write_index[env_ids]
        read_index = (write_index - self._delay_steps[env_ids]) % self._delay_history_length
        uninitialized_mask = ~self._delay_history_initialized[env_ids]

        for name, history in self._delay_history.items():
            current = self._data.output[name][env_ids].clone()
            self._raw_output[name][env_ids] = current
            if uninitialized_mask.any():
                init_env_ids = env_ids[uninitialized_mask]
                init_current = current[uninitialized_mask].unsqueeze(0).expand(
                    self._delay_history_length,
                    -1,
                    *current.shape[1:],
                )
                history[:, init_env_ids] = init_current
            history[write_index, env_ids] = current
            self._data.output[name][env_ids] = history[read_index, env_ids]

        self._delay_history_initialized[env_ids] = True
        self._delay_write_index[env_ids] = (write_index + 1) % self._delay_history_length

    def _resolve_env_ids(self, env_ids: Sequence[int] | None) -> torch.Tensor:
        if env_ids is None:
            return self._ALL_INDICES
        if isinstance(env_ids, slice):
            return self._ALL_INDICES[env_ids]
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self._device, dtype=torch.long)
        return torch.tensor(env_ids, device=self._device, dtype=torch.long)

    def _sample_delay_steps(self, num_envs: int) -> torch.Tensor:
        if self._max_delay_steps <= 0:
            return torch.zeros(num_envs, dtype=torch.long, device=self._device)
        return torch.randint(0, self._max_delay_steps + 1, (num_envs,), device=self._device)

    def get_output(self, data_type: str, use_delay: bool = True) -> torch.Tensor:
        """Return delayed or raw output for an enabled ray-caster camera data type."""
        data = self.data
        if use_delay or not hasattr(self, "_raw_output"):
            return data.output[data_type]
        return self._raw_output[data_type]

    def _uses_random_intrinsics(self) -> bool:
        return self.cfg.horizontal_fov_range is not None or self.cfg.vertical_fov_range is not None

    def _randomize_intrinsics(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        env_ids = self._resolve_env_ids(env_ids)
        intrinsic_matrices = self._sample_intrinsic_matrices(len(env_ids))
        self.set_intrinsic_matrices(intrinsic_matrices, focal_length=1.0, env_ids=env_ids)

    def _sample_intrinsic_matrices(self, num_envs: int) -> torch.Tensor:
        horizontal_fov_range = self.cfg.horizontal_fov_range
        vertical_fov_range = self.cfg.vertical_fov_range
        if horizontal_fov_range is None or vertical_fov_range is None:
            raise ValueError("Both horizontal_fov_range and vertical_fov_range must be set for FOV randomization.")

        horizontal_fov = torch.empty(num_envs, device=self._device).uniform_(*horizontal_fov_range)
        vertical_fov = torch.empty(num_envs, device=self._device).uniform_(*vertical_fov_range)
        intrinsic_width = float(self.cfg.intrinsic_width or self.cfg.pattern_cfg.width)
        intrinsic_height = float(self.cfg.intrinsic_height or self.cfg.pattern_cfg.height)
        crop_width = float(self.cfg.pattern_cfg.width)
        crop_height = float(self.cfg.pattern_cfg.height)

        fx = intrinsic_width / (2.0 * torch.tan(torch.deg2rad(horizontal_fov) * 0.5))
        fy = intrinsic_height / (2.0 * torch.tan(torch.deg2rad(vertical_fov) * 0.5))
        cx = crop_width * 0.5
        cy = crop_height * 0.5
        if self.cfg.principal_point_jitter > 0.0:
            cx = (crop_width + torch.empty(num_envs, device=self._device).uniform_(
                -self.cfg.principal_point_jitter,
                self.cfg.principal_point_jitter,
            )) * 0.5
            cy = (crop_height + torch.empty(num_envs, device=self._device).uniform_(
                -self.cfg.principal_point_jitter,
                self.cfg.principal_point_jitter,
            )) * 0.5
        else:
            cx = torch.full((num_envs,), cx, device=self._device)
            cy = torch.full((num_envs,), cy, device=self._device)

        intrinsic_matrices = torch.zeros(num_envs, 3, 3, device=self._device)
        intrinsic_matrices[:, 0, 0] = fx
        intrinsic_matrices[:, 0, 2] = cx
        intrinsic_matrices[:, 1, 1] = fy
        intrinsic_matrices[:, 1, 2] = cy
        intrinsic_matrices[:, 2, 2] = 1.0
        return intrinsic_matrices


@configclass
class DelayRayCasterCameraCfg(RayCasterCameraCfg):
    """Configuration for :class:`DelayRayCasterCamera`.

    Attributes:
        max_delay: Maximum randomized image delay in seconds. The sampled delay
            is quantized to the sensor update period and resampled per
            environment on reset.
    """

    class_type: type = DelayRayCasterCamera

    max_delay: float = 0.0
    """Maximum randomized image delay in seconds."""

    horizontal_fov_range: tuple[float, float] | None = None
    """Horizontal FOV randomization range in degrees."""

    vertical_fov_range: tuple[float, float] | None = None
    """Vertical FOV randomization range in degrees."""

    intrinsic_width: int | None = None
    """Image width used to compute fx before center-crop adjustment."""

    intrinsic_height: int | None = None
    """Image height used to compute fy before center-crop adjustment."""

    principal_point_jitter: float = 0.0
    """Jitter used as ``(crop_size +/- jitter) / 2`` for the crop-adjusted principal point."""

    randomize_intrinsics_on_reset: bool = False
    """Whether to resample camera intrinsics every reset instead of only at initialization."""

    def __post_init__(self):
        super().__post_init__()
        if self.max_delay < 0.0:
            raise ValueError(f"max_delay must be non-negative, got {self.max_delay}.")
        if (self.horizontal_fov_range is None) != (self.vertical_fov_range is None):
            raise ValueError("horizontal_fov_range and vertical_fov_range must be set together.")
        if self.horizontal_fov_range is not None:
            if self.horizontal_fov_range[0] <= 0.0 or self.horizontal_fov_range[1] <= self.horizontal_fov_range[0]:
                raise ValueError(f"Invalid horizontal_fov_range: {self.horizontal_fov_range}.")
            if self.vertical_fov_range[0] <= 0.0 or self.vertical_fov_range[1] <= self.vertical_fov_range[0]:
                raise ValueError(f"Invalid vertical_fov_range: {self.vertical_fov_range}.")
        if self.principal_point_jitter < 0.0:
            raise ValueError(f"principal_point_jitter must be non-negative, got {self.principal_point_jitter}.")


DelayRayCaster = DelayRayCasterCamera
DelayRayCasterCfg = DelayRayCasterCameraCfg


def add_depth_noise(
    depth: torch.Tensor,
    max_depth: float,
    noise_std: float = 0.0,
    dropout_prob: float = 0.0,
) -> torch.Tensor:
    """Apply simple synthetic noise to ray-caster depth images."""
    if noise_std > 0.0:
        depth = depth + torch.randn_like(depth) * noise_std
    if dropout_prob > 0.0:
        depth = depth.masked_fill(torch.rand_like(depth) < dropout_prob, max_depth)
    return depth.clamp_(0.0, max_depth)


def process_depth_image(
    env,
    sensor_cfg: SceneEntityCfg,
    data_type: str = "distance_to_image_plane",
    image_shape: tuple[int, int] | None = None,
    max_depth: float = 10.0,
    normalize: bool = True,
    use_delay: bool = True,
    enable_noise: bool = False,
    enable_augmentation: bool | None = None,
    noise_std: float = 0.0,
    dropout_prob: float = 0.0,
) -> torch.Tensor:
    """Read ray-caster depth, optionally delayed/noisy, and flatten it."""
    camera = env.scene.sensors[sensor_cfg.name]
    if hasattr(camera, "get_output"):
        depth = camera.get_output(data_type, use_delay=use_delay).float()
    else:
        depth = camera.data.output[data_type].float()

    if depth.ndim == 4:
        if depth.shape[-1] == 1:
            depth = depth.squeeze(-1)
        elif depth.shape[1] == 1:
            depth = depth.squeeze(1)
    elif depth.ndim == 2 and image_shape is not None and depth.shape[-1] == image_shape[0] * image_shape[1]:
        depth = depth.reshape(depth.shape[0], *image_shape)
    if depth.ndim != 3:
        raise ValueError(f"Expected depth image with shape [B,H,W], [B,H,W,1], or flat [B,H*W], got {tuple(depth.shape)}")

    depth = torch.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=0.0).clamp_(0.0, max_depth)

    if image_shape is not None and tuple(depth.shape[-2:]) != tuple(image_shape):
        raise ValueError(f"Expected depth image shape {image_shape}, got {tuple(depth.shape[-2:])}.")

    if enable_augmentation is not None:
        enable_noise = enable_augmentation
    if enable_noise:
        depth = add_depth_noise(depth, max_depth=max_depth, noise_std=noise_std, dropout_prob=dropout_prob)

    if normalize:
        depth = depth / max_depth

    return depth.flatten(1)
