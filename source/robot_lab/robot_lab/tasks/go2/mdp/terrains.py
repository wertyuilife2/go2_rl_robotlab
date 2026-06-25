from __future__ import annotations

import numpy as np

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.utils import configclass
from isaaclab.terrains.height_field import hf_terrains
from isaaclab.terrains.height_field.utils import height_field_to_mesh


class PerSubTerrainSlopeThresholdGenerator(TerrainGenerator):
    """Terrain generator that lets each height-field sub-terrain override slope_threshold."""

    def __init__(self, cfg, device: str = "cpu"):
        self._apply_sub_terrain_border_width(cfg)
        super().__init__(cfg, device)

    def _apply_sub_terrain_border_width(self, cfg):
        border_width = getattr(cfg, "sub_terrain_border_width", None)
        if border_width is None:
            return

        for sub_cfg in cfg.sub_terrains.values():
            if hasattr(sub_cfg, "border_width"):
                sub_cfg.border_width = border_width

    def _get_terrain_mesh(self, difficulty, cfg):
        difficulty = self._maybe_use_gym_difficulty(difficulty)

        override = getattr(cfg, "slope_threshold_override", None)
        if override is None:
            return super()._get_terrain_mesh(difficulty, cfg)

        original_slope_threshold = getattr(cfg, "slope_threshold", None)
        cfg.slope_threshold = override
        try:
            return super()._get_terrain_mesh(difficulty, cfg)
        finally:
            cfg.slope_threshold = original_slope_threshold

    def _maybe_use_gym_difficulty(self, difficulty: float) -> float:
        if not getattr(self.cfg, "use_gym_difficulty", False):
            return difficulty

        lower, upper = self.cfg.difficulty_range
        if upper > lower:
            normalized_difficulty = (float(difficulty) - lower) / (upper - lower)
        else:
            normalized_difficulty = float(difficulty)
        normalized_difficulty = np.clip(normalized_difficulty, 0.0, 1.0)

        num_levels = self.cfg.num_rows
        level = min(int(np.floor(normalized_difficulty * num_levels)), num_levels - 1)
        return level / num_levels


@configclass
class Go2TerrainGeneratorCfg(terrain_gen.TerrainGeneratorCfg):
    class_type: type = PerSubTerrainSlopeThresholdGenerator
    sub_terrain_border_width: float | None = None
    use_gym_difficulty: bool = False


def with_slope_threshold(sub_terrain_cfg, slope_threshold: float | None):
    """Attach a per-sub-terrain slope-threshold override to a terrain config."""
    sub_terrain_cfg.slope_threshold_override = slope_threshold
    return sub_terrain_cfg


# -----------------------------------------------------------------------------
# Default RobotLab terrain setup
# -----------------------------------------------------------------------------

DEFAULT_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    class_type=PerSubTerrainSlopeThresholdGenerator,
    size=(8.0, 8.0),
    border_width=25.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.45,
            grid_height_range=(0.025, 0.1),
            platform_width=2.0,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.5),
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.5),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)


# -----------------------------------------------------------------------------
# Go2 Terrain setup
# -----------------------------------------------------------------------------
@height_field_to_mesh
def wave_terrain(difficulty: float, cfg) -> np.ndarray:
    """wave terrain: wave plus random uniform roughness."""
    wave = hf_terrains.wave_terrain.__wrapped__(difficulty, cfg)
    rough = hf_terrains.random_uniform_terrain.__wrapped__(difficulty, cfg)
    return np.rint(wave + rough).astype(np.int16)


@height_field_to_mesh
def rough_slope_terrain(difficulty: float, cfg) -> np.ndarray:
    """rough slope terrain: slope plus random uniform roughness."""
    slope = hf_terrains.pyramid_sloped_terrain.__wrapped__(difficulty, cfg)
    rough = hf_terrains.random_uniform_terrain.__wrapped__(difficulty, cfg)
    return np.rint(slope + rough).astype(np.int16)


@configclass
class WaveTerrainCfg(terrain_gen.HfWaveTerrainCfg):
    function = wave_terrain
    amplitude_range: tuple[float, float] = (0.1, 0.28)
    num_waves: int = 5
    noise_range: tuple[float, float] = (-0.05, 0.05)
    noise_step: float = 0.005
    downsampled_scale: float = 0.2


@configclass
class RoughSlopeTerrainCfg(terrain_gen.HfPyramidSlopedTerrainCfg):
    function = rough_slope_terrain
    slope_range: tuple[float, float] = (0.1, 0.568)
    platform_width: float = 3.0
    noise_range: tuple[float, float] = (-0.05, 0.05)
    noise_step: float = 0.005
    downsampled_scale: float = 0.2


TERRAIN_CFG = Go2TerrainGeneratorCfg(
    size=(9.0, 9.0),  # 8.0 terrain + 0.5*2 sub_terrain_border_width
    border_width=25.0,
    sub_terrain_border_width=0.5,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    # slope correction = 0.75 ~ 36.9 degrees by default,
    # but recommended to set for each terrain type separately using with_slope_threshold
    slope_threshold=0.75,
    use_gym_difficulty=False,
    use_cache=False,
    sub_terrains={
        "wave": with_slope_threshold(
            WaveTerrainCfg(
                proportion=0.05,
            ),
            10.0, # effectively disable slope correction for wave terrain
        ),
        "slope_up": with_slope_threshold(
            terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.10,
                slope_range=(0.1, 0.568),
                platform_width=3.0,
            ),
            10.0, # effectively disable slope correction for slope_up terrain
        ),
        "slope_down": with_slope_threshold(
            terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.10,
                slope_range=(0.1, 0.568),
                platform_width=3.0,
            ),
            10.0, # effectively disable slope correction for slope_down terrain
        ),
        "rough_slope": with_slope_threshold(
            RoughSlopeTerrainCfg(
                proportion=0.05,
            ),
            10.0, # effectively disable slope correction for rough_slope terrain
        ),
        "stairs_up": with_slope_threshold(
            terrain_gen.HfInvertedPyramidStairsTerrainCfg(
                proportion=0.25,
                step_height_range=(0.05, 0.257),
                step_width=0.31,
                platform_width=3.0,
            ),
            0.25, # enable slope correction for rough_slope terrain by 14.0 degrees, which is recommended for stairs terrain
        ),
        "stairs_down": with_slope_threshold(
            terrain_gen.HfPyramidStairsTerrainCfg(
                proportion=0.10,
                step_height_range=(0.05, 0.257),
                step_width=0.31,
                platform_width=3.0,
            ),
            0.25, # enable slope correction for rough_slope terrain by 14.0 degrees, which is recommended for stairs terrain
        ),
        "obstacles": with_slope_threshold(
            terrain_gen.HfDiscreteObstaclesTerrainCfg(
                proportion=0.20,
                obstacle_width_range=(1.0, 2.0),
                obstacle_height_range=(0.05, 0.275),
                num_obstacles=20,
                platform_width=3.0,
            ),
            0.25, # enable slope correction for rough_slope terrain by 14.0 degrees
        ),
        "stepping_stones": with_slope_threshold(
            terrain_gen.HfSteppingStonesTerrainCfg(
                proportion=0.0,
                stone_height_max=0.0,
                stone_width_range=(0.075, 1.575),
                stone_distance_range=(0.05, 0.10),
                holes_depth=-10.0,
                platform_width=4.0,
            ),
            0.25, # enable slope correction for rough_slope terrain by 14.0 degrees
        ),
        "gap": terrain_gen.MeshGapTerrainCfg(
            proportion=0.0,
            gap_width_range=(0.0, 0.9),
            platform_width=3.0,
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
    },
)
