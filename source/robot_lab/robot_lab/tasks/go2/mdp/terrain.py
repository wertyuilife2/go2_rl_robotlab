from __future__ import annotations

import numpy as np

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass
from isaaclab.terrains.height_field import hf_terrains
from isaaclab.terrains.height_field.utils import height_field_to_mesh


# -----------------------------------------------------------------------------
# Default RobotLab terrain setup
# -----------------------------------------------------------------------------

DEFAULT_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
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
# Gym-aligned terrain setup
# -----------------------------------------------------------------------------


def _signed_pyramid_slope_height_field(
    difficulty: float,
    cfg: terrain_gen.HfPyramidSlopedTerrainCfg,
    random_inverted: bool,
) -> np.ndarray:
    """Generate a pyramid slope with optional random sign."""
    slope = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    if random_inverted and np.random.rand() < 0.5:
        slope = -slope

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_max = int(slope * cfg.size[0] / 2 / cfg.vertical_scale)
    center_x = int(width_pixels / 2)
    center_y = int(length_pixels / 2)

    x = np.arange(0, width_pixels)
    y = np.arange(0, length_pixels)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = (center_x - np.abs(center_x - xx)) / center_x
    yy = (center_y - np.abs(center_y - yy)) / center_y
    xx = xx.reshape(width_pixels, 1)
    yy = yy.reshape(1, length_pixels)

    hf_raw = height_max * xx * yy

    platform_width = int(cfg.platform_width / cfg.horizontal_scale / 2)
    x_pf = width_pixels // 2 - platform_width
    y_pf = length_pixels // 2 - platform_width
    z_pf = hf_raw[x_pf, y_pf]
    hf_raw = np.clip(hf_raw, min(0, z_pf), max(0, z_pf))

    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def wave_terrain(difficulty: float, cfg) -> np.ndarray:
    """wave terrain: wave plus random uniform roughness."""
    wave = hf_terrains.wave_terrain.__wrapped__(difficulty, cfg)
    rough = hf_terrains.random_uniform_terrain.__wrapped__(difficulty, cfg)
    return np.rint(wave + rough).astype(np.int16)


@height_field_to_mesh
def slope_terrain(difficulty: float, cfg) -> np.ndarray:
    """slope terrain with 50/50 positive and negative slopes."""
    return _signed_pyramid_slope_height_field(difficulty, cfg, random_inverted=True)


@height_field_to_mesh
def rough_slope_terrain(difficulty: float, cfg) -> np.ndarray:
    """rough slope terrain: slope plus random uniform roughness."""
    slope = _signed_pyramid_slope_height_field(difficulty, cfg, random_inverted=False)
    rough = hf_terrains.random_uniform_terrain.__wrapped__(difficulty, cfg)
    return np.rint(slope + rough).astype(np.int16)


@configclass
class WaveTerrainCfg(terrain_gen.HfTerrainBaseCfg):
    function = wave_terrain
    amplitude_range: tuple[float, float] = (0.1, 0.28)
    num_waves: int = 5
    noise_range: tuple[float, float] = (-0.05, 0.05)
    noise_step: float = 0.005
    downsampled_scale: float = 0.2


@configclass
class SlopeTerrainCfg(terrain_gen.HfTerrainBaseCfg):
    function = slope_terrain
    slope_range: tuple[float, float] = (0.1, 0.568)
    platform_width: float = 3.0


@configclass
class RoughSlopeTerrainCfg(terrain_gen.HfTerrainBaseCfg):
    function = rough_slope_terrain
    slope_range: tuple[float, float] = (0.1, 0.568)
    platform_width: float = 3.0
    noise_range: tuple[float, float] = (-0.05, 0.05)
    noise_step: float = 0.005
    downsampled_scale: float = 0.2


TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=25.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "wave": WaveTerrainCfg(
            proportion=0.05,
            border_width=0.0,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
        ),
        "slope": SlopeTerrainCfg(
            proportion=0.20,
            border_width=0.0,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
        ),
        "rough_slope": RoughSlopeTerrainCfg(
            proportion=0.05,
            border_width=0.0,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
        ),
        "stairs_up": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.05, 0.257),
            step_width=0.31,
            platform_width=3.0,
            border_width=0.0,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
        ),
        "stairs_down": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.257),
            step_width=0.31,
            platform_width=3.0,
            border_width=0.0,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
        ),
        "obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.20,
            obstacle_width_range=(1.0, 2.0),
            obstacle_height_range=(0.05, 0.275),
            num_obstacles=20,
            platform_width=3.0,
            border_width=0.0,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
        ),
        # "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
        #     proportion=0.0,
        #     stone_height_max=0.0,
        #     stone_width_range=(0.075, 1.575),
        #     stone_distance_range=(0.05, 0.10),
        #     holes_depth=-10.0,
        #     platform_width=4.0,
        #     border_width=0.0,
        #     horizontal_scale=0.1,
        #     vertical_scale=0.005,
        #     slope_threshold=0.75,
        # ),
        # "gap": terrain_gen.MeshGapTerrainCfg(
        #     proportion=0.0,
        #     gap_width_range=(0.0, 0.9),
        #     platform_width=3.0,
        # ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
    },
)
