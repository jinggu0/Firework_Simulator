from __future__ import annotations

import io
import math
import urllib.request
from dataclasses import dataclass

import numpy as np
from PIL import Image

EARTH_RADIUS_M = 6_378_137.0


@dataclass(frozen=True, slots=True)
class TerrainSurface:
    """CPU view of the exact height field rendered by the GPU.

    The navigation camera, acoustics and future physics queries must not each
    invent their own interpretation of the terrain texture.  Samples are
    bilinear over the georeferenced texel centres used by ``land.vert`` and
    ``scene.vert``.  The optional river mask distinguishes solid ground from
    the water surface, whose rendered elevation is the simulator datum.
    """

    height_m: np.ndarray
    bounds: np.ndarray
    water_mask: np.ndarray | None = None
    water_bounds: np.ndarray | None = None

    def __post_init__(self) -> None:
        height = np.asarray(self.height_m, dtype=np.float32)
        bounds = np.asarray(self.bounds, dtype=np.float32)
        if height.ndim != 2 or min(height.shape) < 1:
            raise ValueError("terrain height field must be a non-empty 2D array")
        if bounds.shape != (4,) or not np.all(bounds[2:] > bounds[:2]):
            raise ValueError("terrain bounds must be [min_x, min_z, max_x, max_z]")
        if not np.isfinite(height).all() or not np.isfinite(bounds).all():
            raise ValueError("terrain contains non-finite values")
        object.__setattr__(self, "height_m", height)
        object.__setattr__(self, "bounds", bounds)
        if self.water_mask is None:
            return
        mask = np.asarray(self.water_mask, dtype=np.uint8)
        mask_bounds = np.asarray(
            self.water_bounds if self.water_bounds is not None else bounds,
            dtype=np.float32,
        )
        if mask.ndim != 2 or min(mask.shape) < 1:
            raise ValueError("water mask must be a non-empty 2D array")
        if mask_bounds.shape != (4,) or not np.all(mask_bounds[2:] > mask_bounds[:2]):
            raise ValueError("water bounds must be [min_x, min_z, max_x, max_z]")
        object.__setattr__(self, "water_mask", mask)
        object.__setattr__(self, "water_bounds", mask_bounds)

    @staticmethod
    def _grid_coordinate(
        value: float, minimum: float, maximum: float, count: int
    ) -> float:
        if count == 1:
            return 0.0
        unit = (float(value) - minimum) / (maximum - minimum)
        return float(np.clip(unit, 0.0, 1.0)) * (count - 1)

    def height_at(self, x_m: float, z_m: float) -> float:
        """Return bilinearly interpolated terrain height in local EUS metres."""

        rows, columns = self.height_m.shape
        x = self._grid_coordinate(
            x_m, float(self.bounds[0]), float(self.bounds[2]), columns
        )
        z = self._grid_coordinate(
            z_m, float(self.bounds[1]), float(self.bounds[3]), rows
        )
        x0, z0 = int(math.floor(x)), int(math.floor(z))
        x1, z1 = min(x0 + 1, columns - 1), min(z0 + 1, rows - 1)
        tx, tz = x - x0, z - z0
        lower = self.height_m[z0, x0] * (1.0 - tx) + self.height_m[z0, x1] * tx
        upper = self.height_m[z1, x0] * (1.0 - tx) + self.height_m[z1, x1] * tx
        return float(lower * (1.0 - tz) + upper * tz)

    def normal_at(self, x_m: float, z_m: float) -> np.ndarray:
        """Return the central-difference normal used for draped surfaces."""

        rows, columns = self.height_m.shape
        step_x = float(self.bounds[2] - self.bounds[0]) / max(columns - 1, 1)
        step_z = float(self.bounds[3] - self.bounds[1]) / max(rows - 1, 1)
        dh_dx = (
            self.height_at(x_m + step_x, z_m)
            - self.height_at(x_m - step_x, z_m)
        ) / (2.0 * step_x)
        dh_dz = (
            self.height_at(x_m, z_m + step_z)
            - self.height_at(x_m, z_m - step_z)
        ) / (2.0 * step_z)
        normal = np.array([-dh_dx, 1.0, -dh_dz], dtype=np.float32)
        return normal / np.linalg.norm(normal)

    def is_water(self, x_m: float, z_m: float) -> bool:
        """Nearest-sample classification matching the geographic river mask."""

        if self.water_mask is None or self.water_bounds is None:
            return False
        rows, columns = self.water_mask.shape
        x = self._grid_coordinate(
            x_m,
            float(self.water_bounds[0]),
            float(self.water_bounds[2]),
            columns,
        )
        z = self._grid_coordinate(
            z_m,
            float(self.water_bounds[1]),
            float(self.water_bounds[3]),
            rows,
        )
        return bool(self.water_mask[int(round(z)), int(round(x))] >= 128)

    def collision_height_at(self, x_m: float, z_m: float) -> float:
        """Height of the opaque ground or water boundary seen by the camera."""

        return 0.0 if self.is_water(x_m, z_m) else self.height_at(x_m, z_m)


def sample_heightmap_array(
    height_m: np.ndarray,
    bounds: np.ndarray,
    positions_xz_m: np.ndarray,
) -> np.ndarray:
    """Vectorised bilinear sampling over georeferenced texel centres."""

    height = np.asarray(height_m, dtype=np.float64)
    extent = np.asarray(bounds, dtype=np.float64)
    positions = np.asarray(positions_xz_m, dtype=np.float64)
    if height.ndim != 2 or min(height.shape) < 1:
        raise ValueError("height field must be a non-empty 2D array")
    if extent.shape != (4,) or not np.all(extent[2:] > extent[:2]):
        raise ValueError("bounds must be [min_x, min_z, max_x, max_z]")
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("sample positions must have shape (n, 2)")
    rows, columns = height.shape
    x = np.clip(
        (positions[:, 0] - extent[0]) / (extent[2] - extent[0]), 0.0, 1.0
    ) * (columns - 1)
    z = np.clip(
        (positions[:, 1] - extent[1]) / (extent[3] - extent[1]), 0.0, 1.0
    ) * (rows - 1)
    x0, z0 = np.floor(x).astype(np.int32), np.floor(z).astype(np.int32)
    x1, z1 = np.minimum(x0 + 1, columns - 1), np.minimum(z0 + 1, rows - 1)
    tx, tz = x - x0, z - z0
    lower = height[z0, x0] * (1.0 - tx) + height[z0, x1] * tx
    upper = height[z1, x0] * (1.0 - tx) + height[z1, x1] * tx
    return lower * (1.0 - tz) + upper * tz


def constrained_heightmap(
    baseline_height_m: np.ndarray,
    bounds: np.ndarray,
    sample_positions_xz_m: np.ndarray,
    sample_height_m: np.ndarray,
    water_mask: np.ndarray,
    resolution: tuple[int, int],
) -> tuple[np.ndarray, float]:
    """Rasterise official contour/spot heights without inventing detail.

    A piecewise-linear Delaunay surface honours every source sample and cannot
    overshoot their elevation range. The older regional DEM is used only
    outside the convex hull of official data. The returned fraction reports
    how much of the output grid is supported by the official interpolation.

    SciPy is imported lazily because this path is an offline asset builder;
    running the simulator requires only the generated NumPy height field.
    """

    try:
        from scipy.interpolate import LinearNDInterpolator
        from scipy.spatial import Delaunay
    except ImportError as error:  # pragma: no cover - dependency message
        raise RuntimeError(
            "terrain refinement requires the optional terrain dependencies"
        ) from error

    positions = np.asarray(sample_positions_xz_m, dtype=np.float64)
    values = np.asarray(sample_height_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("terrain constraints must have shape (n, 2)")
    if values.shape != (len(positions),):
        raise ValueError("each terrain constraint needs one elevation")
    if len(positions) < 3 or not np.isfinite(positions).all() or not np.isfinite(values).all():
        raise ValueError("at least three finite terrain constraints are required")
    width, height = (int(value) for value in resolution)
    if width < 2 or height < 2:
        raise ValueError("terrain output resolution must be at least 2 x 2")

    triangulation = Delaunay(positions)
    interpolator = LinearNDInterpolator(
        triangulation, values, fill_value=np.nan
    )
    extent = np.asarray(bounds, dtype=np.float64)
    grid_x = np.linspace(extent[0], extent[2], width, dtype=np.float64)
    grid_z = np.linspace(extent[1], extent[3], height, dtype=np.float64)
    output = np.empty((height, width), dtype=np.float32)
    supported_count = 0
    for row_start in range(0, height, 64):
        rows = grid_z[row_start : row_start + 64]
        positions_xz = np.column_stack(
            (np.tile(grid_x, len(rows)), np.repeat(rows, width))
        )
        official = np.asarray(interpolator(positions_xz), dtype=np.float64)
        supported = np.isfinite(official)
        supported_count += int(np.count_nonzero(supported))
        if not np.all(supported):
            fallback = sample_heightmap_array(
                baseline_height_m, extent, positions_xz
            )
            official = np.where(supported, official, fallback)
        output[row_start : row_start + len(rows)] = official.reshape(
            len(rows), width
        )

    river = np.asarray(
        Image.fromarray(np.asarray(water_mask, dtype=np.uint8)).resize(
            (width, height), resample=Image.Resampling.NEAREST
        )
    ) >= 128
    output[river] = 0.0
    return output, supported_count / float(width * height)


def _mercator_pixel(
    latitude_deg: np.ndarray, longitude_deg: np.ndarray, zoom: int
) -> tuple[np.ndarray, np.ndarray]:
    scale = (2**zoom) * 256.0
    latitude_rad = np.radians(np.clip(latitude_deg, -85.05112878, 85.05112878))
    x = (longitude_deg + 180.0) / 360.0 * scale
    y = (
        1.0 - np.arcsinh(np.tan(latitude_rad)) / np.pi
    ) * 0.5 * scale
    return x, y


def _download_tile(zoom: int, x: int, y: int) -> np.ndarray:
    url = (
        "https://s3.amazonaws.com/elevation-tiles-prod/"
        f"terrarium/{zoom}/{x}/{y}.png"
    )
    request = urllib.request.Request(
        url, headers={"User-Agent": "FireworkSimulator/0.1 (local research project)"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        image = Image.open(io.BytesIO(response.read())).convert("RGB")
    rgb = np.asarray(image, dtype=np.float32)
    return rgb[:, :, 0] * 256.0 + rgb[:, :, 1] + rgb[:, :, 2] / 256.0 - 32768.0


def build_terrain_heightmap(
    origin_latitude_deg: float,
    origin_longitude_deg: float,
    bounds: tuple[float, float, float, float],
    water_mask: np.ndarray,
    resolution: tuple[int, int] = (512, 512),
    zoom: int = 12,
) -> tuple[np.ndarray, float]:
    """Sample Terrarium tiles into local EUS coordinates.

    Returned elevations are relative to the median DEM elevation under the
    supplied river mask, making the simulated water plane y=0.
    """

    minimum_x, minimum_z, maximum_x, maximum_z = bounds
    width, height = resolution
    local_x = np.linspace(minimum_x, maximum_x, width, dtype=np.float64)
    local_z = np.linspace(minimum_z, maximum_z, height, dtype=np.float64)
    xx, zz = np.meshgrid(local_x, local_z)
    latitude = origin_latitude_deg - np.degrees(zz / EARTH_RADIUS_M)
    longitude = origin_longitude_deg + np.degrees(
        xx / (EARTH_RADIUS_M * math.cos(math.radians(origin_latitude_deg)))
    )
    pixel_x, pixel_y = _mercator_pixel(latitude, longitude, zoom)
    tile_min_x = int(np.floor(pixel_x.min() / 256.0))
    tile_max_x = int(np.floor(pixel_x.max() / 256.0))
    tile_min_y = int(np.floor(pixel_y.min() / 256.0))
    tile_max_y = int(np.floor(pixel_y.max() / 256.0))
    mosaic = np.empty(
        ((tile_max_y - tile_min_y + 1) * 256,
         (tile_max_x - tile_min_x + 1) * 256),
        dtype=np.float32,
    )
    for tile_y in range(tile_min_y, tile_max_y + 1):
        for tile_x in range(tile_min_x, tile_max_x + 1):
            row = (tile_y - tile_min_y) * 256
            column = (tile_x - tile_min_x) * 256
            mosaic[row : row + 256, column : column + 256] = _download_tile(
                zoom, tile_x, tile_y
            )

    sample_x = pixel_x - tile_min_x * 256
    sample_y = pixel_y - tile_min_y * 256
    x0 = np.clip(np.floor(sample_x).astype(np.int32), 0, mosaic.shape[1] - 2)
    y0 = np.clip(np.floor(sample_y).astype(np.int32), 0, mosaic.shape[0] - 2)
    tx = (sample_x - x0).astype(np.float32)
    ty = (sample_y - y0).astype(np.float32)
    heightmap = (
        mosaic[y0, x0] * (1 - tx) * (1 - ty)
        + mosaic[y0, x0 + 1] * tx * (1 - ty)
        + mosaic[y0 + 1, x0] * (1 - tx) * ty
        + mosaic[y0 + 1, x0 + 1] * tx * ty
    )

    mask_image = Image.fromarray(water_mask, mode="L").resize(
        resolution, resample=Image.Resampling.NEAREST
    )
    river = np.asarray(mask_image) >= 128
    datum_m = float(np.median(heightmap[river])) if np.any(river) else 0.0
    relative = np.clip(heightmap - datum_m, -1.0, 400.0).astype(np.float32)
    # Water geometry owns the river surface; prevent DEM artefacts below it
    # from creating a rim at the geographic mask boundary.
    relative[river] = 0.0
    return relative, datum_m
