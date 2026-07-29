from __future__ import annotations

import io
import math
import urllib.request

import numpy as np
from PIL import Image

EARTH_RADIUS_M = 6_378_137.0


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
