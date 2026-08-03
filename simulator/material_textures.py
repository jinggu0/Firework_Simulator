"""Photo-scanned PBR surface sets shared by the static scene shader.

The geometry remains tied to dated spatial data.  These textures supply the
sub-centimetre albedo, OpenGL tangent-space normal, ambient-occlusion and
roughness variation that a footprint or a 30 m DEM cannot contain.  They are
generic CC0 scans, not samples taken at Yeouido, so the calibrated material
colour remains authoritative and the scan contributes only normalized detail.
"""

from __future__ import annotations

from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

SCANNED_MATERIAL_UNIT = 14
# Eight taps cover the 6.75:1 footprint reached around 10 m while keeping the
# request below the 16x ceiling common on desktop GPUs. ModernGL/driver clamps
# the value to the device limit and exposes the applied value for the audit.
SCANNED_MATERIAL_ANISOTROPY = 8.0
ANISOTROPY_EXTENSIONS = frozenset(
    {
        "GL_ARB_texture_filter_anisotropic",
        "GL_EXT_texture_filter_anisotropic",
    }
)

MATERIAL_TEXTURE_DIRECTORY = (
    Path(__file__).resolve().parent.parent / "assets" / "materials"
)

SCANNED_MATERIALS: tuple[tuple[str, float], ...] = (
    ("asphalt_04", 4.0),
    ("concrete_pavers", 1.9),
    ("leafy_grass", 2.0),
    ("concrete", 4.0),
)
"""Poly Haven asset id and real-world tile width in metres."""

MAP_SUFFIXES = ("diff", "nor_gl", "arm")


class ScannedMaterialTextures:
    """One mipmapped texture array containing diffuse, normal and ARM maps."""

    def __init__(
        self,
        ctx: moderngl.Context,
        directory: Path = MATERIAL_TEXTURE_DIRECTORY,
    ) -> None:
        layers: list[np.ndarray] = []
        diffuse_means: list[np.ndarray] = []
        size: tuple[int, int] | None = None
        for asset_id, _ in SCANNED_MATERIALS:
            maps: list[np.ndarray] = []
            for suffix in MAP_SUFFIXES:
                path = directory / f"{asset_id}_{suffix}_1k.jpg"
                with Image.open(path) as source:
                    image = source.convert("RGB").transpose(
                        Image.Transpose.FLIP_TOP_BOTTOM
                    )
                    if size is None:
                        size = image.size
                    elif image.size != size:
                        raise ValueError(
                            f"material map {path} is {image.size}, expected {size}"
                        )
                    maps.append(np.asarray(image, dtype=np.uint8))
            # Layers are grouped by asset, then map: shader layer = asset*3+map.
            layers.extend(maps)
            linear_diffuse = np.power(
                maps[0].astype(np.float32) / 255.0, 2.2
            )
            diffuse_means.append(linear_diffuse.mean(axis=(0, 1)))

        if size is None:
            raise ValueError("no scanned material textures were loaded")
        packed = np.stack(layers, axis=0)
        self.texture = ctx.texture_array(
            (size[0], size[1], len(layers)),
            components=3,
            data=packed.tobytes(),
            alignment=1,
        )
        self.texture.repeat_x = True
        self.texture.repeat_y = True
        self.texture.filter = (
            moderngl.LINEAR_MIPMAP_LINEAR,
            moderngl.LINEAR,
        )
        self.texture.build_mipmaps()
        self.anisotropy_supported = bool(
            ANISOTROPY_EXTENSIONS.intersection(ctx.extensions)
        )
        if self.anisotropy_supported:
            self.texture.anisotropy = SCANNED_MATERIAL_ANISOTROPY
            self.applied_anisotropy = float(self.texture.anisotropy)
        else:
            self.applied_anisotropy = 1.0
        self.diffuse_means = np.asarray(diffuse_means, dtype=np.float32)

    def bind(self) -> None:
        self.texture.use(SCANNED_MATERIAL_UNIT)


def texture_widths_m() -> np.ndarray:
    return np.asarray(
        [width_m for _, width_m in SCANNED_MATERIALS], dtype=np.float32
    )
