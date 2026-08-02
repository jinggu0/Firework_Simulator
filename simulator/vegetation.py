"""Vegetation level of detail, keyed to observation distance.

Grass blades were generated once at build time inside a fixed 1,200 m radius of
the scene origin and then drawn at every distance with no further gating. Two
things were wrong with that:

* The radius is measured from the **origin**, not from the observer, so it is a
  detail *budget*, not a level of detail. It neither removes geometry the
  camera cannot resolve nor adds any where the camera actually goes.
* At 1,200 m a 0.04 m blade subtends 0.03 pixels. A triangle narrower than a
  pixel is sampled at pixel centres, so it flickers or vanishes depending on
  where the samples land. That is aliasing, not detail, and it costs a draw.

This module derives the distances at which vegetation stops being resolvable
from the camera's own optics rather than from a chosen constant, so changing
the sensor, the focal length, or the resolution moves the bands with it.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .camera_optics import vertical_fov_deg
from .config import PhysicalCameraConfig, RenderConfig

BLADE_WIDTH_M = 0.035
"""Representative rendered grass-tuft width.

The builder draws crossed cards 0.024 to 0.050 m across to represent small
tufts; this is near the middle of that range. Using one representative width
keeps the band a per-frame scalar instead of a per-vertex computation, at the
cost of the narrowest tufts fading slightly late and the widest slightly early.
"""

FULL_DETAIL_PIXELS = 1.5
"""Subtended width above which a blade is drawn at full height.

Above roughly a pixel and a half a blade covers enough samples to rasterise
stably from frame to frame.
"""

CUTOFF_PIXELS = 0.5
"""Subtended width below which a blade is collapsed entirely.

Below half a pixel the triangle is narrower than the sample spacing, so its
coverage is decided by where the pixel centre happens to fall.
"""

TREE_SWAY_PIXELS = 2.0
"""Subtended sway amplitude below which crown animation stops.

Crown geometry is metres across and stays resolvable far beyond this; only the
motion is gated, because a displacement smaller than a pixel cannot be seen but
still forces the vertex stage to recompute it.
"""

TREE_SWAY_AMPLITUDE_M = 0.13
"""Peak horizontal crown displacement, matching the vertex shader's clamp."""

EVENT_SITE_DETAIL_RADIUS_M = 1_200.0
"""Radius around the scene origin within which the builder may place blades.

This is a **budget**, not a level of detail: it decides where geometry is
authored, while the bands below decide whether authored geometry is drawn.
Naming the two separately is the point — they were previously the same number
doing both jobs badly.
"""


def angular_resolution_rad_per_pixel(
    camera: PhysicalCameraConfig, viewport_height_px: int
) -> float:
    """Vertical angle one pixel subtends, from the physical camera model."""

    if viewport_height_px <= 0:
        raise ValueError(
            f"viewport height must be positive, got {viewport_height_px}"
        )
    return math.radians(vertical_fov_deg(camera)) / viewport_height_px


def subpixel_distance_m(
    feature_size_m: float,
    angular_resolution_rad: float,
    pixels: float = 1.0,
) -> float:
    """Distance at which ``feature_size_m`` subtends ``pixels`` pixels.

    Small-angle approximation, which holds to better than a part in a million
    for anything this function is used on.
    """

    if feature_size_m <= 0.0 or pixels <= 0.0:
        raise ValueError("feature size and pixel count must be positive")
    return feature_size_m / (pixels * angular_resolution_rad)


@dataclass(frozen=True, slots=True)
class VegetationLod:
    """Observation distances at which each vegetation detail is dropped."""

    blade_full_detail_m: float
    blade_cutoff_m: float
    tree_sway_cutoff_m: float
    angular_resolution_rad_per_pixel: float

    @classmethod
    def from_camera(
        cls,
        camera: PhysicalCameraConfig,
        render: RenderConfig,
        blade_width_m: float = BLADE_WIDTH_M,
    ) -> "VegetationLod":
        resolution = angular_resolution_rad_per_pixel(camera, render.height)
        return cls(
            blade_full_detail_m=subpixel_distance_m(
                blade_width_m, resolution, FULL_DETAIL_PIXELS
            ),
            blade_cutoff_m=subpixel_distance_m(
                blade_width_m, resolution, CUTOFF_PIXELS
            ),
            tree_sway_cutoff_m=subpixel_distance_m(
                TREE_SWAY_AMPLITUDE_M, resolution, TREE_SWAY_PIXELS
            ),
            angular_resolution_rad_per_pixel=resolution,
        )

    def __post_init__(self) -> None:
        if not self.blade_full_detail_m < self.blade_cutoff_m:
            raise ValueError(
                "blades must reach full detail nearer than they are cut off: "
                f"{self.blade_full_detail_m} >= {self.blade_cutoff_m}"
            )

    def blade_detail_fraction(self, distance_m: float) -> float:
        """Blade height scale at an observation distance, in ``[0, 1]``.

        The same ramp the vertex shader applies, exposed so the band can be
        tested and reported without a GPU.
        """

        span = self.blade_cutoff_m - self.blade_full_detail_m
        alpha = (distance_m - self.blade_full_detail_m) / span
        clamped = min(max(alpha, 0.0), 1.0)
        # Smoothstep, so a blade neither pops out nor shrinks linearly into a
        # visible seam at the band edges.
        return 1.0 - clamped * clamped * (3.0 - 2.0 * clamped)

    def summary(self) -> dict[str, float]:
        return {
            "blade_full_detail_m": self.blade_full_detail_m,
            "blade_cutoff_m": self.blade_cutoff_m,
            "tree_sway_cutoff_m": self.tree_sway_cutoff_m,
            "milliradians_per_pixel": (
                self.angular_resolution_rad_per_pixel * 1e3
            ),
        }
