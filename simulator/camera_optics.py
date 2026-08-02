"""Lens and sensor stages between scene radiance and a display-referred image.

Three stages were missing and are added here: the lens's geometric distortion,
the white balance that undoes the sensor's own spectral response, and — as a
recorded absence rather than an implementation — the camera colour matrix.

**Why there is no colour matrix.** A camera matrix converts a sensor's native
spectral basis into a standard colour space. This renderer's radiance values
are already in linear sRGB primaries: ``simulator.color.wavelength_rgb`` and
``blackbody_rgb`` both emit sRGB, so every authored and derived colour in the
scene is expressed there. Building a matrix from the three channel wavelengths
and the CIE observer gives, in linear sRGB,

    [[ 2.472 -0.126 -0.008]
     [-0.030  1.445 -0.098]
     [-0.046 -0.169  1.771]]

which is far from identity because it would be a *second* conversion of colour
that has already been converted once. Applying it would be a large unexplained
hue and exposure shift dressed as physics. Closing this properly needs a
spectral renderer and measured sensor sensitivities; neither is held, so the
gap is recorded instead of filled.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from .color import planckian_linear_srgb
from .config import PhysicalCameraConfig
from .provenance import ConfidenceGrade, DataRecord, file_checksum

PLANCK_CONSTANT_J_S = 6.62607015e-34
SPEED_OF_LIGHT_M_S = 299_792_458.0

DISTORTION_INVERSE_ITERATIONS = 5
"""Fixed-point steps used to invert the Brown-Conrady polynomial.

The forward model has no closed-form inverse. Five iterations is what OpenCV's
``undistortPoints`` uses and reaches float32 precision for lens coefficients in
the ordinary range; ``LensDistortion.inverse_residual`` measures it rather than
assuming it, and the shader runs the same count so the two agree.
"""


def vertical_fov_deg(config: PhysicalCameraConfig) -> float:
    return math.degrees(
        2.0
        * math.atan(
            config.sensor_height_mm / (2.0 * config.focal_length_mm)
        )
    )


def photon_to_electron_scale(config: PhysicalCameraConfig) -> np.ndarray:
    aperture_factor = (
        math.pi
        / (4.0 * config.f_number**2)
        * config.lens_transmission
    )
    pixel_area_m2 = (config.pixel_pitch_um * 1e-6) ** 2
    wavelengths_m = (
        np.asarray(config.wavelength_rgb_nm, dtype=np.float64) * 1e-9
    )
    quantum_efficiency = np.asarray(
        config.quantum_efficiency_rgb, dtype=np.float64
    )
    photons_per_joule = wavelengths_m / (
        PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT_M_S
    )
    return (
        aperture_factor
        * config.shutter_time_s
        * pixel_area_m2
        * photons_per_joule
        * quantum_efficiency
    ).astype(np.float32)


def analog_gain(config: PhysicalCameraConfig) -> float:
    return config.iso / config.base_iso


def white_balance_gains(config: PhysicalCameraConfig) -> np.ndarray:
    """Per-channel gains that render the reference illuminant neutral.

    The sensor stage was applying its spectral response and never undoing it.
    Quantum efficiency (0.42 / 0.52 / 0.36) and photon energy — a red photon
    carries less energy than a blue one, so a watt of red light frees more
    electrons — together turn a neutral scene into electrons in the ratio
    0.896 : 1 : 0.579. Every frame therefore left the camera with a yellow-green
    cast that nothing in the pipeline corrected.

    A von Kries balance in the camera's own channel space fixes that: the gains
    are the reciprocal of the response to the reference illuminant, a Planckian
    radiator at ``white_balance_temperature_k``. Green is normalised to unity,
    the usual convention, because exposure is set on the luminance-carrying
    channel and the balance should not silently change it.

    At the shipped 6504 K the gains are about (1.055, 1, 1.642). The reference
    is not exactly (1, 1, 1) because a 6504 K Planckian is not the D65
    *illuminant* — D65 includes atmospheric and solar line structure that a
    black body does not — and that 5% difference is the model being honest
    rather than a rounding.

    The temperature is an **operator setting, not a measurement**. 6504 K is the
    default because it is the only value derivable from the pipeline's own
    colour space: it makes an sRGB-neutral scene render neutral, so the balance
    is a correction rather than a look. A videographer shooting a warm-lit night
    city would plausibly have set 3200-4000 K, which the config allows.
    """

    reference = planckian_linear_srgb(config.white_balance_temperature_k)
    response = reference * photon_to_electron_scale(config).astype(np.float64)
    green = float(response[1])
    if green <= 0.0:
        return np.ones(3, dtype=np.float32)
    return (green / np.maximum(response, 1e-30)).astype(np.float32)


@dataclass(frozen=True, slots=True)
class LensDistortion:
    """Brown-Conrady radial and tangential distortion.

    Coefficients follow the OpenCV convention exactly — ``k1, k2, p1, p2, k3``
    in ``cv::calibrateCamera`` order, on normalised image coordinates
    ``x = X/Z`` — because that is what any calibration this project could
    obtain would report. Adopting a private convention would mean silently
    reinterpreting someone else's measurement.

    **The shipped default is identity.** No lens calibration for the footage
    exists, and inventing coefficients would put a fabricated optical claim
    into every frame. The model is here so that a real calibration can be
    loaded, so V-04's reprojection check has something to reproject through,
    and so Phase 10's video work has a lens to undistort with.
    """

    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    @classmethod
    def from_config(cls, config: PhysicalCameraConfig) -> "LensDistortion":
        return cls(
            k1=config.distortion_k1,
            k2=config.distortion_k2,
            k3=config.distortion_k3,
            p1=config.distortion_p1,
            p2=config.distortion_p2,
        )

    @property
    def is_identity(self) -> bool:
        return not any((self.k1, self.k2, self.k3, self.p1, self.p2))

    def distort(self, ideal: np.ndarray) -> np.ndarray:
        """Where an ideal ray actually lands, in normalised image coordinates.

        This is the forward physical model: the lens maps the pinhole position
        to a displaced one. Accepts an ``(..., 2)`` array.
        """

        ideal = np.asarray(ideal, dtype=np.float64)
        x, y = ideal[..., 0], ideal[..., 1]
        r2 = x * x + y * y
        radial = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3))
        tangential_x = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x)
        tangential_y = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y
        return np.stack(
            [x * radial + tangential_x, y * radial + tangential_y], axis=-1
        )

    def undistort(
        self,
        distorted: np.ndarray,
        iterations: int = DISTORTION_INVERSE_ITERATIONS,
    ) -> np.ndarray:
        """Which ideal ray landed here — the inverse of :meth:`distort`.

        This is the direction the renderer needs. The frame is rendered through
        an ideal pinhole, so producing the distorted image means asking, for
        each output pixel, which undistorted position to sample. The polynomial
        has no closed-form inverse, so it is iterated; with identity
        coefficients the first step is exact and the result is bit-identical to
        the input, which is what keeps the default camera path unperturbed.
        """

        distorted = np.asarray(distorted, dtype=np.float64)
        ideal = distorted.copy()
        for _ in range(max(iterations, 1)):
            x, y = ideal[..., 0], ideal[..., 1]
            r2 = x * x + y * y
            radial = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3))
            tangential_x = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x)
            tangential_y = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y
            ideal = (
                distorted - np.stack([tangential_x, tangential_y], axis=-1)
            ) / radial[..., None]
        return ideal

    def inverse_residual(self, half_extent: tuple[float, float]) -> float:
        """Worst round-trip error over the frame, in normalised coordinates.

        Reported rather than assumed: the fixed-point inversion converges for
        ordinary lenses but not for arbitrary coefficients, and a calibration
        that does not converge must be visible as a number.
        """

        grid = self._frame_grid(half_extent)
        return float(
            np.abs(self.distort(self.undistort(grid)) - grid).max()
        )

    def required_overscan(self, half_extent: tuple[float, float]) -> float:
        """How much wider than the sensor the scene must be rendered.

        The frame is formed by asking, for each output pixel, which undistorted
        direction landed there. Barrel distortion pulls the image inward, so
        those directions run past the edge of an ideal render and the corners
        sample scene that was never drawn. Rendering the field this many times
        wider puts them back inside it.

        Returns exactly 1.0 for an identity or pincushion lens, so the shipped
        camera path is untouched — the render is widened only when a loaded
        calibration actually needs it.
        """

        source = self.undistort(self._frame_grid(half_extent))
        needed = np.maximum(
            np.abs(source[..., 0]) / max(half_extent[0], 1e-12),
            np.abs(source[..., 1]) / max(half_extent[1], 1e-12),
        )
        widest = float(needed.max())
        if widest <= 1.0:
            return 1.0
        # A hair beyond the extreme sample. Scaling by exactly the maximum puts
        # that sample on the boundary, where the containment test loses to
        # floating point and coverage lands at 0.9998 instead of 1. The margin
        # is applied only when the lens needs widening at all, so an identity
        # or pincushion calibration still returns exactly 1.0 and leaves the
        # render untouched.
        return widest * (1.0 + 1e-6)

    def frame_coverage(
        self,
        half_extent: tuple[float, float],
        rendered_extent: tuple[float, float] | None = None,
    ) -> float:
        """Fraction of output pixels whose source lies inside the rendered frame.

        Barrel distortion pulls the image inward, so the corners of the output
        sample beyond what an ideal render of the sensor's own field contains.
        Below 1.0 the border is clamped rather than correct.

        ``half_extent`` is the sensor's field, which sets the output grid.
        ``rendered_extent`` is the field actually drawn, wider when the renderer
        overscans; it defaults to the sensor's, which is the coverage *without*
        overscan and therefore the size of the problem overscan solves.
        """

        rendered = rendered_extent or half_extent
        source = self.undistort(self._frame_grid(half_extent))
        inside = (np.abs(source[..., 0]) <= rendered[0]) & (
            np.abs(source[..., 1]) <= rendered[1]
        )
        return float(inside.mean())

    @staticmethod
    def _frame_grid(
        half_extent: tuple[float, float], samples: int = 129
    ) -> np.ndarray:
        x = np.linspace(-half_extent[0], half_extent[0], samples)
        y = np.linspace(-half_extent[1], half_extent[1], samples)
        return np.stack(np.meshgrid(x, y, indexing="xy"), axis=-1)

    def uniforms(self) -> dict[str, tuple[float, float, float] | tuple[float, float]]:
        """Uniform names ``tonemap.frag`` declares."""

        return {
            "distortion_radial": (self.k1, self.k2, self.k3),
            "distortion_tangential": (self.p1, self.p2),
        }


def frame_half_extent(config: PhysicalCameraConfig) -> tuple[float, float]:
    """Half-width and half-height of the frame in normalised coordinates.

    ``tan(theta/2)`` on each axis: the same quantity the projection matrix and
    the shader's ``sensor_position`` are built from, so the CPU and GPU
    distortion operate on identical coordinates.
    """

    half_height = math.tan(math.radians(vertical_fov_deg(config)) * 0.5)
    aspect = config.sensor_width_mm / config.sensor_height_mm
    return (half_height * aspect, half_height)


REQUIRED_CALIBRATION_FIELDS = (
    "source_id",
    "captured_at",
    "license",
    "image_width_px",
    "image_height_px",
    "principal_point_px",
    "distortion_coefficients",
)
"""Every field a lens calibration must carry to be loadable.

A calibration without provenance is a set of numbers claiming to be a
measurement, which is worse than no calibration at all: the renderer would
apply it and the report would say the optics were characterised.
"""


def load_lens_calibration(path: Path) -> tuple[LensDistortion, DataRecord]:
    """Read an OpenCV-convention lens calibration with its provenance.

    Rejects a calibration whose **principal point is not the image centre**.
    An off-centre principal point displaces the projection as well as the
    distortion, and only the distortion is modelled here; applying half of a
    decentred calibration would be worse than refusing it. Recentring the
    projection matrix is the fix, and it is not implemented.
    """

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = [name for name in REQUIRED_CALIBRATION_FIELDS if name not in payload]
    if missing:
        raise ValueError(
            f"lens calibration {path.name} is missing required provenance or "
            f"intrinsics: {', '.join(missing)}"
        )
    width = float(payload["image_width_px"])
    height = float(payload["image_height_px"])
    centre_x, centre_y = (float(value) for value in payload["principal_point_px"])
    offset_px = max(abs(centre_x - width * 0.5), abs(centre_y - height * 0.5))
    # Half a pixel is the quantisation of the calibration's own image grid.
    if offset_px > 0.5:
        raise ValueError(
            f"lens calibration {path.name} has a principal point "
            f"{offset_px:.2f} px off centre. The projection matrix would have "
            "to be recentred as well, which is not implemented; only the "
            "distortion polynomial is applied here."
        )
    coefficients = [float(value) for value in payload["distortion_coefficients"]]
    if len(coefficients) < 4:
        raise ValueError(
            f"lens calibration {path.name} needs at least k1, k2, p1, p2 in "
            "OpenCV order; found "
            f"{len(coefficients)} coefficient(s)"
        )
    k1, k2, p1, p2 = coefficients[:4]
    k3 = coefficients[4] if len(coefficients) > 4 else 0.0
    record = DataRecord(
        confidence_grade=ConfidenceGrade(
            str(payload.get("confidence_grade", ConfidenceGrade.RECONSTRUCTED.value))
        ),
        source_id=str(payload["source_id"]),
        source_url=str(payload.get("source_url", "")),
        license=str(payload["license"]),
        captured_at=str(payload["captured_at"]),
        units="normalised image coordinates, OpenCV k1 k2 p1 p2 k3 order",
        uncertainty=str(payload.get("reprojection_error_px", "")),
        checksum=file_checksum(path),
        notes=str(payload.get("notes", "")),
    )
    return LensDistortion(k1=k1, k2=k2, k3=k3, p1=p1, p2=p2), record
