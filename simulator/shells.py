"""Data-driven aerial shell library.

The simulator previously held a single global :class:`~simulator.config.ShellConfig`
— one shell type, always fired vertically from the origin, with 8,000
identically coloured stars on an isotropic sphere. That is a peony and nothing
else.

This module expresses a shell as a *profile*: ballistics, a break pattern, an
optical description, and an energy budget. Break patterns are star **emission
distributions**, not new solvers, so they reuse the existing structure-of-arrays
integration in :mod:`simulator.physics` and the energy-conserving radiant power
solve in :mod:`simulator.lighting`.

Nothing here describes a chemical composition, formulation, or manufacturing
procedure. Colour is a dominant emission wavelength or a colour temperature;
energy is a specific energy in J/kg; a break is a velocity distribution. These
are optical and mechanical outcomes expressed as abstract parameters.

**Every profile in the shipped library is confidence grade D.** No measured
shell record for the 2024-10-05 performance has been obtained, so these are
archetypes chosen to reproduce the documented visual behaviour of each named
effect. They are not a claim about any shell that was actually fired.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import math

import numpy as np

from .color import blackbody_rgb, wavelength_rgb
from .config import ShellConfig
from .provenance import ConfidenceGrade


class BreakPattern(Enum):
    """Geometry of the star velocity distribution at the break."""

    PEONY = "peony"
    CHRYSANTHEMUM = "chrysanthemum"
    WILLOW = "willow"
    PALM = "palm"
    RING = "ring"
    CROSSETTE = "crossette"
    HORSETAIL = "horsetail"
    COMET = "comet"
    MINE = "mine"
    FAN = "fan"
    WATERFALL = "waterfall"


class FlickerMode(Enum):
    """Temporal modulation applied to a star's radiant output."""

    STEADY = "steady"
    STROBE = "strobe"
    """Deep, near-square modulation: the star visibly extinguishes between pulses."""

    CRACKLE = "crackle"
    """Fast, shallow, irregular modulation from many small emitters."""


# Dominant emission wavelengths in nanometres. These are appearance parameters:
# each names a colour and the band it peaks in, with no statement about how the
# emission is produced.
EMISSION_NM_DEEP_RED = 640.0
EMISSION_NM_ORANGE = 597.0
EMISSION_NM_YELLOW = 589.0
EMISSION_NM_GREEN = 515.0
EMISSION_NM_BLUE = 452.0
EMISSION_NM_VIOLET = 430.0

# Incandescent effects are genuinely broadband, so a colour temperature is the
# correct description for them rather than a dominant wavelength.
INCANDESCENT_GOLD_K = 2_300.0
INCANDESCENT_SILVER_K = 5_200.0


@dataclass(frozen=True, slots=True)
class SecondaryBreak:
    """A break carried by a primary star and released at the end of its burn.

    Used by crossette and multi-break shells. The children draw from their own
    declared composition mass, so adding a secondary break cannot silently
    inflate the parent shell's energy or smoke budget.
    """

    pattern: BreakPattern
    star_count: int
    parent_fraction: float = 1.0
    speed_mean_mps: float = 22.0
    speed_std_mps: float = 2.5
    lifetime_mean_s: float = 0.85
    lifetime_std_s: float = 0.08
    drag_time_s: float = 0.9
    composition_mass_kg: float = 0.06
    emission_wavelength_nm: float | None = None
    color_temperature_k: float = INCANDESCENT_GOLD_K

    def __post_init__(self) -> None:
        if self.star_count <= 0:
            raise ValueError(
                f"secondary break needs a positive star_count, got {self.star_count}"
            )
        if not 0.0 < self.parent_fraction <= 1.0:
            raise ValueError(
                f"parent_fraction must be in (0, 1], got {self.parent_fraction}"
            )


@dataclass(frozen=True, slots=True)
class ShellProfile:
    """A calibratable aerial shell expressed in SI units.

    Field names carry their units. Every field that is not a pure geometry
    choice is a physical quantity that a measured shell record can replace
    without changing any solver.
    """

    profile_id: str
    display_name: str
    pattern: BreakPattern = BreakPattern.PEONY

    # -- ballistics --------------------------------------------------------
    dry_mass_kg: float = 1.15
    diameter_m: float = 0.10
    launch_speed_mps: float = 78.0
    fuse_delay_s: float = 3.05
    drag_coefficient: float = 0.47

    # -- break distribution ------------------------------------------------
    burst_star_count: int = 8_000
    star_speed_mean_mps: float = 30.0
    star_speed_std_mps: float = 4.5
    star_lifetime_mean_s: float = 2.25
    star_lifetime_std_s: float = 0.22
    star_drag_time_s: float = 1.35
    pattern_spread_deg: float = 12.0
    """Angular half-width for directional patterns; ignored by isotropic ones."""

    # -- optics ------------------------------------------------------------
    emission_wavelength_nm: float | None = None
    """Dominant emission wavelength. ``None`` selects incandescent colour."""

    color_temperature_k: float = INCANDESCENT_GOLD_K
    secondary_emission_wavelength_nm: float | None = None
    """Colour a colour-changing star transitions to, if any."""

    color_change_fraction: float = 0.55
    """Normalised burn age at which a colour-changing star switches."""

    flicker: FlickerMode = FlickerMode.STEADY
    flicker_hz: float = 0.0

    # -- energy budget -----------------------------------------------------
    star_composition_mass_kg: float = 0.78
    star_radiative_energy_fraction: float = 0.15
    star_smoke_yield_fraction: float = 0.24
    star_specific_energy_j_kg: float = 4_200_000.0
    star_post_combustion_thermal_fraction: float = 0.06
    burst_charge_mass_kg: float = 0.085
    burst_specific_energy_j_kg: float = 3_000_000.0
    smoke_yield_fraction: float = 0.12
    post_blast_thermal_fraction: float = 0.18

    # -- composition -------------------------------------------------------
    secondary: SecondaryBreak | None = None

    # -- provenance --------------------------------------------------------
    confidence_grade: ConfidenceGrade = ConfidenceGrade.ARTISTIC
    source_note: str = (
        "Synthetic archetype. No measured shell record has been obtained."
    )

    def __post_init__(self) -> None:
        if self.burst_star_count <= 0:
            raise ValueError(
                f"{self.profile_id}: burst_star_count must be positive"
            )
        if self.star_composition_mass_kg <= 0.0:
            raise ValueError(
                f"{self.profile_id}: star_composition_mass_kg must be positive"
            )
        if not 0.0 <= self.star_radiative_energy_fraction <= 1.0:
            raise ValueError(
                f"{self.profile_id}: star_radiative_energy_fraction must be a fraction"
            )
        if self.flicker is not FlickerMode.STEADY and self.flicker_hz <= 0.0:
            raise ValueError(
                f"{self.profile_id}: {self.flicker.value} requires a positive flicker_hz"
            )

    # -- derived optics ----------------------------------------------------

    def primary_color_linear(self) -> np.ndarray:
        if self.emission_wavelength_nm is None:
            return blackbody_rgb(self.color_temperature_k)
        return wavelength_rgb(self.emission_wavelength_nm)

    def secondary_color_linear(self) -> np.ndarray | None:
        if self.secondary_emission_wavelength_nm is None:
            return None
        return wavelength_rgb(self.secondary_emission_wavelength_nm)

    @property
    def changes_color(self) -> bool:
        return self.secondary_emission_wavelength_nm is not None

    # -- energy ------------------------------------------------------------

    @property
    def star_chemical_energy_j(self) -> float:
        return self.star_composition_mass_kg * self.star_specific_energy_j_kg

    @property
    def burst_chemical_energy_j(self) -> float:
        return self.burst_charge_mass_kg * self.burst_specific_energy_j_kg

    def total_composition_mass_kg(self) -> float:
        """Declared reactive mass including any secondary break."""

        total = self.star_composition_mass_kg
        if self.secondary is not None:
            total += self.secondary.composition_mass_kg
        return total

    # -- interoperation with the legacy single-shell config -----------------

    @classmethod
    def from_shell_config(
        cls, config: ShellConfig, profile_id: str = "legacy_default"
    ) -> "ShellProfile":
        """Wrap the historical global shell config as a peony profile.

        Reproduces the previous behaviour exactly, so existing callers and
        tests that build a :class:`~simulator.config.ShellConfig` keep working.
        """

        return cls(
            profile_id=profile_id,
            display_name="Legacy development shell",
            pattern=BreakPattern.PEONY,
            dry_mass_kg=config.dry_mass_kg,
            diameter_m=config.diameter_m,
            launch_speed_mps=config.launch_speed_mps,
            fuse_delay_s=config.fuse_delay_s,
            drag_coefficient=config.drag_coefficient,
            burst_star_count=config.burst_star_count,
            star_speed_mean_mps=config.star_speed_mean_mps,
            star_speed_std_mps=config.star_speed_std_mps,
            star_lifetime_mean_s=config.star_lifetime_mean_s,
            star_lifetime_std_s=config.star_lifetime_std_s,
            star_drag_time_s=config.star_drag_time_s,
            emission_wavelength_nm=None,
            color_temperature_k=config.color_temperature_k,
            star_composition_mass_kg=config.star_composition_mass_kg,
            star_radiative_energy_fraction=config.star_radiative_energy_fraction,
            star_smoke_yield_fraction=config.star_smoke_yield_fraction,
            star_specific_energy_j_kg=config.star_specific_energy_j_kg,
            star_post_combustion_thermal_fraction=(
                config.star_post_combustion_thermal_fraction
            ),
            burst_charge_mass_kg=config.burst_charge_mass_kg,
            burst_specific_energy_j_kg=config.burst_specific_energy_j_kg,
            smoke_yield_fraction=config.smoke_yield_fraction,
            post_blast_thermal_fraction=config.post_blast_thermal_fraction,
            confidence_grade=ConfidenceGrade.MODELLED,
            source_note=(
                "Historical development shell retained for continuity with "
                "previously recorded measurements."
            ),
        )

    @classmethod
    def coerce(cls, config: "ShellProfile | ShellConfig") -> "ShellProfile":
        if isinstance(config, cls):
            return config
        return cls.from_shell_config(config)


# ---------------------------------------------------------------------------
# Emission geometry
# ---------------------------------------------------------------------------


def _isotropic(count: int, generator: np.random.Generator) -> np.ndarray:
    """Uniform directions on the sphere.

    Draw order is preserved from the original peony implementation so a fixed
    seed reproduces the previously recorded burst.
    """

    directions = generator.normal(size=(count, 3)).astype(np.float32)
    directions /= np.maximum(
        np.linalg.norm(directions, axis=1, keepdims=True), 1e-7
    )
    return directions


def _orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane normal to ``axis``."""

    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    reference = (
        np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(axis[1])) > 0.9
        else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    )
    first = np.cross(axis, reference)
    first /= max(float(np.linalg.norm(first)), 1e-9)
    second = np.cross(axis, first)
    return first.astype(np.float32), second.astype(np.float32)


def _cone(
    count: int,
    generator: np.random.Generator,
    axis: np.ndarray,
    half_angle_deg: float,
) -> np.ndarray:
    """Directions uniformly distributed inside a cone about ``axis``."""

    axis = (axis / max(float(np.linalg.norm(axis)), 1e-9)).astype(np.float32)
    first, second = _orthonormal_basis(axis)
    cos_limit = math.cos(math.radians(max(half_angle_deg, 1e-3)))
    cosine = generator.uniform(cos_limit, 1.0, count).astype(np.float32)
    sine = np.sqrt(np.maximum(1.0 - cosine * cosine, 0.0))
    phi = generator.uniform(0.0, 2.0 * math.pi, count).astype(np.float32)
    return (
        axis[None, :] * cosine[:, None]
        + first[None, :] * (sine * np.cos(phi))[:, None]
        + second[None, :] * (sine * np.sin(phi))[:, None]
    ).astype(np.float32)


def _annulus(
    count: int,
    generator: np.random.Generator,
    axis: np.ndarray,
    thickness_deg: float,
) -> np.ndarray:
    """Directions near the great circle normal to ``axis``."""

    axis = (axis / max(float(np.linalg.norm(axis)), 1e-9)).astype(np.float32)
    first, second = _orthonormal_basis(axis)
    phi = generator.uniform(0.0, 2.0 * math.pi, count).astype(np.float32)
    tilt = np.radians(
        generator.normal(0.0, max(thickness_deg, 1e-3) / 3.0, count)
    ).astype(np.float32)
    return (
        first[None, :] * (np.cos(tilt) * np.cos(phi))[:, None]
        + second[None, :] * (np.cos(tilt) * np.sin(phi))[:, None]
        + axis[None, :] * np.sin(tilt)[:, None]
    ).astype(np.float32)


def _hemisphere(
    count: int, generator: np.random.Generator, axis: np.ndarray, bias: float
) -> np.ndarray:
    """Isotropic directions pushed toward ``axis`` by ``bias`` in [0, 1]."""

    axis = (axis / max(float(np.linalg.norm(axis)), 1e-9)).astype(np.float32)
    directions = _isotropic(count, generator)
    directions = directions * (1.0 - bias) + axis[None, :] * bias
    return (
        directions
        / np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-7)
    ).astype(np.float32)


UP_EUS = np.array([0.0, 1.0, 0.0], dtype=np.float32)
DOWN_EUS = np.array([0.0, -1.0, 0.0], dtype=np.float32)


def emission_directions(
    pattern: BreakPattern,
    count: int,
    generator: np.random.Generator,
    spread_deg: float = 12.0,
    axis_eus: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample break directions and per-star speed scales for a pattern.

    Returns ``(directions, speed_scale)`` where ``directions`` is an
    ``(n, 3)`` array of East-Up-South unit vectors and ``speed_scale``
    multiplies the profile's mean speed. Separating the scale from the
    direction lets a pattern be radially graded — a palm's spokes are thicker
    at the base — without a bespoke solver.
    """

    if count <= 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )
    axis = UP_EUS if axis_eus is None else np.asarray(axis_eus, dtype=np.float32)
    ones = np.ones(count, dtype=np.float32)

    if pattern in (BreakPattern.PEONY, BreakPattern.CHRYSANTHEMUM):
        return _isotropic(count, generator), ones

    if pattern is BreakPattern.WILLOW:
        # Slightly upward-weighted so the canopy forms above the break before
        # drag and gravity draw the long-burning stars down into trails.
        return _hemisphere(count, generator, UP_EUS, 0.35), ones

    if pattern is BreakPattern.PALM:
        # A small number of thick radial spokes rather than a uniform shell.
        spokes = max(int(round(360.0 / max(spread_deg, 1.0))), 5)
        spoke_index = generator.integers(0, spokes, count)
        spoke_phi = (spoke_index.astype(np.float32) / spokes) * 2.0 * math.pi
        jitter = generator.normal(0.0, 0.05, (count, 2)).astype(np.float32)
        first, second = _orthonormal_basis(UP_EUS)
        elevation = generator.uniform(0.15, 1.0, count).astype(np.float32)
        radial = np.sqrt(np.maximum(1.0 - elevation * elevation, 0.0))
        directions = (
            UP_EUS[None, :] * elevation[:, None]
            + first[None, :] * (radial * (np.cos(spoke_phi) + jitter[:, 0]))[:, None]
            + second[None, :] * (radial * (np.sin(spoke_phi) + jitter[:, 1]))[:, None]
        )
        directions /= np.maximum(
            np.linalg.norm(directions, axis=1, keepdims=True), 1e-7
        )
        # Spokes thin toward their tips: outer stars carry more speed.
        return directions.astype(np.float32), (0.75 + 0.5 * elevation).astype(
            np.float32
        )

    if pattern is BreakPattern.RING:
        return _annulus(count, generator, axis, spread_deg), ones

    if pattern is BreakPattern.CROSSETTE:
        # Few, heavy, evenly spaced primaries that split later.
        return _annulus(count, generator, axis, spread_deg * 0.5), ones

    if pattern is BreakPattern.HORSETAIL:
        return _hemisphere(count, generator, DOWN_EUS, 0.55), (
            0.55 + 0.45 * generator.random(count)
        ).astype(np.float32)

    if pattern is BreakPattern.COMET:
        return _cone(count, generator, axis, max(spread_deg * 0.25, 0.5)), ones

    if pattern is BreakPattern.MINE:
        # Ground-launched cone: a wide upward spray from the firing position.
        return _cone(count, generator, UP_EUS, spread_deg), (
            0.6 + 0.8 * generator.random(count)
        ).astype(np.float32)

    if pattern is BreakPattern.FAN:
        # Planar sector: a fan sweeps within one vertical plane.
        first, second = _orthonormal_basis(axis)
        half = math.radians(max(spread_deg, 1.0))
        angle = generator.uniform(-half, half, count).astype(np.float32)
        thickness = np.radians(
            generator.normal(0.0, 1.5, count)
        ).astype(np.float32)
        directions = (
            first[None, :] * (np.cos(angle) * np.cos(thickness))[:, None]
            + second[None, :] * (np.sin(angle) * np.cos(thickness))[:, None]
            + axis[None, :] * np.sin(thickness)[:, None]
        )
        directions /= np.maximum(
            np.linalg.norm(directions, axis=1, keepdims=True), 1e-7
        )
        return directions.astype(np.float32), ones

    if pattern is BreakPattern.WATERFALL:
        # Wide, slow, downward curtain.
        directions = _hemisphere(count, generator, DOWN_EUS, 0.72)
        return directions, (0.25 + 0.35 * generator.random(count)).astype(
            np.float32
        )

    raise ValueError(f"unhandled break pattern {pattern!r}")


# ---------------------------------------------------------------------------
# Shipped library
# ---------------------------------------------------------------------------


def _profile(**kwargs) -> ShellProfile:
    return ShellProfile(**kwargs)


_LIBRARY: tuple[ShellProfile, ...] = (
    _profile(
        profile_id="peony_100mm_gold",
        display_name="100 mm gold peony",
        pattern=BreakPattern.PEONY,
        color_temperature_k=INCANDESCENT_GOLD_K,
    ),
    _profile(
        profile_id="peony_150mm_red",
        display_name="150 mm red peony",
        pattern=BreakPattern.PEONY,
        diameter_m=0.15,
        dry_mass_kg=3.4,
        launch_speed_mps=96.0,
        fuse_delay_s=4.1,
        burst_star_count=11_000,
        star_speed_mean_mps=38.0,
        star_composition_mass_kg=1.6,
        emission_wavelength_nm=EMISSION_NM_DEEP_RED,
    ),
    _profile(
        profile_id="chrysanthemum_125mm_silver",
        display_name="125 mm silver chrysanthemum",
        pattern=BreakPattern.CHRYSANTHEMUM,
        diameter_m=0.125,
        dry_mass_kg=2.1,
        launch_speed_mps=87.0,
        fuse_delay_s=3.6,
        burst_star_count=9_500,
        star_lifetime_mean_s=3.1,
        star_lifetime_std_s=0.28,
        # A chrysanthemum differs from a peony by leaving a visible trail:
        # longer burn and stronger drag, not a different break geometry.
        star_drag_time_s=0.95,
        star_composition_mass_kg=1.05,
        color_temperature_k=INCANDESCENT_SILVER_K,
    ),
    _profile(
        profile_id="willow_150mm_gold",
        display_name="150 mm gold willow",
        pattern=BreakPattern.WILLOW,
        diameter_m=0.15,
        dry_mass_kg=3.2,
        launch_speed_mps=92.0,
        fuse_delay_s=4.3,
        burst_star_count=2_600,
        star_speed_mean_mps=16.0,
        star_speed_std_mps=2.0,
        # Long, heavy, slow-burning stars that fall into drooping trails.
        star_lifetime_mean_s=6.4,
        star_lifetime_std_s=0.45,
        star_drag_time_s=3.4,
        star_composition_mass_kg=1.35,
        color_temperature_k=INCANDESCENT_GOLD_K,
    ),
    _profile(
        profile_id="palm_150mm_gold",
        display_name="150 mm gold palm",
        pattern=BreakPattern.PALM,
        diameter_m=0.15,
        dry_mass_kg=3.3,
        launch_speed_mps=90.0,
        fuse_delay_s=4.0,
        burst_star_count=1_400,
        star_speed_mean_mps=27.0,
        star_lifetime_mean_s=3.4,
        star_drag_time_s=2.2,
        pattern_spread_deg=32.0,
        star_composition_mass_kg=1.1,
    ),
    _profile(
        profile_id="ring_125mm_green",
        display_name="125 mm green ring",
        pattern=BreakPattern.RING,
        diameter_m=0.125,
        dry_mass_kg=2.0,
        launch_speed_mps=86.0,
        fuse_delay_s=3.7,
        burst_star_count=900,
        star_speed_mean_mps=31.0,
        star_speed_std_mps=1.1,
        star_lifetime_mean_s=1.9,
        star_lifetime_std_s=0.08,
        pattern_spread_deg=4.0,
        star_composition_mass_kg=0.42,
        emission_wavelength_nm=EMISSION_NM_GREEN,
    ),
    _profile(
        profile_id="crossette_125mm_blue",
        display_name="125 mm blue crossette",
        pattern=BreakPattern.CROSSETTE,
        diameter_m=0.125,
        dry_mass_kg=2.2,
        launch_speed_mps=85.0,
        fuse_delay_s=3.8,
        burst_star_count=48,
        star_speed_mean_mps=26.0,
        star_speed_std_mps=1.0,
        star_lifetime_mean_s=1.35,
        star_lifetime_std_s=0.05,
        star_drag_time_s=2.4,
        pattern_spread_deg=10.0,
        star_composition_mass_kg=0.34,
        emission_wavelength_nm=EMISSION_NM_BLUE,
        secondary=SecondaryBreak(
            pattern=BreakPattern.RING,
            star_count=4,
            speed_mean_mps=19.0,
            speed_std_mps=0.8,
            lifetime_mean_s=0.9,
            composition_mass_kg=0.11,
            emission_wavelength_nm=EMISSION_NM_BLUE,
        ),
    ),
    _profile(
        profile_id="multibreak_150mm_red_green",
        display_name="150 mm red-to-green multi-break",
        pattern=BreakPattern.PEONY,
        diameter_m=0.15,
        dry_mass_kg=3.5,
        launch_speed_mps=94.0,
        fuse_delay_s=4.2,
        burst_star_count=2_200,
        star_speed_mean_mps=24.0,
        star_lifetime_mean_s=1.6,
        star_composition_mass_kg=0.95,
        emission_wavelength_nm=EMISSION_NM_DEEP_RED,
        secondary=SecondaryBreak(
            pattern=BreakPattern.PEONY,
            star_count=6,
            parent_fraction=0.25,
            speed_mean_mps=17.0,
            lifetime_mean_s=1.2,
            composition_mass_kg=0.38,
            emission_wavelength_nm=EMISSION_NM_GREEN,
        ),
    ),
    _profile(
        profile_id="horsetail_150mm_gold",
        display_name="150 mm gold horsetail",
        pattern=BreakPattern.HORSETAIL,
        diameter_m=0.15,
        dry_mass_kg=3.6,
        launch_speed_mps=74.0,
        fuse_delay_s=3.2,
        burst_star_count=1_100,
        star_speed_mean_mps=13.0,
        star_lifetime_mean_s=4.2,
        star_drag_time_s=3.0,
        star_composition_mass_kg=1.2,
    ),
    _profile(
        profile_id="strobe_125mm_white",
        display_name="125 mm white strobe",
        pattern=BreakPattern.PEONY,
        diameter_m=0.125,
        dry_mass_kg=2.1,
        launch_speed_mps=86.0,
        fuse_delay_s=3.7,
        burst_star_count=1_800,
        star_speed_mean_mps=22.0,
        star_lifetime_mean_s=4.0,
        star_lifetime_std_s=0.4,
        star_composition_mass_kg=0.8,
        color_temperature_k=INCANDESCENT_SILVER_K,
        flicker=FlickerMode.STROBE,
        flicker_hz=6.5,
    ),
    _profile(
        profile_id="crackling_100mm_gold",
        display_name="100 mm crackling pistil",
        pattern=BreakPattern.PEONY,
        burst_star_count=5_200,
        star_speed_mean_mps=19.0,
        star_lifetime_mean_s=1.1,
        star_lifetime_std_s=0.16,
        star_drag_time_s=0.8,
        star_composition_mass_kg=0.52,
        flicker=FlickerMode.CRACKLE,
        flicker_hz=42.0,
    ),
    _profile(
        profile_id="colorchange_125mm_green_red",
        display_name="125 mm colour-changing green to red",
        pattern=BreakPattern.PEONY,
        diameter_m=0.125,
        dry_mass_kg=2.0,
        launch_speed_mps=86.0,
        fuse_delay_s=3.7,
        burst_star_count=4_800,
        star_lifetime_mean_s=2.8,
        star_composition_mass_kg=0.9,
        emission_wavelength_nm=EMISSION_NM_GREEN,
        secondary_emission_wavelength_nm=EMISSION_NM_DEEP_RED,
        color_change_fraction=0.55,
    ),
    _profile(
        profile_id="comet_75mm_gold",
        display_name="75 mm gold comet",
        pattern=BreakPattern.COMET,
        diameter_m=0.075,
        dry_mass_kg=0.62,
        launch_speed_mps=68.0,
        fuse_delay_s=0.35,
        burst_star_count=160,
        star_speed_mean_mps=6.0,
        star_speed_std_mps=1.2,
        star_lifetime_mean_s=2.6,
        star_drag_time_s=2.6,
        pattern_spread_deg=6.0,
        star_composition_mass_kg=0.16,
    ),
    _profile(
        profile_id="mine_100mm_mixed",
        display_name="100 mm mixed mine",
        pattern=BreakPattern.MINE,
        diameter_m=0.10,
        dry_mass_kg=0.9,
        # A mine has no lift charge and no time fuse: it opens at the tube.
        launch_speed_mps=0.0,
        fuse_delay_s=0.0,
        burst_star_count=2_400,
        star_speed_mean_mps=54.0,
        star_speed_std_mps=7.0,
        star_lifetime_mean_s=2.1,
        star_drag_time_s=1.8,
        pattern_spread_deg=26.0,
        star_composition_mass_kg=0.62,
        emission_wavelength_nm=EMISSION_NM_ORANGE,
    ),
    _profile(
        profile_id="fan_100mm_yellow",
        display_name="100 mm yellow fan",
        pattern=BreakPattern.FAN,
        burst_star_count=1_600,
        star_speed_mean_mps=33.0,
        star_lifetime_mean_s=2.0,
        pattern_spread_deg=42.0,
        star_composition_mass_kg=0.55,
        emission_wavelength_nm=EMISSION_NM_YELLOW,
    ),
    _profile(
        profile_id="waterfall_bridge_gold",
        display_name="Bridge waterfall curtain",
        pattern=BreakPattern.WATERFALL,
        diameter_m=0.05,
        dry_mass_kg=0.4,
        launch_speed_mps=0.0,
        fuse_delay_s=0.0,
        burst_star_count=3_600,
        star_speed_mean_mps=4.5,
        star_speed_std_mps=1.0,
        star_lifetime_mean_s=7.5,
        star_lifetime_std_s=0.8,
        star_drag_time_s=4.5,
        star_composition_mass_kg=1.4,
        # A curtain hangs from a structure rather than bursting in air, so its
        # lift and burst charges are zero.
        burst_charge_mass_kg=0.0,
        smoke_yield_fraction=0.0,
        post_blast_thermal_fraction=0.0,
    ),
    _profile(
        profile_id="violet_125mm",
        display_name="125 mm violet peony",
        pattern=BreakPattern.PEONY,
        diameter_m=0.125,
        dry_mass_kg=2.0,
        launch_speed_mps=86.0,
        fuse_delay_s=3.7,
        burst_star_count=4_600,
        star_composition_mass_kg=0.86,
        emission_wavelength_nm=EMISSION_NM_VIOLET,
    ),
)


class ShellLibrary:
    """Immutable lookup from profile id to :class:`ShellProfile`."""

    def __init__(self, profiles: tuple[ShellProfile, ...] = _LIBRARY) -> None:
        self._profiles: dict[str, ShellProfile] = {}
        for profile in profiles:
            if profile.profile_id in self._profiles:
                raise ValueError(f"duplicate profile id {profile.profile_id!r}")
            self._profiles[profile.profile_id] = profile

    def __contains__(self, profile_id: object) -> bool:
        return profile_id in self._profiles

    def __len__(self) -> int:
        return len(self._profiles)

    def __iter__(self):
        return iter(self._profiles.values())

    def get(self, profile_id: str) -> ShellProfile:
        try:
            return self._profiles[profile_id]
        except KeyError:
            raise KeyError(
                f"unknown shell profile {profile_id!r}; have "
                f"{sorted(self._profiles)}"
            ) from None

    def ids(self) -> list[str]:
        return sorted(self._profiles)

    def patterns(self) -> set[BreakPattern]:
        return {profile.pattern for profile in self._profiles.values()}

    def with_profile(self, profile: ShellProfile) -> "ShellLibrary":
        """Return a new library with ``profile`` added or replacing its id."""

        merged = {**self._profiles, profile.profile_id: profile}
        return ShellLibrary(tuple(merged.values()))


SHELL_LIBRARY = ShellLibrary()

__all__ = [
    "BreakPattern",
    "FlickerMode",
    "SHELL_LIBRARY",
    "SecondaryBreak",
    "ShellLibrary",
    "ShellProfile",
    "emission_directions",
    "replace",
]
