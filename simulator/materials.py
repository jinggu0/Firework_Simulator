"""Surface materials as data rather than a shader branch chain.

``scene.frag`` carried a 140-line if/else chain in which every surface's colour,
pattern scale, and blend factor was a literal buried in GLSL. Adding a material
meant editing a shader; auditing one meant reading it. This module lifts those
values into a table with the channel set the product requirements call for, so a
material is a row that can be inspected, graded, and replaced.

Channels
--------
Each material carries base colour, normal strength, roughness, metallic, height,
ambient occlusion, emissive, and transmission. Every one of them is consumed by
the shader; none is a reserved placeholder.

Extension channels named in the architecture — spectral reflectance, index of
refraction, subsurface scattering, anisotropy, clear coat, wetness, weathering —
are **not** present. Adding a field the renderer ignores would misrepresent the
material model, so they arrive with the transport that uses them.

Confidence
----------
No measured reflectance for any surface at Yeouido has been obtained. Every
material here is therefore an appearance calibration, graded ``D``, except where
a value follows from a published physical constant. Nothing in this table is
evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .provenance import ConfidenceGrade
from .scene import (
    SURFACE_BRIDGE,
    SURFACE_CONCRETE,
    SURFACE_CYCLEWAY,
    SURFACE_EARTHWORK,
    SURFACE_FOLIAGE,
    SURFACE_FOOTWAY,
    SURFACE_GARDEN,
    SURFACE_GRASS_BLADE,
    SURFACE_LAMP,
    SURFACE_METAL,
    SURFACE_PLAYGROUND,
    SURFACE_ROAD,
    SURFACE_ROOF,
    SURFACE_RETAINING_WALL,
    SURFACE_SPORT,
    SURFACE_TRAIL,
    SURFACE_VEGETATION,
    SURFACE_WALL,
    SURFACE_WOOD,
)

MATERIAL_SLOTS = 19
"""One row per surface code, 0 through 18."""

DIELECTRIC_NORMAL_REFLECTANCE = 0.04
"""Normal-incidence reflectance of a typical dielectric.

Follows from the Fresnel equation at an index of refraction near 1.5, which
covers paint, concrete, glass, and plastics. Used as the specular base for any
material whose metallic channel is zero.
"""


class PatternKind(IntEnum):
    """How a material's two base colours are blended across the surface.

    These are the procedural forms the shader already implemented; naming them
    turns a branch into a parameter.
    """

    UNIFORM = 0
    """No variation."""

    HASH_CELL = 1
    """Value noise over metric ground cells."""

    HASH_UV = 2
    """Value noise over the surface's own metric coordinates."""

    HASH_VOLUME = 3
    """Value noise including height, so canopy layers differ."""

    GRID = 4
    """Rectangular tiles with gaps, as in rubber playground matting."""

    PANELS = 5
    """Panel field with seams along both axes."""

    STRIPE = 6
    """Painted line along the first surface axis."""

    JOINTS = 7
    """Paving joints along both surface axes."""

    LANE = 8
    """Directional roadway hint in world coordinates."""

    FRESNEL = 9
    """View-dependent blend, for grazing-angle brightening on metal."""

    FACADE = 10
    """Base colour comes from the facade-family path, not from this table.

    A building elevation is an assembly — slabs, mullions, glazing, balconies —
    not a material. Its PBR channels still come from here.
    """


@dataclass(frozen=True, slots=True)
class MaterialChannels:
    """One surface's appearance, in the channel set the renderer consumes."""

    name: str
    surface_code: float

    # -- base colour -------------------------------------------------------
    base_color_primary: tuple[float, float, float] = (0.2, 0.2, 0.2)
    base_color_secondary: tuple[float, float, float] = (0.2, 0.2, 0.2)
    pattern: PatternKind = PatternKind.UNIFORM
    pattern_scale: tuple[float, float] = (1.0, 1.0)
    pattern_mix: float = 1.0
    """How much of the pattern reaches the blend; 1.0 uses its full range."""

    # -- reflectance -------------------------------------------------------
    roughness: float = 0.9
    metallic: float = 0.0
    ambient_occlusion: float = 1.0
    transmission: float = 0.0
    """Diffuse transmission, which is how thin foliage lights from behind."""

    # -- relief ------------------------------------------------------------
    normal_strength: float = 0.0
    """Scales any procedural normal perturbation the surface defines."""

    height_scale: float = 0.0
    """Depth of the pattern read as relief, in metres."""

    # -- emission ----------------------------------------------------------
    emissive_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    emissive_scale: float = 0.0
    """Multiplies the luminaire window radiance, keeping emission radiometric."""

    # -- provenance --------------------------------------------------------
    confidence_grade: ConfidenceGrade = ConfidenceGrade.ARTISTIC
    source_note: str = "Appearance calibration; no measured reflectance held."

    def __post_init__(self) -> None:
        for field_name in (
            "roughness",
            "metallic",
            "ambient_occlusion",
            "transmission",
        ):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{self.name}: {field_name} must be a fraction, got {value}"
                )
        if self.emissive_scale < 0.0:
            raise ValueError(f"{self.name}: emissive_scale must be non-negative")


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------
# Colours and pattern parameters are transcribed from the shader branch chain
# they replace, so moving them here changes no pixels. Roughness, metallic,
# ambient occlusion, and transmission are new: the previous model was purely
# Lambertian and had no way to express them.

_MATERIALS: tuple[MaterialChannels, ...] = (
    MaterialChannels(
        name="facade_wall",
        surface_code=SURFACE_WALL,
        # Generic office cladding. The facade path overrides this per family,
        # because an elevation is an assembly rather than one material.
        base_color_primary=(0.18, 0.20, 0.23),
        base_color_secondary=(0.025, 0.042, 0.075),
        pattern=PatternKind.FACADE,
        # Curtain wall and cladding are smooth enough to glint under a burst.
        roughness=0.35,
        ambient_occlusion=0.92,
    ),
    MaterialChannels(
        name="facade_roof",
        surface_code=SURFACE_ROOF,
        # Generic roof deck; the gold and Parc.1 families override it.
        base_color_primary=(0.12, 0.13, 0.14),
        base_color_secondary=(0.12, 0.13, 0.14),
        pattern=PatternKind.FACADE,
        roughness=0.78,
        ambient_occlusion=0.85,
    ),
    MaterialChannels(
        name="bridge_deck",
        surface_code=SURFACE_BRIDGE,
        base_color_primary=(0.22, 0.24, 0.26),
        base_color_secondary=(0.22, 0.24, 0.26),
        pattern=PatternKind.UNIFORM,
        roughness=0.82,
        ambient_occlusion=0.90,
    ),
    MaterialChannels(
        name="asphalt_road",
        surface_code=SURFACE_ROAD,
        base_color_primary=(0.035, 0.038, 0.042),
        base_color_secondary=(0.035, 0.038, 0.042),
        # Lane geometry follows each road's longitudinal UV in scene.frag;
        # the old world-axis pattern bent or crossed markings on rotated ways.
        pattern=PatternKind.UNIFORM,
        roughness=0.88,
        ambient_occlusion=0.94,
    ),
    MaterialChannels(
        name="grass_sward",
        surface_code=SURFACE_VEGETATION,
        # Early-October event photographs show maintained turf as a subdued
        # olive green under daylight, not the saturated spring green formerly
        # used here.  This is photographic appearance calibration (grade D),
        # not a reflectance measurement.
        base_color_primary=(0.042, 0.105, 0.026),
        base_color_secondary=(0.125, 0.175, 0.045),
        pattern=PatternKind.HASH_CELL,
        pattern_scale=(0.15, 0.15),
        pattern_mix=0.58,
        roughness=0.95,
        # Wind-driven travelling-wave normals; the strength is applied to the
        # existing ripple rather than replacing it.
        normal_strength=1.0,
        transmission=0.10,
        ambient_occlusion=0.88,
    ),
    MaterialChannels(
        name="concrete_footway",
        surface_code=SURFACE_FOOTWAY,
        base_color_primary=(0.24, 0.23, 0.21),
        base_color_secondary=(0.09, 0.09, 0.09),
        pattern=PatternKind.JOINTS,
        pattern_scale=(0.42, 0.28),
        roughness=0.86,
        height_scale=0.004,
        ambient_occlusion=0.93,
    ),
    MaterialChannels(
        name="cycleway_surface",
        surface_code=SURFACE_CYCLEWAY,
        base_color_primary=(0.19, 0.045, 0.032),
        base_color_secondary=(0.19, 0.045, 0.032),
        pattern=PatternKind.UNIFORM,
        pattern_scale=(0.0, 0.0),
        pattern_mix=0.0,
        roughness=0.87,
        ambient_occlusion=0.94,
    ),
    MaterialChannels(
        name="sports_surface",
        surface_code=SURFACE_SPORT,
        base_color_primary=(0.035, 0.13, 0.075),
        base_color_secondary=(0.75, 0.75, 0.75),
        pattern=PatternKind.STRIPE,
        pattern_scale=(0.05, 0.05),
        roughness=0.90,
        ambient_occlusion=0.92,
    ),
    MaterialChannels(
        name="painted_metal",
        surface_code=SURFACE_METAL,
        base_color_primary=(0.18, 0.18, 0.18),
        base_color_secondary=(0.42, 0.42, 0.42),
        pattern=PatternKind.FRESNEL,
        # Railings and posts: the one genuinely conductive surface in the set.
        roughness=0.30,
        metallic=0.85,
        ambient_occlusion=0.88,
    ),
    MaterialChannels(
        name="timber",
        surface_code=SURFACE_WOOD,
        base_color_primary=(0.12, 0.050, 0.018),
        base_color_secondary=(0.28, 0.13, 0.045),
        pattern=PatternKind.HASH_UV,
        pattern_scale=(1.8, 0.18),
        roughness=0.80,
        height_scale=0.002,
        ambient_occlusion=0.90,
    ),
    MaterialChannels(
        name="lamp_head",
        surface_code=SURFACE_LAMP,
        base_color_primary=(0.22, 0.22, 0.22),
        base_color_secondary=(0.22, 0.22, 0.22),
        pattern=PatternKind.UNIFORM,
        roughness=0.40,
        metallic=0.30,
        # Warm luminaire appearance; the radiance itself comes from the LED
        # energy budget rather than from this tint.
        emissive_color=(1.0, 0.48, 0.13),
        emissive_scale=1.8,
        ambient_occlusion=1.0,
    ),
    MaterialChannels(
        name="tree_foliage",
        surface_code=SURFACE_FOLIAGE,
        base_color_primary=(0.026, 0.078, 0.015),
        base_color_secondary=(0.24, 0.155, 0.024),
        pattern=PatternKind.HASH_VOLUME,
        pattern_scale=(0.65, 0.65),
        pattern_mix=0.42,
        roughness=0.94,
        # Leaves are thin enough to glow when lit from behind.
        transmission=0.22,
        ambient_occlusion=0.72,
    ),
    MaterialChannels(
        name="concrete_facility",
        surface_code=SURFACE_CONCRETE,
        base_color_primary=(0.21, 0.21, 0.21),
        base_color_secondary=(0.15, 0.15, 0.15),
        pattern=PatternKind.PANELS,
        pattern_scale=(0.22, 0.22),
        pattern_mix=0.12,
        roughness=0.84,
        height_scale=0.006,
        ambient_occlusion=0.90,
    ),
    MaterialChannels(
        name="playground_matting",
        surface_code=SURFACE_PLAYGROUND,
        base_color_primary=(0.12, 0.035, 0.025),
        base_color_secondary=(0.035, 0.10, 0.12),
        pattern=PatternKind.GRID,
        pattern_scale=(0.25, 0.25),
        roughness=0.92,
        height_scale=0.003,
        ambient_occlusion=0.90,
    ),
    MaterialChannels(
        name="garden_bed",
        surface_code=SURFACE_GARDEN,
        base_color_primary=(0.035, 0.11, 0.025),
        base_color_secondary=(0.17, 0.055, 0.025),
        pattern=PatternKind.HASH_CELL,
        pattern_scale=(0.32, 0.32),
        pattern_mix=0.34,
        roughness=0.93,
        transmission=0.08,
        ambient_occlusion=0.82,
    ),
    MaterialChannels(
        name="compacted_trail",
        surface_code=SURFACE_TRAIL,
        base_color_primary=(0.16, 0.13, 0.095),
        base_color_secondary=(0.26, 0.22, 0.16),
        pattern=PatternKind.HASH_CELL,
        pattern_scale=(1.35, 1.35),
        roughness=0.91,
        ambient_occlusion=0.92,
    ),
    MaterialChannels(
        name="grass_blade",
        surface_code=SURFACE_GRASS_BLADE,
        base_color_primary=(0.030, 0.086, 0.016),
        base_color_secondary=(0.12, 0.17, 0.036),
        pattern=PatternKind.HASH_CELL,
        pattern_scale=(9.0, 9.0),
        pattern_mix=0.62,
        roughness=0.96,
        # Individual blades are the most translucent surface in the scene.
        transmission=0.35,
        ambient_occlusion=0.80,
    ),
    MaterialChannels(
        name="retaining_wall_concrete",
        surface_code=SURFACE_RETAINING_WALL,
        base_color_primary=(0.185, 0.19, 0.185),
        base_color_secondary=(0.105, 0.115, 0.105),
        pattern=PatternKind.PANELS,
        pattern_scale=(0.42, 0.20),
        pattern_mix=0.18,
        roughness=0.91,
        height_scale=0.008,
        ambient_occlusion=0.86,
        source_note=(
            "Weathered retaining-concrete appearance calibration; geometry "
            "requires separate surveyed evidence."
        ),
    ),
    MaterialChannels(
        name="earthwork_slope",
        surface_code=SURFACE_EARTHWORK,
        base_color_primary=(0.070, 0.105, 0.035),
        base_color_secondary=(0.20, 0.145, 0.075),
        pattern=PatternKind.HASH_CELL,
        pattern_scale=(0.36, 0.36),
        pattern_mix=0.45,
        roughness=0.96,
        height_scale=0.010,
        ambient_occlusion=0.84,
        source_note=(
            "Mixed turf and exposed-soil appearance calibration; surveyed "
            "crest/toe geometry remains independent."
        ),
    ),
)


class MaterialLibrary:
    """Surface-code indexed materials, uploadable as shader uniform arrays."""

    def __init__(
        self, materials: tuple[MaterialChannels, ...] = _MATERIALS
    ) -> None:
        self._by_code: dict[int, MaterialChannels] = {}
        for material in materials:
            code = int(round(material.surface_code))
            if code in self._by_code:
                raise ValueError(
                    f"duplicate surface code {code} "
                    f"({material.name} and {self._by_code[code].name})"
                )
            if not 0 <= code < MATERIAL_SLOTS:
                raise ValueError(
                    f"{material.name}: surface code {code} outside "
                    f"0..{MATERIAL_SLOTS - 1}"
                )
            self._by_code[code] = material
        missing = sorted(set(range(MATERIAL_SLOTS)) - set(self._by_code))
        if missing:
            raise ValueError(f"no material for surface codes {missing}")

    def __len__(self) -> int:
        return len(self._by_code)

    def __iter__(self):
        return iter(self._by_code[code] for code in sorted(self._by_code))

    def get(self, surface_code: float) -> MaterialChannels:
        code = int(round(surface_code))
        if code not in self._by_code:
            raise KeyError(f"no material for surface code {code}")
        return self._by_code[code]

    def by_name(self, name: str) -> MaterialChannels:
        for material in self:
            if material.name == name:
                return material
        raise KeyError(f"no material named {name!r}")

    # -- packing -----------------------------------------------------------

    def base_primary(self) -> np.ndarray:
        return np.array(
            [material.base_color_primary for material in self], dtype=np.float32
        )

    def base_secondary(self) -> np.ndarray:
        return np.array(
            [material.base_color_secondary for material in self],
            dtype=np.float32,
        )

    def pattern(self) -> np.ndarray:
        """kind, scale u, scale v, mix."""

        return np.array(
            [
                (
                    float(material.pattern),
                    material.pattern_scale[0],
                    material.pattern_scale[1],
                    material.pattern_mix,
                )
                for material in self
            ],
            dtype=np.float32,
        )

    def reflectance(self) -> np.ndarray:
        """roughness, metallic, ambient occlusion, transmission."""

        return np.array(
            [
                (
                    material.roughness,
                    material.metallic,
                    material.ambient_occlusion,
                    material.transmission,
                )
                for material in self
            ],
            dtype=np.float32,
        )

    def emissive(self) -> np.ndarray:
        """emissive rgb, scale."""

        return np.array(
            [
                (*material.emissive_color, material.emissive_scale)
                for material in self
            ],
            dtype=np.float32,
        )

    def relief(self) -> np.ndarray:
        """normal strength, height scale in metres."""

        return np.array(
            [
                (material.normal_strength, material.height_scale)
                for material in self
            ],
            dtype=np.float32,
        )

    def upload(self, program) -> None:
        """Write every channel array into a shader program's uniforms."""

        program["material_base_primary"].write(self.base_primary().tobytes())
        program["material_base_secondary"].write(
            self.base_secondary().tobytes()
        )
        program["material_pattern"].write(self.pattern().tobytes())
        program["material_reflectance"].write(self.reflectance().tobytes())
        program["material_emissive"].write(self.emissive().tobytes())
        program["material_relief"].write(self.relief().tobytes())

    def summary(self) -> dict[str, object]:
        grades: dict[str, int] = {}
        for material in self:
            key = material.confidence_grade.value
            grades[key] = grades.get(key, 0) + 1
        return {
            "materials": len(self),
            "patterns_used": sorted(
                {material.pattern.name for material in self}
            ),
            "confidence_grades": grades,
            "measured": any(
                material.confidence_grade.is_evidence for material in self
            ),
        }


MATERIAL_LIBRARY = MaterialLibrary()
