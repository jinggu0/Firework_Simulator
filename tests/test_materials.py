import re

import numpy as np
import pytest

from simulator import shaders
from simulator.materials import (
    MATERIAL_LIBRARY,
    MATERIAL_SLOTS,
    MaterialChannels,
    MaterialLibrary,
    PatternKind,
)
from simulator.provenance import ConfidenceGrade
from simulator.scene import (
    SURFACE_GRASS_BLADE,
    SURFACE_LAMP,
    SURFACE_METAL,
    SURFACE_ROAD,
    SURFACE_WALL,
)


def test_every_surface_code_has_a_material() -> None:
    assert len(MATERIAL_LIBRARY) == MATERIAL_SLOTS
    for code in range(MATERIAL_SLOTS):
        assert MATERIAL_LIBRARY.get(code) is not None


def test_a_gap_in_the_table_is_rejected() -> None:
    partial = tuple(MATERIAL_LIBRARY)[:-1]
    with pytest.raises(ValueError, match="no material for surface codes"):
        MaterialLibrary(partial)


def test_duplicate_surface_codes_are_rejected() -> None:
    material = MATERIAL_LIBRARY.get(SURFACE_ROAD)
    with pytest.raises(ValueError, match="duplicate surface code"):
        MaterialLibrary((material, material))


def test_channel_fractions_are_validated() -> None:
    for channel in ("roughness", "metallic", "ambient_occlusion", "transmission"):
        with pytest.raises(ValueError, match=channel):
            MaterialChannels(name="bad", surface_code=0.0, **{channel: 1.5})
    with pytest.raises(ValueError, match="emissive_scale"):
        MaterialChannels(name="bad", surface_code=0.0, emissive_scale=-1.0)


def test_nothing_in_the_table_claims_to_be_measured() -> None:
    # No measured reflectance for any Yeouido surface has been obtained.
    for material in MATERIAL_LIBRARY:
        assert not material.confidence_grade.is_evidence, material.name
        assert material.confidence_grade is ConfidenceGrade.ARTISTIC
        assert material.source_note
    assert MATERIAL_LIBRARY.summary()["measured"] is False


# --- channel semantics ------------------------------------------------------


def test_only_metal_is_conductive() -> None:
    metallic = {
        material.name: material.metallic for material in MATERIAL_LIBRARY
    }
    assert metallic["painted_metal"] > 0.5
    for name, value in metallic.items():
        if name not in ("painted_metal", "lamp_head"):
            assert value == 0.0, name


def test_vegetation_transmits_and_pavement_does_not() -> None:
    by_name = {material.name: material for material in MATERIAL_LIBRARY}
    # Thin leaves and blades glow when backlit; concrete does not.
    assert by_name["grass_blade"].transmission > by_name["tree_foliage"].transmission
    assert by_name["tree_foliage"].transmission > 0.0
    assert by_name["asphalt_road"].transmission == 0.0
    assert by_name["concrete_footway"].transmission == 0.0


def test_facades_are_smoother_than_ground_surfaces() -> None:
    by_name = {material.name: material for material in MATERIAL_LIBRARY}
    assert by_name["facade_wall"].roughness < by_name["asphalt_road"].roughness
    assert by_name["painted_metal"].roughness < by_name["facade_wall"].roughness


def test_only_the_lamp_head_emits() -> None:
    emitters = [
        material.name
        for material in MATERIAL_LIBRARY
        if material.emissive_scale > 0.0
    ]
    assert emitters == ["lamp_head"]
    lamp = MATERIAL_LIBRARY.get(SURFACE_LAMP)
    # Warm luminaire tint; the radiance itself comes from the LED energy budget.
    assert lamp.emissive_color[0] > lamp.emissive_color[2]


def test_facade_surfaces_defer_their_base_colour() -> None:
    wall = MATERIAL_LIBRARY.get(SURFACE_WALL)
    assert wall.pattern is PatternKind.FACADE
    # Its PBR channels still come from the table even though the colour does not.
    assert 0.0 < wall.roughness < 1.0


def test_occluded_surfaces_are_the_enclosed_ones() -> None:
    by_name = {material.name: material for material in MATERIAL_LIBRARY}
    # A tree crown occludes far more of the sky hemisphere than open asphalt.
    assert by_name["tree_foliage"].ambient_occlusion < by_name[
        "asphalt_road"
    ].ambient_occlusion


# --- packing ----------------------------------------------------------------


def test_packed_arrays_have_the_shapes_the_shader_declares() -> None:
    assert MATERIAL_LIBRARY.base_primary().shape == (MATERIAL_SLOTS, 3)
    assert MATERIAL_LIBRARY.base_secondary().shape == (MATERIAL_SLOTS, 3)
    assert MATERIAL_LIBRARY.pattern().shape == (MATERIAL_SLOTS, 4)
    assert MATERIAL_LIBRARY.reflectance().shape == (MATERIAL_SLOTS, 4)
    assert MATERIAL_LIBRARY.emissive().shape == (MATERIAL_SLOTS, 4)
    assert MATERIAL_LIBRARY.relief().shape == (MATERIAL_SLOTS, 2)
    for array in (
        MATERIAL_LIBRARY.base_primary(),
        MATERIAL_LIBRARY.pattern(),
        MATERIAL_LIBRARY.reflectance(),
    ):
        assert array.dtype == np.float32
        assert np.isfinite(array).all()


def test_rows_are_ordered_by_surface_code() -> None:
    # The shader indexes by surface code, so row order is load-bearing.
    codes = [material.surface_code for material in MATERIAL_LIBRARY]
    assert codes == sorted(codes)
    reflectance = MATERIAL_LIBRARY.reflectance()
    metal_row = int(round(SURFACE_METAL))
    assert reflectance[metal_row][1] == pytest.approx(
        MATERIAL_LIBRARY.get(SURFACE_METAL).metallic
    )


def test_pattern_kinds_round_trip_through_the_packed_float() -> None:
    packed = MATERIAL_LIBRARY.pattern()
    for row, material in enumerate(MATERIAL_LIBRARY):
        assert int(packed[row][0] + 0.5) == int(material.pattern)


# --- shader agreement -------------------------------------------------------


def test_shader_declares_every_material_array_at_the_right_length() -> None:
    source = shaders.source("scene.frag")
    for name in (
        "material_base_primary",
        "material_base_secondary",
        "material_pattern",
        "material_reflectance",
        "material_emissive",
        "material_relief",
    ):
        match = re.search(rf"uniform\s+\w+\s+{name}\[(\d+)\]", source)
        assert match, name
        assert int(match.group(1)) == MATERIAL_SLOTS, name


def test_shader_pattern_constants_match_the_enum() -> None:
    source = shaders.source("scene.frag")
    for kind in PatternKind:
        match = re.search(
            rf"const\s+int\s+PATTERN_{kind.name}\s*=\s*(\d+)\s*;", source
        )
        assert match, kind.name
        assert int(match.group(1)) == int(kind), kind.name


def test_shader_no_longer_hardcodes_the_surface_branch_chain() -> None:
    source = shaders.source("scene.frag")
    # The chain tested `surface > 15.5`, `> 14.5`, and so on down to `> 2.5`.
    remaining = re.findall(r"surface\s*>\s*\d+\.5", source)
    # Only the wall/roof split survives, because a facade is an assembly.
    assert remaining == ["surface > .5"] or remaining == [], remaining
