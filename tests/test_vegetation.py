import math
import re

import numpy as np
import pytest

from simulator import shaders
from simulator.camera_optics import vertical_fov_deg
from simulator.config import PhysicalCameraConfig, RenderConfig
from simulator.scene import SURFACE_GRASS_BLADE, load_scene
from simulator.starcatalogue import DEFAULT_CATALOGUE_PATH
from simulator.vegetation import (
    BLADE_WIDTH_M,
    CUTOFF_PIXELS,
    EVENT_SITE_DETAIL_RADIUS_M,
    FULL_DETAIL_PIXELS,
    VegetationLod,
    angular_resolution_rad_per_pixel,
    subpixel_distance_m,
)

SCENE_ASSET = DEFAULT_CATALOGUE_PATH.parent / "yeouido_scene.npz"


@pytest.fixture(scope="module")
def lod() -> VegetationLod:
    return VegetationLod.from_camera(PhysicalCameraConfig(), RenderConfig())


# --- derivation -------------------------------------------------------------


def test_angular_resolution_follows_the_camera_optics() -> None:
    camera, render = PhysicalCameraConfig(), RenderConfig()
    resolution = angular_resolution_rad_per_pixel(camera, render.height)
    assert resolution == pytest.approx(
        math.radians(vertical_fov_deg(camera)) / render.height
    )
    # 45.75 degrees over 720 rows.
    assert resolution * 1e3 == pytest.approx(1.109, abs=0.01)


def test_a_longer_lens_resolves_further() -> None:
    render = RenderConfig()
    wide = angular_resolution_rad_per_pixel(
        PhysicalCameraConfig(focal_length_mm=24.0), render.height
    )
    telephoto = angular_resolution_rad_per_pixel(
        PhysicalCameraConfig(focal_length_mm=200.0), render.height
    )
    assert telephoto < wide


def test_bands_move_with_resolution_and_lens(lod) -> None:
    # The point of deriving the bands is that they track the camera.
    taller = VegetationLod.from_camera(
        PhysicalCameraConfig(), RenderConfig(height=1440)
    )
    assert taller.blade_cutoff_m > lod.blade_cutoff_m
    longer = VegetationLod.from_camera(
        PhysicalCameraConfig(focal_length_mm=85.0), RenderConfig()
    )
    assert longer.blade_cutoff_m > lod.blade_cutoff_m


def test_subpixel_distance_is_inverse_in_both_arguments() -> None:
    resolution = 1.0e-3
    assert subpixel_distance_m(0.04, resolution, 1.0) == pytest.approx(40.0)
    assert subpixel_distance_m(0.08, resolution, 1.0) == pytest.approx(80.0)
    assert subpixel_distance_m(0.04, resolution, 2.0) == pytest.approx(20.0)


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="viewport height"):
        angular_resolution_rad_per_pixel(PhysicalCameraConfig(), 0)
    with pytest.raises(ValueError, match="must be positive"):
        subpixel_distance_m(0.0, 1e-3)
    with pytest.raises(ValueError, match="must be positive"):
        subpixel_distance_m(0.04, 1e-3, 0.0)


def test_band_edges_match_their_stated_pixel_criteria(lod) -> None:
    resolution = lod.angular_resolution_rad_per_pixel
    assert BLADE_WIDTH_M / (lod.blade_full_detail_m * resolution) == (
        pytest.approx(FULL_DETAIL_PIXELS)
    )
    assert BLADE_WIDTH_M / (lod.blade_cutoff_m * resolution) == (
        pytest.approx(CUTOFF_PIXELS)
    )


def test_band_ordering_is_enforced() -> None:
    with pytest.raises(ValueError, match="full detail nearer"):
        VegetationLod(
            blade_full_detail_m=100.0,
            blade_cutoff_m=50.0,
            tree_sway_cutoff_m=50.0,
            angular_resolution_rad_per_pixel=1e-3,
        )


# --- the ramp ---------------------------------------------------------------


def test_detail_is_full_near_and_zero_beyond_cutoff(lod) -> None:
    assert lod.blade_detail_fraction(0.0) == 1.0
    assert lod.blade_detail_fraction(lod.blade_full_detail_m) == 1.0
    assert lod.blade_detail_fraction(lod.blade_cutoff_m) == 0.0
    assert lod.blade_detail_fraction(10_000.0) == 0.0


def test_detail_decreases_monotonically(lod) -> None:
    distances = np.linspace(0.0, lod.blade_cutoff_m * 1.5, 64)
    values = [lod.blade_detail_fraction(float(d)) for d in distances]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_the_ramp_is_smooth_at_both_edges(lod) -> None:
    # Smoothstep has zero derivative at the band edges, so a blade neither
    # pops out nor shows a linear seam where the band starts.
    epsilon = 0.01
    near = lod.blade_detail_fraction(lod.blade_full_detail_m + epsilon)
    far = lod.blade_detail_fraction(lod.blade_cutoff_m - epsilon)
    assert near > 0.999
    assert far < 0.001


# --- the finding this work exposed -----------------------------------------


def test_a_blade_at_the_authoring_radius_is_far_below_a_pixel(lod) -> None:
    # A 0.04 m blade at the 1,200 m authoring radius subtends 0.03 px, which
    # is why observation-distance gating was needed rather than a wider ring.
    subtended = BLADE_WIDTH_M / (
        EVENT_SITE_DETAIL_RADIUS_M * lod.angular_resolution_rad_per_pixel
    )
    assert subtended < 0.05
    assert lod.blade_detail_fraction(EVENT_SITE_DETAIL_RADIUS_M) == 0.0


@pytest.mark.skipif(
    not SCENE_ASSET.exists(), reason="geospatial scene asset not present"
)
def test_every_authored_blade_currently_sits_beyond_the_cutoff(lod) -> None:
    # Documents a real property of the shipped asset: no landuse=grass polygon
    # falls within 600 m of the scene origin, so every blade is authored around
    # 1.1 km out and none is resolvable from the default camera. The LOD is
    # therefore correct but dormant until blade placement is revisited.
    scene = load_scene(SCENE_ASSET)
    detail = scene.detail_vertices
    blades = detail[np.isclose(detail[:, 6], SURFACE_GRASS_BLADE)]
    if not len(blades):
        pytest.skip("asset contains no grass blades")
    radius = np.linalg.norm(blades[:, [0, 2]], axis=1)
    assert radius.min() > lod.blade_cutoff_m
    assert radius.max() <= EVENT_SITE_DETAIL_RADIUS_M


# --- shader agreement -------------------------------------------------------


def test_vertex_shader_declares_the_band_uniforms() -> None:
    source = shaders.source("scene.vert")
    for name in (
        "blade_full_detail_m",
        "blade_cutoff_m",
        "tree_sway_cutoff_m",
        "camera_position",
    ):
        assert re.search(rf"uniform\s+\w+\s+{name}\s*;", source), name


def test_vertex_shader_gates_on_observation_distance() -> None:
    source = shaders.source("scene.vert")
    assert "distance(camera_position" in source
    # The band must be applied with smoothstep, matching blade_detail_fraction.
    assert re.search(
        r"smoothstep\(\s*blade_full_detail_m,\s*blade_cutoff_m", source
    )
