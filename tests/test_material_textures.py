import hashlib
import json

import numpy as np
from PIL import Image

from simulator import shaders
from simulator.material_textures import (
    MAP_SUFFIXES,
    MATERIAL_TEXTURE_DIRECTORY,
    SCANNED_MATERIALS,
    texture_widths_m,
)


def test_every_scanned_pbr_map_is_a_matching_rgb_tile() -> None:
    sizes: set[tuple[int, int]] = set()
    for asset_id, _ in SCANNED_MATERIALS:
        for suffix in MAP_SUFFIXES:
            path = MATERIAL_TEXTURE_DIRECTORY / f"{asset_id}_{suffix}_1k.jpg"
            assert path.is_file(), path
            with Image.open(path) as image:
                assert image.mode == "RGB"
                sizes.add(image.size)
    assert sizes == {(1024, 1024)}


def test_scanned_material_manifest_verifies_every_map() -> None:
    manifest = json.loads(
        (MATERIAL_TEXTURE_DIRECTORY / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["license"] == "CC0 1.0"
    assert manifest["site_identity_confidence"] == "D"
    for asset_id, _ in SCANNED_MATERIALS:
        record = manifest["assets"][asset_id]
        assert record["page_url"].startswith("https://polyhaven.com/a/")
        for suffix in MAP_SUFFIXES:
            path = MATERIAL_TEXTURE_DIRECTORY / f"{asset_id}_{suffix}_1k.jpg"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == record["sha256"][suffix]


def test_scanned_tiles_have_physical_widths() -> None:
    widths = texture_widths_m()
    assert widths.shape == (len(SCANNED_MATERIALS),)
    assert np.all(widths > 0.0)
    assert widths[0] == 4.0


def test_scene_shader_consumes_all_three_pbr_channels() -> None:
    source = shaders.source("scene.frag")
    assert "uniform sampler2DArray scanned_material_texture;" in source
    assert "scanned_albedo" in source
    assert "tangent_normal" in source
    assert "reflectance.x = mix(reflectance.x, arm.g" in source
    assert "reflectance.z *= mix(1.0, arm.r" in source
