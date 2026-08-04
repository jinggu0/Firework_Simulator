"""V2-2e S-Map orthophoto acquisition stays reproducible.

The registration report records a SHA-256 for every acquired tile, the mosaic,
its world file and the local manifest. Those pixels are deliberately **not**
committed — the provider marks the site all rights reserved and redistribution
was never confirmed — so this verifies them when the local acquisition is
present and skips when it is not.

That split matters. Without it the only evidence that the report describes the
bytes on disk is that someone ran the acquisition once, and the acquisition
tool re-downloads unconditionally: re-running it to check would fetch 36 tiles
from a third party's servers rather than verify anything.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
REPORT = (
    REPOSITORY_ROOT
    / "docs"
    / "validation"
    / "road_marking_registration_v2"
    / "smap_2024_registration_report.json"
)
LOCAL_ROOT = REPOSITORY_ROOT / "local_reference" / "smap_2024_ortho"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def acquisition() -> dict:
    if not REPORT.is_file():
        pytest.skip("V2-2e registration report is absent")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    return report["local_acquisition"]


@pytest.fixture(scope="module")
def tiles(acquisition) -> list[dict]:
    for value in acquisition.values():
        if (
            isinstance(value, list)
            and value
            and isinstance(value[0], dict)
            and "sha256" in value[0]
        ):
            return value
    pytest.fail("the report records no per-tile checksums")


def test_the_report_declares_a_checksum_for_every_tile(acquisition, tiles) -> None:
    # A tile count without per-tile checksums would let a substituted image
    # through, which is the whole point of recording them.
    assert len(tiles) == acquisition["tile_count"]
    assert len({tile["sha256"] for tile in tiles}) == len(tiles)
    for tile in tiles:
        assert len(tile["sha256"]) == 64
        assert tile["local_asset"].startswith(f"{tile['zoom']}/")


def test_the_pixels_are_not_committed(acquisition) -> None:
    # Provider rights were never confirmed for redistribution. The checksums
    # travel in the repository; the bytes do not.
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "local_reference/smap_2024_ortho"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    assert not [line for line in tracked.splitlines() if line.strip()]


@pytest.mark.skipif(
    not LOCAL_ROOT.is_dir(), reason="local S-Map acquisition is absent"
)
def test_every_acquired_tile_matches_its_recorded_checksum(tiles) -> None:
    mismatched = []
    missing = []
    for tile in tiles:
        path = LOCAL_ROOT / tile["local_asset"]
        if not path.is_file():
            missing.append(tile["local_asset"])
        elif _sha256(path) != tile["sha256"]:
            mismatched.append(tile["local_asset"])
    assert not missing, f"acquired tiles absent: {missing[:4]}"
    assert not mismatched, f"acquired tiles altered: {mismatched[:4]}"


@pytest.mark.skipif(
    not LOCAL_ROOT.is_dir(), reason="local S-Map acquisition is absent"
)
def test_the_mosaic_world_file_and_manifest_match(acquisition) -> None:
    for key, name in (
        ("mosaic_sha256", "focus_z10_x634-639_y520-525.png"),
        ("world_file_sha256", "focus_z10_x634-639_y520-525.pgw"),
    ):
        recorded = acquisition.get(key)
        if recorded is None:
            continue
        assert _sha256(LOCAL_ROOT / name) == recorded, name
    manifest = REPOSITORY_ROOT / acquisition["manifest_asset"]
    assert _sha256(manifest) == acquisition["manifest_sha256"]


@pytest.mark.skipif(
    not LOCAL_ROOT.is_dir(), reason="local S-Map acquisition is absent"
)
def test_the_world_file_places_the_crop_where_the_report_says(acquisition) -> None:
    # The registration uses the provider's own grid rather than a GCP fit, so
    # the world file *is* the registration. If it and the report's bounding box
    # ever disagree, every pixel-to-way binding built on it is wrong.
    values = [
        float(line)
        for line in (LOCAL_ROOT / "focus_z10_x634-639_y520-525.pgw")
        .read_text(encoding="utf-8")
        .split()
    ]
    pixel_size, _, _, negative_pixel_size, easting, northing = values
    assert pixel_size == pytest.approx(
        acquisition.get("native_resolution_m_per_pixel", 0.25)
    )
    assert negative_pixel_size == pytest.approx(-pixel_size)

    width, height = acquisition["mosaic_pixel_size"]
    minimum_easting, minimum_northing, maximum_easting, maximum_northing = (
        acquisition["bbox_epsg5186_m"]
    )
    # The world file addresses pixel *centres*; the bounding box its edges.
    assert easting - pixel_size / 2.0 == pytest.approx(minimum_easting, abs=1e-6)
    assert northing + pixel_size / 2.0 == pytest.approx(
        maximum_northing, abs=1e-6
    )
    assert easting - pixel_size / 2.0 + width * pixel_size == pytest.approx(
        maximum_easting, abs=1e-6
    )
    assert northing + pixel_size / 2.0 - height * pixel_size == pytest.approx(
        minimum_northing, abs=1e-6
    )


def test_the_event_date_gate_stays_closed(acquisition) -> None:
    # V2-2f is not finished: the exact acquisition date is still unconfirmed,
    # so no marking may be classified as the event day's.
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["layer_contract"]["exact_imagery_acquisition_date"] is None
    assert (
        report["layer_contract"]["event_date_applicability_confirmed"] is False
    )
    assert (
        report["application_gates"]["event_date_marking_classification_allowed"]
        is False
    )
