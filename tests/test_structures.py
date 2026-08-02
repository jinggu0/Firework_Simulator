from __future__ import annotations

import numpy as np
import pytest

from simulator.scene import SURFACE_EARTHWORK, SURFACE_RETAINING_WALL
from simulator.structures import (
    StructureEvidenceError,
    build_structure_mesh,
    retaining_wall_face_vertices,
    surveyed_slope_vertices,
)


TERRAIN_BOUNDS = np.array([0.0, 0.0, 8.0, 4.0], dtype=np.float32)
FLAT_TERRAIN = np.ones((2, 2), dtype=np.float32)


def _evidence(grade: str = "A") -> dict[str, str]:
    return {
        "confidence_grade": grade,
        "source_id": "synthetic-survey-fixture",
        "source_url": "https://example.test/survey",
        "coordinate_reference_system": "local EUS metres",
        "units": "m",
        "notes": "Test-only surveyed edge-role decision.",
    }


def _feature(
    feature_id: str,
    kind: str,
    points: list[list[float | None]],
) -> dict:
    return {
        "feature_id": feature_id,
        "kind": kind,
        "has_source_elevation": all(point[1] is not None for point in points),
        "points_eus_m": points,
    }


def _registration() -> dict:
    report = {
        "schema_version": 1,
        "registration_id": "event-view",
        "target_event_date": "2024-10-05",
        "passed": True,
        "metrics": {
            "converged": True,
            "control_points": 8,
            "jacobian_rank": 6,
            "reprojection_rmse_px": 0.8,
            "reprojection_p95_px": 1.4,
            "reprojection_max_px": 2.0,
            "control_bbox_fraction": 0.12,
            "minimum_camera_depth_m": 20.0,
        },
    }
    return {"sha256": "abc123", "report": report}


def test_wall_converts_absolute_top_to_terrain_relative_offsets() -> None:
    top = np.array([[0.0, 3.0, 2.0], [8.0, 4.0, 2.0]])

    vertices, segments, skipped = retaining_wall_face_vertices(
        top, FLAT_TERRAIN, TERRAIN_BOUNDS
    )

    assert segments == 2
    assert skipped == 0
    assert vertices.shape == (24, 10)
    assert np.all(vertices[:, 6] == SURFACE_RETAINING_WALL)
    assert vertices[:, 1].min() == pytest.approx(0.0)
    assert vertices[:, 1].max() == pytest.approx(3.0)
    # Every face is duplicated with the opposite normal for an unknown side.
    assert np.allclose(vertices[:6, 3:6], -vertices[6:12, 3:6])


def test_wall_rejects_segments_whose_top_is_not_above_terrain() -> None:
    vertices, segments, skipped = retaining_wall_face_vertices(
        np.array([[0.0, 1.01, 2.0], [4.0, 1.02, 2.0]]),
        FLAT_TERRAIN,
        TERRAIN_BOUNDS,
    )

    assert segments == 1
    assert skipped == 1
    assert vertices.shape == (0, 10)


def test_surveyed_crest_and_toe_form_an_upward_earthwork_surface() -> None:
    crest = np.array([[0.0, 4.0, 1.0], [8.0, 4.0, 1.0]])
    # Opposite input direction exercises endpoint orientation matching.
    toe = np.array([[8.0, 1.5, 3.0], [0.0, 1.5, 3.0]])

    vertices, segments = surveyed_slope_vertices(
        crest, toe, FLAT_TERRAIN, TERRAIN_BOUNDS
    )

    assert segments == 2
    assert vertices.shape == (12, 10)
    assert np.all(vertices[:, 6] == SURFACE_EARTHWORK)
    assert np.all(vertices[:, 4] > 0.0)
    assert vertices[:, 1].max() == pytest.approx(3.0)
    assert vertices[:, 1].min() == pytest.approx(0.5)


def test_slope_normals_use_absolute_survey_geometry_not_stored_offsets() -> None:
    terrain = np.array([[0.0, 2.0], [0.0, 2.0]], dtype=np.float32)
    crest = np.array([[0.0, 4.0, 1.0], [8.0, 4.0, 1.0]])
    toe = np.array([[0.0, 2.0, 3.0], [8.0, 2.0, 3.0]])

    vertices, _ = surveyed_slope_vertices(
        crest, toe, terrain, TERRAIN_BOUNDS
    )

    assert np.max(np.abs(vertices[:, 3])) < 1e-6
    assert np.all(vertices[:, 4] > 0.0)


def test_mesh_builder_requires_elevations_edge_roles_and_evidence() -> None:
    asset = {
        "schema_version": 2,
        "target_event_date": "2024-10-05",
        "temporal_relation": "official_same_year_date_unverified",
        "features": [
            _feature("wall", "retaining_wall", [[0.0, 3.0, 2.0], [4.0, 3.0, 2.0]])
        ],
    }
    base_profile = {
        "feature_id": "wall",
        "mesh_kind": "retaining_wall_face",
        "source_edge_role": "top",
        "lower_edge_source": "official_terrain",
        "evidence": _evidence(),
    }

    result = build_structure_mesh(
        asset,
        {"schema_version": 1, "profiles": [base_profile]},
        FLAT_TERRAIN,
        TERRAIN_BOUNDS,
    )
    assert result.profiles_built == 1
    assert result.rendered_segments == 1
    assert result.vertices.shape == (12, 10)

    modelled = {**base_profile, "evidence": _evidence("C")}
    with pytest.raises(StructureEvidenceError, match="grade A or B"):
        build_structure_mesh(
            asset,
            {"schema_version": 1, "profiles": [modelled]},
            FLAT_TERRAIN,
            TERRAIN_BOUNDS,
        )
    wrong_role = {**base_profile, "source_edge_role": "alignment"}
    with pytest.raises(StructureEvidenceError, match="top role"):
        build_structure_mesh(
            asset,
            {"schema_version": 1, "profiles": [wrong_role]},
            FLAT_TERRAIN,
            TERRAIN_BOUNDS,
        )


def test_mesh_builder_rejects_post_event_data_without_second_gate() -> None:
    asset = {
        "schema_version": 2,
        "target_event_date": "2024-10-05",
        "temporal_relation": "official_post_event",
        "features": [
            _feature("wall", "retaining_wall", [[0.0, 3.0, 2.0], [4.0, 3.0, 2.0]])
        ],
    }
    profiles = {
        "schema_version": 1,
        "profiles": [
            {
                "feature_id": "wall",
                "mesh_kind": "retaining_wall_face",
                "source_edge_role": "top",
                "lower_edge_source": "official_terrain",
                "evidence": _evidence("A"),
            }
        ]
    }

    with pytest.raises(StructureEvidenceError, match="post-event"):
        build_structure_mesh(asset, profiles, FLAT_TERRAIN, TERRAIN_BOUNDS)
    result = build_structure_mesh(
        asset,
        profiles,
        FLAT_TERRAIN,
        TERRAIN_BOUNDS,
        allow_post_event_source=True,
    )
    assert len(result.vertices) == 12


def test_slope_builder_requires_distinct_same_kind_crest_and_toe() -> None:
    asset = {
        "schema_version": 2,
        "target_event_date": "2024-10-05",
        "temporal_relation": "official_pre_event",
        "features": [
            _feature("crest", "embankment", [[0, 4, 1], [8, 4, 1]]),
            _feature("toe", "embankment", [[0, 1.5, 3], [8, 1.5, 3]]),
        ],
    }
    profile = {
        "feature_id": "crest",
        "paired_feature_id": "toe",
        "mesh_kind": "surveyed_slope",
        "source_edge_role": "crest",
        "paired_edge_role": "toe",
        "evidence": _evidence(),
    }

    result = build_structure_mesh(
        asset,
        {"schema_version": 1, "profiles": [profile]},
        FLAT_TERRAIN,
        TERRAIN_BOUNDS,
    )

    assert result.profiles_built == 1
    assert result.rendered_segments == 2
    assert result.vertices.shape == (12, 10)


def test_grade_b_profile_requires_a_passing_checksum_locked_registration() -> None:
    asset = {
        "schema_version": 2,
        "target_event_date": "2024-10-05",
        "temporal_relation": "official_pre_event",
        "features": [
            _feature("wall", "retaining_wall", [[0, 3, 2], [4, 3, 2]])
        ],
    }
    profile = {
        "feature_id": "wall",
        "mesh_kind": "retaining_wall_face",
        "source_edge_role": "top",
        "lower_edge_source": "official_terrain",
        "evidence": _evidence("B"),
    }
    document = {"schema_version": 1, "profiles": [profile]}

    with pytest.raises(StructureEvidenceError, match="registration_id"):
        build_structure_mesh(asset, document, FLAT_TERRAIN, TERRAIN_BOUNDS)

    profile.update(
        registration_id="event-view", registration_report_sha256="abc123"
    )
    result = build_structure_mesh(
        asset,
        document,
        FLAT_TERRAIN,
        TERRAIN_BOUNDS,
        verified_registrations={"event-view": _registration()},
    )
    assert len(result.vertices) == 12

    profile["registration_report_sha256"] = "wrong"
    with pytest.raises(StructureEvidenceError, match="checksum mismatch"):
        build_structure_mesh(
            asset,
            document,
            FLAT_TERRAIN,
            TERRAIN_BOUNDS,
            verified_registrations={"event-view": _registration()},
        )
