from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from simulator.bridge_semantics import (
    BridgePlanSemanticError,
    load_bridge_plan_semantics,
    parse_bridge_plan_semantics,
    replace_seogang_bridge_plan,
    replacement_deck_vertices,
    seogang_bridge_segment_mask,
)
from simulator.scene import load_scene


ASSET = Path("assets/seogang_bridge_semantics_2024-10-05.json")


def _quad(centre_x: float, centre_z: float, width: float = 2.0) -> np.ndarray:
    x0, x1 = centre_x - width * 0.5, centre_x + width * 0.5
    z0, z1 = centre_z - 1.0, centre_z + 1.0
    positions = ((x0, z0), (x1, z0), (x1, z1), (x0, z0), (x1, z1), (x0, z1))
    output = np.zeros((6, 10), dtype=np.float32)
    output[:, [0, 2]] = positions
    output[:, 1] = 7.0
    output[:, 4] = 1.0
    output[:, 6] = 2.0
    return output


def test_shipped_semantics_replace_duplicate_plan_without_vertical_claim() -> None:
    semantics = load_bridge_plan_semantics()
    scene = load_scene(Path("assets/yeouido_scene.npz"))

    replaced, stats = replace_seogang_bridge_plan(scene.bridge_vertices, semantics)

    assert stats.removed_generic_segments > stats.replacement_segments
    assert stats.replacement_segments == 64
    assert stats.output_vertices < stats.input_vertices
    assert np.allclose(replaced[-64 * 6 :, 1], 7.0)
    assert semantics.station_registration_passed is False
    assert semantics.station_length_residual_m > 50.0
    assert semantics.event_inside_paint_contract is True
    assert semantics.construction_visual_state_known is False


def test_replacement_keeps_outside_quads_and_removes_inside_or_outline_quads() -> None:
    document = json.loads(ASSET.read_text(encoding="utf-8"))
    document["deck_outline_xz_m"] = [
        [-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0], [-2.0, -2.0]
    ]
    document["replacement_cross_sections"] = [
        {"side_a_xz_m": [-2.0, -2.0], "side_b_xz_m": [2.0, -2.0]},
        {"side_a_xz_m": [-2.0, 2.0], "side_b_xz_m": [2.0, 2.0]},
    ]
    document["render_policy"]["matching_distance_m"] = 0.1
    document["render_policy"]["minimum_matching_source_segments"] = 2
    semantics = parse_bridge_plan_semantics(document)
    source = np.concatenate((_quad(0.0, 0.0), _quad(2.0, 0.0), _quad(10.0, 0.0)))

    mask = seogang_bridge_segment_mask(source, semantics)
    output, stats = replace_seogang_bridge_plan(source, semantics)

    assert mask.tolist() == [True, True, False]
    assert stats.removed_generic_segments == 2
    assert stats.retained_segments == 1
    assert stats.replacement_segments == 1
    assert len(output) == 12
    assert np.allclose(output[:6], source[-6:])


def test_replacement_deck_is_connected_six_vertex_quads() -> None:
    semantics = load_bridge_plan_semantics()
    vertices = replacement_deck_vertices(semantics).reshape(-1, 6, 10)

    assert np.allclose(vertices[:-1, 2, :3], vertices[1:, 1, :3])
    assert np.allclose(vertices[:-1, 5, :3], vertices[1:, 0, :3])
    assert np.all(vertices[:, :, 4] == 1.0)
    assert np.all(vertices[:, :, 6] == 2.0)


def test_semantics_reject_unclosed_or_wrong_date_outline() -> None:
    document = json.loads(ASSET.read_text(encoding="utf-8"))
    document["deck_outline_xz_m"][-1] = [0.0, 0.0]
    with pytest.raises(BridgePlanSemanticError, match="closed"):
        parse_bridge_plan_semantics(document)

    document = json.loads(ASSET.read_text(encoding="utf-8"))
    document["target_event_date"] = "2025-10-05"
    with pytest.raises(BridgePlanSemanticError, match="2024-10-05"):
        parse_bridge_plan_semantics(document)


def test_semantics_reject_negative_matching_distance() -> None:
    document = json.loads(ASSET.read_text(encoding="utf-8"))
    document["render_policy"]["matching_distance_m"] = -0.1
    with pytest.raises(BridgePlanSemanticError, match="negative"):
        parse_bridge_plan_semantics(document)


def test_replacement_is_not_injected_into_an_unrelated_scene() -> None:
    semantics = load_bridge_plan_semantics()
    source = _quad(0.0, 0.0)

    output, stats = replace_seogang_bridge_plan(source, semantics)

    assert np.array_equal(output, source)
    assert stats.removed_generic_segments == 0
    assert stats.replacement_segments == 0
