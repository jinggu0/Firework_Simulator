"""V0 descriptive frame comparison, checked against constructed answers.

Each measure is exercised on a frame pair whose correct value is known by
construction rather than by running the renderer, so a metric that silently
stops measuring anything cannot pass. Where a measure should stay at zero, a
paired case shows it moving, so "always zero" is never mistaken for "no
difference".
"""

from __future__ import annotations

import numpy as np
import pytest

from simulator.validation.frame_comparison import (
    DEFAULT_EDGE_THRESHOLD,
    compare_frames,
    coverage_mask,
    edge_displacement,
    edge_mask,
    silhouette_iou,
    temporal_stability,
    tone_distribution_shift,
)


def _flat(value: int, size: tuple[int, int] = (64, 64)) -> np.ndarray:
    return np.full((*size, 3), value, dtype=np.uint8)


def _vertical_edge(column: int, size: tuple[int, int] = (64, 64)) -> np.ndarray:
    frame = np.zeros((*size, 3), dtype=np.uint8)
    frame[:, column:] = 255
    return frame


def test_coverage_mask_separates_geometry_from_sky() -> None:
    depth = np.ones((4, 4), dtype=np.float32)
    depth[1:3, 1:3] = 0.5

    mask = coverage_mask(depth)

    assert mask.sum() == 4
    assert mask[1, 1] and not mask[0, 0]


def test_silhouette_iou_matches_a_constructed_overlap() -> None:
    reference = np.zeros((10, 10), dtype=bool)
    candidate = np.zeros((10, 10), dtype=bool)
    reference[0:4, 0:4] = True  # 16 px
    candidate[2:6, 0:4] = True  # 16 px, overlapping in rows 2-3

    result = silhouette_iou(reference, candidate)

    # Intersection 8, union 24.
    assert result["intersection_over_union"] == pytest.approx(8 / 24)
    assert result["added_fraction"] == pytest.approx(8 / 100)
    assert result["removed_fraction"] == pytest.approx(8 / 100)


def test_silhouette_iou_separates_growth_from_shrinkage() -> None:
    # IoU alone cannot tell these apart, which is why direction is reported.
    small = np.zeros((10, 10), dtype=bool)
    small[0:2, 0:2] = True
    large = np.zeros((10, 10), dtype=bool)
    large[0:4, 0:4] = True

    grew = silhouette_iou(small, large)
    shrank = silhouette_iou(large, small)

    assert grew["intersection_over_union"] == shrank["intersection_over_union"]
    assert grew["added_fraction"] > 0 and grew["removed_fraction"] == 0
    assert shrank["removed_fraction"] > 0 and shrank["added_fraction"] == 0


def test_identical_frames_report_no_displacement_and_no_tone_shift() -> None:
    frame = _vertical_edge(32)

    edges = edge_displacement(frame, frame)
    tone = tone_distribution_shift(frame, frame)

    assert edges["mean_displacement_px"] == 0.0
    assert edges["edge_intersection_over_union"] == 1.0
    assert tone["luminance_earth_mover_code_values"] == 0.0
    assert tone["median_luminance_shift"] == 0.0


def test_a_shifted_edge_reports_the_shift_in_pixels() -> None:
    reference = _vertical_edge(32)
    candidate = _vertical_edge(35)

    edges = edge_displacement(reference, candidate)

    # The whole edge moved three columns, so every edge pixel is three from its
    # nearest counterpart. Sobel spreads the edge over adjacent columns, so the
    # measured value brackets the true shift rather than hitting it exactly.
    assert 2.0 <= edges["mean_displacement_px"] <= 4.0
    assert edges["max_displacement_px"] >= 3.0
    assert edges["edge_intersection_over_union"] < 1.0


def test_edges_are_not_invented_at_the_frame_border() -> None:
    # Zero padding a Sobel kernel puts a false edge around every frame. A flat
    # mid-grey image must contain no edges at all.
    assert not edge_mask(_flat(128)).any()
    assert edge_mask(_vertical_edge(32)).any()


def test_losing_edges_entirely_is_not_scored_as_a_match() -> None:
    # A one-directional Chamfer distance scores this zero: the candidate has no
    # edges to be far from anything. The symmetric form must not.
    reference = _vertical_edge(32)
    blank = _flat(128)

    edges = edge_displacement(reference, blank)

    assert edges["candidate_edge_fraction"] == 0.0
    assert edges["mean_displacement_px"] > 0.0
    assert edges["edge_intersection_over_union"] == 0.0
    # Every surviving measurement is an unmatched edge saturated at the frame
    # diagonal, and the report says so rather than hiding it in the mean.
    assert edges["unmatched_edge_fraction"] == 1.0
    assert edges["mean_displacement_px"] == pytest.approx(
        edges["saturation_distance_px"]
    )


def test_a_uniform_grade_shift_reads_as_that_many_code_values() -> None:
    reference = _flat(100)
    candidate = _flat(140)

    tone = tone_distribution_shift(reference, candidate)

    # Every pixel moved 40 code values, so the distributions are 40 apart.
    assert tone["luminance_earth_mover_code_values"] == pytest.approx(40.0, abs=1.0)
    assert tone["median_luminance_shift"] == pytest.approx(40.0, abs=1.0)
    assert tone["median_rgb8_shift"] == [40.0, 40.0, 40.0]


def test_a_grade_that_cancels_on_average_still_registers() -> None:
    # Half the pixels brighten by 40 and half darken by 40. A mean difference
    # would report nothing; the distribution distance must not.
    reference = _flat(128)
    candidate = _flat(128)
    candidate[:32] = 168
    candidate[32:] = 88

    tone = tone_distribution_shift(reference, candidate)

    assert tone["luminance_earth_mover_code_values"] == pytest.approx(40.0, abs=1.0)
    assert abs(tone["median_luminance_shift"]) <= 40.0


def test_a_still_view_is_stable_and_a_flickering_one_is_not() -> None:
    still = [_flat(120) for _ in range(4)]
    flickering = [_flat(120), _flat(130), _flat(120), _flat(130)]

    calm = temporal_stability(still)
    noisy = temporal_stability(flickering)

    assert calm["mean_successive_difference_rgb8"] == 0.0
    assert calm["flickering_pixel_fraction"] == 0.0
    assert noisy["mean_successive_difference_rgb8"] == pytest.approx(10.0)
    assert noisy["flickering_pixel_fraction"] == 1.0
    assert noisy["frame_count"] == 4


def test_quantisation_level_movement_is_not_called_flicker() -> None:
    # One code value of movement is where dither lives, not visible shimmer.
    frames = [_flat(120), _flat(121), _flat(120)]

    result = temporal_stability(frames)

    assert result["mean_successive_difference_rgb8"] == pytest.approx(1.0)
    assert result["flickering_pixel_fraction"] == 0.0


def test_temporal_stability_rejects_a_single_frame() -> None:
    with pytest.raises(ValueError):
        temporal_stability([_flat(120)])


def test_the_bundle_omits_silhouette_rather_than_guessing_it() -> None:
    reference = _vertical_edge(32)
    candidate = _vertical_edge(35)

    without = compare_frames(reference, candidate)
    with_masks = compare_frames(
        reference,
        candidate,
        reference_coverage=np.zeros((64, 64), dtype=bool),
        candidate_coverage=np.ones((64, 64), dtype=bool),
    )

    assert without["silhouette"] is None
    assert without["silhouette_absent_because"]
    assert without["edges"]["edge_threshold"] == DEFAULT_EDGE_THRESHOLD
    assert with_masks["silhouette"]["added_fraction"] == 1.0


def _capture_directory(root, name: str, views: dict[str, int]):
    import json

    from PIL import Image

    directory = root / name
    directory.mkdir()
    for view_id, column in views.items():
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        frame[:, column:] = 255
        Image.fromarray(frame).save(directory / f"{view_id}.sdr.png")
    (directory / "manifest.json").write_text(
        json.dumps({"views": [{"view_id": view} for view in views]}),
        encoding="utf-8",
    )
    return directory


def test_the_report_separates_changed_from_unchanged_views(tmp_path) -> None:
    from tools.report_visual_change import build_report

    before = _capture_directory(
        tmp_path, "before", {"shifted": 40, "identical": 20, "dropped": 10}
    )
    after = _capture_directory(
        tmp_path, "after", {"shifted": 46, "identical": 20, "added": 10}
    )

    report = build_report(before, after)

    assert report["view_count"] == 2
    assert report["changed_view_ids"] == ["shifted"]
    assert report["unchanged_view_ids"] == ["identical"]
    # A view that exists on only one side is named, not quietly dropped.
    assert report["views_only_in_before"] == ["dropped"]
    assert report["views_only_in_after"] == ["added"]
    edges = report["views"]["shifted"]["edges"]
    assert 5.0 <= edges["mean_displacement_px"] <= 7.0
    assert report["views"]["identical"]["edges"]["mean_displacement_px"] == 0.0


def test_a_resolution_change_is_reported_rather_than_compared(tmp_path) -> None:
    import json

    from PIL import Image

    from tools.report_visual_change import build_report

    before = _capture_directory(tmp_path, "before", {"view": 20})
    after = tmp_path / "after"
    after.mkdir()
    Image.fromarray(np.zeros((32, 48, 3), dtype=np.uint8)).save(
        after / "view.sdr.png"
    )
    (after / "manifest.json").write_text(
        json.dumps({"views": [{"view_id": "view"}]}), encoding="utf-8"
    )

    report = build_report(before, after)

    assert report["views"]["view"]["comparable"] is False
    assert "resolution changed" in report["views"]["view"]["reason"]
    assert report["changed_view_ids"] == []


class _StubTexture:
    def __init__(self, depth: np.ndarray) -> None:
        self._depth = np.asarray(depth, dtype=np.float32)
        self.size = (self._depth.shape[1], self._depth.shape[0])

    def read(self) -> bytes:
        return self._depth.tobytes()


class _StubRenderer:
    def __init__(self, depth: np.ndarray) -> None:
        self.scene_depth_texture = _StubTexture(depth)


def test_reading_coverage_corrects_the_opengl_row_order() -> None:
    from simulator.validation.capture import read_scene_coverage

    # OpenGL row 0 is the bottom of the frame. Geometry written to the bottom
    # of the image must come back at the bottom, not flipped to the top, or
    # every silhouette would be upside down while still looking plausible.
    depth = np.ones((4, 3), dtype=np.float32)
    depth[0, :] = 0.25  # GL bottom row

    mask = read_scene_coverage(_StubRenderer(depth))

    assert mask.shape == (4, 3)
    assert mask[-1].all()
    assert not mask[0].any()


def test_reading_coverage_rejects_a_truncated_depth_buffer() -> None:
    from simulator.validation.capture import read_scene_coverage

    renderer = _StubRenderer(np.ones((4, 3), dtype=np.float32))
    renderer.scene_depth_texture.size = (3, 5)

    with pytest.raises(ValueError, match="depth samples"):
        read_scene_coverage(renderer)


def test_a_renderer_without_a_depth_target_says_so() -> None:
    from simulator.validation.capture import read_scene_coverage

    with pytest.raises(AttributeError, match="scene_depth_texture"):
        read_scene_coverage(object())


def test_a_coverage_mask_round_trips_losslessly(tmp_path) -> None:
    from simulator.validation.capture import (
        coverage_statistics,
        load_coverage_mask,
        save_coverage_mask,
    )

    mask = np.zeros((32, 48), dtype=bool)
    mask[8:24, 10:30] = True

    path = save_coverage_mask(mask, tmp_path / "view.coverage.png")
    restored = load_coverage_mask(path)

    assert np.array_equal(mask, restored)
    assert coverage_statistics(mask)["coverage_fraction"] == pytest.approx(
        320 / (32 * 48)
    )


def test_the_report_uses_captured_masks_when_they_exist(tmp_path) -> None:
    from simulator.validation.capture import save_coverage_mask
    from tools.report_visual_change import build_report

    before = _capture_directory(tmp_path, "before", {"view": 40})
    after = _capture_directory(tmp_path, "after", {"view": 40})
    reference = np.zeros((64, 96), dtype=bool)
    reference[:32] = True
    candidate = np.zeros((64, 96), dtype=bool)
    candidate[:48] = True
    save_coverage_mask(reference, before / "view.coverage.png")
    save_coverage_mask(candidate, after / "view.coverage.png")

    report = build_report(before, after)
    silhouette = report["views"]["view"]["silhouette"]

    assert silhouette is not None
    assert silhouette["intersection_over_union"] == pytest.approx(32 / 48)
    assert silhouette["added_fraction"] == pytest.approx(16 / 64)
    assert silhouette["removed_fraction"] == 0.0


def test_a_capture_without_masks_still_reports(tmp_path) -> None:
    # Directories captured before masks existed must not break the report.
    from tools.report_visual_change import build_report

    before = _capture_directory(tmp_path, "before", {"view": 40})
    after = _capture_directory(tmp_path, "after", {"view": 46})

    report = build_report(before, after)

    assert report["views"]["view"]["silhouette"] is None
    assert report["views"]["view"]["edges"]["mean_displacement_px"] > 0.0


def test_mismatched_shapes_are_rejected() -> None:
    with pytest.raises(ValueError):
        tone_distribution_shift(_flat(100), _flat(100, (32, 32)))
    with pytest.raises(ValueError):
        silhouette_iou(np.zeros((4, 4), bool), np.zeros((5, 5), bool))
