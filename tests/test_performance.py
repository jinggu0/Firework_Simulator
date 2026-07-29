import numpy as np

from simulator.performance import FrameTimings


def test_frame_timings_reports_bounded_percentiles() -> None:
    timings = FrameTimings(capacity=4)
    for value in range(6):
        timings.record(value, value * 0.5, value * 0.25)

    snapshot = timings.snapshot()
    assert np.isclose(snapshot["frame"]["mean_ms"], 3.5)
    assert snapshot["frame"]["p95_ms"] < 5.0
    assert snapshot["frame"]["maximum_ms"] == 5.0
    assert snapshot["physics"]["mean_ms"] == 1.75
