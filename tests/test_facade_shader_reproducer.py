from __future__ import annotations

import pytest

from tools.probe_facade_shader_reproducer import (
    STAGES,
    _summarize_states,
    fragment_source,
)


def test_every_stage_builds_a_complete_fragment_shader() -> None:
    for stage in STAGES:
        source = fragment_source(stage)

        assert source.startswith("#version 330")
        assert "void main()" in source
        assert "frag_color =" in source
        assert source.rstrip().endswith("}")


def test_full_stages_retain_zero_count_light_loop_control_flow() -> None:
    for stage in ("full_helper_zero_lights", "final_facade"):
        source = fragment_source(stage)

        assert "if (i >= static_light_count) break;" in source
        assert "if (i >= dynamic_light_count) break;" in source
        assert "full_reflected_radiance" in source

    assert "static_light_count" not in fragment_source("environment_helper").split(
        "void main()", 1
    )[1]


def test_unknown_stage_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown standalone facade stage"):
        fragment_source("typo")


def test_state_summary_reports_stability_and_minority_draws() -> None:
    stable = _summarize_states([b"a", b"a", b"a"], 3)
    varying = _summarize_states([b"a", b"b", b"a"], 3)

    assert stable["bit_deterministic"] is True
    assert stable["unique_states"] == 1
    assert stable["differing_iterations"] == 0
    assert varying["bit_deterministic"] is False
    assert varying["unique_states"] == 2
    assert varying["differing_iterations"] == 1
    assert sum(state["count"] for state in varying["states"]) == 3
