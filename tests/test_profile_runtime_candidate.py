from __future__ import annotations

import json
import sys

import pytest

from simulator import shaders
import tools.profile_runtime as profile_runtime
from tools.shader_candidate import DEFAULT_CANDIDATE_DIRECTORY


# The expectation fade was rejected and never shipped, so its marker shows
# whether the candidate or the shipped shader was served.
CANDIDATE = DEFAULT_CANDIDATE_DIRECTORY / "window_grid_expectation_fade.json"


def _run_main(monkeypatch, capsys, arguments: list[str]) -> tuple[dict, list[str]]:
    seen_sources: list[str] = []

    def fake_profile(frames, fluid_backend="3d"):
        seen_sources.append(shaders.source("scene.frag"))
        return {"case": "integrated_blocking", "frame_p95_ms": 1.0}

    monkeypatch.setattr(profile_runtime, "_profile_integrated", fake_profile)
    monkeypatch.setattr(sys, "argv", ["profile_runtime", *arguments])
    profile_runtime.main()
    return json.loads(capsys.readouterr().out), seen_sources


def test_a_candidate_is_profiled_with_its_shader_served(monkeypatch, capsys) -> None:
    report, seen = _run_main(
        monkeypatch, capsys,
        ["--frames", "3", "--integrated-only", "--candidate", str(CANDIDATE)],
    )

    assert "cell_detail" in seen[0]
    assert report["candidate"]["candidate_id"] == "window_grid_expectation_fade"
    assert len(report["candidate"]["shader_sha256"]["scene.frag"]) == 64


def test_the_shipped_shader_is_back_after_profiling_a_candidate(monkeypatch, capsys) -> None:
    original = shaders.source("scene.frag")

    _run_main(
        monkeypatch, capsys,
        ["--frames", "3", "--integrated-only", "--candidate", str(CANDIDATE)],
    )

    assert shaders.source("scene.frag") == original


def test_without_a_candidate_the_shipped_shader_is_profiled(monkeypatch, capsys) -> None:
    report, seen = _run_main(monkeypatch, capsys, ["--frames", "3", "--integrated-only"])

    assert "cell_detail" not in seen[0]
    assert "candidate" not in report


def test_a_candidate_requires_the_integrated_only_case(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys, "argv", ["profile_runtime", "--frames", "3", "--candidate", str(CANDIDATE)]
    )

    with pytest.raises(SystemExit):
        profile_runtime.main()

    assert "--candidate needs --integrated-only" in capsys.readouterr().err
