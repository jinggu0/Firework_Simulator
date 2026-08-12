from __future__ import annotations

import pytest

from tools.probe_render_determinism import (
    CHECKPOINT_SCENE_TERMS,
    EARLY_SCENE_TERMS,
    FINAL_SCENE_TERMS,
    RADIANCE_CHECKPOINT_TERMS,
    _FINAL_FACADE_OUTPUT,
    _instrument_scene_fragment,
)


def test_every_scene_term_replaces_only_the_final_facade_output() -> None:
    source = f"prefix\n{_FINAL_FACADE_OUTPUT}\nsuffix"

    for term, expression in FINAL_SCENE_TERMS.items():
        instrumented = _instrument_scene_fragment(source, term)

        assert _FINAL_FACADE_OUTPUT not in instrumented
        assert f"frag_color = {expression};" in instrumented
        assert instrumented.startswith("prefix\n")
        assert instrumented.endswith("\nsuffix")


def test_early_scene_terms_return_before_material_branches() -> None:
    source = "void main() {\n    real_shader();\n}"

    for term, expression in EARLY_SCENE_TERMS.items():
        instrumented = _instrument_scene_fragment(source, term)

        assert f"frag_color = {expression};" in instrumented
        assert instrumented.index("return;") < instrumented.index("real_shader();")
        assert "if (time_s > -1e20)" in instrumented


def test_checkpoint_scene_terms_return_immediately_after_their_anchor() -> None:
    for term, (anchor, expression) in CHECKPOINT_SCENE_TERMS.items():
        source = f"prefix\n{anchor}\nreal_shader();"
        instrumented = _instrument_scene_fragment(source, term)

        assert f"frag_color = {expression};" in instrumented
        assert instrumented.index(anchor) < instrumented.index("frag_color")
        assert instrumented.index("return;") < instrumented.index("real_shader();")


def test_radiance_checkpoints_return_from_the_helper_and_remove_emission() -> None:
    for term, (anchor, expression) in RADIANCE_CHECKPOINT_TERMS.items():
        source = f"prefix\n{anchor}\n{_FINAL_FACADE_OUTPUT}\nsuffix"
        instrumented = _instrument_scene_fragment(source, term)

        assert f"return {expression};" in instrumented
        assert "+ emission * landmark_emission_scale" not in instrumented

def test_scene_term_instrumentation_fails_closed_when_shader_drifts() -> None:
    with pytest.raises(RuntimeError, match="expected form"):
        _instrument_scene_fragment("void main() {}", "emission")
