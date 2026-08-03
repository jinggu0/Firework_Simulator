"""Compare the V2-1b material filtering path with the frozen V2-1a baseline."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from simulator.material_textures import SCANNED_MATERIAL_ANISOTROPY
from simulator.scenario import DEFAULT_SCENARIO_PATH
from simulator.validation.material_detail import temporal_delta_metrics
from simulator.validation.views import (
    DEFAULT_VISUAL_VIEWS_PATH,
    load_visual_regression_suite,
)
from tools.audit_material_detail import (
    MOTION_VIEW_IDS,
    capture_motion,
    motion_diagnostic_row,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = Path(
    "docs/validation/material_detail_v1/material_detail_report.json"
)
DEFAULT_BASELINE_DIAGNOSTIC = Path(
    "docs/validation/material_detail_v1/material_motion_diagnostic.png"
)
DEFAULT_OUTPUT = Path(
    "docs/validation/material_filtering_v2/material_filtering_report.json"
)
DEFAULT_DIAGNOSTIC = Path(
    "docs/validation/material_filtering_v2/material_filtering_ab.png"
)
WARP_AMPLITUDE_M = 0.16
WARP_FREQUENCY_PER_M = (0.071, 0.057)
MAX_LOCAL_SCALE_DEVIATION = WARP_AMPLITUDE_M * max(WARP_FREQUENCY_PER_M)


def _repository_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _ratio(current: float, baseline: float) -> float | None:
    if baseline == 0.0:
        return None
    return current / baseline


def _comparison(
    baseline_metrics: dict[str, Any],
    current_metrics: dict[str, Any],
) -> dict[str, Any]:
    keys = (
        "mean_absolute_frame_delta",
        "p95_absolute_frame_delta",
        "normalized_mean_delta_to_signal_p95",
        "two_percent_signal_flip_fraction",
        "high_frequency_delta_fraction",
    )
    return {
        "baseline": {key: baseline_metrics[key] for key in keys},
        "filtered": {key: current_metrics[key] for key in keys},
        "filtered_to_baseline_ratio": {
            key: _ratio(current_metrics[key], baseline_metrics[key])
            for key in keys
        },
    }


def _ab_diagnostic(
    baseline_path: Path,
    filtered_rows: list[Image.Image],
) -> Image.Image:
    with Image.open(baseline_path) as source:
        baseline = source.convert("RGB").copy()
    filtered = Image.new("RGB", (1200, 250 * len(filtered_rows)))
    for index, row in enumerate(filtered_rows):
        filtered.paste(row, (0, index * 250))
    if baseline.size != filtered.size:
        baseline = baseline.resize(filtered.size, Image.Resampling.LANCZOS)
    result = Image.new(
        "RGB", (baseline.width + filtered.width, baseline.height + 25),
        (12, 12, 12),
    )
    draw = ImageDraw.Draw(result)
    draw.text((8, 7), "V2-1a baseline", fill="white")
    draw.text((baseline.width + 8, 7), "V2-1b filtered", fill="white")
    result.paste(baseline, (0, 25))
    result.paste(filtered, (baseline.width, 25))
    return result


def build_report(
    baseline_path: Path = DEFAULT_BASELINE,
    baseline_diagnostic_path: Path = DEFAULT_BASELINE_DIAGNOSTIC,
    scenario_path: Path = DEFAULT_SCENARIO_PATH,
    views_path: Path = DEFAULT_VISUAL_VIEWS_PATH,
    frames: int = 8,
    step_m: float = 0.02,
) -> tuple[dict[str, Any], Image.Image]:
    baseline_path = _repository_path(baseline_path)
    baseline_diagnostic_path = _repository_path(baseline_diagnostic_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline["stage"] != "V2-1a":
        raise ValueError("filtering comparison requires the V2-1a baseline")
    baseline_captures = {
        capture["view_id"]: capture
        for capture in baseline["motion_capture"]["captures"]
    }

    suite = load_visual_regression_suite(views_path)
    suite.verify_scene_asset()
    runtime_contract: dict[str, float | bool] = {}
    comparisons = []
    filtered_rows = []
    for view_id in MOTION_VIEW_IDS:
        view = suite.view(view_id)
        current_frames = capture_motion(
            view,
            scenario_path,
            frames,
            step_m,
            runtime_contract,
        )
        current_metrics = temporal_delta_metrics(current_frames)
        comparisons.append(
            {
                "view_id": view_id,
                **_comparison(
                    baseline_captures[view_id]["metrics"], current_metrics
                ),
            }
        )
        filtered_rows.append(motion_diagnostic_row(current_frames, view_id))

    shader_path = REPOSITORY_ROOT / "simulator/shaders/scene.frag"
    texture_loader_path = REPOSITORY_ROOT / "simulator/material_textures.py"
    shader = shader_path.read_text(encoding="utf-8")
    texture_loader = texture_loader_path.read_text(encoding="utf-8")
    same_motion_contract = (
        baseline["motion_capture"]["frames_per_view"] == frames
        and all(
            capture["motion_step_m_per_frame"] == step_m
            for capture in baseline_captures.values()
        )
    )
    return (
        {
            "schema_version": 1,
            "stage": "V2-1b",
            "baseline": {
                "asset": _display_path(baseline_path),
                "sha256": _digest(baseline_path),
                "diagnostic": _display_path(baseline_diagnostic_path),
                "diagnostic_sha256": _digest(baseline_diagnostic_path),
            },
            "capture_contract": {
                "scenario": _display_path(scenario_path),
                "views": _display_path(views_path),
                "view_ids": list(MOTION_VIEW_IDS),
                "linear_hdr_before_display_transform": True,
                "frames_per_view": frames,
                "motion_step_m_per_frame": step_m,
                "motion_speed_at_60hz_mps": step_m * 60.0,
                "matches_v2_1a": same_motion_contract,
            },
            "implementation": {
                "anisotropy_requested": SCANNED_MATERIAL_ANISOTROPY,
                "anisotropy_applied": runtime_contract["applied_anisotropy"],
                "anisotropy_supported": runtime_contract[
                    "anisotropy_supported"
                ],
                "world_metric_uv_preserved": (
                    "/ scanned_texture_width_m[layer]" in shader
                ),
                "warp_amplitude_m": WARP_AMPLITUDE_M,
                "warp_frequency_per_m": list(WARP_FREQUENCY_PER_M),
                "maximum_local_scale_deviation_fraction": (
                    MAX_LOCAL_SCALE_DEVIATION
                ),
                "macro_albedo_multiplier_range": [0.94, 1.06],
                "macro_roughness_offset_range": [-0.0275, 0.0275],
                "shader": {
                    "asset": _display_path(shader_path),
                    "sha256": _digest(shader_path),
                },
                "texture_loader": {
                    "asset": _display_path(texture_loader_path),
                    "sha256": _digest(texture_loader_path),
                    "explicit_anisotropy_assignment": (
                        ".anisotropy = SCANNED_MATERIAL_ANISOTROPY"
                        in texture_loader
                    ),
                },
            },
            "motion_comparisons": comparisons,
            "gates": {
                "shader_rendered_in_both_views": True,
                "same_motion_contract_as_baseline": same_motion_contract,
                "anisotropy_request_applied": (
                    runtime_contract["applied_anisotropy"] >= 1.0
                ),
                "metric_scale_distortion_below_1_2_percent": (
                    MAX_LOCAL_SCALE_DEVIATION < 0.012
                ),
                "site_colour_tuning_allowed": False,
                "temporal_shimmer_gate_defined": False,
                "interpretation": (
                    "A/B engineering measurement only. Without a registered "
                    "real motion sequence, lower frame delta is not by itself "
                    "a visual-fidelity pass."
                ),
            },
        },
        _ab_diagnostic(baseline_diagnostic_path, filtered_rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--baseline-diagnostic",
        type=Path,
        default=DEFAULT_BASELINE_DIAGNOSTIC,
    )
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO_PATH)
    parser.add_argument("--views", type=Path, default=DEFAULT_VISUAL_VIEWS_PATH)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--step-m", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--diagnostic", type=Path, default=DEFAULT_DIAGNOSTIC)
    arguments = parser.parse_args()
    if arguments.frames < 2:
        parser.error("--frames must be at least two")
    if arguments.step_m <= 0.0:
        parser.error("--step-m must be positive")
    report, diagnostic = build_report(
        arguments.baseline,
        arguments.baseline_diagnostic,
        arguments.scenario,
        arguments.views,
        arguments.frames,
        arguments.step_m,
    )
    output = _repository_path(arguments.output)
    diagnostic_path = _repository_path(arguments.diagnostic)
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic.save(diagnostic_path)
    report["diagnostic"] = {
        "asset": _display_path(diagnostic_path),
        "sha256": _digest(diagnostic_path),
    }
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {output}: comparisons={len(report['motion_comparisons'])}, "
        f"anisotropy={report['implementation']['anisotropy_applied']:.1f}"
    )


if __name__ == "__main__":
    main()
