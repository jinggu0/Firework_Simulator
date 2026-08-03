"""Capture and audit the V2-1 metric-scale ground-material baseline."""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import pygame

from simulator.app import SimulatorApp
from simulator.config import SimulationConfig
from simulator.material_textures import MATERIAL_TEXTURE_DIRECTORY
from simulator.scenario import DEFAULT_SCENARIO_PATH
from simulator.validation.capture import read_linear_hdr
from simulator.validation.material_detail import (
    projection_sampling_report,
    scanned_material_report,
    temporal_delta_metrics,
)
from simulator.validation.views import (
    DEFAULT_VISUAL_VIEWS_PATH,
    VisualRegressionView,
    load_visual_regression_suite,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(
    "docs/validation/material_detail_v1/material_detail_report.json"
)
DEFAULT_DIAGNOSTIC = Path(
    "docs/validation/material_detail_v1/material_motion_diagnostic.png"
)
MOTION_VIEW_IDS = ("grass_close", "road_ground")


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _capture_motion(
    view: VisualRegressionView,
    scenario_path: Path,
    frames: int,
    step_m: float,
) -> np.ndarray:
    base = SimulationConfig()
    config = replace(
        base,
        render=replace(base.render, vsync=False, target_fps=0),
    )
    app = SimulatorApp(config, scenario_path=scenario_path)
    captured: list[np.ndarray] = []
    try:
        app.world.shells.clear()
        app.world.stars.count = 0
        view.apply(app.camera)
        yaw = math.radians(view.yaw_deg)
        camera_right = np.asarray(
            [math.cos(yaw), 0.0, math.sin(yaw)], dtype=np.float32
        )
        for _ in range(4):
            app.renderer.render(
                app.world, app.camera, app.celestial, 1.0 / 60.0, None
            )
        for _ in range(frames):
            app.renderer.render(
                app.world, app.camera, app.celestial, 1.0 / 60.0, None
            )
            app.ctx.finish()
            captured.append(read_linear_hdr(app.renderer).copy())
            app.camera.position_m[:] += camera_right * step_m
        return np.stack(captured, axis=0)
    finally:
        app.audio_executor.shutdown(wait=True, cancel_futures=True)
        pygame.quit()


def _display_panel(frame: np.ndarray, scale: float | None = None) -> Image.Image:
    rgb = np.maximum(frame[..., :3].astype(np.float64), 0.0)
    if scale is None:
        scale = max(float(np.percentile(rgb, 99.5)), 1e-12)
    display = np.power(np.clip(rgb / scale, 0.0, 1.0), 1.0 / 2.2)
    return Image.fromarray(np.rint(display * 255.0).astype(np.uint8), "RGB")


def _diagnostic_row(frames: np.ndarray, label: str) -> Image.Image:
    first = frames[0]
    last = frames[-1]
    shared_scale = max(float(np.percentile(np.maximum(frames[..., :3], 0.0), 99.5)), 1e-12)
    difference = np.abs(last[..., :3] - first[..., :3])
    difference_scale = max(float(np.percentile(difference, 99.5)), 1e-12)
    panels = [
        _display_panel(first, shared_scale),
        _display_panel(last, shared_scale),
        _display_panel(np.dstack([difference, np.ones(difference.shape[:2])]), difference_scale),
    ]
    target_size = (400, 225)
    panels = [panel.resize(target_size, Image.Resampling.LANCZOS) for panel in panels]
    row = Image.new("RGB", (1200, 250), (12, 12, 12))
    draw = ImageDraw.Draw(row)
    for index, panel in enumerate(panels):
        row.paste(panel, (index * 400, 25))
    draw.text((8, 7), f"{label}: first | last | amplified absolute delta", fill="white")
    return row


def build_report(
    scenario_path: Path = DEFAULT_SCENARIO_PATH,
    views_path: Path = DEFAULT_VISUAL_VIEWS_PATH,
    frames: int = 8,
    step_m: float = 0.02,
) -> tuple[dict[str, Any], Image.Image]:
    suite = load_visual_regression_suite(views_path)
    suite.verify_scene_asset()
    projection = projection_sampling_report()
    material_records = scanned_material_report(
        MATERIAL_TEXTURE_DIRECTORY, projection
    )
    captures = []
    diagnostic_rows = []
    for view_id in MOTION_VIEW_IDS:
        view = suite.view(view_id)
        motion_frames = _capture_motion(view, scenario_path, frames, step_m)
        metrics = temporal_delta_metrics(motion_frames)
        captures.append(
            {
                "view_id": view_id,
                "subject": view.subject,
                "position_eus_m": list(view.position_eus_m),
                "target_eus_m": list(view.target_eus_m),
                "motion_axis": "camera right",
                "motion_step_m_per_frame": step_m,
                "motion_speed_at_60hz_mps": step_m * 60.0,
                "metrics": metrics,
            }
        )
        diagnostic_rows.append(_diagnostic_row(motion_frames, view_id))
    diagnostic = Image.new("RGB", (1200, 250 * len(diagnostic_rows)))
    for index, row in enumerate(diagnostic_rows):
        diagnostic.paste(row, (0, index * 250))
    manifest_path = MATERIAL_TEXTURE_DIRECTORY / "manifest.json"
    shader_path = REPOSITORY_ROOT / "simulator/shaders/scene.frag"
    texture_loader_path = REPOSITORY_ROOT / "simulator/material_textures.py"
    shader = shader_path.read_text(encoding="utf-8")
    texture_loader = texture_loader_path.read_text(encoding="utf-8")
    return (
        {
            "schema_version": 1,
            "stage": "V2-1a",
            "scenario": _display_path(scenario_path),
            "visual_views": {
                "asset": _display_path(views_path),
                "sha256": _digest(views_path),
                "view_ids": list(MOTION_VIEW_IDS),
                "historical_observer_positions": False,
            },
            "material_source": {
                "manifest": _display_path(manifest_path),
                "sha256": _digest(manifest_path),
                "site_identity_confidence": "D",
                "sampled_at_yeouido": False,
            },
            "runtime_contract": {
                "world_metric_uv": "metric_material_uv(n) / scanned_texture_width_m[layer]" in shader,
                "mipmaps_built": "self.texture.build_mipmaps()" in texture_loader,
                "trilinear_minification": "moderngl.LINEAR_MIPMAP_LINEAR" in texture_loader,
                "anisotropic_filter_explicitly_configured": ".anisotropy" in texture_loader,
                "detail_fade_range_m": [90.0, 420.0],
            },
            "projection_sampling": projection,
            "materials": material_records,
            "motion_capture": {
                "performed": True,
                "linear_hdr_before_display_transform": True,
                "frames_per_view": frames,
                "captures": captures,
                "interpretation": (
                    "Repeatable baseline only. Frame deltas include geometry and "
                    "perspective motion; a registered real sequence is required "
                    "before defining a pass/fail shimmer threshold."
                ),
            },
            "gates": {
                "physical_tile_scale_documented": all(
                    record["texel_pitch_mm"] > 0.0 for record in material_records
                ),
                "runtime_motion_baseline_captured": True,
                "site_colour_tuning_allowed": False,
                "temporal_shimmer_gate_defined": False,
                "blocking_reasons": [
                    "all four scans are generic CC0 samples, not Yeouido measurements",
                    "no registered real motion sequence defines an acceptable shimmer threshold",
                    "anisotropic filtering is not explicitly configured on the scanned texture array",
                    "a single tile repeats 19.5 to 41.1 times across the 2-80 m inspection span",
                ],
            },
        },
        diagnostic,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        arguments.scenario, arguments.views, arguments.frames, arguments.step_m
    )
    output = arguments.output if arguments.output.is_absolute() else REPOSITORY_ROOT / arguments.output
    image_path = arguments.diagnostic if arguments.diagnostic.is_absolute() else REPOSITORY_ROOT / arguments.diagnostic
    output.parent.mkdir(parents=True, exist_ok=True)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic.save(image_path)
    report["motion_capture"]["diagnostic"] = {
        "asset": _display_path(image_path),
        "sha256": _digest(image_path),
    }
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {output}: materials={len(report['materials'])}, "
        f"motion_views={len(report['motion_capture']['captures'])}"
    )


if __name__ == "__main__":
    main()
