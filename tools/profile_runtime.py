from __future__ import annotations

import argparse
from dataclasses import replace
import json
import time

import numpy as np
import pygame

from simulator.app import SimulatorApp
from simulator.config import SimulationConfig


def _prepare_populated_app(fluid_backend: str = "3d") -> SimulatorApp:
    base = SimulationConfig()
    config = replace(
        base,
        render=replace(base.render, vsync=False, target_fps=0),
        smoke=replace(
            base.smoke,
            prefer_3d_gpu_solver=fluid_backend == "3d",
            prefer_gpu_solver=fluid_backend in ("3d", "2d"),
        ),
    )
    app = SimulatorApp(config)
    physics_dt = 1.0 / config.render.physics_hz
    for _ in range(int(3.1 / physics_dt)):
        if app.environment is not None:
            app.world.atmosphere = app.environment.sample(app.event_timestamp)
            app.event_timestamp += physics_dt
        app.world.update(physics_dt)
        for burst in app.world.consume_burst_events():
            app.smoke.inject_burst(
                burst.position_m,
                burst.smoke_mass_kg,
                burst.post_blast_thermal_energy_j,
            )
    count = app.world.stars.count
    if count:
        app.world.stars.lifetime_s[:count] = 100.0
        app.world.stars.age_s[:count] = 0.2
    smoke_warmup_s = 4.0 / config.smoke.update_hz
    for _ in range(round(smoke_warmup_s * app.smoke.update_hz)):
        app.smoke.step(1.0 / app.smoke.update_hz)
    return app


def _close(app: SimulatorApp) -> None:
    app.audio_executor.shutdown(wait=True, cancel_futures=True)
    pygame.quit()


def _profile_case(
    name: str,
    frames: int,
    include_smoke: bool,
    moving_camera: bool,
    fluid_backend: str = "3d",
) -> dict[str, float | str]:
    app = _prepare_populated_app(fluid_backend)
    frame_dt = 1.0 / 60.0
    smoke = app.smoke if include_smoke else None
    for _ in range(12):
        if moving_camera:
            app.camera.yaw_deg += 0.12
        app.renderer.render(
            app.world, app.camera, app.celestial, frame_dt, smoke
        )
        pygame.display.flip()
    app.ctx.finish()

    cpu_ms: list[float] = []
    gpu_queries = []
    for _ in range(frames):
        if moving_camera:
            app.camera.yaw_deg += 0.12
        query = app.ctx.query(time=True)
        started = time.perf_counter()
        with query:
            app.renderer.render(
                app.world, app.camera, app.celestial, frame_dt, smoke
            )
        pygame.display.flip()
        cpu_ms.append((time.perf_counter() - started) * 1000.0)
        gpu_queries.append(query)
    app.ctx.finish()
    gpu_ms = np.asarray(
        [query.elapsed / 1_000_000.0 for query in gpu_queries],
        dtype=np.float64,
    )
    cpu = np.asarray(cpu_ms, dtype=np.float64)
    result: dict[str, float | str] = {
        "case": name,
        "fluid_backend": getattr(app.smoke, "backend_name", "cpu"),
        "stars": float(app.world.stars.count),
        "cpu_mean_ms": float(cpu.mean()),
        "cpu_p95_ms": float(np.percentile(cpu, 95)),
        "gpu_mean_ms": float(gpu_ms.mean()),
        "gpu_p95_ms": float(np.percentile(gpu_ms, 95)),
    }
    _close(app)
    return result


def _profile_integrated(
    frames: int, fluid_backend: str = "3d"
) -> dict[str, object]:
    """Measure the coupled fixed-step simulation and blocking visual path."""

    app = _prepare_populated_app(fluid_backend)
    frame_dt_s = 1.0 / 60.0
    physics_dt_s = 1.0 / app.config.render.physics_hz
    physics_ms: list[float] = []
    visual_ms: list[float] = []
    frame_ms: list[float] = []
    physics_breakdown_ms = {
        "world_update": [],
        "emission_build": [],
        "emission_inject": [],
        "smoke_step_submit": [],
    }
    smoke_gpu_queries = []
    for _ in range(frames):
        frame_started = time.perf_counter()
        app.camera.yaw_deg += 0.12
        physics_started = time.perf_counter()
        world_update_s = 0.0
        emission_build_s = 0.0
        emission_inject_s = 0.0
        smoke_step_s = 0.0
        for _ in range(2):
            world_started = time.perf_counter()
            if app.environment is not None:
                app.world.atmosphere = app.environment.sample(
                    app.event_timestamp
                )
                app.event_timestamp += physics_dt_s
            app.world.update(physics_dt_s)
            world_update_s += time.perf_counter() - world_started
            for burst in app.world.consume_burst_events():
                app.smoke.inject_burst(
                    burst.position_m,
                    burst.smoke_mass_kg,
                    burst.post_blast_thermal_energy_j,
                )
            app.smoke_accumulator_s += physics_dt_s
            smoke_dt_s = 1.0 / app.smoke.update_hz
            while app.smoke_accumulator_s >= smoke_dt_s:
                emission_started = time.perf_counter()
                emissions = app.world.consume_combustion_emissions()
                emission_build_s += time.perf_counter() - emission_started
                inject_started = time.perf_counter()
                for emission in emissions:
                    app.smoke.inject_particles(
                        emission.position_m,
                        emission.smoke_mass_kg,
                        emission.thermal_energy_j,
                    )
                emission_inject_s += time.perf_counter() - inject_started
                app.smoke.set_atmosphere(app.world.atmosphere)
                smoke_started = time.perf_counter()
                smoke_query = app.ctx.query(time=True)
                with smoke_query:
                    app.smoke.step(smoke_dt_s)
                smoke_gpu_queries.append(smoke_query)
                smoke_step_s += time.perf_counter() - smoke_started
                app.smoke_accumulator_s -= smoke_dt_s
        physics_finished = time.perf_counter()
        app.renderer.render(
            app.world, app.camera, app.celestial, frame_dt_s, app.smoke
        )
        app.ctx.finish()
        frame_finished = time.perf_counter()
        physics_ms.append((physics_finished - physics_started) * 1000.0)
        physics_breakdown_ms["world_update"].append(world_update_s * 1000.0)
        physics_breakdown_ms["emission_build"].append(
            emission_build_s * 1000.0
        )
        physics_breakdown_ms["emission_inject"].append(
            emission_inject_s * 1000.0
        )
        physics_breakdown_ms["smoke_step_submit"].append(smoke_step_s * 1000.0)
        visual_ms.append((frame_finished - physics_finished) * 1000.0)
        frame_ms.append((frame_finished - frame_started) * 1000.0)
    result: dict[str, object] = {
        "case": "integrated_blocking",
        "fluid_backend": getattr(app.smoke, "backend_name", "cpu"),
    }
    for name, values in (
        ("frame", frame_ms),
        ("physics", physics_ms),
        ("visual", visual_ms),
    ):
        samples = np.asarray(values, dtype=np.float64)
        result[f"{name}_mean_ms"] = float(samples.mean())
        result[f"{name}_p95_ms"] = float(np.percentile(samples, 95))
        result[f"{name}_p99_ms"] = float(np.percentile(samples, 99))
    breakdown = {}
    for name, values in physics_breakdown_ms.items():
        samples = np.asarray(values, dtype=np.float64)
        active = samples[samples > 0.0]
        breakdown[name] = {
            "mean_all_frames_ms": float(samples.mean()),
            "p95_all_frames_ms": float(np.percentile(samples, 95)),
            "active_frames": int(len(active)),
            "mean_active_ms": float(active.mean()) if len(active) else 0.0,
            "p95_active_ms": (
                float(np.percentile(active, 95)) if len(active) else 0.0
            ),
        }
    smoke_gpu = np.asarray(
        [query.elapsed / 1_000_000.0 for query in smoke_gpu_queries],
        dtype=np.float64,
    )
    breakdown["smoke_step_gpu"] = {
        "active_frames": int(len(smoke_gpu)),
        "mean_active_ms": float(smoke_gpu.mean()) if len(smoke_gpu) else 0.0,
        "p95_active_ms": (
            float(np.percentile(smoke_gpu, 95)) if len(smoke_gpu) else 0.0
        ),
    }
    result["physics_breakdown"] = breakdown
    _close(app)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile visual CPU submission and GPU latency."
    )
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument(
        "--fluid-backend",
        choices=("3d", "2d", "cpu"),
        default="3d",
        help="Select the fluid backend for this isolated process.",
    )
    parser.add_argument(
        "--integrated-only",
        action="store_true",
        help="Run only the coupled blocking case for clean A/B comparison.",
    )
    args = parser.parse_args()
    integrated = _profile_integrated(args.frames, args.fluid_backend)
    if args.integrated_only:
        print(json.dumps({
            "frames": args.frames,
            "integrated": integrated,
            "note": (
                "Run GPU and CPU backends in separate processes to avoid "
                "cross-context driver contamination."
            ),
        }, indent=2))
        return
    cases = [
        _profile_case(
            "full_moving", args.frames, True, True, args.fluid_backend
        ),
        _profile_case(
            "no_smoke_moving", args.frames, False, True, args.fluid_backend
        ),
        _profile_case(
            "full_static", args.frames, True, False, args.fluid_backend
        ),
    ]
    by_name = {str(case["case"]): case for case in cases}
    full = by_name["full_moving"]
    no_smoke = by_name["no_smoke_moving"]
    static = by_name["full_static"]
    report = {
        "frames_per_case": args.frames,
        "cases": cases,
        "integrated": integrated,
        "derived_p95_ms": {
            "smoke_gpu": max(
                float(full["gpu_p95_ms"]) - float(no_smoke["gpu_p95_ms"]),
                0.0,
            ),
            "moving_reflection_gpu": max(
                float(full["gpu_p95_ms"]) - float(static["gpu_p95_ms"]),
                0.0,
            ),
        },
        "notes": [
            "GPU time uses GL timestamp queries around Renderer.render.",
            "Moving camera invalidates planar reflection at its configured rate.",
            "Static camera retains dynamic water normals without redrawing static city geometry.",
            "Integrated timing blocks on GPU completion and advances 120 Hz ballistics plus configured smoke cadence.",
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
