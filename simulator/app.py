from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

import moderngl
import pygame

from .camera import FreeCamera
from .astronomy import AstronomyModel
from .clock import FixedStepClock
from .config import SimulationConfig
from .environment import EnvironmentTimeline
from .fluid import SmokeFluid2D
from .physics import FireworkWorld
from .renderer import Renderer


class SimulatorApp:
    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        render = self.config.render
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
        )
        pygame.display.set_mode(
            (render.width, render.height),
            pygame.OPENGL | pygame.DOUBLEBUF,
            vsync=1 if render.vsync else 0,
        )
        pygame.display.set_caption("Yeouido Fireworks Simulator")
        self.ctx = moderngl.create_context()
        environment_path = (
            Path(__file__).resolve().parent.parent
            / "assets"
            / "yeouido_2024-10-05_environment.json"
        )
        self.environment = (
            EnvironmentTimeline.load(environment_path)
            if environment_path.exists()
            else None
        )
        self.event_timestamp = (
            self.environment.show_start_timestamp if self.environment else 0.0
        )
        initial_atmosphere = (
            self.environment.sample(self.event_timestamp)
            if self.environment
            else self.config.atmosphere
        )
        self.world = FireworkWorld(
            initial_atmosphere, self.config.shell,
            render.max_particles, self.config.random_seed
        )
        self.astronomy = AstronomyModel(37.529, 126.935, 5.0)
        self.celestial = self.astronomy.sample(self.event_timestamp)
        self.next_astronomy_update = self.event_timestamp + 1.0
        self.renderer = Renderer(
            self.ctx, render, initial_atmosphere, self.config.smoke
        )
        self.smoke = SmokeFluid2D(self.config.smoke, initial_atmosphere)
        self.smoke_accumulator_s = 0.0
        self.camera = FreeCamera(config=self.config.camera)
        self.physics_clock = FixedStepClock(render.physics_hz)
        self.frame_clock = pygame.time.Clock()
        self.running = True
        self.mouse_captured = True
        self.title_timer_s = 0.0
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        pygame.mouse.get_rel()
        self.world.launch()

    def run(self, max_frames: int | None = None) -> None:
        previous = time.perf_counter()
        frame_count = 0
        while self.running:
            now = time.perf_counter()
            dt_s, previous = now - previous, now
            self._events()
            self._update_camera(min(dt_s, 0.1))
            steps, _alpha = self.physics_clock.consume(dt_s)
            for _ in range(steps):
                if self.environment:
                    self.world.atmosphere = self.environment.sample(
                        self.event_timestamp
                    )
                    self.event_timestamp += self.physics_clock.step_s
                self.world.update(self.physics_clock.step_s)
                for burst in self.world.consume_burst_events():
                    self.smoke.inject_burst(
                        burst.position_m,
                        burst.smoke_mass_kg,
                        burst.post_blast_thermal_energy_j,
                    )
                self.smoke_accumulator_s += self.physics_clock.step_s
                smoke_step_s = 1.0 / self.config.smoke.update_hz
                while self.smoke_accumulator_s >= smoke_step_s:
                    self.smoke.set_atmosphere(self.world.atmosphere)
                    self.smoke.step(smoke_step_s)
                    self.smoke_accumulator_s -= smoke_step_s
            if self.event_timestamp >= self.next_astronomy_update:
                self.celestial = self.astronomy.sample(self.event_timestamp)
                self.next_astronomy_update = self.event_timestamp + 1.0
            self.renderer.render(
                self.world, self.camera, self.celestial, dt_s, self.smoke
            )
            pygame.display.flip()
            # Swap interval is the primary limiter. Applying SDL's millisecond
            # timer after a blocking swap produces systematic sub-60 FPS pacing.
            self.frame_clock.tick(
                0 if self.config.render.vsync else self.config.render.target_fps
            )
            self._title(dt_s)
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                self.running = False
        pygame.quit()

    def _events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.mouse_captured:
                        self._set_mouse_capture(False)
                    else:
                        self.running = False
                elif event.key == pygame.K_TAB:
                    self._set_mouse_capture(not self.mouse_captured)
                elif event.key == pygame.K_SPACE:
                    self.world.launch()
            elif event.type == pygame.MOUSEBUTTONDOWN and not self.mouse_captured:
                if event.button == 1:
                    self._set_mouse_capture(True)
            elif event.type == pygame.MOUSEMOTION and self.mouse_captured:
                self.camera.look(*event.rel)

    def _set_mouse_capture(self, captured: bool) -> None:
        self.mouse_captured = captured
        pygame.event.set_grab(captured)
        pygame.mouse.set_visible(not captured)
        pygame.mouse.get_rel()

    def _update_camera(self, dt_s: float) -> None:
        keys = pygame.key.get_pressed()
        local_input = (
            float(keys[pygame.K_d]) - float(keys[pygame.K_a]),
            float(keys[pygame.K_e]) - float(keys[pygame.K_q]),
            float(keys[pygame.K_w]) - float(keys[pygame.K_s]),
        )
        sprint = bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])
        self.camera.update(dt_s, local_input, sprint)

    def _title(self, dt_s: float) -> None:
        self.title_timer_s += dt_s
        if self.title_timer_s >= .5:
            event_time = ""
            weather = ""
            if self.environment:
                kst = timezone(timedelta(hours=9))
                event_time = datetime.fromtimestamp(
                    self.event_timestamp, kst
                ).strftime(" | %H:%M:%S KST")
                wind = self.world.atmosphere.wind_velocity_mps
                weather = (
                    f" | {self.world.atmosphere.temperature_k - 273.15:.1f} C"
                    f" | wind {(wind[0] ** 2 + wind[2] ** 2) ** 0.5:.1f} m/s"
                )
            pygame.display.set_caption(
                f"Yeouido Fireworks | {self.frame_clock.get_fps():.1f} FPS"
                f" | {self.world.stars.count:,} stars"
                f"{event_time}"
                f"{weather}"
                f" | XYZ {self.camera.position_m[0]:.0f},"
                f"{self.camera.position_m[1]:.0f},"
                f"{self.camera.position_m[2]:.0f} m"
            )
            self.title_timer_s = 0.0


def run(max_frames: int | None = None) -> None:
    SimulatorApp().run(max_frames=max_frames)
