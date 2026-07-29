from __future__ import annotations

import time

import moderngl
import pygame

from .clock import FixedStepClock
from .config import SimulationConfig
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
            vsync=1,
        )
        pygame.display.set_caption("Yeouido Fireworks Simulator")
        self.ctx = moderngl.create_context()
        self.world = FireworkWorld(
            self.config.atmosphere, self.config.shell,
            render.max_particles, self.config.random_seed
        )
        self.renderer = Renderer(self.ctx, render)
        self.physics_clock = FixedStepClock(render.physics_hz)
        self.frame_clock = pygame.time.Clock()
        self.running = True
        self.title_timer_s = 0.0
        self.world.launch()

    def run(self, max_frames: int | None = None) -> None:
        previous = time.perf_counter()
        frame_count = 0
        while self.running:
            now = time.perf_counter()
            dt_s, previous = now - previous, now
            self._events()
            steps, _alpha = self.physics_clock.consume(dt_s)
            for _ in range(steps):
                self.world.update(self.physics_clock.step_s)
            self.renderer.render(self.world, dt_s)
            pygame.display.flip()
            self.frame_clock.tick(self.config.render.target_fps)
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
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.world.launch()

    def _title(self, dt_s: float) -> None:
        self.title_timer_s += dt_s
        if self.title_timer_s >= .5:
            pygame.display.set_caption(
                f"Yeouido Fireworks | {self.frame_clock.get_fps():.1f} FPS"
                f" | {self.world.stars.count:,} stars"
            )
            self.title_timer_s = 0.0


def run(max_frames: int | None = None) -> None:
    SimulatorApp().run(max_frames=max_frames)
