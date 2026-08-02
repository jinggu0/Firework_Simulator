from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import time

import moderngl
import pygame

from .acoustics import FireworkAcoustics, SoundArrival
from .camera import FreeCamera
from .astronomy import AstronomyModel
from .clock import FixedStepClock
from .config import SimulationConfig
from .environment import EnvironmentTimeline
from .fluid import SmokeFluid2D
from .gpu_fluid import create_smoke_solver
from .physics import FireworkWorld
from .performance import FrameTimings
from .renderer import Renderer
from .scenario import DEFAULT_SCENARIO_PATH, Scenario
from .shells import SHELL_LIBRARY
from .show import ShowScheduler
from .terrain import TerrainSurface


class SimulatorApp:
    def __init__(
        self,
        config: SimulationConfig | None = None,
        scenario_path: Path | None = None,
    ) -> None:
        self.config = config or SimulationConfig()
        render = self.config.render
        pygame.mixer.pre_init(
            self.config.acoustics.sample_rate_hz, -16, 2, 512
        )
        pygame.init()
        self.requested_gl_version = self._create_display(render)
        pygame.display.set_caption("Yeouido Fireworks Simulator")
        self.ctx = moderngl.create_context()
        assets = Path(__file__).resolve().parent.parent / "assets"
        # The scenario file, not this module, is the authority for the observer
        # position and the show's absolute time. Loading fails loudly on a
        # missing or timezone-naive epoch; there is deliberately no fallback,
        # because the previous default silently evaluated the sky at
        # 1970-01-01T00:00:00Z whenever the weather asset was absent.
        self.scenario = Scenario.load(scenario_path or DEFAULT_SCENARIO_PATH)
        # Playback position zero is the show start; the scenario's reference
        # epoch is a separate reporting instant, not the playback origin.
        self.clock = self.scenario.make_clock(render.physics_hz)
        environment_path = assets / "yeouido_2024-10-05_environment.json"
        self.environment = (
            EnvironmentTimeline.load(environment_path)
            if environment_path.exists()
            else None
        )
        initial_atmosphere = (
            self.environment.sample(self.event_timestamp)
            if self.environment
            else self.config.atmosphere
        )
        self.world = FireworkWorld(
            initial_atmosphere, self.config.shell,
            render.max_particles,
            self.scenario.seeds.derive("shell_burst"),
        )
        observer = self.scenario.default_observer
        self.observer_position_eus_m = self.scenario.observer_position_eus_m(
            observer.observer_id
        )
        self.astronomy = AstronomyModel(
            observer.position.latitude_deg,
            observer.position.longitude_deg,
            observer.position.altitude_m,
        )
        self.celestial = self.astronomy.sample(self.event_timestamp)
        self.next_astronomy_update = self.event_timestamp + 1.0
        self.renderer = Renderer(
            self.ctx,
            render,
            initial_atmosphere,
            self.config.smoke,
            self.config.lighting,
            self.config.physical_camera,
        )
        self.smoke = create_smoke_solver(
            self.ctx, self.config.smoke, initial_atmosphere
        )
        self.smoke_accumulator_s = 0.0
        self.camera = FreeCamera(config=self.config.camera)
        scene_data = self.renderer.scene.data
        self.terrain_surface = TerrainSurface(
            scene_data.terrain_height_m,
            scene_data.terrain_bounds,
            scene_data.water_mask,
            scene_data.water_mask_bounds,
        )
        self.acoustics = FireworkAcoustics(
            self.config.acoustics, self.scenario.seeds.derive("acoustics")
        )
        self.audio_enabled = pygame.mixer.get_init() is not None
        self.active_sounds: list[pygame.mixer.Sound] = []
        self.audio_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="firework-audio"
        )
        self.pending_audio: dict[int, Future] = {}
        self.audio_due: set[int] = set()
        self.last_sound_level_db: float | None = None
        self.physics_clock = FixedStepClock(render.physics_hz)
        self.frame_clock = pygame.time.Clock()
        self.scheduler = ShowScheduler(self.scenario)
        self.scheduler.seek_to(self.clock.absolute_time)
        self.manual_profile_ids = SHELL_LIBRARY.ids()
        self.manual_profile_index = 0
        self.running = True
        self.mouse_captured = True
        self.title_timer_s = 0.0
        self.frame_timings = FrameTimings()
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        pygame.mouse.get_rel()
        if not self.scheduler.launches:
            # The historical scenario carries no firing timeline, so nothing
            # would ever be visible. One development shell keeps the scene
            # inspectable without implying it is part of the performance.
            self.world.launch()

    @property
    def event_timestamp(self) -> float:
        """Absolute event time in POSIX seconds, owned by the simulation clock.

        Kept as a float property so existing float-based call sites, including
        ``tools/profile_runtime.py``, continue to work while the clock remains
        the single authority.
        """

        return self.clock.posix_timestamp

    @event_timestamp.setter
    def event_timestamp(self, timestamp: float) -> None:
        self.clock.seek_to_posix(timestamp)

    @staticmethod
    def _create_display(render) -> tuple[int, int]:
        last_error: pygame.error | None = None
        for major, minor in ((4, 3), (3, 3)):
            try:
                pygame.display.gl_set_attribute(
                    pygame.GL_CONTEXT_MAJOR_VERSION, major
                )
                pygame.display.gl_set_attribute(
                    pygame.GL_CONTEXT_MINOR_VERSION, minor
                )
                pygame.display.gl_set_attribute(
                    pygame.GL_CONTEXT_PROFILE_MASK,
                    pygame.GL_CONTEXT_PROFILE_CORE,
                )
                pygame.display.set_mode(
                    (render.width, render.height),
                    pygame.OPENGL | pygame.DOUBLEBUF,
                    vsync=1 if render.vsync else 0,
                )
                return major, minor
            except pygame.error as error:
                last_error = error
                pygame.display.quit()
                pygame.display.init()
        raise pygame.error(
            f"OpenGL 3.3 core context unavailable: {last_error}"
        )

    def run(self, max_frames: int | None = None) -> None:
        previous = time.perf_counter()
        frame_count = 0
        while self.running:
            frame_started = time.perf_counter()
            now = time.perf_counter()
            dt_s, previous = now - previous, now
            self._events()
            self._update_camera(min(dt_s, 0.1))
            physics_started = time.perf_counter()
            steps, _alpha = self.physics_clock.consume(dt_s)
            for _ in range(steps):
                if self.environment:
                    self.world.atmosphere = self.environment.sample(
                        self.event_timestamp
                    )
                # Absolute time advances whether or not the weather asset is
                # present, so the celestial state can no longer freeze at the
                # scenario epoch when an asset is missing.
                self.clock.advance_steps(1)
                for launch in self.scheduler.due(self.clock.absolute_time):
                    self.world.launch(
                        tuple(float(value) for value in launch.position_eus_m),
                        profile=launch.profile,
                        azimuth_deg=launch.azimuth_deg,
                        elevation_deg=launch.elevation_deg,
                        event_id=launch.event_id,
                    )
                self.world.update(self.physics_clock.step_s)
                for burst in self.world.consume_burst_events():
                    self.smoke.inject_burst(
                        burst.position_m,
                        burst.smoke_mass_kg,
                        burst.post_blast_thermal_energy_j,
                    )
                    self.acoustics.emit(
                        burst.position_m, burst.chemical_energy_j
                    )
                for upcoming in self.acoustics.prepare_upcoming(
                    0.12,
                    self.camera.position_m,
                    self.camera.right,
                    self.world.atmosphere,
                ):
                    self._prepare_arrival(upcoming)
                arrivals = self.acoustics.update(
                    self.physics_clock.step_s,
                    self.camera.position_m,
                    self.camera.right,
                    self.world.atmosphere,
                )
                for arrival in arrivals:
                    self._play_arrival(arrival)
                self.smoke_accumulator_s += self.physics_clock.step_s
                smoke_step_s = 1.0 / self.smoke.update_hz
                while self.smoke_accumulator_s >= smoke_step_s:
                    emissions = self.world.consume_combustion_emissions()
                    for emission in emissions:
                        self.smoke.inject_particles(
                            emission.position_m,
                            emission.smoke_mass_kg,
                            emission.thermal_energy_j,
                        )
                    self.smoke.set_atmosphere(self.world.atmosphere)
                    self.smoke.step(smoke_step_s)
                    self.smoke_accumulator_s -= smoke_step_s
            physics_finished = time.perf_counter()
            if self.event_timestamp >= self.next_astronomy_update:
                self.celestial = self.astronomy.sample(self.event_timestamp)
                self.next_astronomy_update = self.event_timestamp + 1.0
            self.renderer.render(
                self.world, self.camera, self.celestial, dt_s, self.smoke
            )
            self._flush_audio()
            pygame.display.flip()
            frame_finished = time.perf_counter()
            self.frame_timings.record(
                (frame_finished - frame_started) * 1000.0,
                (physics_finished - physics_started) * 1000.0,
                (frame_finished - physics_finished) * 1000.0,
            )
            # Swap interval is the primary limiter. Applying SDL's millisecond
            # timer after a blocking swap produces systematic sub-60 FPS pacing.
            self.frame_clock.tick(
                0 if self.config.render.vsync else self.config.render.target_fps
            )
            self._title(dt_s)
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                self.running = False
        self.audio_executor.shutdown(wait=True, cancel_futures=True)
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
                    self.world.launch(profile=self.manual_profile())
                elif event.key == pygame.K_v:
                    self.renderer.toggle_display_mode()
                elif event.key == pygame.K_g:
                    self.camera.set_walking(
                        not self.camera.walking, self.terrain_surface
                    )
                elif event.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                    step = 1 if event.key == pygame.K_RIGHTBRACKET else -1
                    self.manual_profile_index = (
                        self.manual_profile_index + step
                    ) % len(self.manual_profile_ids)
            elif event.type == pygame.MOUSEBUTTONDOWN and not self.mouse_captured:
                if event.button == 1:
                    self._set_mouse_capture(True)
            elif event.type == pygame.MOUSEMOTION and self.mouse_captured:
                self.camera.look(*event.rel)

    def manual_profile(self):
        """Shell profile fired by the manual launch key."""

        return SHELL_LIBRARY.get(
            self.manual_profile_ids[self.manual_profile_index]
        )

    def _set_mouse_capture(self, captured: bool) -> None:
        self.mouse_captured = captured
        pygame.event.set_grab(captured)
        pygame.mouse.set_visible(not captured)
        pygame.mouse.get_rel()

    def _play_arrival(self, arrival: SoundArrival) -> None:
        self.last_sound_level_db = arrival.sound_pressure_level_db
        if not self.audio_enabled:
            return
        self._prepare_arrival(arrival)
        self.audio_due.add(arrival.seed)

    def _prepare_arrival(self, arrival: SoundArrival) -> None:
        if not self.audio_enabled or arrival.seed in self.pending_audio:
            return
        self.pending_audio[arrival.seed] = self.audio_executor.submit(
            self.acoustics.synthesize_pcm, arrival
        )

    def _flush_audio(self) -> None:
        waiting: dict[int, Future] = {}
        for seed, future in self.pending_audio.items():
            if seed not in self.audio_due or not future.done():
                waiting[seed] = future
                continue
            try:
                sound = pygame.sndarray.make_sound(future.result())
                sound.play()
                self.active_sounds.append(sound)
            except (pygame.error, RuntimeError):
                self.audio_enabled = False
            self.audio_due.discard(seed)
        self.pending_audio = waiting
        self.active_sounds = self.active_sounds[-8:]

    def _update_camera(self, dt_s: float) -> None:
        keys = pygame.key.get_pressed()
        local_input = (
            float(keys[pygame.K_d]) - float(keys[pygame.K_a]),
            float(keys[pygame.K_e]) - float(keys[pygame.K_q]),
            float(keys[pygame.K_w]) - float(keys[pygame.K_s]),
        )
        sprint = bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])
        self.camera.update(dt_s, local_input, sprint, self.terrain_surface)

    def _title(self, dt_s: float) -> None:
        self.title_timer_s += dt_s
        if self.title_timer_s >= .5:
            event_time = self.clock.local_time.strftime(" | %H:%M:%S KST")
            weather = ""
            if self.environment:
                wind = self.world.atmosphere.wind_velocity_mps
                weather = (
                    f" | {self.world.atmosphere.temperature_k - 273.15:.1f} C"
                    f" | wind {(wind[0] ** 2 + wind[2] ** 2) ** 0.5:.1f} m/s"
                    f" | Hs {self.renderer.significant_wave_height_m * 100:.1f} cm"
                    # Modelled, not observed: the weather record carries no
                    # visibility field. Shown because it is what the frame is
                    # actually drawn with.
                    f" | vis {self.renderer.visibility_m / 1000.0:.1f} km*"
                )
            sound_level = (
                f" | last boom {self.last_sound_level_db:.1f} dB"
                if self.last_sound_level_db is not None else ""
            )
            if self.scheduler.launches:
                show = (
                    f" | shot {self.scheduler.fired_count}"
                    f"/{len(self.scheduler)}"
                )
            else:
                show = f" | manual {self.manual_profile().profile_id}"
            pygame.display.set_caption(
                f"Yeouido Fireworks | {self.frame_clock.get_fps():.1f} FPS"
                f" | p95 {self.frame_timings.snapshot()['frame']['p95_ms']:.1f} ms"
                f"{show}"
                f" | {self.world.stars.count:,} stars"
                f" | fluid {getattr(self.smoke, 'backend_name', 'cpu')}"
                f"{sound_level}"
                f"{event_time}"
                f"{weather}"
                f" | camera {'walk' if self.camera.walking else 'free'}"
                f" | XYZ {self.camera.position_m[0]:.0f},"
                f"{self.camera.position_m[1]:.0f},"
                f"{self.camera.position_m[2]:.0f} m"
            )
            self.title_timer_s = 0.0


def run(
    max_frames: int | None = None, scenario_path: Path | None = None
) -> None:
    SimulatorApp(scenario_path=scenario_path).run(max_frames=max_frames)
