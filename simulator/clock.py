from __future__ import annotations


class FixedStepClock:
    """Accumulator clock keeping simulation results independent of frame rate."""

    def __init__(self, physics_hz: int, max_catch_up_steps: int = 8) -> None:
        self.step_s = 1.0 / physics_hz
        self.max_catch_up_steps = max_catch_up_steps
        self.accumulator_s = 0.0

    def consume(self, frame_time_s: float) -> tuple[int, float]:
        self.accumulator_s += min(frame_time_s, 0.25)
        steps = min(
            int(self.accumulator_s / self.step_s), self.max_catch_up_steps
        )
        self.accumulator_s -= steps * self.step_s
        interpolation = self.accumulator_s / self.step_s
        return steps, interpolation

