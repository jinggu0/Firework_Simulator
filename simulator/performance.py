from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class FrameTimings:
    """Bounded runtime timing history reported in milliseconds."""

    capacity: int = 240
    _frame_ms: deque[float] = field(init=False)
    _physics_ms: deque[float] = field(init=False)
    _render_ms: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        self._frame_ms = deque(maxlen=self.capacity)
        self._physics_ms = deque(maxlen=self.capacity)
        self._render_ms = deque(maxlen=self.capacity)

    def record(
        self, frame_ms: float, physics_ms: float, render_ms: float
    ) -> None:
        self._frame_ms.append(float(frame_ms))
        self._physics_ms.append(float(physics_ms))
        self._render_ms.append(float(render_ms))

    @staticmethod
    def _stats(samples: deque[float]) -> dict[str, float]:
        if not samples:
            return {"mean_ms": 0.0, "p95_ms": 0.0, "maximum_ms": 0.0}
        values = np.fromiter(samples, dtype=np.float64)
        return {
            "mean_ms": float(values.mean()),
            "p95_ms": float(np.percentile(values, 95)),
            "maximum_ms": float(values.max()),
        }

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {
            "frame": self._stats(self._frame_ms),
            "physics": self._stats(self._physics_ms),
            "render": self._stats(self._render_ms),
        }

