"""Absolute event time separated from playback time.

Before this module the simulator held one bare float,
``SimulatorApp.event_timestamp``, which was advanced only when the weather
asset happened to be present. With the asset missing it stayed at ``0.0`` and
the astronomical state was silently evaluated at 1970-01-01T00:00:00Z. There is
no default epoch here: a clock without an explicit timezone-aware epoch cannot
be constructed.

Two playback modes exist and they are deliberately different:

``REALTIME``
    Wall-clock driven, matching the existing :class:`~simulator.clock.FixedStepClock`
    behaviour including its catch-up cap. Simulated time may fall behind wall
    time under load; the deficit is discarded. Correct for interactive viewing.

``DETERMINISTIC``
    Exactly one fixed step per call, wall time ignored. Required for
    frame-by-frame comparison against reference footage and for the replay
    regression test.

Playback position is stored as an integer step count rather than an accumulated
float, so advancing N steps and then seeking back gives bit-identical times.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib

from .provenance import require_aware_timestamp

KST = timezone(timedelta(hours=9), name="Asia/Seoul")
"""Display and input timezone. Korea has observed no DST since 1988."""

UTC = timezone.utc


class PlaybackMode(Enum):
    REALTIME = "realtime"
    DETERMINISTIC = "deterministic"


def parse_event_time(text: str) -> datetime:
    """Parse an ISO-8601 event timestamp into UTC, rejecting naive input."""

    return require_aware_timestamp(text, "event time").astimezone(UTC)


def format_kst(moment: datetime) -> str:
    """Render an absolute instant in the project's display timezone."""

    return moment.astimezone(KST).isoformat()


@dataclass(frozen=True, slots=True)
class SeedRegistry:
    """Named deterministic sub-seeds derived from one scenario master seed.

    Every stochastic subsystem draws from a *named* sub-seed rather than the
    master, so introducing a new random consumer cannot perturb the sequence
    seen by an existing one. Without that property an A/B comparison of a
    single change is impossible, because the change silently reseeds everything
    downstream of it.

    ``blake2b`` is used rather than :func:`hash` because the built-in string
    hash is randomised per process and would break reproducibility.
    """

    master_seed: int

    def derive(self, name: str) -> int:
        digest = hashlib.blake2b(
            f"{self.master_seed}:{name}".encode("utf-8"), digest_size=8
        ).digest()
        return int.from_bytes(digest, "big") & 0x7FFF_FFFF

    def numpy_seed(self, name: str) -> int:
        return self.derive(name)


class SimulationClock:
    """Authority for absolute event time and playback position.

    Parameters
    ----------
    epoch:
        Timezone-aware instant that playback offset zero refers to.
    physics_hz:
        Fixed step rate. The step is exact (``1 / physics_hz``) and never
        derived from measured frame time.
    mode:
        See :class:`PlaybackMode`.
    """

    __slots__ = (
        "_epoch",
        "_step_s",
        "_step_index",
        "_mode",
        "_rate",
        "_paused",
        "_accumulator_s",
        "_max_catch_up_steps",
    )

    def __init__(
        self,
        epoch: datetime,
        physics_hz: int = 120,
        mode: PlaybackMode = PlaybackMode.REALTIME,
        max_catch_up_steps: int = 8,
    ) -> None:
        if epoch.tzinfo is None or epoch.tzinfo.utcoffset(epoch) is None:
            raise ValueError(
                "SimulationClock requires a timezone-aware epoch; there is no "
                "default epoch"
            )
        if physics_hz <= 0:
            raise ValueError(f"physics_hz must be positive, got {physics_hz}")
        self._epoch = epoch.astimezone(UTC)
        self._step_s = 1.0 / physics_hz
        self._step_index = 0
        self._mode = mode
        self._rate = 1.0
        self._paused = False
        self._accumulator_s = 0.0
        self._max_catch_up_steps = max_catch_up_steps

    # -- configuration -------------------------------------------------

    @property
    def epoch(self) -> datetime:
        return self._epoch

    @property
    def step_s(self) -> float:
        return self._step_s

    @property
    def mode(self) -> PlaybackMode:
        return self._mode

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def rate(self) -> float:
        return self._rate

    def set_paused(self, paused: bool) -> None:
        self._paused = bool(paused)

    def set_rate(self, rate: float) -> None:
        if rate < 0.0:
            raise ValueError(f"playback rate must be non-negative, got {rate}")
        self._rate = float(rate)

    # -- position ------------------------------------------------------

    @property
    def step_index(self) -> int:
        """Number of fixed steps elapsed since the epoch."""

        return self._step_index

    @property
    def playback_time_s(self) -> float:
        """Seconds since the epoch, reconstructed exactly from the step count."""

        return self._step_index * self._step_s

    @property
    def absolute_time(self) -> datetime:
        """Current absolute event instant, in UTC."""

        return self._epoch + timedelta(seconds=self.playback_time_s)

    @property
    def local_time(self) -> datetime:
        """Current absolute event instant, in ``Asia/Seoul``."""

        return self.absolute_time.astimezone(KST)

    @property
    def posix_timestamp(self) -> float:
        """POSIX seconds, for interoperation with the existing float API."""

        return self._epoch.timestamp() + self.playback_time_s

    # -- advancing -----------------------------------------------------

    def advance_steps(self, steps: int = 1) -> int:
        """Advance by exact fixed steps. Wall time is not consulted."""

        if steps < 0:
            raise ValueError(f"cannot advance by {steps} steps; use seek")
        if self._paused:
            return 0
        self._step_index += steps
        return steps

    def consume_frame(self, frame_time_s: float) -> int:
        """Advance for one rendered frame and return the number of steps taken.

        In ``DETERMINISTIC`` mode this is exactly one step regardless of
        ``frame_time_s``, which is what makes replay reproducible. In
        ``REALTIME`` mode it mirrors :class:`~simulator.clock.FixedStepClock`:
        the frame time is clamped against the spiral of death and the catch-up
        burst is capped, so simulated time may lag wall time under load.
        """

        if self._paused:
            return 0
        if self._mode is PlaybackMode.DETERMINISTIC:
            self._step_index += 1
            return 1
        self._accumulator_s += min(max(frame_time_s, 0.0), 0.25) * self._rate
        steps = min(
            int(self._accumulator_s / self._step_s), self._max_catch_up_steps
        )
        self._accumulator_s -= steps * self._step_s
        self._step_index += steps
        return steps

    # -- seeking -------------------------------------------------------

    def seek_to_step(self, step_index: int) -> None:
        if step_index < 0:
            raise ValueError(f"step index must be non-negative, got {step_index}")
        self._step_index = int(step_index)
        self._accumulator_s = 0.0

    def seek_to_playback_s(self, playback_time_s: float) -> None:
        """Seek to the nearest fixed step at or before ``playback_time_s``."""

        if playback_time_s < 0.0:
            raise ValueError(
                f"playback time must be non-negative, got {playback_time_s}"
            )
        self.seek_to_step(int(round(playback_time_s / self._step_s)))

    def seek_to_absolute(self, moment: datetime) -> None:
        if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
            raise ValueError("seek target must be timezone-aware")
        self.seek_to_playback_s(
            (moment.astimezone(UTC) - self._epoch).total_seconds()
        )

    def seek_to_posix(self, timestamp: float) -> None:
        """Seek using POSIX seconds, for the existing float-based call sites."""

        self.seek_to_playback_s(timestamp - self._epoch.timestamp())

    def reset(self) -> None:
        self._step_index = 0
        self._accumulator_s = 0.0

    def state(self) -> dict[str, object]:
        """Serialisable snapshot used by reports and determinism checks."""

        return {
            "epoch_utc": self._epoch.isoformat(),
            "epoch_kst": format_kst(self._epoch),
            "mode": self._mode.value,
            "physics_step_s": self._step_s,
            "step_index": self._step_index,
            "playback_time_s": self.playback_time_s,
            "absolute_time_utc": self.absolute_time.isoformat(),
            "absolute_time_kst": format_kst(self.absolute_time),
            "posix_timestamp": self.posix_timestamp,
        }
