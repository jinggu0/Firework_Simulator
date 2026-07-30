"""Performance scheduler: fires scenario events on the absolute clock.

The simulator previously had no timeline at all — one development shell was
launched at startup and further shells only on a key press. This module turns
the scenario's `events` list into launches keyed to absolute event time, so a
reconstruction plays the performance rather than a demonstration.

The scheduler holds no randomness. Every launch is determined by the event
record and the clock, which is what allows the replay to be compared frame by
frame against a recording.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

import numpy as np

from .scenario import Scenario, ShowEvent
from .shells import SHELL_LIBRARY, ShellLibrary, ShellProfile


@dataclass(frozen=True, slots=True)
class ScheduledLaunch:
    """A resolved firing instruction ready to hand to the physics world."""

    event: ShowEvent
    profile: ShellProfile
    position_eus_m: np.ndarray
    azimuth_deg: float
    elevation_deg: float

    @property
    def event_id(self) -> str:
        return self.event.event_id


def _apply_event_overrides(
    profile: ShellProfile, event: ShowEvent
) -> ShellProfile:
    """Let a per-shot record override the profile's nominal values.

    A firing record may know the exact calibre, muzzle velocity, or fuse delay
    of one shot while the rest of the shell matches the library archetype.
    Fields left as ``None`` in the event keep the profile's value, so "not
    recorded" stays distinguishable from "recorded as zero".
    """

    overrides: dict[str, float] = {}
    if event.calibre_mm is not None:
        overrides["diameter_m"] = event.calibre_mm / 1_000.0
    if event.muzzle_velocity_mps is not None:
        overrides["launch_speed_mps"] = event.muzzle_velocity_mps
    if event.fuse_delay_s is not None:
        overrides["fuse_delay_s"] = event.fuse_delay_s
    if not overrides:
        return profile
    return replace(profile, **overrides)


class ShowScheduler:
    """Releases scenario events as their absolute launch times arrive."""

    __slots__ = ("_launches", "_next_index", "_library")

    def __init__(
        self, scenario: Scenario, library: ShellLibrary = SHELL_LIBRARY
    ) -> None:
        self._library = library
        launches: list[ScheduledLaunch] = []
        for event in scenario.events:
            profile = _apply_event_overrides(
                library.get(event.shell_profile_id), event
            )
            position = scenario.launch_site_position_eus_m(event.launch_site_id)
            launches.append(
                ScheduledLaunch(
                    event=event,
                    profile=profile,
                    position_eus_m=position.astype(np.float32),
                    # A tube with no recorded heading is treated as vertical,
                    # which is the common case and is visibly wrong if the real
                    # heading later proves otherwise.
                    azimuth_deg=(
                        0.0 if event.tube_azimuth_deg is None
                        else event.tube_azimuth_deg
                    ),
                    elevation_deg=(
                        90.0 if event.tube_elevation_deg is None
                        else event.tube_elevation_deg
                    ),
                )
            )
        # Stable sort keeps the scenario's ordering for simultaneous events, so
        # two shells fired at the same instant always resolve in file order.
        launches.sort(key=lambda launch: launch.event.launch_time_utc)
        self._launches: tuple[ScheduledLaunch, ...] = tuple(launches)
        self._next_index = 0

    def __len__(self) -> int:
        return len(self._launches)

    @property
    def launches(self) -> tuple[ScheduledLaunch, ...]:
        return self._launches

    @property
    def fired_count(self) -> int:
        return self._next_index

    @property
    def remaining(self) -> int:
        return len(self._launches) - self._next_index

    @property
    def finished(self) -> bool:
        return self._next_index >= len(self._launches)

    def next_launch_time(self) -> datetime | None:
        if self.finished:
            return None
        return self._launches[self._next_index].event.launch_time_utc

    def due(self, absolute_time: datetime) -> list[ScheduledLaunch]:
        """Return every launch whose time has arrived since the last call."""

        if absolute_time.tzinfo is None:
            raise ValueError("scheduler requires a timezone-aware instant")
        released: list[ScheduledLaunch] = []
        while (
            self._next_index < len(self._launches)
            and self._launches[self._next_index].event.launch_time_utc
            <= absolute_time
        ):
            released.append(self._launches[self._next_index])
            self._next_index += 1
        return released

    def seek_to(self, absolute_time: datetime) -> None:
        """Move the cursor without firing, for a clock seek or a replay restart.

        Events before ``absolute_time`` are marked consumed rather than fired,
        so seeking forward does not dump the whole show into one frame.
        """

        if absolute_time.tzinfo is None:
            raise ValueError("scheduler requires a timezone-aware instant")
        self._next_index = sum(
            1
            for launch in self._launches
            if launch.event.launch_time_utc <= absolute_time
        )

    def reset(self) -> None:
        self._next_index = 0

    def summary(self) -> dict[str, object]:
        """Counts by profile and pattern, for reports and validation."""

        by_profile: dict[str, int] = {}
        by_pattern: dict[str, int] = {}
        for launch in self._launches:
            by_profile[launch.profile.profile_id] = (
                by_profile.get(launch.profile.profile_id, 0) + 1
            )
            pattern = launch.profile.pattern.value
            by_pattern[pattern] = by_pattern.get(pattern, 0) + 1
        first = self._launches[0].event.launch_time_utc if self._launches else None
        last = self._launches[-1].event.launch_time_utc if self._launches else None
        return {
            "event_count": len(self._launches),
            "first_launch_utc": first.isoformat() if first else "",
            "last_launch_utc": last.isoformat() if last else "",
            "duration_s": (last - first).total_seconds() if first and last else 0.0,
            "by_profile": dict(sorted(by_profile.items())),
            "by_pattern": dict(sorted(by_pattern.items())),
            "total_stars": sum(
                launch.profile.burst_star_count for launch in self._launches
            ),
        }
