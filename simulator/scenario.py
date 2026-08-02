"""Historical scenario: the single authority for what, where, and when.

Before this module the observer position lived at ``simulator/app.py`` as a
literal, the launch point was the ``(0, 0, 0)`` default argument of
``FireworkWorld.launch``, and the OSM bounding box lived in
``tools/import_osm_scene.py``. ``docs/ARCHITECTURE.md`` already stated that
"camera calibration and launch coordinates belong in scenario data rather than
source code"; this module is that scenario data.

Everything the file asserts carries a provenance record, so a consumer can ask
what grade of evidence stands behind any value. Fields for which no source
exists — notably the firing timeline and the launch positions — are present in
the schema as empty typed collections with grade ``U`` records. They are not
populated with plausible invented values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .geodesy import LocalTangentPlane
from .provenance import ConfidenceGrade, Provenance
from .timebase import PlaybackMode, SeedRegistry, SimulationClock, parse_event_time

SCHEMA_VERSION = 1
"""Incremented when a change would invalidate an existing scenario file."""


@dataclass(frozen=True, slots=True)
class GeodeticPosition:
    """A WGS84 position with an explicitly named vertical datum.

    ``vertical_datum`` is mandatory and free-text on purpose. The project's
    render datum is the median DEM elevation beneath the Han River mask, whose
    own geodetic datum is not documented by the upstream tile source. Naming it
    ``wamis_hangang_bridge_2024-10-05_19:20`` names the station and instant
    that define the shipped event-water reference plane.
    """

    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0
    vertical_datum: str = "wgs84_ellipsoid"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GeodeticPosition":
        try:
            latitude = float(data["latitude_deg"])
            longitude = float(data["longitude_deg"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"position requires numeric latitude_deg and longitude_deg: {data!r}"
            ) from error
        if not -90.0 <= latitude <= 90.0:
            raise ValueError(f"latitude_deg out of range: {latitude}")
        if not -180.0 <= longitude <= 180.0:
            raise ValueError(f"longitude_deg out of range: {longitude}")
        return cls(
            latitude_deg=latitude,
            longitude_deg=longitude,
            altitude_m=float(data.get("altitude_m", 0.0)),
            vertical_datum=str(data.get("vertical_datum", "wgs84_ellipsoid")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "latitude_deg": self.latitude_deg,
            "longitude_deg": self.longitude_deg,
            "altitude_m": self.altitude_m,
            "vertical_datum": self.vertical_datum,
        }


@dataclass(frozen=True, slots=True)
class Observer:
    """A named viewing position the scenario can be reconstructed from."""

    observer_id: str
    name: str
    position: GeodeticPosition
    notes: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Observer":
        if "observer_id" not in data:
            raise ValueError(f"observer requires observer_id: {data!r}")
        return cls(
            observer_id=str(data["observer_id"]),
            name=str(data.get("name", data["observer_id"])),
            position=GeodeticPosition.from_dict(data.get("position", {})),
            notes=str(data.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "observer_id": self.observer_id,
            "name": self.name,
            "position": self.position.to_dict(),
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class LaunchSite:
    """A firing position or barge.

    No dated source publishing 2024-10-05 barge coordinates has been located,
    so the shipped scenario carries an empty list rather than an estimate.
    """

    site_id: str
    name: str
    position: GeodeticPosition
    notes: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LaunchSite":
        if "site_id" not in data:
            raise ValueError(f"launch site requires site_id: {data!r}")
        return cls(
            site_id=str(data["site_id"]),
            name=str(data.get("name", data["site_id"])),
            position=GeodeticPosition.from_dict(data.get("position", {})),
            notes=str(data.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "site_id": self.site_id,
            "name": self.name,
            "position": self.position.to_dict(),
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class ShowEvent:
    """One scheduled firing in the performance timeline.

    The field set follows the reconstruction brief. Values that a partial
    record cannot supply stay ``None`` rather than being defaulted, so a
    consumer can distinguish "zero degrees of tilt" from "tilt unknown".
    """

    event_id: str
    launch_time_utc: datetime
    launch_site_id: str
    shell_profile_id: str
    confidence_grade: ConfidenceGrade = ConfidenceGrade.UNVERIFIED
    tube_azimuth_deg: float | None = None
    tube_elevation_deg: float | None = None
    calibre_mm: float | None = None
    muzzle_velocity_mps: float | None = None
    fuse_delay_s: float | None = None
    seed_name: str = ""
    evidence_ref: str = ""
    notes: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ShowEvent":
        for required in ("event_id", "launch_time_utc", "launch_site_id",
                         "shell_profile_id"):
            if required not in data:
                raise ValueError(f"show event requires {required}: {data!r}")

        def optional_float(key: str) -> float | None:
            value = data.get(key)
            return None if value is None else float(value)

        return cls(
            event_id=str(data["event_id"]),
            launch_time_utc=parse_event_time(str(data["launch_time_utc"])),
            launch_site_id=str(data["launch_site_id"]),
            shell_profile_id=str(data["shell_profile_id"]),
            confidence_grade=ConfidenceGrade(data.get("confidence_grade", "U")),
            tube_azimuth_deg=optional_float("tube_azimuth_deg"),
            tube_elevation_deg=optional_float("tube_elevation_deg"),
            calibre_mm=optional_float("calibre_mm"),
            muzzle_velocity_mps=optional_float("muzzle_velocity_mps"),
            fuse_delay_s=optional_float("fuse_delay_s"),
            seed_name=str(data.get("seed_name", "")),
            evidence_ref=str(data.get("evidence_ref", "")),
            notes=str(data.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "launch_time_utc": self.launch_time_utc.isoformat(),
            "launch_site_id": self.launch_site_id,
            "shell_profile_id": self.shell_profile_id,
            "confidence_grade": self.confidence_grade.value,
            "tube_azimuth_deg": self.tube_azimuth_deg,
            "tube_elevation_deg": self.tube_elevation_deg,
            "calibre_mm": self.calibre_mm,
            "muzzle_velocity_mps": self.muzzle_velocity_mps,
            "fuse_delay_s": self.fuse_delay_s,
            "seed_name": self.seed_name,
            "evidence_ref": self.evidence_ref,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class Scenario:
    """A loaded, validated historical scenario."""

    scenario_id: str
    schema_version: int
    reference_epoch: datetime
    show_start: datetime
    show_end: datetime
    origin: GeodeticPosition
    render_vertical_datum: str
    observers: tuple[Observer, ...]
    launch_sites: tuple[LaunchSite, ...]
    events: tuple[ShowEvent, ...]
    seeds: SeedRegistry
    provenance: Provenance
    source_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    # -- loading -------------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> "Scenario":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"scenario file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        scenario = cls.from_dict(data)
        return dataclass_replace_source(scenario, path)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Scenario":
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported scenario schema_version {version!r}; "
                f"this build understands {SCHEMA_VERSION}"
            )
        for required in ("scenario_id", "reference_epoch", "origin"):
            if required not in data:
                raise ValueError(f"scenario requires {required}")

        reference_epoch = parse_event_time(str(data["reference_epoch"]))
        show = data.get("show", {})
        show_start = parse_event_time(
            str(show.get("start", data["reference_epoch"]))
        )
        show_end = parse_event_time(str(show.get("end", data["reference_epoch"])))
        if show_end < show_start:
            raise ValueError(
                f"show end {show_end.isoformat()} precedes start "
                f"{show_start.isoformat()}"
            )

        observers = tuple(
            Observer.from_dict(entry) for entry in data.get("observers", [])
        )
        if not observers:
            raise ValueError("scenario requires at least one observer")
        identifiers = [observer.observer_id for observer in observers]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"duplicate observer_id in {identifiers}")

        launch_sites = tuple(
            LaunchSite.from_dict(entry) for entry in data.get("launch_sites", [])
        )
        known_sites = {site.site_id for site in launch_sites}
        events = tuple(
            ShowEvent.from_dict(entry) for entry in data.get("events", [])
        )
        # Resolving references at load time turns a typo into a load error
        # rather than a shell that silently never fires.
        from .shells import SHELL_LIBRARY

        known_events: set[str] = set()
        for event in events:
            if event.event_id in known_events:
                raise ValueError(f"duplicate event_id {event.event_id!r}")
            known_events.add(event.event_id)
            if event.launch_site_id not in known_sites:
                raise ValueError(
                    f"event {event.event_id} references unknown launch site "
                    f"{event.launch_site_id!r}"
                )
            if event.shell_profile_id not in SHELL_LIBRARY:
                raise ValueError(
                    f"event {event.event_id} references unknown shell profile "
                    f"{event.shell_profile_id!r}; have {SHELL_LIBRARY.ids()}"
                )

        seeds_data = data.get("seeds", {})
        if "master_seed" not in seeds_data:
            raise ValueError("scenario requires seeds.master_seed")
        seeds = SeedRegistry(int(seeds_data["master_seed"]))

        known_keys = {
            "scenario_id", "schema_version", "reference_epoch", "show",
            "origin", "render_vertical_datum", "observers", "launch_sites",
            "events", "seeds", "provenance",
        }
        return cls(
            scenario_id=str(data["scenario_id"]),
            schema_version=SCHEMA_VERSION,
            reference_epoch=reference_epoch,
            show_start=show_start,
            show_end=show_end,
            origin=GeodeticPosition.from_dict(data["origin"]),
            render_vertical_datum=str(
                data.get("render_vertical_datum", "wgs84_ellipsoid")
            ),
            observers=observers,
            launch_sites=launch_sites,
            events=events,
            seeds=seeds,
            provenance=Provenance.from_dict(data.get("provenance", {})),
            # Unknown keys are preserved rather than dropped so a file written
            # by a newer tool survives a round trip through this build.
            extra={
                key: value for key, value in data.items() if key not in known_keys
            },
        )

    # -- queries -------------------------------------------------------

    @property
    def tangent_plane(self) -> LocalTangentPlane:
        """Local East-Up-South frame anchored at the scenario origin."""

        return LocalTangentPlane(
            self.origin.latitude_deg,
            self.origin.longitude_deg,
            self.origin.altitude_m,
        )

    def observer(self, observer_id: str) -> Observer:
        for candidate in self.observers:
            if candidate.observer_id == observer_id:
                return candidate
        available = [candidate.observer_id for candidate in self.observers]
        raise KeyError(f"unknown observer {observer_id!r}; have {available}")

    @property
    def default_observer(self) -> Observer:
        return self.observers[0]

    def observer_position_eus_m(self, observer_id: str) -> np.ndarray:
        """Observer position in local East-Up-South metres."""

        position = self.observer(observer_id).position
        return self.tangent_plane.to_local(
            position.latitude_deg, position.longitude_deg, position.altitude_m
        )

    def launch_site_position_eus_m(self, site_id: str) -> np.ndarray:
        for site in self.launch_sites:
            if site.site_id == site_id:
                return self.tangent_plane.to_local(
                    site.position.latitude_deg,
                    site.position.longitude_deg,
                    site.position.altitude_m,
                )
        raise KeyError(f"unknown launch site {site_id!r}")

    def make_clock(
        self,
        physics_hz: int = 120,
        mode: PlaybackMode = PlaybackMode.REALTIME,
        epoch: datetime | None = None,
    ) -> SimulationClock:
        """Build the simulation clock.

        The clock's epoch is the *playback origin* and defaults to the show
        start, so playback position zero is the moment the performance begins.
        ``reference_epoch`` is a separate concept: the canonical instant this
        scenario is reported and validated at, which for this show sits ten
        minutes into the window. Conflating the two would force a negative
        playback position.
        """

        return SimulationClock(epoch or self.show_start, physics_hz, mode)

    def show_duration_s(self) -> float:
        return (self.show_end - self.show_start).total_seconds()

    def missing_datasets(self) -> list[str]:
        """Report the scenario collections that no source has yet populated.

        This is deliberately explicit: an empty firing timeline is a known
        missing dataset, not an implicit "no fireworks".
        """

        missing: list[str] = []
        if not self.launch_sites:
            missing.append("launch_sites")
        if not self.events:
            missing.append("events")
        return missing


def dataclass_replace_source(scenario: Scenario, path: Path) -> Scenario:
    """Attach the on-disk path to a loaded scenario."""

    return Scenario(
        scenario_id=scenario.scenario_id,
        schema_version=scenario.schema_version,
        reference_epoch=scenario.reference_epoch,
        show_start=scenario.show_start,
        show_end=scenario.show_end,
        origin=scenario.origin,
        render_vertical_datum=scenario.render_vertical_datum,
        observers=scenario.observers,
        launch_sites=scenario.launch_sites,
        events=scenario.events,
        seeds=scenario.seeds,
        provenance=scenario.provenance,
        source_path=path,
        extra=scenario.extra,
    )


DEFAULT_SCENARIO_PATH = (
    Path(__file__).resolve().parent.parent
    / "assets"
    / "scenario_yeouido_2024-10-05.json"
)


def load_default_scenario() -> Scenario:
    """Load the shipped 2024-10-05 Yeouido scenario."""

    return Scenario.load(DEFAULT_SCENARIO_PATH)
