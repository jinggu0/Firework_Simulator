"""Import surveyed linear structures from NGII 1:1,000 ASCII DXF sheets.

This importer deliberately stops short of generating render meshes.  A mapped
retaining-wall line without a measured top/bottom elevation is planimetric
evidence, not permission to invent its height.  The normalized JSON retains
that distinction for the later evidence-gated scene merge.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence
import zipfile

import numpy as np

from simulator.geodesy import LocalTangentPlane
from simulator.scene import load_scene


DEFAULT_LAYER_KINDS = {
    "C0050000": "embankment",
    "F0030000": "cut_or_fill",
    "F0040000": "retaining_wall",
}
EVENT_DATE = date(2024, 10, 5)


@dataclass(frozen=True, slots=True)
class DxfPolyline:
    """One source entity in its unmodified projected coordinate system."""

    layer: str
    entity_type: str
    points: tuple[tuple[float, float, float | None], ...]


def _pairs(text: str) -> list[tuple[int, str]]:
    lines = text.splitlines()
    if len(lines) % 2:
        raise ValueError("malformed ASCII DXF: group-code line has no value")
    result: list[tuple[int, str]] = []
    for offset in range(0, len(lines), 2):
        try:
            code = int(lines[offset].strip())
        except ValueError as error:
            raise ValueError(
                f"malformed ASCII DXF group code on line {offset + 1}"
            ) from error
        result.append((code, lines[offset + 1].strip()))
    return result


def _value(groups: Sequence[tuple[int, str]], code: int) -> str | None:
    return next((value for group_code, value in groups if group_code == code), None)


def _number(
    groups: Sequence[tuple[int, str]], code: int, *, required: bool = False
) -> float | None:
    raw = _value(groups, code)
    if raw is None:
        if required:
            raise ValueError(f"DXF entity is missing required group {code}")
        return None
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"invalid numeric value for DXF group {code}: {raw!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"non-finite numeric value for DXF group {code}")
    return value


def _integer(groups: Sequence[tuple[int, str]], code: int, default: int = 0) -> int:
    raw = _value(groups, code)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"invalid integer value for DXF group {code}: {raw!r}") from error


def _entity_groups(
    pairs: Sequence[tuple[int, str]], start: int
) -> tuple[list[tuple[int, str]], int]:
    end = start
    while end < len(pairs) and pairs[end][0] != 0:
        end += 1
    return list(pairs[start:end]), end


def _line(groups: Sequence[tuple[int, str]]) -> tuple[tuple[float, float, float | None], ...]:
    x0 = _number(groups, 10, required=True)
    y0 = _number(groups, 20, required=True)
    x1 = _number(groups, 11, required=True)
    y1 = _number(groups, 21, required=True)
    assert x0 is not None and y0 is not None and x1 is not None and y1 is not None
    return ((x0, y0, _number(groups, 30)), (x1, y1, _number(groups, 31)))


def _lwpolyline(
    groups: Sequence[tuple[int, str]],
) -> tuple[tuple[float, float, float | None], ...]:
    elevation = _number(groups, 38)
    points: list[list[float | None]] = []
    bulges: list[float] = []
    for code, raw in groups:
        if code == 10:
            try:
                x = float(raw)
            except ValueError as error:
                raise ValueError(f"invalid LWPOLYLINE x coordinate: {raw!r}") from error
            if not math.isfinite(x):
                raise ValueError("non-finite LWPOLYLINE x coordinate")
            points.append([x, None, elevation])
            bulges.append(0.0)
        elif code == 20:
            if not points or points[-1][1] is not None:
                raise ValueError("LWPOLYLINE y coordinate has no matching x coordinate")
            try:
                y = float(raw)
            except ValueError as error:
                raise ValueError(f"invalid LWPOLYLINE y coordinate: {raw!r}") from error
            if not math.isfinite(y):
                raise ValueError("non-finite LWPOLYLINE y coordinate")
            points[-1][1] = y
        elif code == 42:
            if not points:
                raise ValueError("LWPOLYLINE bulge has no matching vertex")
            try:
                bulge = float(raw)
            except ValueError as error:
                raise ValueError(f"invalid LWPOLYLINE bulge: {raw!r}") from error
            if not math.isfinite(bulge):
                raise ValueError("non-finite LWPOLYLINE bulge")
            bulges[-1] = bulge
    if len(points) < 2 or any(point[1] is None for point in points):
        raise ValueError("LWPOLYLINE must contain at least two complete vertices")
    vertices = tuple((float(x), float(y), z) for x, y, z in points)
    return _expand_bulges(vertices, bulges, bool(_integer(groups, 70) & 1))


def _expand_bulges(
    vertices: Sequence[tuple[float, float, float | None]],
    bulges: Sequence[float],
    closed: bool,
    maximum_angle_deg: float = 5.0,
) -> tuple[tuple[float, float, float | None], ...]:
    """Tessellate DXF bulge arcs without replacing source vertical evidence."""

    if len(vertices) < 2:
        return tuple(vertices)
    segment_count = len(vertices) if closed else len(vertices) - 1
    output = [vertices[0]]
    for index in range(segment_count):
        start = vertices[index]
        end = vertices[(index + 1) % len(vertices)]
        bulge = bulges[index] if index < len(bulges) else 0.0
        dx, dy = end[0] - start[0], end[1] - start[1]
        chord = math.hypot(dx, dy)
        if chord <= 1e-9:
            if end != output[-1]:
                output.append(end)
            continue
        if abs(bulge) <= 1e-12:
            output.append(end)
            continue

        angle = 4.0 * math.atan(bulge)
        radius = chord * (1.0 + bulge * bulge) / (4.0 * abs(bulge))
        centre_offset = chord * (1.0 - bulge * bulge) / (4.0 * bulge)
        midpoint_x = (start[0] + end[0]) * 0.5
        midpoint_y = (start[1] + end[1]) * 0.5
        centre_x = midpoint_x - dy / chord * centre_offset
        centre_y = midpoint_y + dx / chord * centre_offset
        start_angle = math.atan2(start[1] - centre_y, start[0] - centre_x)
        subdivisions = max(
            2, math.ceil(abs(math.degrees(angle)) / maximum_angle_deg)
        )
        for step in range(1, subdivisions + 1):
            if step == subdivisions:
                output.append(end)
                continue
            fraction = step / subdivisions
            sample_angle = start_angle + angle * fraction
            if start[2] is not None and end[2] is not None:
                elevation = start[2] + (end[2] - start[2]) * fraction
            else:
                elevation = None
            output.append(
                (
                    centre_x + radius * math.cos(sample_angle),
                    centre_y + radius * math.sin(sample_angle),
                    elevation,
                )
            )
    return tuple(output)


def parse_ascii_dxf(text: str) -> list[DxfPolyline]:
    """Parse the linear entities needed by Korean digital topographic maps."""

    pairs = _pairs(text)
    output: list[DxfPolyline] = []
    index = 0
    section: str | None = None
    while index < len(pairs):
        code, raw_type = pairs[index]
        entity_type = raw_type.upper()
        if code == 0 and entity_type == "SECTION":
            section = pairs[index + 1][1].upper() if index + 1 < len(pairs) else None
            index += 2
            continue
        if code == 0 and entity_type == "ENDSEC":
            section = None
            index += 1
            continue
        if code != 0 or section != "ENTITIES":
            index += 1
            continue

        if entity_type in {"LINE", "LWPOLYLINE"}:
            groups, index = _entity_groups(pairs, index + 1)
            layer = (_value(groups, 8) or "").strip().upper()
            points = _line(groups) if entity_type == "LINE" else _lwpolyline(groups)
            output.append(DxfPolyline(layer, entity_type, points))
            continue

        if entity_type == "POLYLINE":
            header, index = _entity_groups(pairs, index + 1)
            layer = (_value(header, 8) or "").strip().upper()
            header_elevation = _number(header, 30)
            vertices: list[tuple[float, float, float | None]] = []
            bulges: list[float] = []
            while index < len(pairs) and pairs[index][0] == 0:
                nested_type = pairs[index][1].upper()
                if nested_type == "VERTEX":
                    groups, index = _entity_groups(pairs, index + 1)
                    x = _number(groups, 10, required=True)
                    y = _number(groups, 20, required=True)
                    assert x is not None and y is not None
                    vertex_elevation = _number(groups, 30)
                    vertices.append(
                        (
                            x,
                            y,
                            header_elevation
                            if vertex_elevation is None
                            else vertex_elevation,
                        )
                    )
                    bulges.append(_number(groups, 42) or 0.0)
                    continue
                if nested_type == "SEQEND":
                    _, index = _entity_groups(pairs, index + 1)
                break
            if len(vertices) >= 2:
                output.append(
                    DxfPolyline(
                        layer,
                        entity_type,
                        _expand_bulges(vertices, bulges, bool(_integer(header, 70) & 1)),
                    )
                )
            continue

        _, index = _entity_groups(pairs, index + 1)
    return output


def resolve_layer_kind(layer: str, mapping: Mapping[str, str]) -> str | None:
    normalized = layer.strip().upper()
    if normalized in mapping:
        return mapping[normalized]
    for code, kind in mapping.items():
        if any(normalized.startswith(code + separator) for separator in ("_", "-", " ")):
            return kind
    return None


def validate_source_year(
    source_year: int,
    event_date: date = EVENT_DATE,
    *,
    allow_post_event_source: bool = False,
) -> str:
    if source_year < 1900 or source_year > date.today().year:
        raise ValueError(f"implausible source year: {source_year}")
    if source_year > event_date.year and not allow_post_event_source:
        raise ValueError(
            f"source year {source_year} is later than event date {event_date}; "
            "pass --allow-post-event-source only after documenting the temporal mismatch"
        )
    if source_year > event_date.year:
        return "official_post_event"
    if source_year == event_date.year:
        return "official_same_year_date_unverified"
    return "official_pre_event"


def _decode(data: bytes, label: str) -> str:
    for encoding in ("utf-8-sig", "cp949", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"unable to decode {label} as ASCII-compatible DXF")


def iter_dxf_sources(paths: Iterable[Path]) -> Iterator[tuple[str, bytes]]:
    for path in paths:
        candidates = (
            sorted(
                candidate
                for candidate in path.rglob("*")
                if candidate.suffix.casefold() in {".dxf", ".zip"}
            )
            if path.is_dir()
            else [path]
        )
        for candidate in candidates:
            if candidate.suffix.casefold() == ".zip":
                with zipfile.ZipFile(candidate) as archive:
                    for member in sorted(archive.namelist()):
                        if member.casefold().endswith(".dxf"):
                            yield f"{candidate.name}:{member}", archive.read(member)
            elif candidate.suffix.casefold() == ".dxf":
                yield str(candidate), candidate.read_bytes()


def _bbox_intersects(points: Sequence[Sequence[float]], bounds: Sequence[float]) -> bool:
    x = [point[0] for point in points]
    z = [point[2] for point in points]
    return not (
        max(x) < bounds[0]
        or min(x) > bounds[2]
        or max(z) < bounds[1]
        or min(z) > bounds[3]
    )


def build_normalized_asset(
    sources: Iterable[tuple[str, bytes]],
    *,
    source_crs: str,
    source_year: int,
    scene_path: Path,
    layer_kinds: Mapping[str, str] = DEFAULT_LAYER_KINDS,
    allow_post_event_source: bool = False,
) -> dict[str, object]:
    """Transform selected entities to the runtime East-Up-South frame."""

    try:
        import pyproj
    except ImportError as error:
        raise RuntimeError("install requirements-terrain.txt for DXF reprojection") from error

    temporal_relation = validate_source_year(
        source_year, allow_post_event_source=allow_post_event_source
    )
    scene = load_scene(scene_path)
    transformer = pyproj.Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    plane = LocalTangentPlane(scene.origin_latitude_deg, scene.origin_longitude_deg)
    source_records: list[dict[str, object]] = []
    features: list[dict[str, object]] = []
    total_length_m = 0.0

    for label, raw in sources:
        source_records.append(
            {"label": label, "sha256": sha256(raw).hexdigest(), "bytes": len(raw)}
        )
        source_checksum = sha256(raw).hexdigest()
        for entity_index, entity in enumerate(parse_ascii_dxf(_decode(raw, label))):
            kind = resolve_layer_kind(entity.layer, layer_kinds)
            if kind is None:
                continue
            east = np.array([point[0] for point in entity.points], dtype=np.float64)
            north = np.array([point[1] for point in entity.points], dtype=np.float64)
            longitude, latitude = transformer.transform(east, north)
            local = plane.to_local_array(np.asarray(latitude), np.asarray(longitude))
            normalized_points: list[list[float | None]] = []
            has_elevation = any(point[2] is not None for point in entity.points)
            for source_point, local_point in zip(entity.points, local, strict=True):
                relative_height = (
                    float(source_point[2]) - scene.elevation_datum_m
                    if source_point[2] is not None
                    else None
                )
                normalized_points.append(
                    [float(local_point[0]), relative_height, float(local_point[2])]
                )
            if not _bbox_intersects(normalized_points, scene.terrain_bounds):
                continue
            planar = np.array([[point[0], point[2]] for point in normalized_points])
            length_m = float(np.linalg.norm(np.diff(planar, axis=0), axis=1).sum())
            total_length_m += length_m
            features.append(
                {
                    "feature_id": sha256(
                        f"{source_checksum}:{entity_index}:{entity.layer}:"
                        f"{entity.entity_type}".encode("utf-8")
                    ).hexdigest()[:24],
                    "kind": kind,
                    "source_layer": entity.layer,
                    "source_entity": entity.entity_type,
                    "has_source_elevation": has_elevation,
                    "length_m": length_m,
                    "points_eus_m": normalized_points,
                }
            )

    counts = {kind: 0 for kind in sorted(set(layer_kinds.values()))}
    for feature in features:
        counts[str(feature["kind"])] += 1
    return {
        "schema_version": 2,
        "target_event_date": EVENT_DATE.isoformat(),
        "source_year": source_year,
        "temporal_relation": temporal_relation,
        "source_crs": source_crs,
        "coordinate_frame": "local East-Up-South metres; y=0 at scene elevation datum",
        "scene_origin": {
            "latitude_deg": scene.origin_latitude_deg,
            "longitude_deg": scene.origin_longitude_deg,
            "elevation_datum_m": scene.elevation_datum_m,
        },
        "source_files": source_records,
        "layer_mapping": dict(layer_kinds),
        "features": features,
        "summary": {
            "feature_count": len(features),
            "feature_count_by_kind": counts,
            "planar_length_m": total_length_m,
            "features_with_source_elevation": sum(
                bool(feature["has_source_elevation"]) for feature in features
            ),
        },
        "mesh_policy": (
            "No wall height is inferred. A render mesh may be generated only from "
            "source elevation or separately cited survey/photogrammetry."
        ),
    }


def _layer_overrides(values: Sequence[str]) -> dict[str, str]:
    mapping = dict(DEFAULT_LAYER_KINDS)
    for value in values:
        if "=" not in value:
            raise ValueError(f"layer override must be CODE=kind, got {value!r}")
        code, kind = value.split("=", 1)
        code, kind = code.strip().upper(), kind.strip()
        if not code or not kind:
            raise ValueError(f"layer override must be CODE=kind, got {value!r}")
        mapping[code] = kind
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="+", type=Path, help="DXF, ZIP, or directory")
    parser.add_argument("--source-crs", required=True, help="confirmed projected CRS")
    parser.add_argument("--source-year", required=True, type=int)
    parser.add_argument("--scene", type=Path, default=Path("assets/yeouido_scene.npz"))
    parser.add_argument(
        "--output", type=Path, default=Path("assets/yeouido_ngii_structures.json")
    )
    parser.add_argument("--layer", action="append", default=[], metavar="CODE=KIND")
    parser.add_argument("--allow-post-event-source", action="store_true")
    arguments = parser.parse_args()

    sources = list(iter_dxf_sources(arguments.input))
    if not sources:
        parser.error("no .dxf files found in the supplied paths")
    try:
        asset = build_normalized_asset(
            sources,
            source_crs=arguments.source_crs,
            source_year=arguments.source_year,
            scene_path=arguments.scene,
            layer_kinds=_layer_overrides(arguments.layer),
            allow_post_event_source=arguments.allow_post_event_source,
        )
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(asset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = asset["summary"]
    print(
        f"wrote {arguments.output}: {summary['feature_count']} features, "
        f"{summary['planar_length_m']:.1f} m"
    )


if __name__ == "__main__":
    main()
