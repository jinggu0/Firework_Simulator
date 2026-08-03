"""Bind NGII DBF height/edge-role attributes to normalized structure features."""

from __future__ import annotations

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
from typing import Any, Iterable
import zipfile

from simulator.ngii_delivery import (
    DEFAULT_NGII_DELIVERY_RECEIPT_PATH,
    load_ngii_delivery_receipt,
    validate_delivery_packages,
)
from tools.import_ngii_structures import parse_shp_polylines


DEFAULT_STRUCTURES = Path("assets/yeouido_ngii_structures.json")
DEFAULT_OUTPUT = Path("assets/yeouido_ngii_structure_attributes.json")
STRUCTURE_CODES = {"C0050000", "F0030000", "F0040000"}
FIELD_ROLE = "\uc0c1\ud558\uad6c\ubd84"
FIELD_HEIGHT = "\ub192\uc774"
FIELD_LENGTH = "\uc5f0\uc7a5"
FIELD_UFID = "UFID"
ROLE_MAP = {"\uc0c1\ub2e8": "upper", "\ud558\ub2e8": "lower"}


def _digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def _path_digest(path: Path) -> str:
    return _digest(path.read_bytes())


def _package_paths(paths: Iterable[Path]) -> list[Path]:
    output: list[Path] = []
    for supplied in paths:
        if supplied.is_dir():
            output.extend(
                path
                for path in supplied.rglob("*")
                if path.is_file()
                and path.suffix.casefold() in {".zip", ".dxf", ".ngi", ".xml"}
            )
        elif supplied.is_file():
            output.append(supplied)
    return output


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def build_attribute_audit(
    input_paths: Iterable[Path],
    *,
    structures_path: Path = DEFAULT_STRUCTURES,
    receipt_path: Path = DEFAULT_NGII_DELIVERY_RECEIPT_PATH,
) -> dict[str, Any]:
    """Extract provider attributes without treating relative height as elevation."""

    try:
        import shapefile
    except ImportError as error:
        raise RuntimeError("install requirements-terrain.txt for DBF audit") from error

    supplied = list(input_paths)
    receipt = load_ngii_delivery_receipt(receipt_path)
    validate_delivery_packages(receipt, supplied)
    structures_raw = structures_path.read_bytes()
    structures = json.loads(structures_raw.decode("utf-8"))
    normalized_features = {
        str(feature["feature_id"]): feature for feature in structures["features"]
    }
    matched: set[str] = set()
    source_members: list[dict[str, Any]] = []
    attributes: list[dict[str, Any]] = []

    for package in sorted(_package_paths(supplied), key=lambda path: path.name):
        if package.suffix.casefold() != ".zip":
            continue
        with zipfile.ZipFile(package) as archive:
            names = set(archive.namelist())
            shp_members = sorted(
                name
                for name in names
                if name.upper().endswith(".SHP")
                and any(name.upper().endswith(f"_{code}.SHP") for code in STRUCTURE_CODES)
            )
            for shp_member in shp_members:
                base = shp_member[:-4]
                companions = [base + extension for extension in (".shp", ".shx", ".dbf", ".prj")]
                missing = [name for name in companions if name not in names]
                if missing:
                    raise ValueError(
                        f"{package.name}:{shp_member} is missing companions: {missing}"
                    )
                raw_members = {name: archive.read(name) for name in companions}
                source_members.append(
                    {
                        "package": package.name,
                        "shp_member": shp_member,
                        "members": [
                            {
                                "member_path": name,
                                "sha256": _digest(raw_members[name]),
                                "bytes": len(raw_members[name]),
                            }
                            for name in companions
                        ],
                    }
                )
                reader = shapefile.Reader(
                    shp=io.BytesIO(raw_members[base + ".shp"]),
                    shx=io.BytesIO(raw_members[base + ".shx"]),
                    dbf=io.BytesIO(raw_members[base + ".dbf"]),
                    encoding="cp949",
                )
                fields = [field[0] for field in reader.fields[1:]]
                label = f"{package.name}:{shp_member}"
                parsed = parse_shp_polylines(raw_members[base + ".shp"], label)
                entity_index = 0
                for shape_record in reader.iterShapeRecords():
                    record = dict(zip(fields, shape_record.record))
                    part_count = len(shape_record.shape.parts)
                    for _ in range(part_count):
                        entity = parsed[entity_index]
                        feature_id = _digest(
                            f"{_digest(raw_members[base + '.shp'])}:{entity_index}:"
                            f"{entity.layer}:{entity.entity_type}".encode("utf-8")
                        )[:24]
                        entity_index += 1
                        feature = normalized_features.get(feature_id)
                        if feature is None:
                            raise ValueError(
                                f"DBF record does not resolve to normalized feature {feature_id}"
                            )
                        if feature_id in matched:
                            raise ValueError(f"duplicate attribute match for {feature_id}")
                        matched.add(feature_id)
                        declared_height = _number(record.get(FIELD_HEIGHT))
                        raw_role = str(record.get(FIELD_ROLE, "")).strip()
                        attributes.append(
                            {
                                "feature_id": feature_id,
                                "kind": feature["kind"],
                                "source_member": label,
                                "record_index": entity_index - 1,
                                "ufid": str(record.get(FIELD_UFID, "")).strip(),
                                "edge_role": ROLE_MAP.get(raw_role, "unknown"),
                                "provider_edge_role": raw_role or None,
                                "declared_relative_height_m": declared_height,
                                "declared_length_m": _number(record.get(FIELD_LENGTH)),
                                "height_usable_for_absolute_elevation": False,
                            }
                        )
                if entity_index != len(parsed):
                    raise ValueError(f"shape/record part mismatch in {label}")

    missing_features = sorted(set(normalized_features) - matched)
    if missing_features:
        raise ValueError(
            f"attribute audit did not cover {len(missing_features)} normalized features"
        )
    positive_heights = [
        item for item in attributes
        if item["declared_relative_height_m"] is not None
        and item["declared_relative_height_m"] > 0.0
    ]
    return {
        "schema_version": 1,
        "target_event_date": structures["target_event_date"],
        "source_year": structures["source_year"],
        "temporal_relation": structures["temporal_relation"],
        "structure_asset": structures_path.as_posix(),
        "structure_asset_sha256": _digest(structures_raw),
        "delivery_receipt": receipt_path.as_posix(),
        "delivery_receipt_sha256": _path_digest(receipt_path),
        "source_members": source_members,
        "features": attributes,
        "summary": {
            "feature_count": len(attributes),
            "upper_edge_count": sum(item["edge_role"] == "upper" for item in attributes),
            "lower_edge_count": sum(item["edge_role"] == "lower" for item in attributes),
            "unknown_edge_role_count": sum(
                item["edge_role"] == "unknown" for item in attributes
            ),
            "relative_height_count": sum(
                item["declared_relative_height_m"] is not None for item in attributes
            ),
            "positive_relative_height_count": len(positive_heights),
            "relative_height_range_m": [
                min(item["declared_relative_height_m"] for item in positive_heights),
                max(item["declared_relative_height_m"] for item in positive_heights),
            ] if positive_heights else None,
        },
        "application": {
            "mesh_merge_allowed": False,
            "absolute_elevation_available": False,
            "policy": (
                "Provider height is a relative structure dimension, not an absolute "
                "Seoul EL or runtime Y coordinate. Edge roles and heights are retained "
                "as evidence but cannot populate points_eus_m.y without an audited "
                "vertical anchoring rule."
            ),
            "blockers": [
                "DXF structure Z coordinates are uniformly 0.0 placeholders",
                "DBF height does not state an absolute elevation datum",
                "upper/lower edge roles are not yet registered to a terrain-side anchor",
                "2025 attributes are not verified as unchanged on 2024-10-05",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="+", type=Path)
    parser.add_argument("--structures", type=Path, default=DEFAULT_STRUCTURES)
    parser.add_argument(
        "--delivery-receipt", type=Path, default=DEFAULT_NGII_DELIVERY_RECEIPT_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    try:
        audit = build_attribute_audit(
            arguments.input,
            structures_path=arguments.structures,
            receipt_path=arguments.delivery_receipt,
        )
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = audit["summary"]
    print(
        f"wrote {arguments.output}: {summary['feature_count']} attributed features, "
        f"{summary['positive_relative_height_count']} positive relative heights; "
        "mesh merge remains blocked"
    )


if __name__ == "__main__":
    main()
