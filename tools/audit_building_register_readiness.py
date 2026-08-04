"""Audit the building-register route, and derive from it once data arrives.

The facade module dimensions and the untagged height default are the two
weakest numbers in the building geometry: nine assumed floor heights and one
flat 12.0 m constant that sets a sixth of the scene's building vertices. The
official register publishes `heit` and `grndFlrCnt` per building, which makes
`heit / grndFlrCnt` a floor-to-floor height derived from two official values —
the only route out of grade D.

Nothing here fabricates that data. Without a recorded response the audit
reports the route as ready and everything downstream as blocked. With one, it
computes the distribution and the derived floor heights, so the arrival of a
key is the only manual step between here and the upgrade.

The service key is a credential and never enters the repository; the audit
fails if one appears.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import median, quantiles
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUEST = Path("assets/yeouido_building_register_request.json")
DEFAULT_RESPONSE = Path("assets/yeouido_building_register_2024-10-05.json")
DEFAULT_OUTPUT = Path(
    "docs/validation/facade_modules_v3/building_register_report.json"
)

#: A floor-to-floor height outside this range means the record pairs a height
#: with a floor count that does not belong to it — a mixed-use tower counted as
#: one storey, or a height in the wrong unit. Such records are excluded from the
#: derivation rather than silently averaged in.
PLAUSIBLE_FLOOR_HEIGHT_M = (2.0, 12.0)
MINIMUM_DERIVATION_SAMPLE = 20

#: Anything that looks like a portal credential. data.go.kr keys are long
#: URL-encoded tokens, so a long opaque run of key characters is the signal.
_CREDENTIAL = re.compile(r"[A-Za-z0-9%+/=]{40,}")
_CREDENTIAL_KEYS = {"servicekey", "service_key", "apikey", "api_key", "인증키"}


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def find_credentials(value: Any, path: str = "") -> list[str]:
    """Locate anything in the package that looks like a service key."""

    found: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            where = f"{path}.{key}" if path else str(key)
            if str(key).lower() in _CREDENTIAL_KEYS and isinstance(nested, str):
                if nested.strip():
                    found.append(where)
            found.extend(find_credentials(nested, where))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found.extend(find_credentials(nested, f"{path}[{index}]"))
    elif isinstance(value, str) and _CREDENTIAL.search(value):
        found.append(path)
    return found


def _number(value: Any) -> float | None:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result


def derive_from_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Heights, floor counts and the floor-to-floor heights they imply."""

    heights: list[float] = []
    derived: list[dict[str, Any]] = []
    rejected = {"no_height": 0, "no_floor_count": 0, "implausible_floor_height": 0}
    for record in records:
        height = _number(record.get("heit"))
        floors = _number(record.get("grndFlrCnt"))
        if height is None or height <= 0.0:
            rejected["no_height"] += 1
            continue
        heights.append(height)
        if floors is None or floors < 1.0:
            rejected["no_floor_count"] += 1
            continue
        floor_height = height / floors
        low, high = PLAUSIBLE_FLOOR_HEIGHT_M
        if not low <= floor_height <= high:
            rejected["implausible_floor_height"] += 1
            continue
        derived.append(
            {
                "height_m": height,
                "above_ground_floors": floors,
                "floor_height_m": round(floor_height, 4),
                "main_purpose": record.get("mainPurpsCdNm"),
            }
        )

    floor_heights = [row["floor_height_m"] for row in derived]
    by_purpose: dict[str, list[float]] = {}
    for row in derived:
        by_purpose.setdefault(str(row["main_purpose"]), []).append(
            row["floor_height_m"]
        )

    return {
        "record_count": len(records),
        "records_with_height": len(heights),
        "records_usable_for_derivation": len(derived),
        "rejected": rejected,
        "height_median_m": round(median(heights), 2) if heights else None,
        "height_quartiles_m": (
            [round(value, 2) for value in quantiles(heights, n=4)]
            if len(heights) >= 4
            else None
        ),
        "floor_height_median_m": (
            round(median(floor_heights), 3) if floor_heights else None
        ),
        "floor_height_quartiles_m": (
            [round(value, 3) for value in quantiles(floor_heights, n=4)]
            if len(floor_heights) >= 4
            else None
        ),
        "floor_height_by_main_purpose": {
            purpose: {
                "sample": len(values),
                "median_m": round(median(values), 3),
            }
            for purpose, values in sorted(by_purpose.items())
            if len(values) >= MINIMUM_DERIVATION_SAMPLE
        },
        "derivation_sample_sufficient": len(derived) >= MINIMUM_DERIVATION_SAMPLE,
    }


def build_report(
    request_path: Path = DEFAULT_REQUEST,
    response_path: Path | None = DEFAULT_RESPONSE,
) -> dict[str, Any]:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    response = (
        json.loads(response_path.read_text(encoding="utf-8"))
        if response_path is not None and response_path.is_file()
        else None
    )

    credentials = find_credentials(request)
    route = request["verified_route"]
    parameters = request["request_parameters"]
    code_basis = parameters["code_basis"]

    route_valid = bool(
        route.get("base_url") == "https://apis.data.go.kr/1613000/BldRgstHubService"
        and route.get("endpoint") == "/getBrTitleInfo"
        and route.get("browser_verified_utc")
        and route.get("terms", {}).get("approval")
    )
    codes_valid = bool(
        code_basis.get("bjdong_code_10_digit") == "1156011000"
        and code_basis.get("source")
        and parameters.get("sigunguCd")
        + parameters.get("bjdongCd")
        == code_basis["bjdong_code_10_digit"]
    )

    derivation = (
        derive_from_records(response["records"]) if response is not None else None
    )
    # A retrieved response only upgrades anything if it is large enough to carry
    # a distribution and its parameter split was confirmed against the service.
    upgrade_available = bool(
        derivation is not None
        and derivation["derivation_sample_sufficient"]
        and response.get("parameter_split_confirmed") is True
    )

    blockers: list[str] = []
    if credentials:
        blockers.append(
            "the request package contains something that looks like a service "
            "key: " + ", ".join(credentials)
        )
    if not route_valid:
        blockers.append("the verified route is incomplete or has been altered")
    if not codes_valid:
        blockers.append(
            "the district and legal-dong codes do not reconstruct the verified "
            "10-digit code"
        )
    if response is None:
        blockers.append(
            "no register response is recorded, so height and floor count remain "
            "unavailable and the facade dimensions stay at grade D"
        )
    elif not upgrade_available:
        blockers.append(
            "the recorded response is too small to derive a distribution, or its "
            "parameter split has not been confirmed against the service"
        )

    return {
        "schema_version": 1,
        "stage": "V3-1e",
        "request_asset": _display_path(request_path),
        "request_sha256": _digest(request_path),
        "response_asset": (
            _display_path(response_path) if response is not None else None
        ),
        "checks": {
            "route_verified": route_valid,
            "codes_reconstruct_the_verified_value": codes_valid,
            "credential_like_values_found": credentials,
            "no_credential_committed": not credentials,
            "response_recorded": response is not None,
        },
        "route": {
            "portal_url": route.get("portal_url"),
            "base_url": route.get("base_url"),
            "endpoint": route.get("endpoint"),
            "approval": route.get("terms", {}).get("approval"),
            "cost": route.get("terms", {}).get("cost"),
            "use_scope": route.get("terms", {}).get("use_scope"),
            "sigunguCd": parameters.get("sigunguCd"),
            "bjdongCd": parameters.get("bjdongCd"),
        },
        "join_assessment": request["join_assessment"],
        "derivation": derivation,
        "application_gates": {
            "route_ready_for_key_issue": route_valid and codes_valid and not credentials,
            "register_data_recorded": response is not None,
            "grade_a_upgrade_available": upgrade_available,
            "scene_vertices_modified": 0,
            "runtime_geometry_changed_by_this_stage": False,
        },
        "blocking_reasons": blockers,
        "next_evidence_gate": (
            "Issuing the service key is the project owner's action; the dataset "
            "auto-approves, is free and carries no use-scope restriction. Once a "
            "response is recorded, heit / grndFlrCnt supplies floor-to-floor "
            "heights from two official values and replaces the grade-D assumptions."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, default=DEFAULT_REQUEST)
    parser.add_argument("--response", type=Path, default=DEFAULT_RESPONSE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    report = build_report(arguments.request, arguments.response)
    output = arguments.output
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {output}: route_ready_for_key_issue="
        f"{report['application_gates']['route_ready_for_key_issue']}, "
        f"grade_a_upgrade_available="
        f"{report['application_gates']['grade_a_upgrade_available']}"
    )


if __name__ == "__main__":
    main()
