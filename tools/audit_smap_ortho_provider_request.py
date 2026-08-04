"""Audit the S-Map orthophoto provider enquiry without submitting it.

V2-2e registered the provider's own EPSG:5186 tile grid directly, so the world
file *is* the registration. What it could not establish is when the imagery was
flown, whether the crop can be checked against anything independent, and what
may be done with tiles pulled from the tile service. Those three are questions
for the operator, not computations, so this audit checks that the enquiry
package actually asks them and that nothing downstream opens until a reply
arrives.

The load-bearing check is `gate_classification_complete`. Every gate the V2-2e
report publishes must be classified here as provider-answerable, downstream, or
a status flag. If a later stage adds a gate and nobody classifies it, the audit
blocks rather than quietly ignoring it — a silent omission is exactly how an
open question turns into an assumed answer.
"""

from __future__ import annotations

import argparse
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUEST = Path("assets/yeouido_smap_ortho_provider_request.json")
DEFAULT_REGISTRATION_REPORT = Path(
    "docs/validation/road_marking_registration_v2/smap_2024_registration_report.json"
)
DEFAULT_OUTPUT = Path(
    "docs/validation/road_marking_registration_v2/"
    "smap_2024_provider_request_report.json"
)

EVENT_DATE = date(2024, 10, 5)
MAXIMUM_CHECK_POINT_RESIDUAL_M = 1.0

#: Gates the S-Map operator can actually answer. Each must be addressed by at
#: least one question while it is still closed.
PROVIDER_ANSWERABLE_GATES = {
    "exact_imagery_acquisition_date_confirmed",
    "independent_spatial_check_passes",
    "raw_tile_redistribution_authorised",
}
#: Outcomes of the gates above rather than questions in their own right. They
#: open only when their inputs open, so asking about them directly would be
#: circular.
DOWNSTREAM_GATES = {"event_date_marking_classification_allowed"}
#: Recorded state, not something to unlock. `runtime_geometry_changed_by_this
#: _stage` being false is the desired condition, so treating false as "open"
#: would invert its meaning.
STATUS_FLAGS = {
    "provider_pixels_acquired_locally",
    "provider_crs_and_grid_explicit",
    "native_25cm_resolution_confirmed",
    "runtime_geometry_changed_by_this_stage",
}

REQUIRED_DELIVERABLES = {
    "exact acquisition date or date range covering the requested bbox",
    "single-sortie or multi-date mosaic composition for that bbox",
    "source agency and applicable copyright policy for the layer",
    "positional accuracy document or check-point residuals",
    "orthorectification reference surface",
    "local research use scope for tiles retrieved from the tile service",
    "required attribution wording",
    "official reply reference or receipt",
}
FORBIDDEN_PERSONAL_KEYS = {
    "name",
    "contact",
    "email",
    "phone",
    "dob",
    "birth",
    "applicant_name",
    "address",
    "이름",
    "연락처",
    "이메일",
    "전화번호",
    "생년월일",
}


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _collect_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key).lower() for key in value}
        for nested in value.values():
            keys.update(_collect_keys(nested))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for nested in value:
            keys.update(_collect_keys(nested))
        return keys
    return set()


def _published_gates(registration: dict[str, Any]) -> dict[str, bool]:
    gates = dict(registration["application_gates"])
    gates.pop("reason", None)
    # The redistribution question lives in `source`, but it gates application
    # exactly like the rest, so the classification must cover it too.
    gates["raw_tile_redistribution_authorised"] = registration["source"][
        "raw_tile_redistribution_authorised"
    ]
    return {key: value for key, value in gates.items() if isinstance(value, bool)}


def _parse_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def evaluate_reply(reply: dict[str, Any] | None) -> dict[str, Any]:
    """Decide what a provider reply does and does not unlock.

    Kept separate from the request checks so the decision can be exercised
    against replies that never touch the repository.
    """

    if reply is None:
        return {
            "reply_received": False,
            "acquisition_date_confirmed": False,
            "event_date_applicability_confirmed": False,
            "independent_spatial_check_passes": False,
            "tile_local_research_use_confirmed": False,
            "latest_acquisition_date": None,
            "reasons": ["no provider reply has been received"],
        }

    reasons: list[str] = []
    coverage = reply.get("coverage", {})
    acquisition = reply.get("acquisition", {})
    accuracy = reply.get("positional_accuracy", {})
    use_scope = reply.get("use_scope", {})

    latest = _parse_date(acquisition.get("date_latest"))
    earliest = _parse_date(acquisition.get("date_earliest"))
    basis = acquisition.get("basis")

    date_confirmed = bool(
        coverage.get("covers_requested_bbox") is True
        and latest is not None
        and earliest is not None
        and basis in {"provider_statement", "attached_document"}
    )
    if coverage.get("covers_requested_bbox") is not True:
        reasons.append("the reply does not cover the requested bounding box")
    if latest is None or earliest is None:
        reasons.append("the reply states no acquisition date range")
    if basis not in {"provider_statement", "attached_document"}:
        reasons.append("the reply gives no basis for the acquisition date")

    # A multi-date mosaic qualifies only if its *latest* contributing frame
    # predates the event. Comparing the earliest would let imagery captured
    # after the event in as though it were contemporaneous.
    event_applicable = bool(date_confirmed and latest is not None and latest <= EVENT_DATE)
    if date_confirmed and not event_applicable:
        reasons.append(
            f"the latest acquisition date {acquisition.get('date_latest')} "
            f"is after the {EVENT_DATE.isoformat()} event"
        )

    residual = accuracy.get("independent_check_point_residual_m")
    check_passes = bool(
        accuracy.get("document_provided") is True
        and isinstance(residual, (int, float))
        and not isinstance(residual, bool)
        and residual <= MAXIMUM_CHECK_POINT_RESIDUAL_M
    )
    if not check_passes:
        reasons.append(
            "no independent check-point residual at or below "
            f"{MAXIMUM_CHECK_POINT_RESIDUAL_M} m is documented"
        )

    tile_use_confirmed = use_scope.get("tile_service_local_research_use_allowed") is True
    if not tile_use_confirmed:
        reasons.append("local research use of tile-service tiles is not confirmed")

    return {
        "reply_received": True,
        "acquisition_date_confirmed": date_confirmed,
        "event_date_applicability_confirmed": event_applicable,
        "independent_spatial_check_passes": check_passes,
        "tile_local_research_use_confirmed": tile_use_confirmed,
        "latest_acquisition_date": acquisition.get("date_latest"),
        "reasons": reasons,
    }


def build_report(
    request_path: Path = DEFAULT_REQUEST,
    registration_report_path: Path = DEFAULT_REGISTRATION_REPORT,
    reply_path: Path | None = None,
) -> dict[str, Any]:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    registration = json.loads(registration_report_path.read_text(encoding="utf-8"))
    reply = (
        json.loads(reply_path.read_text(encoding="utf-8"))
        if reply_path is not None and reply_path.is_file()
        else None
    )

    published = _published_gates(registration)
    classified = PROVIDER_ANSWERABLE_GATES | DOWNSTREAM_GATES | STATUS_FLAGS
    unclassified = sorted(set(published) - classified)
    missing_from_report = sorted(classified - set(published))
    gate_classification_complete = not unclassified and not missing_from_report

    open_gates = {
        gate
        for gate in PROVIDER_ANSWERABLE_GATES
        if published.get(gate) is False
    }
    questions = request.get("questions", [])
    addressed = {question.get("addresses_gate") for question in questions}
    unaddressed_gates = sorted(open_gates - addressed)
    # A question aimed at something already satisfied, or at a name that is not
    # a gate at all, means the package has drifted from the report it cites.
    stale_questions = sorted(
        str(question.get("addresses_gate"))
        for question in questions
        if question.get("addresses_gate") not in open_gates
    )
    questions_well_formed = all(
        isinstance(question.get("question_ko"), str)
        and question["question_ko"].strip()
        and isinstance(question.get("why"), str)
        and question["why"].strip()
        for question in questions
    )

    evidence = request.get("evidence", {})
    registration_link_matches = bool(
        evidence.get("registration_report")
        == _display_path(registration_report_path)
        and evidence.get("registration_report_sha256")
        == _digest(registration_report_path)
    )

    submission = request.get("submission", {})
    route_valid = bool(
        submission.get("primary_route") == "https://smap.seoul.go.kr/guide/qna.html"
        and submission.get("provider") == "서울특별시 공간정보과"
        and submission.get("primary_route_verified_utc")
    )
    personal_keys = sorted(_collect_keys(request) & FORBIDDEN_PERSONAL_KEYS)
    privacy_safe = bool(
        not personal_keys
        and submission.get("applicant_personal_information_stored") is False
    )
    external_action_safe = bool(
        submission.get("credentials_entered_by_agent") is False
        and submission.get("captcha_entered_by_agent") is False
        and submission.get("provider_attachments_downloaded_by_agent") is False
        and submission.get("submission_is_user_gated") is True
    )
    deliverables_complete = (
        set(request.get("required_deliverables", [])) == REQUIRED_DELIVERABLES
    )

    outcome = evaluate_reply(reply)
    # The floor guarantee: a favourable date alone never unlocks the event-day
    # reading. Every input gate must pass, which is why an answered enquiry can
    # still leave the marking gate shut on the check-point evidence.
    marking_allowed = bool(
        outcome["event_date_applicability_confirmed"]
        and outcome["independent_spatial_check_passes"]
        and outcome["tile_local_research_use_confirmed"]
    )

    payload_ready = bool(
        request.get("status")
        in {
            "ready_for_manual_submission_not_submitted",
            "submitted_pending_provider_reply",
            "reply_received",
        }
        and gate_classification_complete
        and not unaddressed_gates
        and not stale_questions
        and questions_well_formed
        and registration_link_matches
        and route_valid
        and privacy_safe
        and external_action_safe
        and deliverables_complete
        and request.get("acceptance_gate", {}).get(
            "event_date_marking_classification_allowed"
        )
        is False
        and request.get("acceptance_gate", {}).get("scene_vertices_modified") == 0
    )

    blockers: list[str] = []
    if unclassified:
        blockers.append(
            "the registration report publishes unclassified gates: "
            + ", ".join(unclassified)
        )
    if missing_from_report:
        blockers.append(
            "classified gates are absent from the registration report: "
            + ", ".join(missing_from_report)
        )
    if unaddressed_gates:
        blockers.append(
            "open gates without a question: " + ", ".join(unaddressed_gates)
        )
    if stale_questions:
        blockers.append(
            "questions aimed at gates that are not open: "
            + ", ".join(stale_questions)
        )
    if not questions_well_formed:
        blockers.append("a question is missing its text or its rationale")
    if not registration_link_matches:
        blockers.append("registration report link or checksum does not match")
    if not route_valid:
        blockers.append("the verified provider route is missing or altered")
    if not privacy_safe:
        blockers.append("the request package contains a personal-information field")
    if not external_action_safe:
        blockers.append("an external action is not recorded as user-gated")
    if not deliverables_complete:
        blockers.append("the required provider evidence list is incomplete")
    blockers.extend(outcome["reasons"])

    return {
        "schema_version": 1,
        "stage": "V2-2f",
        "request_asset": _display_path(request_path),
        "request_sha256": _digest(request_path),
        "registration_report": _display_path(registration_report_path),
        "registration_report_sha256": _digest(registration_report_path),
        "reply_asset": _display_path(reply_path) if reply is not None else None,
        "checks": {
            "gate_classification_complete": gate_classification_complete,
            "unclassified_gates": unclassified,
            "classified_gates_absent_from_report": missing_from_report,
            "open_provider_gates": sorted(open_gates),
            "open_gates_without_a_question": unaddressed_gates,
            "stale_questions": stale_questions,
            "questions_well_formed": questions_well_formed,
            "question_count": len(questions),
            "registration_link_matches": registration_link_matches,
            "provider_route_valid": route_valid,
            "personal_information_keys_found": personal_keys,
            "privacy_safe": privacy_safe,
            "external_action_safe": external_action_safe,
            "required_deliverables_complete": deliverables_complete,
        },
        "reply_outcome": outcome,
        "submission_payload_ready": payload_ready,
        "ready_for_manual_submission": payload_ready
        and submission.get("performed") is not True,
        "external_submission_authorized": submission.get("authorized") is True,
        "external_submission_performed": submission.get("performed") is True,
        "event_date_marking_classification_allowed": marking_allowed,
        "scene_vertices_modified": 0,
        "runtime_geometry_changed_by_this_stage": False,
        "blocking_reasons": blockers,
        "next_evidence_gate": (
            "Submission is the project owner's action. When a reply arrives, record "
            "it against assets/smap_ortho_provider_reply.schema.json and re-run this "
            "audit. Event-day marking classification additionally needs an "
            "independent check point at or below "
            f"{MAXIMUM_CHECK_POINT_RESIDUAL_M} m, which no held source supplies."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, default=DEFAULT_REQUEST)
    parser.add_argument(
        "--registration-report", type=Path, default=DEFAULT_REGISTRATION_REPORT
    )
    parser.add_argument(
        "--reply",
        type=Path,
        default=None,
        help="provider reply recorded against the reply schema, when one exists",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    report = build_report(
        arguments.request, arguments.registration_report, arguments.reply
    )
    output = arguments.output
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {output}: ready_for_manual_submission="
        f"{report['ready_for_manual_submission']}, "
        f"event_date_marking_classification_allowed="
        f"{report['event_date_marking_classification_allowed']}"
    )


if __name__ == "__main__":
    main()
