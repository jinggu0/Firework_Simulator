# V1-10a NGII authenticated-delivery intake audit

This audit separates a safe, reproducible intake path from the still-missing
official delivery. It does **not** claim that a 2024-or-earlier map has been
acquired.

## Official acquisition state

- Provider: 국토지리정보원 국토정보플랫폼
- Product: Seoul 1:1,000 digital topographic map, DXF or NGI
- Required current-stage sheet: Seoul2447 (`376082447`)
- Event-area request set: Seoul2447, Seoul2448, Seoul2457, Seoul2458
- Maximum accepted production year: 2024
- Portal state observed on 2026-08-03: download requires login; this repository
  has no authenticated delivery
- Currently listed event-area sheets: 2025, so they are not silently treated as
  the 2024-10-05 state

The acquisition route and login requirement follow the
[Seoul digital-topographic-map notice](https://news.seoul.go.kr/gov/archives/528208)
and the [NGII platform](https://map.ngii.go.kr/ms/map/NlipMap.do).

## Implemented gate

`assets/yeouido_ngii_delivery_receipt.json` is a deliberately pending receipt.
It becomes importable only after all of the following evidence is recorded:

1. authenticated portal download and acquisition timestamp;
2. production year no later than 2024;
3. SHA-256 and byte count of each downloaded package;
4. SHA-256 and byte count of every extracted DXF/NGI member;
5. an explicit projected EPSG code from embedded metadata, provider sidecar, or
   provider receipt—not a catalogue label or nearby survey-control CRS;
6. verified terms allowing local derived use; and
7. coverage of Seoul2447.

The structure importer compares the complete multiset of extracted DXF hashes
and byte counts with the receipt. A missing, added, duplicated, or modified DXF
is rejected even if the file name looks correct.

## Reproduction

```powershell
python -m tools.audit_ngii_delivery_readiness
python -m pytest tests/test_ngii_delivery.py tests/test_ngii_structures.py tests/test_ngii_control_evidence.py -q
```

Expected pending result:

- `safety_gate_passed=true`: the repository refuses unsupported geometry;
- `stage_complete=false`: the authenticated delivery is still absent;
- scene vertices modified: `0`;
- runtime frame path changed: `false`.

After an authorised download, keep the raw package outside version control,
record its checksums and provider evidence in the receipt, and rerun the audit
before invoking `tools.import_ngii_structures`.
