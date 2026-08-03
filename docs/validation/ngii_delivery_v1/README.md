# V1-10d NGII authenticated-delivery audit

This audit locks the authenticated NGII delivery that the project owner chose
to adopt. It permits planimetric normalization, but it does **not** claim that
the 2025 production represents the exact 2024-10-05 event state.

## Adopted delivery

- Provider: 국토지리정보원 국토정보플랫폼
- Product: 1:1,000 digital topographic map
- Sheets: Seoul2447, Seoul2448, Seoul2457, Seoul2458
- Production year: 2025
- Raw evidence retained outside version control: four DXFs, four digital-map
  2.0 ZIPs, and eight XML metadata sidecars
- Receipt lock: SHA-256 and byte count for all 16 raw artifacts
- Normalization input: eight structure-layer SHP members from the four ZIPs
- Projected CRS: EPSG:5186, resolved from the provider PRJ sidecars
- Structure layers: `C0050000`, `F0030000`, and `F0040000`

The receipt records the project owner's explicit post-event authorization and
sets `historical_identity_claim=false`. Raw NGII files remain in the user's
Downloads directory and are not redistributed through this repository.

## Normalized result

`assets/yeouido_ngii_structures.json` contains 71 planimetric features:

- 15 embankments;
- 6 cut/fill lines;
- 50 retaining-wall lines; and
- 3,560.14 m total planar length.

None of the selected SHPs contain source elevation. Every normalized height is
therefore `null`; no wall height or vertical profile is inferred. Runtime scene
vertices remain unchanged until separate elevation evidence is supplied.

## Reproduction

Keep the original 16 files in the Downloads directory, then run:

```powershell
$ngiiFiles = Get-ChildItem -LiteralPath "$env:USERPROFILE\Downloads" -File |
  Where-Object { $_.Name -like '(B010)수치지도_3760824*' }
python -m tools.import_ngii_structures $ngiiFiles.FullName `
  --source-crs EPSG:5186 --source-year 2025 --allow-post-event-source
python -m tools.audit_ngii_delivery_readiness
python -m pytest tests/test_ngii_delivery.py tests/test_ngii_structures.py -q
```

Expected audit result:

- `safety_gate_passed=true`;
- `stage_complete=true` for authenticated intake and normalization readiness;
- `historical_fidelity_complete=false`;
- scene vertices modified: `0`; and
- runtime frame path changed: `false`.

The next merge gate requires cited vertical profiles and a 2024-10-05 change
audit. The archived-delivery request remains available in
`ARCHIVE_REQUEST.md` if exact event-date evidence is later pursued.
