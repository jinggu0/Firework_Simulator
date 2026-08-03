# V1-11b NGII vertical-anchor audit

This audit tests whether the already authenticated NGII 1:1,000 delivery can
provide absolute vertical anchors for the normalized structure lines.

## Evidence examined

- 35 `F0010000` contour features;
- 12 `F0020000` spot-height features;
- 71 attributed structure lines, including nine lower-edge features; and
- the existing 2023 1:5,000 derived terrain and its fit metrics.

All SHP/SHX/DBF/PRJ evidence is inside the checksum-locked provider ZIPs. Raw
members are not copied into the repository.

## Result

- Absolute-elevation candidate range: 2.977–23.84 m
- Resampled contour constraints: 2,825
- Structure-point convex-hull coverage: 88.03%
- Strong upper/lower plan pairs: 4 of 9 lower edges
- Contour-to-spot cross-validation: RMSE 4.027 m, MAE 2.794 m, p95 7.215 m
- Passed vertical anchors: 0

Only five of the twelve spot heights lie inside the contour convex hull. The
cross-validation error and the existing terrain error are too large for the
0.3–8.0 m structure dimensions. The provider member metadata also does not
explicitly name the absolute vertical datum. Consequently the audit retains
`mesh_merge_allowed=false` and modifies no scene vertices.

## Reproduction

```powershell
$ngiiFiles = Get-ChildItem -LiteralPath "$env:USERPROFILE\Downloads" -File |
  Where-Object { $_.Name -like '(B010)수치지도_3760824*' }
python -m tools.audit_ngii_vertical_anchors $ngiiFiles.FullName
python -m pytest tests/test_ngii_vertical_anchors.py -q
```

The next gate requires a provider-documented high-resolution DEM or surveyed
structure elevations with explicit vertical datum, plan RMSE no greater than
0.25 m, and vertical uncertainty no greater than 0.10 m.
