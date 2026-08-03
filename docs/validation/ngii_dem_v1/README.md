# V1-11c authenticated NGII DEM intake

The NGII portal catalogue exposes a public DEM candidate matching the event
year and area: `2024 서울 37608`.

## Current state

- Product: 공개DEM
- Sheet: Seoul `37608`
- Production year: 2024
- Portal selection: complete
- Authentication: confirmed
- Application: purpose, detail, and terms prepared
- Download: waiting for the user to enter the required date of birth
- Runtime impact: none

The portal also lists 2014, 2021, 2022, 2023, and 2025 versions. The 2024
version is selected because it matches the target event year without the
post-event temporal override required by the 2025 structure delivery.

The date of birth is intentionally neither inferred nor stored in the
repository. The prepared Chrome form is retained for direct user entry.

## Intake gate

After login, retain the downloaded package outside version control and record:

1. package name, byte count, and SHA-256;
2. projected CRS from the delivered metadata;
3. explicit vertical datum;
4. grid spacing and NoData value; and
5. provider terms permitting local derived use.

Do not merge the DEM or structure meshes until plan registration RMSE is at
most 0.25 m and vertical uncertainty is at most 0.10 m. The pending state is
machine-readable in `assets/yeouido_ngii_dem_request.json`.
