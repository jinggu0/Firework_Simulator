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
- Submission attempts: 4 complete forms across the retained and official
  `?tabGb=total` entry paths
- Portal response: HTTP 200 `text/html` page-not-found document from the DEM
  submission endpoint instead of the JSON `orderDownList` expected by the
  portal's own `download.js`
- My page state: public DEM download count remains zero
- Download: blocked by the provider endpoint; no raster package was delivered
- Runtime impact: none

The portal also lists 2014, 2021, 2022, 2023, and 2025 versions. The 2024
version is selected because it matches the target event year without the
post-event temporal override required by the 2025 structure delivery.

The user authorized reuse of the value they entered for subsequent portal
attempts. It was reused only inside the active browser session and is neither
printed nor stored in the repository.

The portal's modal closes when a complete request is submitted. Network tracing
shows that the front end posts to `shbtInsertTritVidoDem.do` and expects JSON,
but the server returns the site's page-not-found HTML with HTTP 200. This is not
a queued request or a large-file-transfer handoff.

Two official, login-free metadata CSVs were downloaded from the public data
portal and locked by checksum. The performance snapshot identifies six 2009
airborne-LiDAR, 1 m ASCII candidates overlapping the full project bbox:
`서울087`, `서울088`, `서울089`, `서울097`, `서울098`, and `서울099`.
They use orthometric heights relative to Incheon mean sea level. Their
published accuracy fields are blank, and the raster files themselves were not
delivered, so they cannot satisfy the 0.10 m vertical uncertainty gate. See
`ngii_dem_metadata_report.json` for the machine-readable audit.

## Intake gate

After login, retain the downloaded package outside version control and record:

1. package name, byte count, and SHA-256;
2. projected CRS from the delivered metadata;
3. explicit vertical datum;
4. grid spacing and NoData value; and
5. provider terms permitting local derived use.

Do not merge the DEM or structure meshes until plan registration RMSE is at
most 0.25 m and vertical uncertainty is at most 0.10 m. The blocked state is
machine-readable in `assets/yeouido_ngii_dem_request.json`.
