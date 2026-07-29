# Geospatial data attribution

`yeouido_scene.npz` is derived from OpenStreetMap data downloaded through the
Overpass API.

Copyright © OpenStreetMap contributors. The database is made available under
the Open Data Commons Open Database License (ODbL).

- Source: https://www.openstreetmap.org/
- Licence: https://www.openstreetmap.org/copyright
- Extract bounding box: 37.515–37.545 N, 126.910–126.960 E
- Local scene origin: 37.529 N, 126.935 E
- Historical data timestamp: 2024-10-05 10:20 UTC (19:20 KST)
- Snapshot retrieved: 2026-07-29 through an Overpass attic-capable endpoint
- Han River water multipolygon: OpenStreetMap relation `152336`

The shipped snapshot contains the OSM state at show start rather than the
current map state. It includes building footprints and parts, tagged heights,
bridge and road ways, parks, grass, forest, and the Han River multipolygon.
OSM remains a community dataset: missing heights use documented fallbacks and
do not establish survey-grade equality with every October 5, 2024 structure.

## Landmark facade references

Historical OSM names, heights, uses, material tags, and colour tags select the
runtime facade family. The following references constrain distinctive
landmark treatments:

- IFC Seoul describes its three office towers and Conrad hotel as a
  crystalline glass composition with reflected light and shadow:
  https://www.ifcseoul.com/en/BD_02_00.asp
- The Parc.1 architectural report describes the externally expressed red
  structural columns and their relationship to the Han River and Yeouido
  Park:
  https://r.yna.co.kr/www/imazine/202004/232.pdf
- The historical OSM snapshot identifies 63 City as a 252 m glass tower and
  tags its building colour as gold.

These sources calibrate procedural material families; no copyrighted facade
photographs are packaged as textures. Window occupancy is deterministic
simulation data and is not a record of which rooms were illuminated during
the 2024 event.

## Elevation

Terrain elevation is derived from the Mapzen Terrain Tiles dataset hosted in
the AWS Registry of Open Data.

- Dataset: https://registry.opendata.aws/terrain-tiles/
- Tile format: Terrarium PNG, Web Mercator
- Zoom level: 12
- Snapshot accessed: 2026-07-29
- Dataset-specific source licences:
  https://github.com/tilezen/joerd/blob/master/docs/attribution.md

The AWS registry describes this as a global bare-earth elevation dataset.
Higher zoom tiles can be resampled from coarser sources, so zoom level must not
be interpreted as source survey resolution.

## Historical weather

The event environment snapshot is derived from hourly Seoul station records
distributed by Meteostat.

- Station: Seoul `47108`
- Event observations: October 5, 2024, 18:00–22:00 KST
- Endpoint: `https://data.meteostat.net/hourly/2024/47108.csv.gz`
- Documentation: https://dev.meteostat.net/data/timeseries/hourly
- Parameter units: https://dev.meteostat.net/parameters
- Snapshot accessed: 2026-07-29

Temperature, humidity, pressure, wind, and cloud-cover values used here report
`isd_lite` as their field source. Meteostat wind speed is distributed in km/h
and is converted to m/s by the importer.

The 19:20–20:30 show interval is taken from the Seoul Metropolitan Government
culture listing:
https://culture.seoul.go.kr/culture/culture/cultureEvent/view.do?cultcode=149765.&menuNo=200010

The same official listing and event map identify the fireworks presentation
area around Wonhyo Bridge. This reference constrains scene placement; it does
not provide individual barge coordinates or surveyed temporary structures.

## Astronomy

Topocentric Sun and Moon positions, lunar phase, distance, and apparent
magnitude are calculated with Astronomy Engine:

- Project: https://github.com/cosinekitty/astronomy
- Python package: `astronomy-engine`
- Licence: MIT
- Observer: 37.529 N, 126.935 E, 5 m

Moonlight uses Astronomy Engine apparent magnitude and a documented 0.25 lux
near-zenith full-moon reference. Twilight illuminance is a bounded piecewise
log interpolation across the civil, nautical, and astronomical thresholds; it
is a rendering calibration model rather than a full atmospheric
radiative-transfer solution.

## Firework photometry references

The initial star-expansion and drag model is informed by the observational
description in:

- "Fireworks on Weather Radar and Camera", Bulletin of the American
  Meteorological Society, 2020:
  https://journals.ametsoc.org/view/journals/bams/101/2/bams-d-18-0248.1.xml
- "Burning characteristics of fireworks stars", Science and Technology of
  Energetic Materials 67(1), 2006:
  https://www.jes.or.jp/mag/stem/Vol.67/documents/Vol.67%2CNo.1%2Cp.43-47.pdf

These references support spherical post-burst expansion, quadratic drag, and
finite measured star burn time. Current luminous-power values remain
calibration parameters rather than laboratory measurements of the exact 2024
shell compositions.

## Fluid-solver design reference

The post-blast smoke solver design was informed by CubbyFlow's open-source
grid-fluid architecture, including staggered grids, semi-Lagrangian advection,
pressure projection, smoke buoyancy, and fixed/adaptive substeps:

- Project: https://github.com/utilForever/CubbyFlow
- Licence: MIT
- Reference consulted: 2026-07-29

The simulator contains an independent Python/NumPy implementation rather than
copied CubbyFlow source. CubbyFlow is used as an architectural and numerical
reference.
