# Geospatial data attribution

`scenario_yeouido_2024-10-05.json` carries a machine-readable provenance record
for every source listed in this file, including licence, capture time, validity
window, coordinate reference system, units, confidence grade, uncertainty, and
checksum. This document remains the human-readable companion; where the two
differ, the scenario file is authoritative for grading and the text below is
authoritative for licence wording.

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

`yeouido_detail_osm_2024-10-05.json` is a second query fixed to the same
historical timestamp. It preserves 131 mapped garden, pitch, track,
playground, grass, scrub, forest, and wood polygons used by the detailed
ground-cover pass. Individual tree locations were not present at that
historical date; the runtime therefore places deterministic tree geometry
inside the dated `natural=wood` boundaries without claiming surveyed trunk
positions.

`yeouido_official_facilities.json` contains 121 coordinates retrieved from the
Seoul Future Hangang Headquarters' official Yeouido facility map on
2026-07-29:

- Facility dashboard:
  https://hangang.seoul.go.kr/www/park/dashboard.do?mid=474&opt1=Hzone007
- Facility map data:
  https://hangang.seoul.go.kr/www/facility/mapList.layer

This is an official current inventory, not a timestamped 2024 archive. The
scene uses 54 coordinates for visible toilets, shops, information/safety
centres, drinking fountains, smoking booths, rental kiosks, observation
structures, and playground equipment after suppressing positions already
covered by an OSM building footprint. Facility dimensions and headings remain
procedural approximations until survey drawings or photogrammetry are
available.

Street lamps and benches have no individual point coordinates in either
historical source. Their meshes are explicitly classified as inferred LOD:
they follow dated path centre-lines with deterministic spacing and do not
claim object-by-object identity. The riverside railing follows the geographic
Han River water/land boundary. Its height and post spacing are calibrated
visual dimensions, not a municipal railing survey.

The Seoul Metropolitan Government's October 3, 2024 safety notice independently
confirms the event extent from Mapo Bridge to 63 City, the Wonhyo Bridge
restrictions, and the event-day installation of 90 temporary toilets plus
additional waste facilities:
https://culture.seoul.go.kr/culture/bbs/B0000000/view.do?menuNo=200050&nttId=13326
The notice does not publish coordinates for each temporary unit, so those
objects are not presented as surveyed placements.

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

## Star catalogue

`star_catalogue.npz` is **not committed to this repository**. It is generated by
`python -m tools.import_star_catalogue`, which writes a provenance sidecar
alongside it recording the source URL, SHA-256 checksum, retrieval time, record
count, and magnitude limit.

- Source: Yale Bright Star Catalogue, 5th Revised Edition (Preliminary Version)
- Authors: Hoffleit D., Warren Jr W.H., Astronomical Data Center, NSSDC/ADC
  (1991)
- Retrieved through: https://cdsarc.cds.unistra.fr/ftp/V/50/catalog.gz
- Documentation: https://cdsarc.cds.unistra.fr/ftp/V/50/ReadMe
- Positions and proper motions: J2000 equinox and epoch, FK5 system. The
  catalogue's own note states that its RA proper motion is the
  cos(declination)-projected motion.

**The CDS ReadMe states no explicit licence.** The catalogue is therefore not
redistributed here; whether to commit a derived copy is a decision for the
repository owner. Cite as: Hoffleit D., Warren Jr W.H., The Bright Star
Catalogue, 5th Revised Edition (Preliminary Version), Astronomical Data Center,
NSSDC/ADC (1991), retrieved through the CDS archive, Strasbourg.

Until the importer is run, the renderer uses
`simulator.environmental_optics.procedural_star_catalogue`. Those 3,500
positions and magnitudes are **invented** and carry confidence grade D.

Star colour is derived from the catalogue's Johnson B-V index using the
temperature relation of Ballesteros F.J., "New insights into black bodies",
Europhysics Letters 97(3), 2012.

## Atmospheric optics

Molecular scattering optical depth follows Bodhaine B.A., Wood N.B., Dutton
E.G., Slusser J.R., "On Rayleigh Optical Depth Calculations", Journal of
Atmospheric and Oceanic Technology 16(11), 1999.

Relative optical air mass follows Kasten F., Young A.T., "Revised optical air
mass tables and approximation formula", Applied Optics 28(22), 1989.

Aerosol optical depth uses the Ångström turbidity form. Its exponent and
turbidity coefficient are **documented urban estimates, not measurements of
Seoul on 2024-10-05**, and carry confidence grade C.

Ozone absorption is wired in but inactive: neither the Chappuis-band absorption
cross-sections nor the column ozone amount for the event date has been obtained,
so the term is zero and graded U.

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
