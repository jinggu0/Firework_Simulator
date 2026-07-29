# Geospatial data attribution

`yeouido_scene.npz` is derived from OpenStreetMap data downloaded through the
Overpass API.

Copyright © OpenStreetMap contributors. The database is made available under
the Open Data Commons Open Database License (ODbL).

- Source: https://www.openstreetmap.org/
- Licence: https://www.openstreetmap.org/copyright
- Extract bounding box: 37.515–37.545 N, 126.910–126.960 E
- Local scene origin: 37.529 N, 126.935 E
- Snapshot retrieved: 2026-07-29
- Han River water multipolygon: OpenStreetMap relation `152336`

This snapshot is an initial geometry reference. It does not establish that
every feature or height matches October 5, 2024. Production validation must
compare it with dated Seoul/VWorld data and event footage.

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
