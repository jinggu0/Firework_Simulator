"""Import a real star catalogue, replacing the procedural placeholder.

Fetches the Yale Bright Star Catalogue, 5th Revised Edition (Hoffleit & Warren
1991) from the CDS archive, parses its fixed-width records, and writes a compact
subset with full provenance.

    python -m tools.import_star_catalogue
    python -m tools.import_star_catalogue --magnitude-limit 6.5

The output is **not committed to the repository**. The CDS ReadMe for this
catalogue states no explicit licence, so redistributing a derived copy is a
decision for the repository owner rather than something this tool makes on their
behalf. Until it is run, the renderer falls back to the procedural field and
reports that it is doing so.

Byte offsets follow the catalogue's own byte-by-byte description:
https://cdsarc.cds.unistra.fr/ftp/V/50/ReadMe
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np

from simulator.provenance import ConfidenceGrade
from simulator.starcatalogue import DEFAULT_CATALOGUE_PATH, StarCatalogue

SOURCE_URL = "https://cdsarc.cds.unistra.fr/ftp/V/50/catalog.gz"
README_URL = "https://cdsarc.cds.unistra.fr/ftp/V/50/ReadMe"
SOURCE_ID = "cds-V/50-bsc5"
CITATION = (
    "Hoffleit D., Warren Jr W.H., The Bright Star Catalogue, 5th Revised "
    "Edition (Preliminary Version), Astronomical Data Center, NSSDC/ADC "
    "(1991). Retrieved through the CDS archive, Strasbourg."
)


def _slice_float(line: str, start: int, end: int) -> float:
    """Read a 1-indexed inclusive byte range as a float, blank meaning absent."""

    text = line[start - 1 : end].strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def parse_catalog(text: str, magnitude_limit: float) -> dict[str, np.ndarray]:
    """Parse BSC5 fixed-width records into ICRS arrays."""

    right_ascension: list[float] = []
    declination: list[float] = []
    pm_ra: list[float] = []
    pm_dec: list[float] = []
    magnitude: list[float] = []
    color_index: list[float] = []

    for line in text.splitlines():
        if len(line) < 160:
            continue
        visual = _slice_float(line, 103, 107)
        if not np.isfinite(visual) or visual > magnitude_limit:
            continue
        hours = _slice_float(line, 76, 77)
        minutes = _slice_float(line, 78, 79)
        seconds = _slice_float(line, 80, 83)
        if not (
            np.isfinite(hours) and np.isfinite(minutes) and np.isfinite(seconds)
        ):
            # The 14 retained non-stellar entries carry no J2000 position.
            continue
        degrees = _slice_float(line, 85, 86)
        arcminutes = _slice_float(line, 87, 88)
        arcseconds = _slice_float(line, 89, 90)
        if not np.isfinite(degrees):
            continue
        sign = -1.0 if line[83] == "-" else 1.0

        right_ascension.append((hours + minutes / 60.0 + seconds / 3600.0) * 15.0)
        declination.append(
            sign * (degrees + arcminutes / 60.0 + arcseconds / 3600.0)
        )
        # pmRA in this catalogue is already the cos(dec)-projected motion.
        motion_ra = _slice_float(line, 149, 154)
        motion_dec = _slice_float(line, 155, 160)
        pm_ra.append(0.0 if not np.isfinite(motion_ra) else motion_ra)
        pm_dec.append(0.0 if not np.isfinite(motion_dec) else motion_dec)
        magnitude.append(visual)
        bv = _slice_float(line, 110, 114)
        # A missing colour index defaults to a solar-type value rather than to
        # zero, which would render an unmeasured star as hot blue-white.
        color_index.append(0.65 if not np.isfinite(bv) else bv)

    return {
        "right_ascension_deg": np.asarray(right_ascension, dtype=np.float64),
        "declination_deg": np.asarray(declination, dtype=np.float64),
        "proper_motion_ra_cosdec_arcsec_yr": np.asarray(pm_ra, dtype=np.float32),
        "proper_motion_dec_arcsec_yr": np.asarray(pm_dec, dtype=np.float32),
        "visual_magnitude": np.asarray(magnitude, dtype=np.float32),
        "color_index_bv": np.asarray(color_index, dtype=np.float32),
    }


def download(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "FireworkSimulator/0.2 (local research project)"
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import the Bright Star Catalogue as astrometric data."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_CATALOGUE_PATH)
    parser.add_argument(
        "--magnitude-limit",
        type=float,
        default=6.5,
        help=(
            "Faintest visual magnitude to retain. 6.5 is the catalogue's own "
            "completeness limit and roughly the naked-eye limit at a dark site."
        ),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Parse a local copy instead of downloading.",
    )
    args = parser.parse_args()

    retrieved_at = datetime.now(timezone.utc).isoformat()
    if args.source is not None:
        raw = args.source.read_bytes()
        origin = str(args.source)
    else:
        raw = download(SOURCE_URL)
        origin = SOURCE_URL
    checksum = f"sha256:{hashlib.sha256(raw).hexdigest()}"
    # The archive serves the catalogue gzipped; a local copy may be either.
    if raw[:2] == b"\x1f\x8b":
        raw_text = gzip.decompress(raw)
    else:
        raw_text = raw
    columns = parse_catalog(raw_text.decode("latin-1"), args.magnitude_limit)

    catalogue = StarCatalogue(
        **columns,
        source_id=SOURCE_ID,
        source_url=SOURCE_URL,
        license=(
            "No explicit licence stated in the CDS ReadMe. Cite as: " + CITATION
        ),
        retrieved_at=retrieved_at,
        reference_epoch="J2000.0",
        # Measured astrometry from a published catalogue.
        confidence_grade=ConfidenceGrade.MEASURED,
        notes=(
            f"Retrieved from {origin}. Magnitude limit {args.magnitude_limit}. "
            "Proper motion is the cos(dec)-projected motion in RA, per the "
            "catalogue's own note. Positions are J2000 equinox and epoch."
        ),
    )
    path = catalogue.save(args.output)
    sidecar = path.with_suffix(".provenance.json")
    sidecar.write_text(
        json.dumps(
            {
                **catalogue.summary(),
                "source_checksum": checksum,
                "source_bytes": len(raw),
                "readme_url": README_URL,
                "citation": CITATION,
                "magnitude_limit": args.magnitude_limit,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stars": len(catalogue),
                "output": str(path),
                "output_bytes": path.stat().st_size,
                "provenance": str(sidecar),
                "source_checksum": checksum,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
