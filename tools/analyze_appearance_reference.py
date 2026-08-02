"""Measure local event photographs without adding them to the repository.

Example::

    python -m tools.analyze_appearance_reference \
      --image news1_night_wide=C:\\tmp\\yeouido_news1_night.jpg \
      --image newsis_night_river=C:\\tmp\\yeouido_newsis_night.jpg

The JSON file records source pages, normalized crops, and usage restrictions.
This tool reports simple display-referred statistics for repeatable comparison;
it deliberately does not call JPEG values physical radiance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


DEFAULT_REFERENCE = (
    Path(__file__).resolve().parent.parent
    / "assets"
    / "yeouido_2024-10-05_appearance_reference.json"
)
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def _parse_image(value: str) -> tuple[str, Path]:
    key, separator, path = value.partition("=")
    if not separator or not key or not path:
        raise argparse.ArgumentTypeError("expected SOURCE_KEY=IMAGE_PATH")
    return key, Path(path)


def crop_statistics(image: np.ndarray, bounds: list[float]) -> dict[str, object]:
    height, width = image.shape[:2]
    x0, y0, x1, y1 = bounds
    left, right = round(x0 * width), round(x1 * width)
    top, bottom = round(y0 * height), round(y1 * height)
    pixels = image[top:bottom, left:right].reshape(-1, 3).astype(np.float64)
    if not len(pixels):
        raise ValueError(f"empty crop {bounds} for {width}x{height} image")
    luminance = pixels @ LUMA
    return {
        "pixel_bounds": [left, top, right, bottom],
        "median_srgb_8bit": np.median(pixels, axis=0).round(2).tolist(),
        "mean_srgb_8bit": pixels.mean(axis=0).round(2).tolist(),
        "luminance_p05_p50_p95": np.percentile(
            luminance, [5, 50, 95]
        ).round(2).tolist(),
    }


def analyze(reference_path: Path, images: list[tuple[str, Path]]) -> dict:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    output: dict[str, object] = {
        "reference": str(reference_path),
        "warning": "sRGB/JPEG statistics are display-referred, not radiance",
        "sources": {},
    }
    for key, path in images:
        source = reference["sources"].get(key)
        if source is None:
            raise KeyError(f"unknown source key {key!r}")
        if not path.exists():
            raise FileNotFoundError(path)
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        crops = {
            name: crop_statistics(image, bounds)
            for name, bounds in source.get("normalized_crops", {}).items()
        }
        output["sources"][key] = {
            "path": str(path),
            "size": [int(image.shape[1]), int(image.shape[0])],
            "crops": crops,
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure local 2024 Yeouido appearance-reference images."
    )
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--image", action="append", type=_parse_image, default=[],
        metavar="SOURCE_KEY=IMAGE_PATH",
    )
    args = parser.parse_args()
    if not args.image:
        parser.error("at least one --image is required")
    print(json.dumps(analyze(args.reference, args.image), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
