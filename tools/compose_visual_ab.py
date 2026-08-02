"""Compose labelled before/after/difference PNG evidence from two SDR frames."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance


def compose(before_path: Path, after_path: Path, output_path: Path) -> Path:
    with Image.open(before_path) as image:
        before = image.convert("RGB")
    with Image.open(after_path) as image:
        after = image.convert("RGB")
    if before.size != after.size:
        raise ValueError("before and after frames must have the same dimensions")
    display_width = min(before.width, 640)
    display_height = round(before.height * display_width / before.width)
    size = (display_width, display_height)
    before_display = before.resize(size, Image.Resampling.LANCZOS)
    after_display = after.resize(size, Image.Resampling.LANCZOS)
    difference = ImageEnhance.Contrast(
        ImageEnhance.Brightness(
            ImageChops.difference(before_display, after_display)
        ).enhance(4.0)
    ).enhance(1.8)
    header_height = 34
    canvas = Image.new(
        "RGB", (display_width * 3, display_height + header_height), (15, 17, 20)
    )
    for index, (label, panel) in enumerate(
        (("BEFORE", before_display), ("AFTER", after_display), ("DIFF x4", difference))
    ):
        x = index * display_width
        canvas.paste(panel, (x, header_height))
        ImageDraw.Draw(canvas).text((x + 12, 10), label, fill=(235, 237, 240))
    output_path = Path(output_path).with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)
    before_rgb = np.asarray(before, dtype=np.int16)
    after_rgb = np.asarray(after, dtype=np.int16)
    changed = np.any(before_rgb != after_rgb, axis=2)
    print(
        f"changed_pixels={int(changed.sum())} "
        f"changed_fraction={float(changed.mean()):.9f} output={output_path}"
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        compose(arguments.before, arguments.after, arguments.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
