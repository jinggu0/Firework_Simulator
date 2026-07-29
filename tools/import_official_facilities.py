from __future__ import annotations

import argparse
from datetime import date
import html
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request


SOURCE_URL = "https://hangang.seoul.go.kr/www/facility/mapList.layer"
PARK_CODE = "Hzone007"


def download_facilities() -> list[dict[str, object]]:
    request = urllib.request.Request(
        SOURCE_URL,
        data=urllib.parse.urlencode(
            {
                "opt12": PARK_CODE,
                "pageNo": 1,
                "perPageCnt": 250,
            }
        ).encode(),
        headers={
            "User-Agent": "FireworkSimulator/0.1 (local research project)"
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        document = response.read().decode("utf-8")
    facilities: list[dict[str, object]] = []
    for item in re.findall(
        r'<li class="fac-listitem"[\s\S]*?</li>', document
    ):
        code = re.search(r"data-key='([^']+)'", item)
        name = re.search(
            r'<strong class="fac-name">([\s\S]*?)</strong>', item
        )
        coordinates = re.search(
            r"clickMarkerSetting\('([^']*)','([^']*)','", item
        )
        if not (code and name and coordinates):
            continue
        clean_name = html.unescape(re.sub(r"<[^>]+>", "", name.group(1)))
        facilities.append(
            {
                "code": code.group(1),
                "name": clean_name.strip(),
                "latitude": float(coordinates.group(1)),
                "longitude": float(coordinates.group(2)),
            }
        )
    return facilities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/yeouido_official_facilities.json"),
    )
    args = parser.parse_args()
    facilities = download_facilities()
    payload = {
        "source": SOURCE_URL,
        "park_code": PARK_CODE,
        "retrieved_date": date.today().isoformat(),
        "historical_status": (
            "Current official facility positions; not a dated 2024 inventory."
        ),
        "facilities": facilities,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved {len(facilities)} facilities to {args.output}")


if __name__ == "__main__":
    main()
