from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import gzip
import io
import json
from pathlib import Path
import urllib.request

STATION_ID = "47108"
YEAR = 2024
OUTPUT = Path("assets/yeouido_2024-10-05_environment.json")
KST = timezone(timedelta(hours=9))


def download_rows() -> list[dict[str, str]]:
    url = f"https://data.meteostat.net/hourly/{YEAR}/{STATION_ID}.csv.gz"
    request = urllib.request.Request(
        url, headers={"User-Agent": "FireworkSimulator/0.1 (local research project)"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        text = gzip.decompress(response.read()).decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def main() -> None:
    records = []
    for row in download_rows():
        utc = datetime(
            int(row["year"]), int(row["month"]), int(row["day"]), int(row["hour"]),
            tzinfo=timezone.utc,
        )
        local = utc.astimezone(KST)
        if local.date().isoformat() != "2024-10-05" or not 18 <= local.hour <= 22:
            continue
        records.append(
            {
                "time": local.isoformat(timespec="minutes"),
                "temperature_c": float(row["temp"]),
                "relative_humidity_percent": float(row["rhum"]),
                "pressure_hpa": float(row["pres"]),
                "wind_speed_mps": float(row["wspd"]) / 3.6,
                "wind_from_direction_deg": float(row["wdir"]),
                "cloud_cover_oktas": float(row["cldc"]),
                "field_sources": {
                    key: row[f"{key}_source"]
                    for key in ("temp", "rhum", "pres", "wspd", "wdir", "cldc")
                },
            }
        )
    if len(records) != 5:
        raise RuntimeError(f"expected five event records, received {len(records)}")
    document = {
        "event": {
            "name": "2024 Seoul International Fireworks Festival",
            "location": "Yeouido Hangang Park",
            "show_start": "2024-10-05T19:20:00+09:00",
            "show_end": "2024-10-05T20:30:00+09:00",
        },
        "provenance": {
            "weather_source": "Meteostat station 47108 (Seoul), ISD Lite observations",
            "station_id": STATION_ID,
            "retrieved": datetime.now(timezone.utc).date().isoformat(),
            "time_basis": "Source UTC converted to Asia/Seoul (UTC+09:00)",
        },
        "records": records,
    }
    OUTPUT.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved {OUTPUT} with {len(records)} hourly observations")


if __name__ == "__main__":
    main()
