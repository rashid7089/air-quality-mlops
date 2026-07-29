from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import requests

LATITUDE = 24.7136
LONGITUDE = 46.6753
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_json(url: str, params: dict[str, object]) -> dict:
    # Fail fast while still allowing a short retry window.
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if "hourly" not in payload:
        raise ValueError("The API response does not contain hourly data.")
    return payload


def main() -> None:
    end_date = date.today() - timedelta(days=5)
    start_date = end_date - timedelta(days=90)
    common = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timezone": "Asia/Riyadh",
    }

    air = fetch_json(
        AIR_URL,
        common | {"hourly": "pm2_5,pm10"},
    )
    weather = fetch_json(
        WEATHER_URL,
        common | {
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m"
        },
    )

    output = {
        "metadata": {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
        "air_quality": air,
        "weather": weather,
    }
    path = Path("data/raw/air_quality.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved raw data to {path}")


if __name__ == "__main__":
    main()

