"""Collect hourly Riyadh air-quality and weather data from Open-Meteo.

The raw payload is written verbatim, alongside collection metadata, so the
downstream pipeline is always reproducible from an immutable input.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LATITUDE = 24.7136
LONGITUDE = 46.6753
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

# The specification requires at least 60 days of hourly observations.
MINIMUM_DAYS = 60
COLLECTION_DAYS = 90


def fetch_json(url: str, params: dict[str, object]) -> dict:
    """GET a JSON payload and confirm it carries an hourly block."""
    # Fail fast while still allowing a short retry window.
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if "hourly" not in payload:
        raise ValueError("The API response does not contain hourly data.")
    return payload


def validate_hourly(payload: dict, label: str) -> int:
    """Check the hourly block covers enough days without duplicate timestamps."""
    hourly = payload["hourly"]
    if "time" not in hourly:
        raise ValueError(f"{label}: the hourly block has no 'time' series.")

    timestamps = hourly["time"]
    if len(timestamps) != len(set(timestamps)):
        raise ValueError(f"{label}: the response contains duplicate timestamps.")

    days = len(timestamps) / 24
    if days < MINIMUM_DAYS:
        raise ValueError(
            f"{label}: only {days:.1f} days returned, "
            f"but at least {MINIMUM_DAYS} are required."
        )

    for name, series in hourly.items():
        if len(series) != len(timestamps):
            raise ValueError(f"{label}: series '{name}' has a mismatched length.")

    return len(timestamps)


def main() -> None:
    end_date = date.today() - timedelta(days=5)
    start_date = end_date - timedelta(days=COLLECTION_DAYS)
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

    air_hours = validate_hourly(air, "air quality")
    weather_hours = validate_hourly(weather, "weather")

    output = {
        "metadata": {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "collected_at": date.today().isoformat(),
            "air_quality_hours": air_hours,
            "weather_hours": weather_hours,
            "timezone": "Asia/Riyadh",
        },
        "air_quality": air,
        "weather": weather,
    }
    path = Path("data/raw/air_quality.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info(
        "Saved raw data to %s (%d air-quality hours, %.1f days)",
        path,
        air_hours,
        air_hours / 24,
    )


if __name__ == "__main__":
    main()

