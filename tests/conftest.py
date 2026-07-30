"""Shared fixtures.

Tests must not depend on the network or on a running MLflow server, so the raw
Open-Meteo payload is synthesised here in the same shape ``collect.py`` writes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALID_PAYLOAD = {
    "pm2_5": 120.0,
    "pm10": 260.0,
    "temperature_2m": 34.0,
    "relative_humidity_2m": 25.0,
    "wind_speed_10m": 12.0,
    "hour": 14,
    "day_of_week": 2,
    "pm2_5_lag_1": 118.0,
    "pm2_5_lag_3": 110.0,
    "pm2_5_rolling_mean_6": 115.0,
}


def make_raw_payload(hours: int = 500, seed: int = 0) -> dict:
    """Build a raw payload with the same structure as the real API response."""
    rng = np.random.default_rng(seed)
    stamps = pd.date_range("2026-01-01", periods=hours, freq="h")
    times = stamps.strftime("%Y-%m-%dT%H:%M").tolist()
    # A drifting sine keeps the target from collapsing to a single class.
    pm2_5 = 120 + 60 * np.sin(np.arange(hours) / 9.0) + rng.normal(0, 12, hours)
    pm2_5 = np.clip(pm2_5, 1, None)

    return {
        "metadata": {
            "latitude": 24.7136,
            "longitude": 46.6753,
            "start_date": "2026-01-01",
            "end_date": "2026-01-21",
        },
        "air_quality": {
            "hourly": {
                "time": times,
                "pm2_5": pm2_5.round(2).tolist(),
                "pm10": (pm2_5 * 2.1).round(2).tolist(),
            }
        },
        "weather": {
            "hourly": {
                "time": times,
                "temperature_2m": rng.uniform(18, 45, hours).round(2).tolist(),
                "relative_humidity_2m": rng.uniform(5, 60, hours).round(2).tolist(),
                "wind_speed_10m": rng.uniform(0, 30, hours).round(2).tolist(),
            }
        },
    }


@pytest.fixture
def raw_payload() -> dict:
    return make_raw_payload()


@pytest.fixture
def raw_path(tmp_path: Path, raw_payload: dict) -> Path:
    path = tmp_path / "air_quality.json"
    path.write_text(json.dumps(raw_payload), encoding="utf-8")
    return path


@pytest.fixture
def valid_payload() -> dict:
    return dict(VALID_PAYLOAD)
