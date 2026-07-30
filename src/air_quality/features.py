from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

FEATURES = [
    "pm2_5", "pm10", "temperature_2m", "relative_humidity_2m",
    "wind_speed_10m", "hour", "day_of_week", "pm2_5_lag_1",
    "pm2_5_lag_3", "pm2_5_rolling_mean_6",
]

TARGET = "high_pollution_next_hour"

# Riyadh PM2.5 is dust-dominated (median ~123 ug/m3), so the 35 ug/m3 value used
# elsewhere labels 99% of hours as positive and makes F1 meaningless. 100 ug/m3
# sits in the EPA "Unhealthy" band and keeps every chronological split balanced.
POLLUTION_THRESHOLD = 100.0


def build_dataset(
    raw_path: str,
    output_path: str,
    threshold: float = POLLUTION_THRESHOLD,
) -> pd.DataFrame:
    """Build the modelling table from the immutable raw payload.

    The target is next-hour PM2.5 above ``threshold``. Lag and rolling features
    are historical only: the rolling mean is shifted before the window is
    applied, so no future observation can leak into a feature.
    """
    payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    air = pd.DataFrame(payload["air_quality"]["hourly"])
    weather = pd.DataFrame(payload["weather"]["hourly"])

    air["time"] = pd.to_datetime(air["time"])
    weather["time"] = pd.to_datetime(weather["time"])
    df = air.merge(weather, on="time", how="inner").sort_values("time")
    df = df.drop_duplicates("time")

    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["pm2_5_lag_1"] = df["pm2_5"].shift(1)
    df["pm2_5_lag_3"] = df["pm2_5"].shift(3)
    df["pm2_5_rolling_mean_6"] = df["pm2_5"].shift(1).rolling(6).mean()
    df[TARGET] = (df["pm2_5"].shift(-1) > threshold).astype("Int64")

    df = df.dropna(subset=FEATURES + [TARGET])
    df[TARGET] = df[TARGET].astype(int)

    if df.empty:
        raise ValueError("The feature table is empty after dropping missing rows.")
    if df["time"].duplicated().any():
        raise ValueError("Duplicate timestamps remain in the feature table.")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df


if __name__ == "__main__":
    frame = build_dataset(
        "data/raw/air_quality.json",
        "data/processed/model_table.parquet",
    )
    print(frame.shape)
    print(frame[TARGET].value_counts(normalize=True))

