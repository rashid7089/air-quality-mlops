from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

FEATURES = [
    "pm2_5", "pm10", "temperature_2m", "relative_humidity_2m",
    "wind_speed_10m", "hour", "day_of_week", "pm2_5_lag_1",
    "pm2_5_lag_3", "pm2_5_rolling_mean_6",
]


def build_dataset(raw_path: str, output_path: str, threshold: float = 35.0) -> pd.DataFrame:
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
    df["high_pollution_next_hour"] = (df["pm2_5"].shift(-1) > threshold).astype("Int64")

    df = df.dropna(subset=FEATURES + ["high_pollution_next_hour"])
    df["high_pollution_next_hour"] = df["high_pollution_next_hour"].astype(int)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df


if __name__ == "__main__":
    frame = build_dataset(
        "data/raw/air_quality.json",
        "data/processed/model_table.parquet",
    )
    print(frame.shape)
    print(frame["high_pollution_next_hour"].value_counts(normalize=True))

