"""Unit tests for the feature pipeline, focused on leakage prevention."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from air_quality.features import (
    FEATURES,
    POLLUTION_THRESHOLD,
    TARGET,
    build_dataset,
)

from .conftest import make_raw_payload


def build(tmp_path: Path, raw_path: Path, **kwargs) -> pd.DataFrame:
    return build_dataset(str(raw_path), str(tmp_path / "model_table.parquet"), **kwargs)


def test_dataset_is_written_and_non_empty(tmp_path: Path, raw_path: Path) -> None:
    output = tmp_path / "model_table.parquet"
    frame = build_dataset(str(raw_path), str(output))

    assert output.exists()
    assert not frame.empty
    assert pd.read_parquet(output).shape == frame.shape


def test_all_expected_feature_columns_exist(tmp_path: Path, raw_path: Path) -> None:
    frame = build(tmp_path, raw_path)
    assert set(FEATURES + [TARGET]).issubset(frame.columns)


def test_no_missing_values_in_features(tmp_path: Path, raw_path: Path) -> None:
    frame = build(tmp_path, raw_path)
    assert frame[FEATURES + [TARGET]].isna().sum().sum() == 0


def test_rows_are_chronological_and_unique(tmp_path: Path, raw_path: Path) -> None:
    frame = build(tmp_path, raw_path)
    assert frame["time"].is_monotonic_increasing
    assert not frame["time"].duplicated().any()


def test_target_is_next_hour_not_current_hour(tmp_path: Path, raw_path: Path) -> None:
    """The label must describe hour t+1, never hour t."""
    frame = build(tmp_path, raw_path).reset_index(drop=True)

    # Row i's target must match row i+1's pm2_5 crossing the threshold.
    expected = (frame["pm2_5"].shift(-1) > POLLUTION_THRESHOLD).astype("Int64")
    comparable = expected.notna()
    assert (frame.loc[comparable, TARGET] == expected[comparable]).all()


def test_lag_features_are_historical_only(tmp_path: Path, raw_path: Path) -> None:
    """lag_1 and lag_3 must equal past pm2_5 values, never future ones."""
    frame = build(tmp_path, raw_path).reset_index(drop=True)

    assert (frame["pm2_5_lag_1"].iloc[1:].values == frame["pm2_5"].iloc[:-1].values).all()
    assert (frame["pm2_5_lag_3"].iloc[3:].values == frame["pm2_5"].iloc[:-3].values).all()


def test_rolling_mean_excludes_the_current_hour(tmp_path: Path, raw_path: Path) -> None:
    """The 6-hour mean is shifted before the window, so hour t is not included."""
    frame = build(tmp_path, raw_path).reset_index(drop=True)

    manual = frame["pm2_5"].shift(1).rolling(6).mean()
    aligned = manual.notna()
    assert frame.loc[aligned, "pm2_5_rolling_mean_6"].round(6).equals(
        manual[aligned].round(6)
    )

    # A window that wrongly included hour t would differ from the shifted one.
    leaky = frame["pm2_5"].rolling(6).mean()
    assert not frame["pm2_5_rolling_mean_6"].round(6).equals(leaky.round(6))


def test_future_pm2_5_is_not_a_feature() -> None:
    """No forward-looking column may appear in the model's feature list."""
    assert not any("next" in name or "lead" in name for name in FEATURES)
    assert TARGET not in FEATURES


def test_threshold_controls_the_label(tmp_path: Path, raw_path: Path) -> None:
    low = build(tmp_path, raw_path, threshold=20.0)[TARGET].mean()
    high = build(tmp_path, raw_path, threshold=250.0)[TARGET].mean()
    assert low > high


def test_default_threshold_keeps_both_classes(tmp_path: Path, raw_path: Path) -> None:
    """A degenerate single-class target makes F1 meaningless."""
    rate = build(tmp_path, raw_path)[TARGET].mean()
    assert 0.05 < rate < 0.95


def test_time_features_match_the_timestamp(tmp_path: Path, raw_path: Path) -> None:
    frame = build(tmp_path, raw_path)
    assert (frame["hour"] == frame["time"].dt.hour).all()
    assert (frame["day_of_week"] == frame["time"].dt.dayofweek).all()
    assert frame["hour"].between(0, 23).all()
    assert frame["day_of_week"].between(0, 6).all()


def test_duplicate_timestamps_are_dropped(tmp_path: Path) -> None:
    payload = make_raw_payload(hours=200)
    for section in ("air_quality", "weather"):
        hourly = payload[section]["hourly"]
        for key in hourly:
            hourly[key] = hourly[key] + hourly[key][:50]

    raw = tmp_path / "dupes.json"
    raw.write_text(json.dumps(payload), encoding="utf-8")

    frame = build_dataset(str(raw), str(tmp_path / "out.parquet"))
    assert not frame["time"].duplicated().any()


def test_empty_input_raises(tmp_path: Path) -> None:
    payload = make_raw_payload(hours=2)
    raw = tmp_path / "tiny.json"
    raw.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        build_dataset(str(raw), str(tmp_path / "out.parquet"))
