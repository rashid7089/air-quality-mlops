from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_PATH = (
    PROJECT_ROOT / "data" / "processed" / "model_table.parquet"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "monitoring" / "predictions.csv"
)

REFERENCE_PATH = (
    PROJECT_ROOT / "data" / "monitoring" / "reference.csv"
)

REPORT_PATH = (
    PROJECT_ROOT / "reports" / "monitoring.html"
)


# Features used by the trained model
FEATURES = [
    "pm2_5",
    "pm10",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "hour",
    "day_of_week",
    "pm2_5_lag_1",
    "pm2_5_lag_3",
    "pm2_5_rolling_mean_6",
]


def create_reference_data() -> pd.DataFrame:
    """Create reference data from the final 15% of processed data."""

    processed_data = pd.read_parquet(PROCESSED_PATH)

    test_start = int(len(processed_data) * 0.85)

    reference_data = processed_data.iloc[test_start:][FEATURES].copy()

    REFERENCE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_data.to_csv(
        REFERENCE_PATH,
        index=False,
    )

    return reference_data


def main() -> None:
    """Generate the Evidently monitoring report."""

    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            "Processed data was not found: "
            f"{PROCESSED_PATH}"
        )

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "No prediction logs were found. "
            "Make predictions using Streamlit first."
        )

    # Create or recreate the reference batch
    reference_data = create_reference_data()

    # Read current production prediction inputs
    prediction_logs = pd.read_csv(PREDICTIONS_PATH)

    current_data = prediction_logs[FEATURES].copy()

    if current_data.empty:
        raise ValueError(
            "The prediction log is empty. "
            "Make at least one prediction first."
        )

    if len(current_data) < 5:
        print(
            "Warning: the current dataset contains fewer than "
            "5 rows. The report will run, but drift results "
            "will be more meaningful with additional predictions."
        )

    # Create the Evidently report
    report = Report(
        [
            DataSummaryPreset(),
            DataDriftPreset(),
        ]
    )

    snapshot = report.run(
        current_data=current_data,
        reference_data=reference_data,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot.save_html(str(REPORT_PATH))

    print(f"Reference rows: {len(reference_data)}")
    print(f"Current rows: {len(current_data)}")
    print(f"Saved monitoring report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()