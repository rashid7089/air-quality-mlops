"""Generate the Evidently drift report from reference and current batches."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from air_quality.features import FEATURES

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

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


def load_reference_data() -> pd.DataFrame:
    """Load the reference batch, rebuilding it from processed data if absent.

    ``train.py`` writes this file from the held-out test slice. The fallback
    below reproduces that same final-15% slice so the report can still be
    generated from a processed table alone.
    """
    if REFERENCE_PATH.exists():
        reference_data = pd.read_csv(REFERENCE_PATH)
        missing = [column for column in FEATURES if column not in reference_data]
        if not missing:
            return reference_data[FEATURES].copy()
        logger.warning("Reference file is missing %s; rebuilding it.", missing)

    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Neither a reference batch nor processed data was found: {PROCESSED_PATH}"
        )

    processed_data = pd.read_parquet(PROCESSED_PATH)
    test_start = int(len(processed_data) * 0.85)
    reference_data = processed_data.iloc[test_start:][FEATURES].copy()

    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    reference_data.to_csv(REFERENCE_PATH, index=False)

    return reference_data


def main() -> None:
    """Generate the Evidently monitoring report."""

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "No prediction logs were found. Submit at least one prediction "
            "through FastAPI or Streamlit first."
        )

    reference_data = load_reference_data()

    # Read current production prediction inputs
    prediction_logs = pd.read_csv(PREDICTIONS_PATH)

    missing = [column for column in FEATURES if column not in prediction_logs]
    if missing:
        raise ValueError(f"The prediction log is missing feature columns: {missing}")

    current_data = prediction_logs[FEATURES].copy()

    if current_data.empty:
        raise ValueError(
            "The prediction log is empty. Make at least one prediction first."
        )

    if len(current_data) < 5:
        logger.warning(
            "The current batch has only %d rows. The report will still be "
            "generated, but drift results need more predictions to be meaningful.",
            len(current_data),
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

    logger.info("Reference rows: %d", len(reference_data))
    logger.info("Current rows: %d", len(current_data))
    logger.info("Saved monitoring report to: %s", REPORT_PATH)


if __name__ == "__main__":
    main()