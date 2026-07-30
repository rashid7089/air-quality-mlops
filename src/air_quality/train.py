"""Train, track, and select the next-hour pollution classifier.

Splits are chronological (70/15/15). Model choice and the decision threshold are
both made on the validation slice; the test slice is touched exactly once, to
report the final held-out numbers.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from air_quality.features import FEATURES, POLLUTION_THRESHOLD, TARGET

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "model_table.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
REFERENCE_PATH = PROJECT_ROOT / "data" / "monitoring" / "reference.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "riyadh-air-quality")

# Thresholds scanned on the validation slice only.
CANDIDATE_THRESHOLDS = np.round(np.arange(0.20, 0.81, 0.01), 2)


def split_by_time(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically into train (70%), validation (15%), test (15%)."""
    n = len(df)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)
    return df.iloc[:train_end], df.iloc[train_end:valid_end], df.iloc[valid_end:]


def make_pipeline(model: Any) -> Pipeline:
    """Wrap a classifier in the median-impute + scale preprocessing pipeline."""
    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                FEATURES,
            )
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def score(
    y_true: pd.Series, probability: np.ndarray, threshold: float
) -> dict[str, float]:
    """Compute the metrics the specification asks for at a given threshold."""
    prediction = (probability >= threshold).astype(int)
    return {
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
    }


def choose_threshold(
    y_true: pd.Series, probability: np.ndarray
) -> tuple[float, float]:
    """Pick the decision threshold maximising F1 on the validation slice."""
    best_threshold, best_f1 = 0.50, -1.0
    for candidate in CANDIDATE_THRESHOLDS:
        value = f1_score(
            y_true, (probability >= candidate).astype(int), zero_division=0
        )
        if value > best_f1:
            best_threshold, best_f1 = float(candidate), float(value)
    return best_threshold, best_f1


def build_candidates() -> dict[str, Any]:
    """The baseline plus the two trained models required by the specification."""
    return {
        "baseline_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }


def main() -> None:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {PROCESSED_PATH}. "
            "Run 'uv run python -m air_quality.features' first."
        )

    df = pd.read_parquet(PROCESSED_PATH)
    train, valid, test = split_by_time(df)
    logger.info(
        "Chronological split: train=%d valid=%d test=%d "
        "(positive rate %.3f/%.3f/%.3f)",
        len(train),
        len(valid),
        len(test),
        train[TARGET].mean(),
        valid[TARGET].mean(),
        test[TARGET].mean(),
    )

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        logger.info("Logging runs to MLflow at %s", MLFLOW_TRACKING_URI)
    except Exception as error:  # noqa: BLE001 - training must not depend on MLflow
        logger.warning("MLflow unavailable (%s); continuing without tracking.", error)

    best: dict[str, Any] | None = None

    for name, model in build_candidates().items():
        pipeline = make_pipeline(model)
        pipeline.fit(train[FEATURES], train[TARGET])
        valid_probability = pipeline.predict_proba(valid[FEATURES])[:, 1]

        # Threshold is tuned on validation data only; the test slice is untouched.
        threshold, _ = choose_threshold(valid[TARGET], valid_probability)
        metrics = score(valid[TARGET], valid_probability, threshold)

        logger.info(
            "%-24s valid F1=%.4f recall=%.4f precision=%.4f roc_auc=%.4f "
            "@ threshold %.2f",
            name,
            metrics["f1"],
            metrics["recall"],
            metrics["precision"],
            metrics["roc_auc"],
            threshold,
        )

        try:
            with mlflow.start_run(run_name=name):
                mlflow.log_params(
                    {
                        "model": name,
                        "decision_threshold": threshold,
                        "pollution_threshold": POLLUTION_THRESHOLD,
                        "split": "chronological_70_15_15",
                        "n_train": len(train),
                    }
                )
                mlflow.log_metrics(
                    {f"validation_{k}": v for k, v in metrics.items()}
                )
                # cloudpickle: the default skops backend rejects the
                # numpy.dtype references inside a fitted ColumnTransformer.
                mlflow.sklearn.log_model(
                    pipeline,
                    name="model",
                    serialization_format="cloudpickle",
                )
        except Exception as error:  # noqa: BLE001
            logger.warning("Could not log run '%s' to MLflow: %s", name, error)

        if best is None or metrics["f1"] > best["metrics"]["f1"]:
            best = {
                "name": name,
                "pipeline": pipeline,
                "threshold": threshold,
                "metrics": metrics,
            }

    assert best is not None

    # Single, final evaluation on the held-out test slice.
    test_probability = best["pipeline"].predict_proba(test[FEATURES])[:, 1]
    test_metrics = score(test[TARGET], test_probability, best["threshold"])
    test_prediction = (test_probability >= best["threshold"]).astype(int)
    matrix = confusion_matrix(test[TARGET], test_prediction)

    logger.info(
        "Selected model: %s (threshold %.2f)", best["name"], best["threshold"]
    )
    logger.info(
        "Test F1=%.4f recall=%.4f precision=%.4f roc_auc=%.4f",
        test_metrics["f1"],
        test_metrics["recall"],
        test_metrics["precision"],
        test_metrics["roc_auc"],
    )
    logger.info("Test confusion matrix [[TN FP] [FN TP]]:\n%s", matrix)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": best["pipeline"],
            "threshold": best["threshold"],
            "model_name": best["name"],
            "features": FEATURES,
            "pollution_threshold": POLLUTION_THRESHOLD,
        },
        MODEL_PATH,
    )
    logger.info("Saved model artifact to %s", MODEL_PATH)

    # Reference batch for Evidently: the same held-out slice the model was scored on.
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    test[FEATURES + [TARGET]].to_csv(REFERENCE_PATH, index=False)

    # Evaluation artifacts consumed by the README model card and the presentation.
    false_positive_rate, true_positive_rate, _ = roc_curve(
        test[TARGET], test_probability
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "evaluation.json").write_text(
        json.dumps(
            {
                "selected_model": best["name"],
                "decision_threshold": best["threshold"],
                "pollution_threshold": POLLUTION_THRESHOLD,
                "validation_metrics": best["metrics"],
                "test_metrics": test_metrics,
                "test_confusion_matrix": matrix.tolist(),
                "roc_curve": {
                    "fpr": false_positive_rate.tolist(),
                    "tpr": true_positive_rate.tolist(),
                },
                "split_sizes": {
                    "train": len(train),
                    "valid": len(valid),
                    "test": len(test),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Saved evaluation summary to %s", REPORTS_DIR / "evaluation.json")


if __name__ == "__main__":
    main()
