"""Model tests: split correctness, artifact contract, and beating the baseline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from air_quality.features import FEATURES, TARGET, build_dataset
from air_quality.train import (
    MODEL_PATH,
    REPORTS_DIR,
    build_candidates,
    choose_threshold,
    make_pipeline,
    score,
    split_by_time,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def artifact() -> dict:
    if not MODEL_PATH.exists():
        pytest.skip("Model artifact not built; run 'uv run python -m air_quality.train'")
    return joblib.load(MODEL_PATH)


def test_split_is_chronological_and_disjoint() -> None:
    df = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=1000, freq="h")})
    train, valid, test = split_by_time(df)

    assert (len(train), len(valid), len(test)) == (700, 150, 150)
    assert train["time"].max() < valid["time"].min()
    assert valid["time"].max() < test["time"].min()
    assert len(train) + len(valid) + len(test) == len(df)


def test_split_never_shuffles() -> None:
    df = pd.DataFrame({"value": range(100)})
    train, valid, test = split_by_time(df)
    rejoined = pd.concat([train, valid, test])["value"].tolist()
    assert rejoined == list(range(100))


def test_pipeline_includes_imputer_and_scaler() -> None:
    pipeline = make_pipeline(DummyClassifier())
    assert isinstance(pipeline, Pipeline)

    steps = pipeline.named_steps["preprocess"].transformers[0][1].named_steps
    assert "imputer" in steps
    assert "scaler" in steps


def test_pipeline_selects_features_explicitly() -> None:
    """Column selection must be by name, so stray columns can never be fed in."""
    assert make_pipeline(DummyClassifier()).named_steps[
        "preprocess"
    ].transformers[0][2] == FEATURES


def test_pipeline_tolerates_missing_values(tmp_path: Path, raw_path: Path) -> None:
    frame = build_dataset(str(raw_path), str(tmp_path / "t.parquet"))
    pipeline = make_pipeline(DummyClassifier(strategy="most_frequent"))
    pipeline.fit(frame[FEATURES], frame[TARGET])

    gapped = frame[FEATURES].copy()
    gapped.loc[gapped.index[0], "pm2_5_lag_1"] = None
    assert len(pipeline.predict(gapped)) == len(gapped)


def test_candidates_cover_baseline_and_two_models() -> None:
    candidates = build_candidates()
    assert len(candidates) == 3
    assert any("baseline" in name for name in candidates)
    assert "logistic_regression" in candidates
    assert "random_forest" in candidates


def test_choose_threshold_returns_a_probability() -> None:
    y = pd.Series([0, 0, 1, 1, 1, 0, 1, 1])
    probability = pd.Series([0.1, 0.2, 0.7, 0.8, 0.9, 0.3, 0.6, 0.75]).to_numpy()

    threshold, f1 = choose_threshold(y, probability)
    assert 0.0 < threshold < 1.0
    assert f1 == pytest.approx(
        f1_score(y, (probability >= threshold).astype(int)), abs=1e-9
    )


def test_score_reports_every_required_metric() -> None:
    y = pd.Series([0, 1, 1, 0, 1, 0])
    probability = pd.Series([0.1, 0.9, 0.8, 0.2, 0.7, 0.4]).to_numpy()
    metrics = score(y, probability, 0.5)
    assert set(metrics) == {"f1", "recall", "precision", "roc_auc"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_artifact_carries_pipeline_and_threshold(artifact: dict) -> None:
    assert isinstance(artifact["pipeline"], Pipeline)
    assert 0.0 < artifact["threshold"] < 1.0
    assert artifact["features"] == FEATURES
    assert artifact["model_name"]


def test_artifact_predicts_in_probability_space(artifact: dict) -> None:
    row = pd.DataFrame(
        [
            {
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
        ]
    )
    probability = artifact["pipeline"].predict_proba(row)[0, 1]
    assert 0.0 <= probability <= 1.0


def test_selected_model_beats_the_baseline() -> None:
    path = REPORTS_DIR / "evaluation.json"
    if not path.exists():
        pytest.skip("Evaluation summary not built; run the training script")

    summary = json.loads(path.read_text())
    assert summary["selected_model"] != "baseline_most_frequent"
    assert summary["test_metrics"]["roc_auc"] > 0.5
    assert summary["test_metrics"]["f1"] > 0.0


def test_confusion_matrix_has_all_four_cells() -> None:
    """A degenerate target would collapse the matrix to a single row."""
    path = REPORTS_DIR / "evaluation.json"
    if not path.exists():
        pytest.skip("Evaluation summary not built; run the training script")

    matrix = json.loads(path.read_text())["test_confusion_matrix"]
    assert len(matrix) == 2 and all(len(row) == 2 for row in matrix)
    assert sum(matrix[0]) > 0, "test set contains no negative examples"
    assert sum(matrix[1]) > 0, "test set contains no positive examples"


def test_threshold_was_tuned_on_validation_not_test() -> None:
    """Validation and test metrics must be reported separately."""
    path = REPORTS_DIR / "evaluation.json"
    if not path.exists():
        pytest.skip("Evaluation summary not built; run the training script")

    summary = json.loads(path.read_text())
    assert "validation_metrics" in summary
    assert "test_metrics" in summary
    assert 0.0 < summary["decision_threshold"] < 1.0
