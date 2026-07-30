"""End-to-end test of the vertical slice.

raw JSON -> features -> train -> artifact -> FastAPI -> prediction log ->
Evidently report. Runs entirely on synthetic data in a temporary directory, so
it needs neither the network nor a running MLflow server.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from air_quality.features import FEATURES, TARGET, build_dataset
from air_quality.train import (
    build_candidates,
    choose_threshold,
    make_pipeline,
    score,
    split_by_time,
)
from app.api import main as api

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run collect-output -> features -> train without touching the repo."""
    from .conftest import make_raw_payload

    workspace = tmp_path_factory.mktemp("e2e")
    raw_path = workspace / "air_quality.json"
    raw_path.write_text(json.dumps(make_raw_payload(hours=900)), encoding="utf-8")

    processed_path = workspace / "model_table.parquet"
    frame = build_dataset(str(raw_path), str(processed_path))

    train, valid, test = split_by_time(frame)

    best = None
    for name, model in build_candidates().items():
        model_pipeline = make_pipeline(model)
        model_pipeline.fit(train[FEATURES], train[TARGET])
        probability = model_pipeline.predict_proba(valid[FEATURES])[:, 1]
        threshold, _ = choose_threshold(valid[TARGET], probability)
        metrics = score(valid[TARGET], probability, threshold)

        if best is None or metrics["f1"] > best["metrics"]["f1"]:
            best = {
                "name": name,
                "pipeline": model_pipeline,
                "threshold": threshold,
                "metrics": metrics,
            }

    assert best is not None
    model_path = workspace / "model.joblib"
    joblib.dump(
        {
            "pipeline": best["pipeline"],
            "threshold": best["threshold"],
            "model_name": best["name"],
            "features": FEATURES,
            "pollution_threshold": 100.0,
        },
        model_path,
    )

    reference_path = workspace / "reference.csv"
    test[FEATURES + [TARGET]].to_csv(reference_path, index=False)

    return {
        "workspace": workspace,
        "frame": frame,
        "test": test,
        "model_path": model_path,
        "reference_path": reference_path,
        "best": best,
    }


def test_stage_1_features_produce_a_usable_table(pipeline_run: dict) -> None:
    frame = pipeline_run["frame"]
    assert not frame.empty
    assert frame[TARGET].nunique() == 2, "target collapsed to one class"


def test_stage_2_training_selects_a_real_model(pipeline_run: dict) -> None:
    best = pipeline_run["best"]
    assert best["name"] != "baseline_most_frequent"
    assert best["metrics"]["roc_auc"] > 0.5


def test_stage_3_artifact_round_trips(pipeline_run: dict) -> None:
    artifact = joblib.load(pipeline_run["model_path"])
    assert artifact["features"] == FEATURES

    sample = pipeline_run["test"][FEATURES].head(5)
    probability = artifact["pipeline"].predict_proba(sample)[:, 1]
    assert ((probability >= 0) & (probability <= 1)).all()


def test_stage_4_api_serves_the_artifact(
    pipeline_run: dict, valid_payload: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The API serves whatever the training stage produced."""
    artifact = joblib.load(pipeline_run["model_path"])
    log_path = pipeline_run["workspace"] / "predictions.csv"

    monkeypatch.setattr(api, "model", artifact["pipeline"])
    monkeypatch.setattr(api, "threshold", float(artifact["threshold"]))
    monkeypatch.setattr(api, "LOG_PATH", log_path)

    client = TestClient(api.app)
    assert client.get("/health").json()["model_loaded"] is True

    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    assert 0.0 <= response.json()["probability"] <= 1.0


def test_stage_5_predictions_accumulate_in_the_log(
    pipeline_run: dict, valid_payload: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = joblib.load(pipeline_run["model_path"])
    log_path = pipeline_run["workspace"] / "monitoring_predictions.csv"

    monkeypatch.setattr(api, "model", artifact["pipeline"])
    monkeypatch.setattr(api, "threshold", float(artifact["threshold"]))
    monkeypatch.setattr(api, "LOG_PATH", log_path)

    client = TestClient(api.app)
    for row in pipeline_run["test"][FEATURES].head(20).to_dict(orient="records"):
        payload = {k: float(v) for k, v in row.items()}
        payload["hour"] = int(payload["hour"])
        payload["day_of_week"] = int(payload["day_of_week"])
        assert client.post("/predict", json=payload).status_code == 200

    logged = pd.read_csv(log_path)
    assert len(logged) == 20
    assert set(FEATURES).issubset(logged.columns)

    pipeline_run["log_path"] = log_path


def test_stage_6_evidently_report_compares_both_batches(pipeline_run: dict) -> None:
    """The monitoring report must open and contain reference and current data."""
    from evidently import Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    log_path = pipeline_run.get("log_path")
    if log_path is None or not Path(log_path).exists():
        pytest.skip("Prediction log stage did not run")

    reference = pd.read_csv(pipeline_run["reference_path"])[FEATURES]
    current = pd.read_csv(log_path)[FEATURES]
    assert not reference.empty and not current.empty

    report = Report([DataSummaryPreset(), DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)

    report_path = pipeline_run["workspace"] / "monitoring.html"
    snapshot.save_html(str(report_path))

    assert report_path.exists()
    html = report_path.read_text(encoding="utf-8")
    assert len(html) > 1000
    assert "html" in html[:200].lower()
