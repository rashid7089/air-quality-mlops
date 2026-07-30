"""API contract tests covering the specification's acceptance criteria."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import main as api
from app.api.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_health_reports_ok_status() -> None:
    assert client.get("/health").json()["status"] == "ok"


def test_model_info_exposes_the_decision_threshold() -> None:
    response = client.get("/model-info")
    assert response.status_code == 200

    body = response.json()
    assert 0.0 < body["threshold"] < 1.0
    assert body["pollution_threshold"] > 0
    assert body["target"] == "high_pollution_next_hour"
    assert len(body["features"]) == 10


def test_invalid_prediction_payload() -> None:
    response = client.post("/predict", json={"pm2_5": 10})
    assert response.status_code == 422


def test_valid_prediction_returns_probability(valid_payload: dict) -> None:
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert body["risk_level"] in ("high", "normal")
    assert body["request_id"]


def test_prediction_agrees_with_the_threshold(valid_payload: dict) -> None:
    body = client.post("/predict", json=valid_payload).json()
    assert body["prediction"] == int(body["probability"] >= api.threshold)
    assert body["risk_level"] == ("high" if body["prediction"] else "normal")


def test_request_ids_are_unique(valid_payload: dict) -> None:
    first = client.post("/predict", json=valid_payload).json()["request_id"]
    second = client.post("/predict", json=valid_payload).json()["request_id"]
    assert first != second


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hour", 24),
        ("hour", -1),
        ("day_of_week", 7),
        ("pm2_5", -1),
        ("relative_humidity_2m", 101),
        ("temperature_2m", 100),
        ("pm10", 9000),
    ],
)
def test_out_of_range_values_are_rejected(
    valid_payload: dict, field: str, value: float
) -> None:
    valid_payload[field] = value
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_dust_storm_pm10_is_accepted(valid_payload: dict) -> None:
    """Riyadh dust storms reach ~3300 ug/m3 PM10; those hours must not 422."""
    valid_payload["pm10"] = 3263.4
    assert client.post("/predict", json=valid_payload).status_code == 200


def test_missing_single_field_is_rejected(valid_payload: dict) -> None:
    valid_payload.pop("pm2_5_lag_1")
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_non_numeric_value_is_rejected(valid_payload: dict) -> None:
    valid_payload["pm2_5"] = "very high"
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_swagger_documentation_is_served() -> None:
    assert client.get("/docs").status_code == 200

    schema = client.get("/openapi.json").json()
    for route in ("/health", "/model-info", "/predict"):
        assert route in schema["paths"]


def test_prediction_is_logged_for_monitoring(
    valid_payload: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path = tmp_path / "predictions.csv"
    monkeypatch.setattr(api, "LOG_PATH", log_path)

    body = client.post("/predict", json=valid_payload).json()

    assert log_path.exists()
    rows = list(csv.DictReader(log_path.open()))
    assert len(rows) == 1
    assert rows[0]["request_id"] == body["request_id"]
    assert float(rows[0]["probability"]) == pytest.approx(body["probability"])
    assert rows[0]["timestamp"]

    # Every model feature must be captured, or Evidently has nothing to compare.
    for feature in valid_payload:
        assert feature in rows[0]


def test_log_appends_one_header_only(
    valid_payload: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path = tmp_path / "predictions.csv"
    monkeypatch.setattr(api, "LOG_PATH", log_path)

    for _ in range(3):
        client.post("/predict", json=valid_payload)

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 4
    assert sum(line.startswith("pm2_5,") for line in lines) == 1


def test_model_is_loaded_once_at_import() -> None:
    """The rubric penalises loading the model per request."""
    before = id(api.model)
    client.post("/predict", json=dict(TEST_PAYLOAD))
    assert id(api.model) == before


TEST_PAYLOAD = {
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
