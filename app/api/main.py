"""FastAPI prediction service for next-hour Riyadh PM2.5 risk.

The model artifact is loaded once at import time, never per request.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

API_VERSION = "1.0.0"

# Locate project files
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
LOG_PATH = PROJECT_ROOT / "data" / "monitoring" / "predictions.csv"


# Load the trained model once when the API starts
artifact = joblib.load(MODEL_PATH)
model = artifact["pipeline"]
threshold = float(artifact["threshold"])
model_name = artifact.get("model_name", type(model).__name__)
pollution_threshold = float(artifact.get("pollution_threshold", 100.0))


# Create the FastAPI application
app = FastAPI(
    title="Riyadh Air Quality API",
    description=(
        "Predicts whether the next hour in Riyadh will exceed the "
        "high-PM2.5 pollution threshold."
    ),
    version=API_VERSION,
)


# Define and validate the required input features.
# Bounds are guard rails against nonsense input, not distribution limits. The
# pm10 ceiling is 5000 rather than 1500 because Riyadh dust storms genuinely
# reach 3263 ug/m3 in the collected data; a 1500 cap rejects 26% of real hours.
class PredictionRequest(BaseModel):
    pm2_5: float = Field(ge=0, le=1000)
    pm10: float = Field(ge=0, le=5000)
    temperature_2m: float = Field(ge=-20, le=65)
    relative_humidity_2m: float = Field(ge=0, le=100)
    wind_speed_10m: float = Field(ge=0, le=200)
    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    pm2_5_lag_1: float = Field(ge=0, le=1000)
    pm2_5_lag_3: float = Field(ge=0, le=1000)
    pm2_5_rolling_mean_6: float = Field(ge=0, le=1000)


@app.get("/health")
def health() -> dict[str, object]:
    """Check whether the API and model are working."""

    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.get("/model-info")
def model_info() -> dict[str, object]:
    """Return basic information about the loaded model."""

    return {
        "model_name": model_name,
        "model_type": type(model).__name__,
        "model_version": API_VERSION,
        "version": API_VERSION,
        "target": "high_pollution_next_hour",
        # Probability above which an hour is flagged high risk, tuned on validation.
        "threshold": threshold,
        # PM2.5 concentration (ug/m3) that defines a "high pollution" hour.
        "pollution_threshold": pollution_threshold,
        "features": list(PredictionRequest.model_fields),
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, object]:
    """Predict whether PM2.5 will be high in the next hour."""

    request_id = str(uuid4())

    # Convert the received data into one DataFrame row
    input_data = pd.DataFrame([request.model_dump()])

    try:
        probability = float(model.predict_proba(input_data)[0, 1])
    except Exception as error:  # noqa: BLE001 - surface as a clean 500
        logger.exception("Prediction failed for request %s", request_id)
        raise HTTPException(
            status_code=500, detail="Prediction failed."
        ) from error

    # The decision threshold comes from the artifact, tuned on validation data.
    prediction = int(probability >= threshold)

    # Save the input and prediction for future monitoring. A logging failure
    # must not cost the caller their prediction.
    try:
        log_row = input_data.copy()
        log_row["request_id"] = request_id
        log_row["probability"] = probability
        log_row["prediction"] = prediction
        log_row["timestamp"] = datetime.now(UTC).isoformat()

        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log_row.to_csv(
            LOG_PATH,
            mode="a",
            header=not LOG_PATH.exists(),
            index=False,
        )
    except OSError as error:
        logger.warning("Could not append to prediction log: %s", error)

    return {
        "request_id": request_id,
        "prediction": prediction,
        "probability": probability,
        "risk_level": "high" if prediction == 1 else "normal",
        "model_version": API_VERSION,
    }