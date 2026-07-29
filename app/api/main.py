from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


# Locate project files
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
LOG_PATH = PROJECT_ROOT / "data" / "monitoring" / "predictions.csv"


# Load the trained model once when the API starts
model = joblib.load(MODEL_PATH)


# Create the FastAPI application
app = FastAPI(
    title="Riyadh Air Quality API",
    version="1.0.0",
)


# Define and validate the required input features
class PredictionRequest(BaseModel):
    pm2_5: float = Field(ge=0, le=1000)
    pm10: float = Field(ge=0, le=1500)
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
        "model_type": type(model).__name__,
        "model_version": "1.0.0",
        "target": "high_pollution_next_hour",
        "pollution_threshold": 35.0,
        "features": list(PredictionRequest.model_fields),
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, object]:
    """Predict whether PM2.5 will be high in the next hour."""

    request_id = str(uuid4())

    # Convert the received data into one DataFrame row
    input_data = pd.DataFrame([request.model_dump()])

    # Generate the predicted class
    prediction = int(model.predict(input_data)[0])

    # Generate the probability of the high-pollution class
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_data)[0, 1])
    else:
        probability = float(prediction)

    # Save the input and prediction for future monitoring
    log_row = input_data.copy()
    log_row["request_id"] = request_id
    log_row["probability"] = probability
    log_row["prediction"] = prediction
    log_row["timestamp"] = datetime.now(timezone.utc).isoformat()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    log_row.to_csv(
        LOG_PATH,
        mode="a",
        header=not LOG_PATH.exists(),
        index=False,
    )

    return {
        "request_id": request_id,
        "prediction": prediction,
        "probability": probability,
        "risk_level": "high" if prediction == 1 else "normal",
        "model_version": "1.0.0",
    }

"""To open FastAPI use : uv run uvicorn app.api.main:app \
  --host 127.0.0.1 \
  --port 8000 \
  --reload"""