from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)

# THE COMMAND TO RUN THE TEST (from the root directory)
# uv run python -m pytest tests/ -q

def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_invalid_prediction_payload() -> None:
    response = client.post("/predict", json={"pm2_5": 10})
    assert response.status_code == 422



