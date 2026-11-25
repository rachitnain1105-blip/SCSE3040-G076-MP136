
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction_endpoint():
    payload = {
        "distance_km": 1250,
        "weight_kg": 5.5,
        "hub_backlog": 2500,
        "pin_code_tier": 2,
        "declared_value_rs": 5000,
        "transfers": 2,
        "service": "Standard",
        "destination": "Patna",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "predicted_delivery_days" in response.json()
    assert isinstance(response.json()["predicted_delivery_days"], float)


def test_backlog_validation():
    payload = {
        "distance_km": 1250,
        "weight_kg": 5.5,
        "hub_backlog": 9500,
        "pin_code_tier": 2,
        "declared_value_rs": 5000,
        "transfers": 2,
        "service": "Standard",
        "destination": "Patna",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_transfers_validation():
    payload = {
        "distance_km": 1250,
        "weight_kg": 5.5,
        "hub_backlog": 2500,
        "pin_code_tier": 2,
        "declared_value_rs": 5000,
        "transfers": -1,
        "service": "Standard",
        "destination": "Patna",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
