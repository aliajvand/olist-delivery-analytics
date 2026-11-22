"""Unit and Integration tests for Olist Delivery Analytics FastAPI Service."""
import os
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["feature_count"] == 39
    assert data["lift_at_capacity"] == pytest.approx(2.07, rel=1e-2)

def test_governance_endpoint():
    response = client.get("/governance")
    assert response.status_code == 200
    data = response.json()
    assert "classification_governance" in data
    assert data["regression_mae_days"] == 4.09
    assert data["policy"]["decision_threshold"] == 0.251

def test_predict_delay_endpoint():
    payload = {
        "order_gmv": 150.0,
        "order_weight_g": 3000.0,
        "freight_value": 35.0,
        "distance_km": 800.0,
        "same_state_flag": 0,
        "seller_prior_late_rate": 0.15,
        "estimated_delivery_days": 12.0,
    }
    response = client.post("/predict-delay", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["late_delivery_probability"] <= 1.0
    assert "expedite_recommended" in data
    assert data["expected_transit_days"] > 0

def test_route_risk_endpoint():
    payload = {
        "origin_state": "SP",
        "destination_state": "BA",
    }
    response = client.post("/route-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["origin_state"] == "SP"
    assert data["destination_state"] == "BA"
    assert data["historical_late_rate_pct"] == 14.2
    assert "High Bottleneck Risk" in data["route_status"]

