"""
Tests for the Flask prediction API.

Run from the project root:
  pytest tests/
"""

import pytest

from app.api import app

# A valid sample (first row of data_for_training.csv)
VALID_PAYLOAD = {
    "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
    "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
    "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689,
    "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
    "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0,
    "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0,
}


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ── /health ──────────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_payload(client):
    data = client.get("/health").get_json()
    assert data["status"] == "healthy"
    assert "v1" in data["available_versions"]
    assert "v2" in data["available_versions"]


# ── /predict — happy paths ────────────────────────────────────────────────────

def test_predict_v1_returns_200(client):
    resp = client.post("/predict?version=v1", json=VALID_PAYLOAD)
    assert resp.status_code == 200


def test_predict_v1_shape(client):
    data = client.post("/predict?version=v1", json=VALID_PAYLOAD).get_json()
    assert "prediction" in data
    assert "probability" in data
    assert "model_version" in data


def test_predict_v1_values(client):
    data = client.post("/predict?version=v1", json=VALID_PAYLOAD).get_json()
    assert data["prediction"] in (0, 1)
    assert 0.0 <= data["probability"] <= 1.0
    assert data["model_version"] == "model_v1"


def test_predict_v2_returns_200(client):
    resp = client.post("/predict?version=v2", json=VALID_PAYLOAD)
    assert resp.status_code == 200


def test_predict_v2_version_label(client):
    data = client.post("/predict?version=v2", json=VALID_PAYLOAD).get_json()
    assert data["model_version"] == "model_v2"


def test_predict_default_version_is_v1(client):
    data = client.post("/predict", json=VALID_PAYLOAD).get_json()
    assert data["model_version"] == "model_v1"


# ── /predict — error cases ────────────────────────────────────────────────────

def test_predict_missing_features_returns_400(client):
    resp = client.post("/predict", json={"LIMIT_BAL": 20000})
    assert resp.status_code == 400


def test_predict_missing_features_lists_them(client):
    data = client.post("/predict", json={"LIMIT_BAL": 20000}).get_json()
    assert "details" in data
    assert "SEX" in data["details"]


def test_predict_unknown_version_returns_400(client):
    resp = client.post("/predict?version=v99", json=VALID_PAYLOAD)
    assert resp.status_code == 400


def test_predict_extra_keys_ignored(client):
    """Extra keys in the payload must not cause an error."""
    payload = {**VALID_PAYLOAD, "unexpected_field": "noise"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
