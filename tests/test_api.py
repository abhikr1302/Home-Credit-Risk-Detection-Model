from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as api_main
from app.main import app


class FakeModel:
    pass


class FakePipeline:
    feature_names_in_ = np.array(
        [
            "amt_income_total",
            "amt_credit",
            "credit_income_ratio",
        ]
    )

    named_steps = {
        "model": FakeModel(),
    }

    def predict_proba(self, features):
        return np.array(
            [
                [0.75, 0.25]
                for _ in range(len(features))
            ]
        )


@pytest.fixture
def client(
    monkeypatch,
    tmp_path: Path,
):
    fake_model_path = (
        tmp_path / "fake_model.joblib"
    )

    fake_model_path.touch()

    monkeypatch.setattr(
        api_main,
        "MODEL_PATH",
        fake_model_path,
    )

    monkeypatch.setattr(
        api_main.joblib,
        "load",
        lambda model_path: FakePipeline(),
    )

    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_status": "loaded",
    }


def test_model_information(client):
    response = client.get("/model-info")

    assert response.status_code == 200

    result = response.json()

    assert result["model_type"] == "FakeModel"
    assert result["expected_feature_count"] == 3


def test_features_endpoint(client):
    response = client.get("/features")

    assert response.status_code == 200

    assert response.json()["features"] == [
        "amt_income_total",
        "amt_credit",
        "credit_income_ratio",
    ]


def test_prediction_endpoint(client):
    response = client.post(
        "/predict",
        json={
            "applicant_id": 100001,
            "features": {
                "amt_income_total": 180000,
                "amt_credit": 450000,
                "credit_income_ratio": 2.5,
            },
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["applicant_id"] == 100001
    assert result["default_probability"] == 0.25
    assert result["predicted_class"] == 0
    assert result["risk_level"] == "low"
    assert result["missing_feature_count"] == 0


def test_missing_features_are_imputed(client):
    response = client.post(
        "/predict",
        json={
            "applicant_id": 100002,
            "features": {
                "amt_income_total": 180000
            },
        },
    )

    assert response.status_code == 200
    assert (
        response.json()["missing_feature_count"]
        == 2
    )


def test_empty_features_fail(client):
    response = client.post(
        "/predict",
        json={
            "applicant_id": 100003,
            "features": {},
        },
    )

    assert response.status_code == 422


def test_unknown_features_fail(client):
    response = client.post(
        "/predict",
        json={
            "applicant_id": 100004,
            "features": {
                "unknown_feature": 10
            },
        },
    )

    assert response.status_code == 422