import joblib
from src.config import MODEL_PATH
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import api


BASE_APPLICATION = {
    "age": 35,
    "annual_income": 600000,
    "loan_amount": 300000,
    "monthly_emi": 7500,
    "owns_car": True,
    "owns_property": False,
    "total_credit_accounts": 5,
    "active_credit_accounts": 2,
    "previous_applications": 3,
    "previous_approved": 2,
    "previous_refused": 1,
}


@pytest.fixture
def client(monkeypatch):
    class Pipeline:
        def predict_proba(self, frame):
            assert list(frame.columns) == api.EXPECTED_FEATURES
            assert frame.iloc[0]["flag_own_car"] == "Y"
            assert frame.iloc[0]["flag_own_realty"] == "N"
            return np.array([[0.25, 0.75]])

    monkeypatch.setattr(api, "model", Pipeline())
    with TestClient(api.app) as client:
        yield client


def test_banking_prediction_contract(client):
    response = client.post("/predict", json=BASE_APPLICATION)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["risk_probability"] == 0.75
    assert result["risk_level"] == "HIGH"
    assert result["decision"] == "HIGH RISK"
    assert result["backend_processing"]["features_sent_to_model"] == 30
    assert result["affordability"]["monthly_income"] == 50000


@pytest.mark.parametrize(
    "changes",
    [
        {"age": 17},
        {"annual_income": 0},
        {"monthly_emi": 0},
        {"active_credit_accounts": 6},
        {"previous_approved": 4},
        {"credit_card_balance": 100, "credit_card_limit": 0},
        {"external_score_1": 1.5},
        {"total_outstanding_debt": -1},
    ],
)
def test_invalid_applications_are_rejected(client, changes):
    response = client.post("/predict", json=BASE_APPLICATION | changes)
    assert response.status_code == 422


def test_nonfinite_inputs_are_rejected():
    with pytest.raises(ValueError):
        api.CreditApplication(**(BASE_APPLICATION | {"annual_income": float("inf")}))


def test_missing_model_is_unhealthy(client, monkeypatch):
    monkeypatch.setattr(api, "model", None)
    assert client.get("/health").status_code == 503
    assert client.post("/predict", json=BASE_APPLICATION).status_code == 503


def test_invalid_model_probability_is_not_reported_as_low_risk(client, monkeypatch):
    class InvalidPipeline:
        def predict_proba(self, frame):
            return np.array([[np.nan, np.nan]])

    monkeypatch.setattr(api, "model", InvalidPipeline())
    assert client.post("/predict", json=BASE_APPLICATION).status_code == 500


def test_training_category_and_ratio_parity():
    application = api.CreditApplication(
        **(
            BASE_APPLICATION
            | {
                "credit_card_balance": 150,
                "credit_card_limit": 100,
                "external_score_1": 0.3,
                "external_score_2": 0.6,
                "external_score_3": 0.9,
            }
        )
    )
    features = api.engineer_features(application).iloc[0]
    assert features["flag_own_car"] == "Y"
    assert features["flag_own_realty"] == "N"
    assert features["credit_card_avg_utilization"] == 1.5
    assert features["ext_source_mean"] == pytest.approx(0.6)
    assert features["previous_refusal_rate"] == pytest.approx(1 / 3)
    assert features["active_bureau_account_ratio"] == pytest.approx(2 / 5)


def test_undefined_ratios_reach_the_fitted_imputer():
    application = api.CreditApplication(
        age=35, annual_income=600000, loan_amount=300000, monthly_emi=7500
    )
    row = api.engineer_features(application).iloc[0]
    for name in [
        "previous_approval_rate",
        "previous_refusal_rate",
        "credit_card_avg_utilization",
        "active_bureau_account_ratio",
    ]:
        assert np.isnan(row[name])


@pytest.mark.parametrize("probability", [0, 0.299, 0.3, 0.599, 0.6, 1])
def test_threshold_decision_consistency(probability):
    high = probability >= api.THRESHOLD
    assert (api.calculate_risk_level(probability) == "HIGH") == high
    assert (api.calculate_decision(probability) == "HIGH RISK") == high


@pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="Local model artifact is not available."
)
def test_saved_pipeline_predicts_banking_inputs(monkeypatch):
    pipeline = joblib.load(MODEL_PATH)
    assert list(pipeline.feature_names_in_) == api.EXPECTED_FEATURES
    base = {
        "age": 35,
        "annual_income": 600000,
        "loan_amount": 300000,
        "monthly_emi": 7500,
    }
    probabilities = []
    monkeypatch.setattr(api, "model", pipeline)
    with TestClient(api.app) as client:
        assert client.get("/health").status_code == 200
        for score in [0.1, 0.5, 0.9]:
            payload = base | {f"external_score_{i}": score for i in range(1, 4)}
            response = client.post("/predict", json=payload)
            assert response.status_code == 200, response.text
            probability = response.json()["risk_probability"]
            assert np.isfinite(probability) and 0 <= probability <= 1
            probabilities.append(probability)
    assert probabilities[0] > probabilities[-1]
