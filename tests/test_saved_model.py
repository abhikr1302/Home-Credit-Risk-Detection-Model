import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import api
from src.config import MODEL_PATH


@pytest.mark.skipif(not MODEL_PATH.exists(), reason='Local model artifact is not available.')
def test_saved_pipeline_predicts_banking_inputs(monkeypatch):
    pipeline = joblib.load(MODEL_PATH)
    assert list(pipeline.feature_names_in_) == api.EXPECTED_FEATURES
    base = {'age': 35, 'annual_income': 600000, 'loan_amount': 300000, 'monthly_emi': 7500}
    probabilities = []
    monkeypatch.setattr(api, 'model', pipeline)
    with TestClient(api.app) as client:
        assert client.get('/health').status_code == 200
        for score in [0.1, 0.5, 0.9]:
            payload = base | {f'external_score_{i}': score for i in range(1, 4)}
            response = client.post('/predict', json=payload)
            assert response.status_code == 200, response.text
            probability = response.json()['risk_probability']
            assert np.isfinite(probability) and 0 <= probability <= 1
            probabilities.append(probability)
    assert probabilities[0] > probabilities[-1]
