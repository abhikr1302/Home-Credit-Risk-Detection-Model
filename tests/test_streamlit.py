from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from src import api


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('entrypoint', ['app.py', 'ui/streamlit_app.py'])
def test_ui_submits_and_displays_backend_probability(monkeypatch, entrypoint):
    class Model:
        def predict_proba(self, frame):
            import numpy as np
            return np.array([[0.25, 0.75]])

    monkeypatch.setattr(api, 'model', Model())
    with TestClient(api.app) as client:
        monkeypatch.setattr(requests, 'get', lambda url, **kwargs: client.get('/health'))
        monkeypatch.setattr(requests, 'post', lambda url, **kwargs: client.post('/predict', json=kwargs['json']))
        app = AppTest.from_file(str(ROOT / entrypoint), default_timeout=30).run()
        assert not app.exception
        app.button[0].click().run()
        assert not app.exception
        assert not app.error or any('HIGH' in error.value for error in app.error)
        metrics = {metric.label: metric.value for metric in app.metric}
        assert metrics['Estimated Default Probability'] == '75.00%'
        assert metrics['Model Decision'] == 'HIGH RISK'
