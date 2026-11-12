import json

import joblib
import numpy as np
import pandas as pd

from src import train_model
from src.config import PRODUCTION_FEATURES


def test_production_training_saves_reloadable_pipeline(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)
    data = pd.DataFrame({name: rng.random(120) for name in PRODUCTION_FEATURES})
    for name in [
        "name_education_type",
        "code_gender",
        "name_income_type",
        "name_housing_type",
        "flag_own_car",
        "name_type_suite",
        "name_family_status",
        "name_contract_type",
        "flag_own_realty",
    ]:
        data[name] = np.where(np.arange(120) % 2, "A", "B")
    data["target"] = np.tile([0, 0, 0, 1], 30)
    data["sk_id_curr"] = np.arange(120)
    data_path = tmp_path / "training.parquet"
    data.to_parquet(data_path)
    paths = {
        "MODEL_TRAIN_FEATURES_PATH": data_path,
        "MODEL_PATH": tmp_path / "model.joblib",
        "FINAL_MODEL_METRICS_PATH": tmp_path / "metrics.json",
    }
    for name, path in paths.items():
        monkeypatch.setattr(train_model, name, path)
    monkeypatch.setattr(
        train_model,
        "XGB_PARAMS",
        train_model.XGB_PARAMS | {"n_estimators": 3, "n_jobs": 1},
    )
    train_model.main()
    pipeline = joblib.load(paths["MODEL_PATH"])
    assert list(pipeline.feature_names_in_) == PRODUCTION_FEATURES
    assert np.isfinite(pipeline.predict_proba(data[PRODUCTION_FEATURES].head())).all()
    metrics = json.loads(paths["FINAL_MODEL_METRICS_PATH"].read_text())
    assert metrics["training_rows"] == 96
    assert metrics["validation_rows"] == 24
    assert metrics["feature_count"] == 30
