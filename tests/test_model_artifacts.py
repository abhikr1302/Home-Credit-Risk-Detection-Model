from pathlib import Path

import pandas as pd


MODEL_PATH = Path(
    "models/final_credit_risk_model.joblib"
)

IMPORTANCE_PATH = Path(
    "reports/shap_feature_importance.csv"
)


def test_final_model_exists():
    assert MODEL_PATH.exists()


def test_feature_importance_exists():
    assert IMPORTANCE_PATH.exists()

    importance = pd.read_csv(IMPORTANCE_PATH)

    assert not importance.empty

    assert {
        "feature",
        "mean_absolute_shap",
    }.issubset(importance.columns)

    assert (
        importance["mean_absolute_shap"] >= 0
    ).all()