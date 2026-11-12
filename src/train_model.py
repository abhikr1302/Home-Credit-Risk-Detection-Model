"""Train, evaluate, and save the production pipeline: python -m src.train_model."""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.config import (
    CLASSIFICATION_THRESHOLD,
    FINAL_MODEL_METRICS_PATH,
    MODEL_PATH,
    MODEL_TRAIN_FEATURES_PATH,
    PRODUCTION_FEATURES,
)

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}


def build_preprocessor(features):
    numeric = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [name for name in features if name not in numeric]
    return ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


def main():
    if not MODEL_TRAIN_FEATURES_PATH.exists():
        raise FileNotFoundError("Run python -m src.feature_engineering first.")
    data = pd.read_parquet(MODEL_TRAIN_FEATURES_PATH)
    required = set(PRODUCTION_FEATURES) | {"target", "sk_id_curr"}
    if missing := required - set(data.columns):
        raise ValueError(f"Missing training columns: {sorted(missing)}")
    if data["sk_id_curr"].duplicated().any() or data["sk_id_curr"].isna().any():
        raise ValueError("Applicant IDs must be unique and non-missing.")
    target = data["target"]
    if target.isna().any() or set(target.unique()) != {0, 1}:
        raise ValueError("Target must contain both 0 and 1 with no missing values.")

    features = data[PRODUCTION_FEATURES].replace([np.inf, -np.inf], np.nan)
    train, valid, y_train, y_valid = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    model = XGBClassifier(
        **XGB_PARAMS,
        scale_pos_weight=int((y_train == 0).sum()) / int((y_train == 1).sum()),
    )
    pipeline = Pipeline([("preprocessor", build_preprocessor(train)), ("model", model)])
    pipeline.fit(train, y_train)
    probabilities = pipeline.predict_proba(valid)[:, 1]
    predictions = (probabilities >= CLASSIFICATION_THRESHOLD).astype(int)
    metrics = {
        "model_type": "XGBoost",
        "feature_count": len(PRODUCTION_FEATURES),
        "training_rows": len(train),
        "validation_rows": len(valid),
        "random_state": 42,
        "threshold": CLASSIFICATION_THRESHOLD,
        "roc_auc": float(roc_auc_score(y_valid, probabilities)),
        "pr_auc": float(average_precision_score(y_valid, probabilities)),
        "precision": float(precision_score(y_valid, predictions, zero_division=0)),
        "recall": float(recall_score(y_valid, predictions, zero_division=0)),
        "f1": float(f1_score(y_valid, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_valid, predictions).tolist(),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    reloaded = joblib.load(MODEL_PATH)
    np.testing.assert_allclose(
        reloaded.predict_proba(valid.head()), pipeline.predict_proba(valid.head())
    )
    FINAL_MODEL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_MODEL_METRICS_PATH.write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved and verified {MODEL_PATH}")


if __name__ == "__main__":
    main()
