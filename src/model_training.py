"""
Home Credit Risk Detection Model
--------------------------------
Production model training using the original 30 recommended features.

Strategy:
- Keep all engineered backend features in the dataset.
- Use ONLY the original 30 recommended features for the production model.
- Use a fixed stratified validation split.
- Train XGBoost with the existing configuration.
- Save the production pipeline and validation metrics.
- Verify that the saved model reloads successfully.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier




# ============================================================
# CONFIG
# ============================================================

try:
    from src.config import (
        MODEL_TRAIN_FEATURES_PATH,
        RECOMMENDED_FEATURES_JSON,
        MODEL_PATH,
        FINAL_MODEL_METRICS_PATH,
        FINAL_MODEL_FEATURES_PATH,
    )
except ModuleNotFoundError:
    from config import (
        MODEL_TRAIN_FEATURES_PATH,
        RECOMMENDED_FEATURES_JSON,
        MODEL_PATH,
        FINAL_MODEL_METRICS_PATH,
        FINAL_MODEL_FEATURES_PATH,
    )


# ============================================================
# CONSTANTS
# ============================================================

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

# Existing XGBoost configuration
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "min_child_weight": 5,
    "reg_alpha": 0.10,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "tree_method": "hist",
}

# Thresholds used for evaluation
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


# ============================================================
# 30 PRODUCTION FEATURES
# ============================================================

from src.model_contract import PRODUCTION_FEATURES


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def ensure_parent_directory(path):
    """Create parent directory if it does not exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_recommended_features():
    """
    Load the original recommended feature list from JSON.

    This is used only as a validation check. The production
    feature list remains explicitly fixed to PRODUCTION_FEATURES.
    """

    path = Path(RECOMMENDED_FEATURES_JSON)

    if not path.exists():
        print(f"WARNING: Recommended feature file not found: {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in [
            "recommended_features",
            "features",
            "selected_features",
        ]:
            if key in data and isinstance(data[key], list):
                return data[key]

    return []


def validate_production_features(df):
    """Ensure all 30 production features exist in the dataset."""

    missing = [
        feature
        for feature in PRODUCTION_FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            "Production features missing from dataset:\n"
            + "\n".join(f" - {feature}" for feature in missing)
        )

    if len(PRODUCTION_FEATURES) != 30:
        raise ValueError(
            f"Expected exactly 30 production features, "
            f"found {len(PRODUCTION_FEATURES)}."
        )

    print("Production feature validation: PASSED")
    print(f"Production feature count: {len(PRODUCTION_FEATURES)}")


def build_preprocessor(X):
    """Build preprocessing pipeline."""

    numeric_features = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_features = [
        column
        for column in X.columns
        if column not in numeric_features
    ]

    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


def calculate_metrics(y_true, probabilities, threshold=0.50):
    """Calculate classification metrics."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
    }


def find_best_f1_threshold(y_true, probabilities):
    """Find threshold producing the highest F1."""

    results = []

    for threshold in THRESHOLDS:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        results.append(
            {
                "threshold": threshold,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

    best = max(
        results,
        key=lambda x: x["f1"]
    )

    return best, results


def get_feature_names_from_preprocessor(preprocessor):
    """
    Return transformed feature names.

    Useful for verifying the preprocessing pipeline and
    documenting the production model.
    """

    try:
        return list(
            preprocessor.get_feature_names_out()
        )
    except Exception:
        return []


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    print("=" * 70)
    print("HOME CREDIT RISK MODEL - PRODUCTION TRAINING")
    print("=" * 70)

    print("\nProduction strategy:")
    print("- 30 original recommended features")
    print("- 43 additional backend features excluded")
    print("- Fixed stratified validation split")
    print("- XGBoost")
    print("- Same configuration used for baseline comparison")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    data_path = Path(MODEL_TRAIN_FEATURES_PATH)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found:\n{data_path}"
        )

    print(f"\nLoading dataset:")
    print(data_path)

    df = pd.read_parquet(data_path)

    print(f"\nDataset shape: {df.shape}")

    if "target" not in df.columns:
        raise ValueError(
            "Target column 'target' was not found."
        )

    # --------------------------------------------------------
    # LOAD AND VALIDATE RECOMMENDED FEATURES
    # --------------------------------------------------------

    recommended_features = load_recommended_features()

    if recommended_features:
        print(
            f"\nRecommended feature JSON contains "
            f"{len(recommended_features)} features."
        )

        if set(recommended_features) != set(PRODUCTION_FEATURES):
            print(
                "WARNING: JSON feature list differs from "
                "the fixed production feature list."
            )

    # --------------------------------------------------------
    # VALIDATE PRODUCTION FEATURES
    # --------------------------------------------------------

    validate_production_features(df)

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    if df["target"].isna().any() or set(df["target"].unique()) != {0, 1}:
        raise ValueError("Target must contain both 0 and 1 with no missing values.")
    if "sk_id_curr" not in df or df["sk_id_curr"].duplicated().any():
        raise ValueError("Training data requires unique applicant IDs.")
    y = df["target"].astype(int)

    X = df[PRODUCTION_FEATURES].replace([np.inf, -np.inf], np.nan).copy()

    print("\nFinal production training dataset:")
    print(f"Rows: {len(X):,}")
    print(f"Features: {X.shape[1]}")

    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    default_rate = y.mean()

    print(
        f"\nDefault rate: {default_rate:.4%}"
    )

    # --------------------------------------------------------
    # CHECK BACKEND FEATURES ARE NOT USED
    # --------------------------------------------------------

    excluded_features = [
        column
        for column in df.columns
        if column not in PRODUCTION_FEATURES
        and column != "target"
    ]

    print(
        f"\nBackend/unused features available: "
        f"{len(excluded_features)}"
    )

    # --------------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # --------------------------------------------------------

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nValidation split:")
    print(f"Training rows:   {len(X_train):,}")
    print(f"Validation rows: {len(X_valid):,}")

    # --------------------------------------------------------
    # CLASS WEIGHT
    # --------------------------------------------------------

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    scale_pos_weight = (
        negative_count / positive_count
    )

    print(
        f"\nScale positive weight: "
        f"{scale_pos_weight:.4f}"
    )

    # --------------------------------------------------------
    # PREPROCESSOR
    # --------------------------------------------------------

    preprocessor = build_preprocessor(X_train)

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model_params = XGB_PARAMS.copy()

    model_params["scale_pos_weight"] = (
        scale_pos_weight
    )

    model = XGBClassifier(
        **model_params
    )

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    print("\nTraining production model...")
    print("-" * 70)

    pipeline.fit(
        X_train,
        y_train,
    )

    print("Training completed.")

    # --------------------------------------------------------
    # VALIDATION PREDICTIONS
    # --------------------------------------------------------

    probabilities = pipeline.predict_proba(
        X_valid
    )[:, 1]

    # --------------------------------------------------------
    # METRICS @ 0.50
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_valid,
        probabilities,
        threshold=0.50,
    )

    # --------------------------------------------------------
    # BEST F1
    # --------------------------------------------------------

    best_threshold, threshold_results = (
        find_best_f1_threshold(
            y_valid,
            probabilities,
        )
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    predictions_050 = (
        probabilities >= 0.50
    ).astype(int)

    cm = confusion_matrix(
        y_valid,
        predictions_050,
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PRODUCTION MODEL RESULTS")
    print("=" * 70)

    print(
        f"ROC-AUC:             {metrics['roc_auc']:.6f}"
    )

    print(
        f"PR-AUC:              {metrics['pr_auc']:.6f}"
    )

    print(
        f"Precision @ 0.50:    {metrics['precision']:.6f}"
    )

    print(
        f"Recall @ 0.50:       {metrics['recall']:.6f}"
    )

    print(
        f"F1 @ 0.50:           {metrics['f1']:.6f}"
    )

    print(
        f"Best F1 threshold:   "
        f"{best_threshold['threshold']:.2f}"
    )

    print(
        f"Best F1:             "
        f"{best_threshold['f1']:.6f}"
    )

    print("\nConfusion Matrix @ 0.50:")
    print(cm)

    # --------------------------------------------------------
    # TRANSFORMED FEATURE COUNT
    # --------------------------------------------------------

    fitted_preprocessor = (
        pipeline.named_steps["preprocessor"]
    )

    transformed_features = (
        get_feature_names_from_preprocessor(
            fitted_preprocessor
        )
    )

    print(
        f"\nOriginal input features: "
        f"{len(PRODUCTION_FEATURES)}"
    )

    print(
        f"Transformed model features: "
        f"{len(transformed_features)}"
    )

    # --------------------------------------------------------
    # SAVE PRODUCTION MODEL
    # --------------------------------------------------------

    ensure_parent_directory(MODEL_PATH)

    joblib.dump(
        pipeline,
        MODEL_PATH,
    )

    print(
        f"\nProduction model saved:"
        f"\n{MODEL_PATH}"
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics_output = {
        "model_type": "XGBoost",
        "model_status": "production",
        "feature_strategy": (
            "Original 30 recommended features"
        ),
        "feature_count": len(PRODUCTION_FEATURES),
        "backend_features_excluded": len(
            excluded_features
        ),
        "production_features": PRODUCTION_FEATURES,
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "precision_at_0_50": metrics["precision"],
        "recall_at_0_50": metrics["recall"],
        "f1_at_0_50": metrics["f1"],
        "best_f1_threshold": (
            best_threshold["threshold"]
        ),
        "best_f1": best_threshold["f1"],
        "confusion_matrix_at_0_50": cm.tolist(),
        "validation_size": VALIDATION_SIZE,
        "random_state": RANDOM_STATE,
        "xgb_parameters": model_params,
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "default_rate": float(default_rate),
        "transformed_feature_count": len(
            transformed_features
        ),
    }

    ensure_parent_directory(
        FINAL_MODEL_METRICS_PATH
    )

    with open(
        FINAL_MODEL_METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics_output,
            f,
            indent=4,
        )

    print(
        f"\nMetrics saved:"
        f"\n{FINAL_MODEL_METRICS_PATH}"
    )

    # --------------------------------------------------------
    # SAVE FEATURE REPORT
    # --------------------------------------------------------

    feature_report = {
        "model_status": "production",
        "selection_method": (
            "Original recommended feature set"
        ),
        "feature_count": len(PRODUCTION_FEATURES),
        "features": PRODUCTION_FEATURES,
        "excluded_backend_features": excluded_features,
        "transformed_feature_count": len(
            transformed_features
        ),
        "random_state": RANDOM_STATE,
        "validation_size": VALIDATION_SIZE,
    }

    ensure_parent_directory(
        FINAL_MODEL_FEATURES_PATH
    )

    with open(
        FINAL_MODEL_FEATURES_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            feature_report,
            f,
            indent=4,
        )

    print(
        f"\nFeature report saved:"
        f"\n{FINAL_MODEL_FEATURES_PATH}"
    )

    # --------------------------------------------------------
    # RELOAD TEST
    # --------------------------------------------------------

    print("\nTesting saved model reload...")

    loaded_pipeline = joblib.load(
        MODEL_PATH
    )

    reload_predictions = (
        loaded_pipeline.predict_proba(
            X_valid.iloc[:5]
        )[:, 1]
    )

    print(
        "Reload test predictions:"
    )

    for index, probability in enumerate(
        reload_predictions,
        start=1,
    ):
        print(
            f"  Sample {index}: "
            f"{probability:.6f}"
        )

    # --------------------------------------------------------
    # FINAL FEATURE CHECK
    # --------------------------------------------------------

    loaded_preprocessor = (
        loaded_pipeline.named_steps[
            "preprocessor"
        ]
    )

    loaded_numeric = (
        loaded_preprocessor.transformers_[0][2]
    )

    loaded_categorical = (
        loaded_preprocessor.transformers_[1][2]
    )

    loaded_input_features = (
        list(loaded_numeric)
        + list(loaded_categorical)
    )

    if set(loaded_input_features) != set(
        PRODUCTION_FEATURES
    ):
        raise RuntimeError(
            "Production model feature validation failed."
        )

    print(
        "\nProduction feature verification: PASSED"
    )

    print(
        f"Verified exactly "
        f"{len(PRODUCTION_FEATURES)} "
        f"input features."
    )

    print("\n" + "=" * 70)
    print(
        "PRODUCTION MODEL TRAINING COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    print("\nProduction model:")
    print(MODEL_PATH)

    print("\nProduction feature count:")
    print(len(PRODUCTION_FEATURES))

    print("\nThe additional 43 backend features remain")
    print("available for future experiments but are")
    print("NOT used by the production model.")


if __name__ == "__main__":
    main()
