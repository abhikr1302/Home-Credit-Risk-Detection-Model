from pathlib import Path
import numpy as np
import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_features_train.parquet"
)

FEATURES_PATH = (
    PROJECT_ROOT
    / "reports"
    / "final_model_features.json"
)

TARGET = "target"
RANDOM_STATE = 42
THRESHOLD = 0.60


# Seven features added to the 30-feature production model.
ADDITIONAL_FEATURES = [
    "income_per_family_member",
    "cnt_children",
    "bureau_total_credit",
    "amt_income_total",
]


MODEL_PARAMS = {
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


# =============================================================================
# BUILD MODEL PIPELINE
# =============================================================================

def build_pipeline(X, scale_pos_weight):

    numeric_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = [
        col
        for col in X.columns
        if col not in numeric_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
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
        ]
    )

    model = XGBClassifier(
        **MODEL_PARAMS,
        scale_pos_weight=scale_pos_weight,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(
    features,
    X_train,
    X_valid,
    y_train,
    y_valid,
    scale_pos_weight,
):

    X_train_model = X_train[features]
    X_valid_model = X_valid[features]

    model = build_pipeline(
        X_train_model,
        scale_pos_weight,
    )

    model.fit(
        X_train_model,
        y_train,
    )

    probabilities = model.predict_proba(
        X_valid_model
    )[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    return {
        "roc_auc": roc_auc_score(
            y_valid,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_valid,
            probabilities,
        ),
        "precision": precision_score(
            y_valid,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_valid,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_valid,
            predictions,
            zero_division=0,
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("HOME CREDIT MODEL - FEATURE ABLATION")
    print("=" * 80)

    print("\nLoading dataset:")
    print(DATA_PATH)

    df = pd.read_parquet(DATA_PATH)

    print(f"\nDataset shape: {df.shape}")

    # -------------------------------------------------------------------------
    # Load production feature list
    # -------------------------------------------------------------------------

    with open(
        FEATURES_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        feature_config = json.load(f)

    production_features = feature_config["features"]

    print(
        f"\nProduction feature count: "
        f"{len(production_features)}"
    )

    # -------------------------------------------------------------------------
    # Build 37-feature set
    # -------------------------------------------------------------------------

    all_features = (
        production_features
        + ADDITIONAL_FEATURES
    )

    all_features = list(
        dict.fromkeys(all_features)
    )

    print(
        f"Experimental feature count: "
        f"{len(all_features)}"
    )

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    missing_features = [
        feature
        for feature in all_features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing features:\n"
            + "\n".join(missing_features)
        )

    # -------------------------------------------------------------------------
    # Fixed validation split
    # -------------------------------------------------------------------------

    X = df[all_features].replace([np.inf, -np.inf], np.nan).copy()
    y = df[TARGET].copy()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("\nValidation split:")
    print(f"Training rows:   {len(X_train):,}")
    print(f"Validation rows: {len(X_valid):,}")

    negative_count = (
        y_train == 0
    ).sum()

    positive_count = (
        y_train == 1
    ).sum()

    scale_pos_weight = (
        negative_count / positive_count
    )

    print(
        f"Scale positive weight: "
        f"{scale_pos_weight:.4f}"
    )

    # -------------------------------------------------------------------------
    # 37-feature baseline
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("BASELINE: ADDITIONAL FEATURES")
    print("=" * 80)

    baseline = evaluate_model(
        all_features,
        X_train,
        X_valid,
        y_train,
        y_valid,
        scale_pos_weight,
    )

    print(
        f"ROC-AUC: {baseline['roc_auc']:.6f}"
    )

    print(
        f"PR-AUC:  {baseline['pr_auc']:.6f}"
    )

    print(
        f"Precision @0.60: "
        f"{baseline['precision']:.6f}"
    )

    print(
        f"Recall @0.60:    "
        f"{baseline['recall']:.6f}"
    )

    print(
        f"F1 @0.60:        "
        f"{baseline['f1']:.6f}"
    )

    # -------------------------------------------------------------------------
    # One-feature-removal experiments
    # -------------------------------------------------------------------------

    results = []

    print()
    print("=" * 80)
    print("ONE-FEATURE-REMOVAL ABLATION")
    print("=" * 80)

    for feature in ADDITIONAL_FEATURES:

        test_features = [
            f
            for f in all_features
            if f != feature
        ]

        metrics = evaluate_model(
            test_features,
            X_train,
            X_valid,
            y_train,
            y_valid,
            scale_pos_weight,
        )

        roc_change = (
            metrics["roc_auc"]
            - baseline["roc_auc"]
        )

        pr_change = (
            metrics["pr_auc"]
            - baseline["pr_auc"]
        )

        precision_change = (
            metrics["precision"]
            - baseline["precision"]
        )

        recall_change = (
            metrics["recall"]
            - baseline["recall"]
        )

        f1_change = (
            metrics["f1"]
            - baseline["f1"]
        )

        results.append(
            {
                "removed_feature": feature,
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc_change": roc_change,
                "pr_auc_change": pr_change,
                "precision_change": precision_change,
                "recall_change": recall_change,
                "f1_change": f1_change,
            }
        )

        print()
        print(f"Removed: {feature}")
        print(
            f"  ROC-AUC: "
            f"{metrics['roc_auc']:.6f} "
            f"({roc_change:+.6f})"
        )
        print(
            f"  PR-AUC:  "
            f"{metrics['pr_auc']:.6f} "
            f"({pr_change:+.6f})"
        )
        print(
            f"  Precision@0.60: "
            f"{metrics['precision']:.6f} "
            f"({precision_change:+.6f})"
        )
        print(
            f"  Recall@0.60: "
            f"{metrics['recall']:.6f} "
            f"({recall_change:+.6f})"
        )
        print(
            f"  F1@0.60: "
            f"{metrics['f1']:.6f} "
            f"({f1_change:+.6f})"
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "f1_change"
    )

    print()
    print("=" * 80)
    print("ABLATION SUMMARY")
    print("=" * 80)

    print(
        results_df[
            [
                "removed_feature",
                "roc_auc_change",
                "pr_auc_change",
                "f1_change",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:+.6f}",
        )
    )

    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    print(
        "\nIf removing a feature causes F1 to decrease,"
    )
    print(
        "that feature is contributing positively "
        "to the experimental model."
    )

    print(
        "\nIf removing a feature causes F1 to increase,"
    )
    print(
        "that feature may be adding noise or redundancy."
    )

    print(
        "\nIMPORTANT:"
    )
    print(
        "This is an experiment only."
    )
    print(
        "The production model was NOT modified."
    )


if __name__ == "__main__":
    main()