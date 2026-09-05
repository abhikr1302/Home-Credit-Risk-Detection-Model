"""
Final Home Credit XGBoost model training.

Pipeline:

Raw feature dataset
        |
        v
50 selected features
        |
        v
Train / validation split
        |
        v
ColumnTransformer
    |             |
    v             v
Numeric        Categorical
Median         Most frequent
imputation     +
               OneHotEncoder
    |             |
    +------v------+
           |
           v
       XGBoost
           |
           v
       Evaluation
           |
           v
Complete Joblib Pipeline


Outputs:

models/xgboost_home_credit_pipeline.joblib

reports/final_model_metrics.json

reports/final_model_features.json
"""

import json
from pathlib import Path

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

from config import (
    MODEL_TRAIN_FEATURES_PATH,
    RECOMMENDED_FEATURES_JSON,
    MODEL_PATH,
    FINAL_MODEL_METRICS_PATH,
    FINAL_MODEL_FEATURES_PATH,
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

EXPECTED_FEATURE_COUNT = 50


# ============================================================
# LOAD DATA
# ============================================================

def load_training_data():

    if not MODEL_TRAIN_FEATURES_PATH.exists():

        raise FileNotFoundError(
            f"Training feature dataset not found:\n"
            f"{MODEL_TRAIN_FEATURES_PATH}\n\n"
            f"Run feature_engineering.py first."
        )

    df = pd.read_parquet(
        MODEL_TRAIN_FEATURES_PATH
    )

    return df


# ============================================================
# LOAD SELECTED FEATURES
# ============================================================

def load_selected_features():

    if not RECOMMENDED_FEATURES_JSON.exists():

        raise FileNotFoundError(
            f"Recommended feature file not found:\n"
            f"{RECOMMENDED_FEATURES_JSON}\n\n"
            f"Run feature_selection.py first."
        )

    with open(
        RECOMMENDED_FEATURES_JSON,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    features = data[
        "recommended_features"
    ]

    if len(features) != EXPECTED_FEATURE_COUNT:

        raise ValueError(
            f"Expected "
            f"{EXPECTED_FEATURE_COUNT} "
            f"features but found "
            f"{len(features)}."
        )

    return features


# ============================================================
# CREATE PREPROCESSOR
# ============================================================

def create_preprocessor(
    X_train: pd.DataFrame,
):

    numeric_features = (
        X_train
        .select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    categorical_features = (
        X_train
        .select_dtypes(
            include=[
                "object",
                "category",
                "bool",
            ]
        )
        .columns
        .tolist()
    )

    print(
        f"Numeric features: "
        f"{len(numeric_features)}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_features)}"
    )

    # --------------------------------------------------------
    # Numeric preprocessing
    # --------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            )
        ]
    )

    # --------------------------------------------------------
    # Categorical preprocessing
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # Combined preprocessor
    # --------------------------------------------------------

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


# ============================================================
# CREATE XGBOOST MODEL
# ============================================================

def create_xgboost_model(
    y_train: pd.Series,
):

    negative_count = (
        y_train == 0
    ).sum()

    positive_count = (
        y_train == 1
    ).sum()

    if positive_count == 0:

        raise ValueError(
            "Training data contains "
            "no positive samples."
        )

    scale_pos_weight = (
        negative_count /
        positive_count
    )

    print(
        f"Negative samples: "
        f"{negative_count:,}"
    )

    print(
        f"Positive samples: "
        f"{positive_count:,}"
    )

    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.6f}"
    )

    model = XGBClassifier(

        n_estimators=300,

        max_depth=5,

        learning_rate=0.05,

        subsample=0.80,

        colsample_bytree=0.80,

        min_child_weight=5,

        reg_alpha=0.10,

        reg_lambda=1.0,

        objective="binary:logistic",

        eval_metric="auc",

        scale_pos_weight=scale_pos_weight,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        tree_method="hist",
    )

    return model


# ============================================================
# BUILD COMPLETE PIPELINE
# ============================================================

def build_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
):

    preprocessor = create_preprocessor(
        X_train
    )

    model = create_xgboost_model(
        y_train
    )

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

    return pipeline


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    pipeline,
    X_valid,
    y_valid,
):

    probabilities = (
        pipeline.predict_proba(
            X_valid
        )[:, 1]
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    roc_auc = roc_auc_score(
        y_valid,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_valid,
        probabilities,
    )

    precision = precision_score(
        y_valid,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_valid,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_valid,
        predictions,
        zero_division=0,
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_valid,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {

        "roc_auc": float(
            roc_auc
        ),

        "pr_auc": float(
            pr_auc
        ),

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1": float(
            f1
        ),

        "confusion_matrix": {

            "true_negative": int(
                tn
            ),

            "false_positive": int(
                fp
            ),

            "false_negative": int(
                fn
            ),

            "true_positive": int(
                tp
            ),
        },

        "validation_rows": int(
            len(y_valid)
        ),
    }

    return metrics


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    data,
    path: Path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
        )


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    print("=" * 70)
    print("HOME CREDIT FINAL MODEL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_training_data()

    print(
        f"\nRows: "
        f"{len(df):,}"
    )

    print(
        f"Columns: "
        f"{len(df.columns)}"
    )

    # --------------------------------------------------------
    # Validate target
    # --------------------------------------------------------

    if "target" not in df.columns:

        raise ValueError(
            "Target column is missing."
        )

    # --------------------------------------------------------
    # Display target distribution
    # --------------------------------------------------------

    target_counts = (
        df["target"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nTarget distribution:"
    )

    for target, count in (
        target_counts.items()
    ):

        percentage = (
            count /
            len(df)
            * 100
        )

        print(
            f"Target {target}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # Load selected features
    # --------------------------------------------------------

    selected_features = (
        load_selected_features()
    )

    print(
        f"\nSelected features: "
        f"{len(selected_features)}"
    )

    # --------------------------------------------------------
    # Validate selected features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in selected_features
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "The following selected "
            "features are missing:\n"
            + "\n".join(
                missing_features
            )
        )

    # --------------------------------------------------------
    # Prepare X and y
    # --------------------------------------------------------

    X = df[
        selected_features
    ].copy()

    y = df[
        "target"
    ].copy()

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    (
        X_train,
        X_valid,
        y_train,
        y_valid,
    ) = train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        stratify=y,

        random_state=RANDOM_STATE,
    )

    print(
        f"\nTraining rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(X_valid):,}"
    )

    # --------------------------------------------------------
    # Build pipeline
    # --------------------------------------------------------

    print(
        "\nBuilding preprocessing "
        "and model pipeline..."
    )

    pipeline = build_pipeline(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print(
        "\nTraining XGBoost..."
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    print(
        "Training completed."
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print(
        "\nEvaluating model..."
    )

    metrics = evaluate_model(
        pipeline,
        X_valid,
        y_valid,
    )

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL MODEL RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"ROC-AUC: "
        f"{metrics['roc_auc']:.6f}"
    )

    print(
        f"PR-AUC: "
        f"{metrics['pr_auc']:.6f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.6f}"
    )

    print(
        f"Recall: "
        f"{metrics['recall']:.6f}"
    )

    print(
        f"F1 Score: "
        f"{metrics['f1']:.6f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    cm = metrics[
        "confusion_matrix"
    ]

    print(
        f"TN: "
        f"{cm['true_negative']:,}"
    )

    print(
        f"FP: "
        f"{cm['false_positive']:,}"
    )

    print(
        f"FN: "
        f"{cm['false_negative']:,}"
    )

    print(
        f"TP: "
        f"{cm['true_positive']:,}"
    )

    # --------------------------------------------------------
    # Add metadata
    # --------------------------------------------------------

    metrics["model"] = (
        "XGBClassifier"
    )

    metrics["feature_count"] = (
        len(selected_features)
    )

    metrics["features"] = (
        selected_features
    )

    metrics["random_state"] = (
        RANDOM_STATE
    )

    metrics["test_size"] = (
        TEST_SIZE
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        MODEL_PATH,
    )

    print(
        f"\nModel saved to:"
    )

    print(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    save_json(
        metrics,
        FINAL_MODEL_METRICS_PATH,
    )

    print(
        "\nMetrics saved to:"
    )

    print(
        FINAL_MODEL_METRICS_PATH
    )

    # --------------------------------------------------------
    # Save final feature list
    # --------------------------------------------------------

    feature_report = {

        "feature_count":
            len(selected_features),

        "features":
            selected_features,

        "model":
            "XGBClassifier",

        "model_path":
            str(MODEL_PATH),
    }

    save_json(
        feature_report,
        FINAL_MODEL_FEATURES_PATH,
    )

    print(
        "\nFeature list saved to:"
    )

    print(
        FINAL_MODEL_FEATURES_PATH
    )

    # --------------------------------------------------------
    # Reload model verification
    # --------------------------------------------------------

    print(
        "\nVerifying saved model..."
    )

    loaded_pipeline = joblib.load(
        MODEL_PATH
    )

    if (
        "preprocessor"
        not in loaded_pipeline.named_steps
    ):

        raise ValueError(
            "Saved pipeline does not "
            "contain preprocessor."
        )

    if (
        "model"
        not in loaded_pipeline.named_steps
    ):

        raise ValueError(
            "Saved pipeline does not "
            "contain model."
        )

    print(
        "Preprocessor:",
        type(
            loaded_pipeline
            .named_steps[
                "preprocessor"
            ]
        ).__name__,
    )

    print(
        "Model:",
        type(
            loaded_pipeline
            .named_steps[
                "model"
            ]
        ).__name__,
    )

    print(
        f"Expected input features: "
        f"{len(selected_features)}"
    )

    print(
        "\nModel reload verification "
        "passed."
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL MODEL READY FOR FASTAPI"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()