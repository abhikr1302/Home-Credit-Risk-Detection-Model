"""
Feature selection for Home Credit Risk Detection.

Evaluates:

- Full feature set
- Top 60 features
- Top 50 features
- Top 40 features

The final recommended feature count is fixed at 50.

Outputs:

    reports/recommended_features.json
    reports/recommended_features.txt
"""

import json
from pathlib import Path

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
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from config import (
    MODEL_TRAIN_FEATURES_PATH,
    RECOMMENDED_FEATURES_JSON,
    RECOMMENDED_FEATURES_TXT,
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

RECOMMENDED_FEATURE_COUNT = 50

EVALUATION_FEATURE_COUNTS = [
    76,
    60,
    50,
    40,
]


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

ID_COLUMN = "sk_id_curr"

TARGET_COLUMN = "target"


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset() -> pd.DataFrame:

    if not MODEL_TRAIN_FEATURES_PATH.exists():

        raise FileNotFoundError(
            f"Feature dataset not found:\n"
            f"{MODEL_TRAIN_FEATURES_PATH}\n\n"
            f"Run feature_engineering.py first."
        )

    df = pd.read_parquet(
        MODEL_TRAIN_FEATURES_PATH
    )

    return df


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor(
    X: pd.DataFrame,
):
    """
    Create preprocessing pipeline.

    Numeric:
        median imputation

    Categorical:
        most frequent imputation
        one-hot encoding
    """

    numeric_features = (
        X.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    categorical_features = (
    X.select_dtypes(
        include=["object", "string", "category", "bool"]
    )
    .columns
    .tolist()
    
    )

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
# MODEL
# ============================================================

def create_model(
    y_train: pd.Series,
) -> XGBClassifier:

    negative_count = (
        y_train == 0
    ).sum()

    positive_count = (
        y_train == 1
    ).sum()

    scale_pos_weight = (
        negative_count /
        positive_count
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
# EVALUATION
# ============================================================

def evaluate_predictions(
    y_true,
    probabilities,
):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
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


# ============================================================
# TRAIN FEATURE IMPORTANCE MODEL
# ============================================================

def train_importance_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
):

    preprocessor = create_preprocessor(
        X_train
    )

    model = create_model(
        y_train
    )

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    model.fit(
        X_train_processed,
        y_train,
    )

    # --------------------------------------------------------
    # Get transformed feature names
    # --------------------------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importances = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return importance_df


# ============================================================
# MAIN FEATURE SELECTION
# ============================================================

def main():

    print("=" * 70)
    print("HOME CREDIT FEATURE SELECTION")
    print("=" * 70)

    df = load_dataset()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Missing target column: "
            f"{TARGET_COLUMN}"
        )

    if ID_COLUMN not in df.columns:
        raise ValueError(
            f"Missing ID column: "
            f"{ID_COLUMN}"
        )

    feature_columns = [
        col
        for col in df.columns
        if col not in [
            ID_COLUMN,
            TARGET_COLUMN,
        ]
    ]

    X = df[feature_columns].copy()

    y = df[TARGET_COLUMN].copy()

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Available features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Train validation split
    # --------------------------------------------------------

    X_train, X_valid, y_train, y_valid = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    print(
        "\nTraining importance model..."
    )

    importance_df = train_importance_model(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Aggregate one-hot feature importance
    # --------------------------------------------------------

    # For categorical features, multiple encoded columns
    # correspond to the same original feature.
    importance_df["original_feature"] = (
        importance_df["feature"]
        .str.replace(
            r"^(numeric|categorical)__",
            "",
            regex=True,
        )
        .str.split("_")
        .str[0]
    )

    # Instead of relying only on the transformed feature
    # names, calculate importance for original columns by
    # prefix matching.

    original_importance = []

    for feature in feature_columns:

        matching = importance_df[
            importance_df["feature"]
            .str.contains(
                feature,
                regex=False,
            )
        ]

        importance = (
            matching["importance"].sum()
        )

        original_importance.append(
            {
                "feature": feature,
                "importance": float(
                    importance
                ),
            }
        )

    original_importance_df = (
        pd.DataFrame(
            original_importance
        )
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Evaluate different feature counts
    # --------------------------------------------------------

    results = []

    for feature_count in EVALUATION_FEATURE_COUNTS:

        if feature_count > len(
            feature_columns
        ):
            continue

        selected_features = (
            original_importance_df
            .head(feature_count)
            ["feature"]
            .tolist()
        )

        print(
            f"\nEvaluating top "
            f"{feature_count} features..."
        )

        X_train_selected = (
            X_train[selected_features]
        )

        X_valid_selected = (
            X_valid[selected_features]
        )

        preprocessor = create_preprocessor(
            X_train_selected
        )

        model = create_model(
            y_train
        )

        X_train_processed = (
            preprocessor.fit_transform(
                X_train_selected
            )
        )

        X_valid_processed = (
            preprocessor.transform(
                X_valid_selected
            )
        )

        model.fit(
            X_train_processed,
            y_train,
        )

        probabilities = (
            model.predict_proba(
                X_valid_processed
            )[:, 1]
        )

        metrics = evaluate_predictions(
            y_valid,
            probabilities,
        )

        metrics["feature_count"] = (
            feature_count
        )

        results.append(
            metrics
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
            f"F1: "
            f"{metrics['f1']:.6f}"
        )

    # --------------------------------------------------------
    # Select final 50
    # --------------------------------------------------------

    recommended_features = (
        original_importance_df
        .head(RECOMMENDED_FEATURE_COUNT)
        ["feature"]
        .tolist()
    )

    if len(
        recommended_features
    ) != RECOMMENDED_FEATURE_COUNT:

        raise ValueError(
            "Unable to select exactly "
            f"{RECOMMENDED_FEATURE_COUNT} "
            "features."
        )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    report = {
        "recommended_feature_count":
            RECOMMENDED_FEATURE_COUNT,

        "recommended_features":
            recommended_features,

        "feature_importance":
            original_importance_df
            .to_dict(
                orient="records"
            ),

        "evaluation_results":
            results,
    }

    RECOMMENDED_FEATURES_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RECOMMENDED_FEATURES_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Save TXT
    # --------------------------------------------------------

    with open(
        RECOMMENDED_FEATURES_TXT,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "HOME CREDIT "
            "RECOMMENDED FEATURES\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Feature count: "
            f"{len(recommended_features)}\n\n"
        )

        for index, feature in enumerate(
            recommended_features,
            start=1,
        ):

            file.write(
                f"{index:02d}. {feature}\n"
            )

        file.write(
            "\n\nEVALUATION RESULTS\n"
        )

        file.write(
            "=" * 60 + "\n"
        )

        for result in results:

            file.write(
                f"\nTop "
                f"{result['feature_count']} features\n"
            )

            file.write(
                f"ROC-AUC: "
                f"{result['roc_auc']:.6f}\n"
            )

            file.write(
                f"PR-AUC: "
                f"{result['pr_auc']:.6f}\n"
            )

            file.write(
                f"Precision: "
                f"{result['precision']:.6f}\n"
            )

            file.write(
                f"Recall: "
                f"{result['recall']:.6f}\n"
            )

            file.write(
                f"F1: "
                f"{result['f1']:.6f}\n"
            )

    # --------------------------------------------------------
    # Display final features
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        f"RECOMMENDED FEATURES "
        f"({len(recommended_features)})"
    )
    print("=" * 70)

    for index, feature in enumerate(
        recommended_features,
        start=1,
    ):

        print(
            f"{index:02d}. {feature}"
        )

    print(
        "\nSaved:"
    )

    print(
        RECOMMENDED_FEATURES_JSON
    )

    print(
        RECOMMENDED_FEATURES_TXT
    )


if __name__ == "__main__":
    main()