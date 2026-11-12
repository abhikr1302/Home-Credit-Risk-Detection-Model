"""
Model training pipeline.

This script:

1. Loads the engineered feature dataset.
2. Loads the existing 30-feature recommendation as the BASELINE.
3. Builds an ENHANCED feature set using all engineered backend features.
4. Uses the exact same train/validation split for both models.
5. Trains both models with identical XGBoost parameters.
6. Compares:
      - ROC-AUC
      - PR-AUC
      - Precision
      - Recall
      - F1
7. Evaluates multiple decision thresholds.
8. Selects the enhanced model only when it provides a measurable
   improvement over the baseline.
9. Saves the selected experimental model separately from production.
10. Saves metrics and feature-selection information.
"""

import json
from pathlib import Path

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
    MODEL_TRAIN_FEATURES_PATH,
    RECOMMENDED_FEATURES_JSON,
    MODEL_PATH,
    FINAL_MODEL_METRICS_PATH,
    FINAL_MODEL_FEATURES_PATH,
)


# ============================================================
# SETTINGS
# ============================================================

# Experimental comparisons must never replace the 30-feature serving pipeline.
MODEL_PATH = MODEL_PATH.parent / "experimental_credit_risk_pipeline.joblib"
FINAL_MODEL_METRICS_PATH = FINAL_MODEL_METRICS_PATH.parent / "experimental_model_metrics.json"
FINAL_MODEL_FEATURES_PATH = FINAL_MODEL_FEATURES_PATH.parent / "experimental_model_features.json"

RANDOM_STATE = 42

VALIDATION_SIZE = 0.20

TARGET_COLUMN = "target"

ID_COLUMN = "sk_id_curr"

# Existing recommended feature count.
# This is used only for the BASELINE model.
EXPECTED_BASELINE_FEATURE_COUNT = 30

# Decision thresholds.
THRESHOLDS = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
]

# Primary metric for comparing models.
PRIMARY_METRIC = "roc_auc"

# Secondary metric.
SECONDARY_METRIC = "pr_auc"


# ============================================================
# XGBOOST SETTINGS
# ============================================================

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


# ============================================================
# GENERAL HELPERS
# ============================================================

def print_section(title):
    """Print a clearly formatted section heading."""

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def clean_features(df):
    """
    Replace infinite values with NaN.

    Missing values are handled later by the sklearn preprocessing
    pipeline.
    """

    df = df.copy()

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return df


def save_json(path, data):
    """Save JSON with readable formatting."""

    path = Path(path)

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
            default=str,
        )


def load_recommended_features(path):
    """
    Load recommended feature names.

    Supports either:

    1. A plain list:
       ["feature_1", "feature_2", ...]

    2. A dictionary containing:
       {
           "features": [...]
       }

    3. A dictionary containing:
       {
           "selected_features": [...]
       }
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Recommended features file not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if isinstance(data, list):
        features = data

    elif isinstance(data, dict):
        if "recommended_features" in data:
            features = data["recommended_features"]
        elif "features" in data:
            features = data["features"]

        elif "selected_features" in data:
            features = data["selected_features"]

        else:
            raise ValueError(
                "Recommended features JSON does not contain "
                "'features' or 'selected_features'."
            )

    else:
        raise ValueError(
            "Recommended features JSON must contain a list "
            "or dictionary."
        )

    features = [
        str(feature)
        for feature in features
        if feature is not None
    ]

    return features


# ============================================================
# DATA LOADING
# ============================================================

def load_training_data():
    """Load the engineered model training dataset."""

    path = Path(MODEL_TRAIN_FEATURES_PATH)

    print_section("LOADING ENGINEERED DATA")

    print(f"Training dataset:")
    print(path)

    if not path.exists():
        raise FileNotFoundError(
            f"\nTraining feature file was not found:\n{path}\n\n"
            "Run feature engineering first."
        )

    df = pd.read_parquet(path)

    print()
    print(f"Dataset shape: {df.shape}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    if ID_COLUMN not in df.columns:
        raise ValueError(
            f"ID column '{ID_COLUMN}' not found."
        )

    return df


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataset(df):
    """Validate the engineered dataset before training."""

    print_section("DATA VALIDATION")

    # --------------------------------------------------------
    # Duplicate IDs
    # --------------------------------------------------------

    duplicate_ids = df[ID_COLUMN].duplicated().sum()

    print(
        f"Duplicate applicant IDs: {duplicate_ids}"
    )

    if duplicate_ids > 0:
        raise ValueError(
            "Duplicate applicant IDs detected."
        )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    target_counts = (
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    print()
    print("Target distribution:")

    for value, count in target_counts.items():
        print(
            f"  target={value}: {count:,}"
        )

    if len(target_counts) != 2:
        raise ValueError(
            "Target must contain exactly two classes."
        )

    negative_count = int(
        target_counts.get(0, 0)
    )

    positive_count = int(
        target_counts.get(1, 0)
    )

    if positive_count == 0:
        raise ValueError(
            "No positive/default observations found."
        )

    default_rate = (
        positive_count /
        (positive_count + negative_count)
    )

    print()
    print(
        f"Default rate: {default_rate:.4%}"
    )

    # --------------------------------------------------------
    # Infinite values
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    infinite_count = np.isinf(
        df[numeric_columns].to_numpy(
            dtype=float,
            na_value=np.nan,
        )
    ).sum()

    print(
        f"Infinite numeric values: {infinite_count:,}"
    )

    # Infinite values are cleaned later rather than failing
    # immediately.

    return


# ============================================================
# FEATURE VALIDATION
# ============================================================

def validate_baseline_features(
    df,
    baseline_features,
):
    """
    Validate the existing recommended feature set.
    """

    print_section("BASELINE FEATURE VALIDATION")

    print(
        f"Expected baseline features: "
        f"{EXPECTED_BASELINE_FEATURE_COUNT}"
    )

    print(
        f"Features found in JSON: "
        f"{len(baseline_features)}"
    )

    if len(baseline_features) != EXPECTED_BASELINE_FEATURE_COUNT:
        raise ValueError(
            "\nThe recommended_features.json file does not "
            f"contain exactly {EXPECTED_BASELINE_FEATURE_COUNT} "
            "features.\n"
            f"Found: {len(baseline_features)}"
        )

    forbidden = {
        TARGET_COLUMN,
        ID_COLUMN,
    }

    invalid_features = [
        feature
        for feature in baseline_features
        if feature in forbidden
    ]

    if invalid_features:
        raise ValueError(
            f"Baseline contains invalid features: "
            f"{invalid_features}"
        )

    missing_features = [
        feature
        for feature in baseline_features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "\nThe following baseline features are missing "
            "from the engineered dataset:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing_features
            )
        )

    duplicate_features = (
        pd.Series(baseline_features)
        .duplicated()
    )

    if duplicate_features.any():
        raise ValueError(
            "Duplicate features detected in baseline."
        )

    print()
    print("Baseline feature validation: PASSED")

    return


def build_enhanced_features(
    df,
    baseline_features,
):
    """
    Build the enhanced backend feature set.

    The enhanced model uses every engineered model feature
    except:

        - target
        - applicant ID

    This allows the new backend features to participate in
    model training without exposing them individually in the
    Streamlit UI.
    """

    all_candidate_features = [
        column
        for column in df.columns
        if column not in {
            TARGET_COLUMN,
            ID_COLUMN,
        }
    ]

    # Preserve dataset order.
    enhanced_features = list(
        dict.fromkeys(all_candidate_features)
    )

    if len(enhanced_features) <= len(baseline_features):
        raise ValueError(
            "Enhanced feature set is not larger than "
            "the baseline feature set."
        )

    new_backend_features = [
        feature
        for feature in enhanced_features
        if feature not in baseline_features
    ]

    print_section("ENHANCED FEATURE SET")

    print(
        f"Baseline feature count: "
        f"{len(baseline_features)}"
    )

    print(
        f"Enhanced feature count: "
        f"{len(enhanced_features)}"
    )

    print(
        f"Additional backend features: "
        f"{len(new_backend_features)}"
    )

    print()
    print("Additional backend features:")

    for feature in new_backend_features:
        print(
            f"  + {feature}"
        )

    return (
        enhanced_features,
        new_backend_features,
    )


# ============================================================
# PREPROCESSING
# ============================================================

def build_preprocessor(X):
    """
    Build preprocessing pipeline.

    Numeric:
        median imputation

    Categorical:
        most-frequent imputation
        one-hot encoding

    This allows the same pipeline to be saved together with
    the model and used later by FastAPI.
    """

    numeric_features = list(
        X.select_dtypes(
            include=[np.number]
        ).columns
    )

    categorical_features = list(
        X.select_dtypes(
            exclude=[np.number]
        ).columns
    )

    print()
    print(
        f"Numeric features: "
        f"{len(numeric_features)}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_features)}"
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
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
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preprocessor


# ============================================================
# MODEL PIPELINE
# ============================================================

def build_model_pipeline(
    X_train,
    y_train,
):
    """
    Build complete preprocessing + XGBoost pipeline.
    """

    preprocessor = build_preprocessor(
        X_train
    )

    negative_count = int(
        (y_train == 0).sum()
    )

    positive_count = int(
        (y_train == 1).sum()
    )

    if positive_count == 0:
        raise ValueError(
            "Training split contains no positive cases."
        )

    scale_pos_weight = (
        negative_count /
        positive_count
    )

    print()
    print(
        f"Training negatives: "
        f"{negative_count:,}"
    )

    print(
        f"Training positives: "
        f"{positive_count:,}"
    )

    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    model_params = XGB_PARAMS.copy()

    model_params[
        "scale_pos_weight"
    ] = scale_pos_weight

    classifier = XGBClassifier(
        **model_params
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

    return pipeline


# ============================================================
# TRAINING
# ============================================================

def train_model(
    model_name,
    df,
    feature_list,
    train_indices,
    validation_indices,
):
    """
    Train one model using a fixed train/validation split.

    Both baseline and enhanced models use exactly the same
    applicant rows in train and validation sets.
    """

    print_section(
        f"TRAINING {model_name.upper()} MODEL"
    )

    X = df[feature_list].copy()

    y = df[TARGET_COLUMN].copy()

    X = clean_features(X)

    X_train = X.iloc[
        train_indices
    ]

    X_validation = X.iloc[
        validation_indices
    ]

    y_train = y.iloc[
        train_indices
    ]

    y_validation = y.iloc[
        validation_indices
    ]

    print(
        f"Features: {len(feature_list)}"
    )

    print(
        f"Training rows: {len(X_train):,}"
    )

    print(
        f"Validation rows: {len(X_validation):,}"
    )

    model = build_model_pipeline(
        X_train,
        y_train,
    )

    print()
    print("Fitting model...")

    model.fit(
        X_train,
        y_train,
    )

    print("Training completed.")

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    validation_probability = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    return (
        model,
        y_validation,
        validation_probability,
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model_name,
    y_validation,
    probabilities,
):
    """
    Evaluate model using threshold-independent and
    threshold-dependent metrics.
    """

    print_section(
        f"{model_name.upper()} MODEL EVALUATION"
    )

    roc_auc = roc_auc_score(
        y_validation,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_validation,
        probabilities,
    )

    # Default 0.50 threshold.
    predictions_050 = (
        probabilities >= 0.50
    ).astype(int)

    precision_050 = precision_score(
        y_validation,
        predictions_050,
        zero_division=0,
    )

    recall_050 = recall_score(
        y_validation,
        predictions_050,
        zero_division=0,
    )

    f1_050 = f1_score(
        y_validation,
        predictions_050,
        zero_division=0,
    )

    cm_050 = confusion_matrix(
        y_validation,
        predictions_050,
    )

    print(
        f"ROC-AUC : {roc_auc:.6f}"
    )

    print(
        f"PR-AUC  : {pr_auc:.6f}"
    )

    print(
        f"Precision @ 0.50: {precision_050:.6f}"
    )

    print(
        f"Recall    @ 0.50: {recall_050:.6f}"
    )

    print(
        f"F1        @ 0.50: {f1_050:.6f}"
    )

    print()
    print("Confusion Matrix @ 0.50:")

    print(cm_050)

    # --------------------------------------------------------
    # Threshold evaluation
    # --------------------------------------------------------

    threshold_results = []

    print()
    print(
        "Threshold analysis:"
    )

    print(
        "-" * 75
    )

    print(
        f"{'Threshold':>10} "
        f"{'Precision':>12} "
        f"{'Recall':>12} "
        f"{'F1':>12}"
    )

    print(
        "-" * 75
    )

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        threshold_results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

        print(
            f"{threshold:>10.2f} "
            f"{precision:>12.6f} "
            f"{recall:>12.6f} "
            f"{f1:>12.6f}"
        )

    # --------------------------------------------------------
    # Best F1 threshold
    # --------------------------------------------------------

    best_threshold_result = max(
        threshold_results,
        key=lambda item: (
            item["f1"],
            item["recall"],
        ),
    )

    print()
    print(
        "Best F1 threshold:"
    )

    print(
        f"  Threshold : "
        f"{best_threshold_result['threshold']:.2f}"
    )

    print(
        f"  Precision : "
        f"{best_threshold_result['precision']:.6f}"
    )

    print(
        f"  Recall    : "
        f"{best_threshold_result['recall']:.6f}"
    )

    print(
        f"  F1        : "
        f"{best_threshold_result['f1']:.6f}"
    )

    metrics = {
        "model_name": model_name,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision_at_0_50": float(
            precision_050
        ),
        "recall_at_0_50": float(
            recall_050
        ),
        "f1_at_0_50": float(
            f1_050
        ),
        "confusion_matrix_at_0_50": (
            cm_050.tolist()
        ),
        "threshold_results": (
            threshold_results
        ),
        "best_f1_threshold": (
            best_threshold_result
        ),
    }

    return metrics


# ============================================================
# MODEL COMPARISON
# ============================================================

def compare_models(
    baseline_metrics,
    enhanced_metrics,
):
    """
    Compare baseline and enhanced models.

    Primary comparison:
        ROC-AUC

    Secondary comparison:
        PR-AUC

    The enhanced model is selected only when its ROC-AUC
    improves, or when ROC-AUC is effectively tied and PR-AUC
    improves.
    """

    print_section(
        "BASELINE VS ENHANCED MODEL"
    )

    baseline_roc = baseline_metrics[
        "roc_auc"
    ]

    enhanced_roc = enhanced_metrics[
        "roc_auc"
    ]

    baseline_pr = baseline_metrics[
        "pr_auc"
    ]

    enhanced_pr = enhanced_metrics[
        "pr_auc"
    ]

    roc_delta = (
        enhanced_roc -
        baseline_roc
    )

    pr_delta = (
        enhanced_pr -
        baseline_pr
    )

    print(
        f"{'Metric':<20}"
        f"{'Baseline':>15}"
        f"{'Enhanced':>15}"
        f"{'Delta':>15}"
    )

    print("-" * 65)

    print(
        f"{'ROC-AUC':<20}"
        f"{baseline_roc:>15.6f}"
        f"{enhanced_roc:>15.6f}"
        f"{roc_delta:>15.6f}"
    )

    print(
        f"{'PR-AUC':<20}"
        f"{baseline_pr:>15.6f}"
        f"{enhanced_pr:>15.6f}"
        f"{pr_delta:>15.6f}"
    )

    print(
        f"{'Precision @ 0.50':<20}"
        f"{baseline_metrics['precision_at_0_50']:>15.6f}"
        f"{enhanced_metrics['precision_at_0_50']:>15.6f}"
        f"{enhanced_metrics['precision_at_0_50'] - baseline_metrics['precision_at_0_50']:>15.6f}"
    )

    print(
        f"{'Recall @ 0.50':<20}"
        f"{baseline_metrics['recall_at_0_50']:>15.6f}"
        f"{enhanced_metrics['recall_at_0_50']:>15.6f}"
        f"{enhanced_metrics['recall_at_0_50'] - baseline_metrics['recall_at_0_50']:>15.6f}"
    )

    print(
        f"{'F1 @ 0.50':<20}"
        f"{baseline_metrics['f1_at_0_50']:>15.6f}"
        f"{enhanced_metrics['f1_at_0_50']:>15.6f}"
        f"{enhanced_metrics['f1_at_0_50'] - baseline_metrics['f1_at_0_50']:>15.6f}"
    )

    # --------------------------------------------------------
    # Model selection
    # --------------------------------------------------------

    # Use a tiny tolerance so that insignificant floating-point
    # differences do not determine the model.
    tolerance = 1e-6

    if enhanced_roc > (
        baseline_roc + tolerance
    ):

        selected_model = "enhanced"

    elif (
        abs(enhanced_roc - baseline_roc)
        <= tolerance
        and
        enhanced_pr > (
            baseline_pr + tolerance
        )
    ):

        selected_model = "enhanced"

    else:

        selected_model = "baseline"

    print()
    print(
        f"Selected model: "
        f"{selected_model.upper()}"
    )

    if selected_model == "enhanced":
        print(
            "The enhanced feature set produced a "
            "measurable improvement based on the "
            "comparison criteria."
        )

    else:
        print(
            "The enhanced feature set did not produce "
            "a measurable improvement over the baseline "
            "on this validation split."
        )

    return selected_model


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def extract_feature_importance(
    model,
    top_n=30,
):
    """
    Extract XGBoost feature importance after preprocessing.

    Because categorical variables are one-hot encoded, the
    resulting importance names may contain encoded category
    values.
    """

    try:

        preprocessor = model.named_steps[
            "preprocessor"
        ]

        classifier = model.named_steps[
            "classifier"
        ]

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        importances = (
            classifier.feature_importances_
        )

        if len(feature_names) != len(importances):
            return []

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
            .head(top_n)
        )

        return [
            {
                "feature": row["feature"],
                "importance": float(
                    row["importance"]
                ),
            }
            for _, row in importance_df.iterrows()
        ]

    except Exception as exc:

        print()
        print(
            "Feature importance extraction skipped:"
        )

        print(exc)

        return []


# ============================================================
# SAVE MODEL
# ============================================================

def save_final_model(
    model,
    selected_model_name,
    selected_features,
    metrics,
    baseline_features,
    enhanced_features,
    new_backend_features,
):
    """
    Save final production model and metadata.
    """

    print_section(
        "SAVING FINAL MODEL"
    )

    model_path = Path(
        MODEL_PATH
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"Model saved to:\n"
        f"{model_path}"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    final_metrics = {
        "selected_model": selected_model_name,
        "primary_metric": PRIMARY_METRIC,
        "secondary_metric": SECONDARY_METRIC,
        "metrics": metrics,
        "random_state": RANDOM_STATE,
        "validation_size": VALIDATION_SIZE,
        "xgboost_parameters": XGB_PARAMS,
    }

    save_json(
        FINAL_MODEL_METRICS_PATH,
        final_metrics,
    )

    print()
    print(
        f"Metrics saved to:\n"
        f"{FINAL_MODEL_METRICS_PATH}"
    )

    # --------------------------------------------------------
    # Feature report
    # --------------------------------------------------------

    feature_importance = (
        extract_feature_importance(
            model,
            top_n=30,
        )
    )

    feature_report = {
        "selected_model": selected_model_name,

        "feature_count": len(
            selected_features
        ),

        "features": selected_features,

        "baseline_feature_count": len(
            baseline_features
        ),

        "baseline_features": baseline_features,

        "enhanced_feature_count": len(
            enhanced_features
        ),

        "enhanced_features": enhanced_features,

        "new_backend_feature_count": len(
            new_backend_features
        ),

        "new_backend_features": (
            new_backend_features
        ),

        "feature_importance_top_30": (
            feature_importance
        ),

        "selection_source": str(
            RECOMMENDED_FEATURES_JSON
        ),

        "model": selected_model_name,

        "random_state": RANDOM_STATE,
    }

    save_json(
        FINAL_MODEL_FEATURES_PATH,
        feature_report,
    )

    print()
    print(
        f"Feature report saved to:\n"
        f"{FINAL_MODEL_FEATURES_PATH}"
    )


# ============================================================
# MODEL RELOAD TEST
# ============================================================

def test_saved_model(
    model_path,
    df,
    selected_features,
):
    """
    Reload the saved model and test predictions.

    This verifies that the complete preprocessing +
    XGBoost pipeline can be serialized and loaded correctly.
    """

    print_section(
        "TESTING SAVED MODEL"
    )

    model_path = Path(
        model_path
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found:\n"
            f"{model_path}"
        )

    loaded_model = joblib.load(
        model_path
    )

    sample = (
        df[selected_features]
        .head(5)
        .copy()
    )

    sample = clean_features(
        sample
    )

    probabilities = (
        loaded_model
        .predict_proba(sample)[:, 1]
    )

    print(
        "Sample predictions:"
    )

    for index, probability in enumerate(
        probabilities,
        start=1,
    ):

        print(
            f"  Applicant {index}: "
            f"{probability:.6f}"
        )

    print()
    print(
        "Saved model reload test: PASSED"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print(
        "HOME CREDIT RISK MODEL TRAINING"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    df = load_training_data()

    # --------------------------------------------------------
    # 2. Validate
    # --------------------------------------------------------

    validate_dataset(
        df
    )

    # --------------------------------------------------------
    # 3. Load baseline feature list
    # --------------------------------------------------------

    baseline_features = (
        load_recommended_features(
            RECOMMENDED_FEATURES_JSON
        )
    )

    validate_baseline_features(
        df,
        baseline_features,
    )

    # --------------------------------------------------------
    # 4. Build enhanced feature set
    # --------------------------------------------------------

    (
        enhanced_features,
        new_backend_features,
    ) = build_enhanced_features(
        df,
        baseline_features,
    )

    # --------------------------------------------------------
    # 5. Clean dataset
    # --------------------------------------------------------

    df = df.copy()

    df = clean_features(
        df
    )

    # --------------------------------------------------------
    # 6. Create ONE fixed validation split
    # --------------------------------------------------------
    #
    # This is important.
    #
    # Both models must be evaluated on exactly the same
    # applicants. Otherwise the comparison would not be fair.
    #

    print_section(
        "CREATING FIXED TRAIN / VALIDATION SPLIT"
    )

    all_indices = np.arange(
        len(df)
    )

    train_indices, validation_indices = (
        train_test_split(
            all_indices,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=df[TARGET_COLUMN],
        )
    )

    print(
        f"Training applicants: "
        f"{len(train_indices):,}"
    )

    print(
        f"Validation applicants: "
        f"{len(validation_indices):,}"
    )

    # --------------------------------------------------------
    # 7. Train BASELINE model
    # --------------------------------------------------------

    (
        baseline_model,
        baseline_y_validation,
        baseline_probabilities,
    ) = train_model(
        model_name="baseline",
        df=df,
        feature_list=baseline_features,
        train_indices=train_indices,
        validation_indices=validation_indices,
    )

    baseline_metrics = evaluate_model(
        model_name="baseline",
        y_validation=baseline_y_validation,
        probabilities=baseline_probabilities,
    )

    # --------------------------------------------------------
    # 8. Train ENHANCED model
    # --------------------------------------------------------

    (
        enhanced_model,
        enhanced_y_validation,
        enhanced_probabilities,
    ) = train_model(
        model_name="enhanced",
        df=df,
        feature_list=enhanced_features,
        train_indices=train_indices,
        validation_indices=validation_indices,
    )

    enhanced_metrics = evaluate_model(
        model_name="enhanced",
        y_validation=enhanced_y_validation,
        probabilities=enhanced_probabilities,
    )

    # --------------------------------------------------------
    # 9. Compare models
    # --------------------------------------------------------

    selected_model_name = compare_models(
        baseline_metrics=baseline_metrics,
        enhanced_metrics=enhanced_metrics,
    )

    # --------------------------------------------------------
    # 10. Select model
    # --------------------------------------------------------

    if selected_model_name == "enhanced":

        selected_model = enhanced_model

        selected_features = enhanced_features

        selected_metrics = enhanced_metrics

    else:

        selected_model = baseline_model

        selected_features = baseline_features

        selected_metrics = baseline_metrics

    # --------------------------------------------------------
    # 11. Save comparison metrics
    # --------------------------------------------------------

    comparison_metrics = {
        "baseline": baseline_metrics,
        "enhanced": enhanced_metrics,

        "selected_model": selected_model_name,

        "roc_auc_delta": (
            enhanced_metrics["roc_auc"]
            -
            baseline_metrics["roc_auc"]
        ),

        "pr_auc_delta": (
            enhanced_metrics["pr_auc"]
            -
            baseline_metrics["pr_auc"]
        ),

        "feature_count_delta": (
            len(enhanced_features)
            -
            len(baseline_features)
        ),
    }

    save_json(
        Path(FINAL_MODEL_METRICS_PATH)
        .parent
        / "model_comparison_metrics.json",
        comparison_metrics,
    )

    print()
    print(
        "Model comparison saved to:"
    )

    print(
        Path(FINAL_MODEL_METRICS_PATH)
        .parent
        / "model_comparison_metrics.json"
    )

    # --------------------------------------------------------
    # 12. Save final model
    # --------------------------------------------------------

    save_final_model(
        model=selected_model,
        selected_model_name=selected_model_name,
        selected_features=selected_features,
        metrics=selected_metrics,
        baseline_features=baseline_features,
        enhanced_features=enhanced_features,
        new_backend_features=new_backend_features,
    )

    # --------------------------------------------------------
    # 13. Reload test
    # --------------------------------------------------------

    test_saved_model(
        model_path=MODEL_PATH,
        df=df,
        selected_features=selected_features,
    )

    # --------------------------------------------------------
    # 14. Final summary
    # --------------------------------------------------------

    print_section(
        "FINAL TRAINING SUMMARY"
    )

    print(
        f"Baseline features : "
        f"{len(baseline_features)}"
    )

    print(
        f"Enhanced features : "
        f"{len(enhanced_features)}"
    )

    print(
        f"Additional backend features : "
        f"{len(new_backend_features)}"
    )

    print()
    print(
        f"Baseline ROC-AUC : "
        f"{baseline_metrics['roc_auc']:.6f}"
    )

    print(
        f"Enhanced ROC-AUC : "
        f"{enhanced_metrics['roc_auc']:.6f}"
    )

    print(
        f"ROC-AUC change   : "
        f"{comparison_metrics['roc_auc_delta']:+.6f}"
    )

    print()
    print(
        f"Baseline PR-AUC  : "
        f"{baseline_metrics['pr_auc']:.6f}"
    )

    print(
        f"Enhanced PR-AUC  : "
        f"{enhanced_metrics['pr_auc']:.6f}"
    )

    print(
        f"PR-AUC change    : "
        f"{comparison_metrics['pr_auc_delta']:+.6f}"
    )

    print()
    print(
        f"FINAL MODEL      : "
        f"{selected_model_name.upper()}"
    )

    print()
    print(
        "Model training completed successfully."
    )

    print(
        f"Experimental model:"
    )

    print(
        MODEL_PATH
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
