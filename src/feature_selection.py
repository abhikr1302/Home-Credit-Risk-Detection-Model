import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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

VALIDATION_SIZE = 0.20

# Candidate feature counts.
#
# The current engineered dataset contains 84 columns including
# ID and target, therefore the actual number of usable features
# is smaller than 84.
#
# The script automatically skips any candidate larger than the
# available feature count.

EVALUATION_FEATURE_COUNTS = [
    75,
    60,
    50,
    40,
    30,
]

# If two feature counts have nearly identical PR-AUC,
# prefer the smaller feature set.
PR_AUC_TOLERANCE = 0.002

# Absolute maximum number of features allowed in the final model.
# This prevents accidentally selecting an unnecessarily large
# feature set if future feature engineering adds many columns.
MAX_RECOMMENDED_FEATURES = 60


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

ID_COLUMN = "sk_id_curr"

TARGET_COLUMN = "target"


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset() -> pd.DataFrame:
    """
    Load the engineered training dataset.
    """

    if not MODEL_TRAIN_FEATURES_PATH.exists():

        raise FileNotFoundError(
            f"\nFeature dataset not found:\n"
            f"{MODEL_TRAIN_FEATURES_PATH}\n\n"
            f"Run feature_engineering.py first."
        )

    print(
        f"\nLoading feature dataset:\n"
        f"{MODEL_TRAIN_FEATURES_PATH}"
    )

    df = pd.read_parquet(
        MODEL_TRAIN_FEATURES_PATH
    )

    if df.empty:

        raise ValueError(
            "Feature dataset is empty."
        )

    return df


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataset(
    df: pd.DataFrame,
) -> None:
    """
    Validate the engineered dataset before feature selection.
    """

    print("\nValidating dataset...")

    required_columns = [
        ID_COLUMN,
        TARGET_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Target validation
    # --------------------------------------------------------

    if df[TARGET_COLUMN].isna().any():

        raise ValueError(
            "Target column contains missing values."
        )

    target_values = set(
        df[TARGET_COLUMN]
        .dropna()
        .unique()
    )

    if not target_values.issubset({0, 1}):

        raise ValueError(
            "Target column must contain only 0 and 1. "
            f"Found: {target_values}"
        )

    # --------------------------------------------------------
    # Duplicate ID check
    # --------------------------------------------------------

    duplicate_ids = (
        df[ID_COLUMN]
        .duplicated()
        .sum()
    )

    if duplicate_ids > 0:

        print(
            f"WARNING: {duplicate_ids:,} "
            "duplicate IDs detected."
        )

    # --------------------------------------------------------
    # Infinite value check
    # --------------------------------------------------------

    numeric_columns = (
        df.select_dtypes(
            include=["number"]
        )
        .columns
    )

    infinite_count = np.isinf(
        df[numeric_columns]
        .to_numpy()
    ).sum()

    if infinite_count > 0:

        print(
            f"WARNING: {infinite_count:,} "
            "infinite values detected."
        )

    else:

        print(
            "Infinite values: 0"
        )

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    target_distribution = (
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    print(
        "\nTarget distribution:"
    )

    for value, count in (
        target_distribution.items()
    ):

        percentage = (
            count / len(df) * 100
        )

        print(
            f"  Target {value}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

    print(
        "\nDataset validation completed."
    )


# ============================================================
# CLEAN DATA
# ============================================================

def clean_features(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Replace positive/negative infinity with NaN.

    NaN values are subsequently handled by the preprocessing
    pipeline using median/mode imputation.
    """

    X = X.copy()

    numeric_columns = (
        X.select_dtypes(
            include=["number"]
        )
        .columns
    )

    if len(numeric_columns) > 0:

        X[numeric_columns] = (
            X[numeric_columns]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

    return X


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:
    """
    Create preprocessing pipeline.

    Numeric:
        median imputation

    Categorical:
        most-frequent imputation
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
            include=[
                "object",
                "string",
                "category",
                "bool",
            ]
        )
        .columns
        .tolist()
    )

    transformers = []

    # --------------------------------------------------------
    # Numeric pipeline
    # --------------------------------------------------------

    if numeric_features:

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

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        )

    # --------------------------------------------------------
    # Categorical pipeline
    # --------------------------------------------------------

    if categorical_features:

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

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    # --------------------------------------------------------
    # Column transformer
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preprocessor


# ============================================================
# MODEL
# ============================================================

def create_model(
    y_train: pd.Series,
) -> XGBClassifier:
    """
    Create XGBoost classifier.

    scale_pos_weight is calculated dynamically from the
    training target distribution to handle class imbalance.
    """

    negative_count = int(
        (y_train == 0).sum()
    )

    positive_count = int(
        (y_train == 1).sum()
    )

    if positive_count == 0:

        raise ValueError(
            "Training data contains no positive target samples."
        )

    scale_pos_weight = (
        negative_count /
        positive_count
    )

    print(
        f"Scale positive weight: "
        f"{scale_pos_weight:.4f}"
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
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float = 0.50,
) -> dict:
    """
    Calculate classification metrics.

    ROC-AUC and PR-AUC use probabilities.

    Precision, recall and F1 use the supplied threshold.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
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

    return metrics


# ============================================================
# TRAIN IMPORTANCE MODEL
# ============================================================

def train_importance_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    """
    Train an XGBoost model on all available features and return
    transformed feature importance.

    This model is used ONLY for feature ranking.
    """

    print(
        "\nCreating preprocessing pipeline "
        "for feature importance..."
    )

    preprocessor = create_preprocessor(
        X_train
    )

    model = create_model(
        y_train
    )

    print(
        "Transforming training features..."
    )

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    print(
        f"Transformed feature matrix shape: "
        f"{X_train_processed.shape}"
    )

    print(
        "\nTraining XGBoost importance model..."
    )

    model.fit(
        X_train_processed,
        y_train,
    )

    # --------------------------------------------------------
    # Transformed feature names
    # --------------------------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importances = (
        model.feature_importances_
    )

    if len(feature_names) != len(importances):

        raise ValueError(
            "Feature name count does not match "
            "feature importance count."
        )

    importance_df = pd.DataFrame(
        {
            "transformed_feature": feature_names,
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

    return (
        importance_df,
        preprocessor,
        model,
    )


# ============================================================
# ORIGINAL FEATURE NAME EXTRACTION
# ============================================================

def get_original_feature_name(
    transformed_feature: str,
) -> str:
    """
    Convert a transformed sklearn feature name back to its
    original dataframe column name.

    Examples
    --------
    numeric__amt_income_total
        -> amt_income_total

    categorical__name_gender_M
        -> name_gender

    categorical__credit_active_ACTIVE
        -> credit_active
    """

    feature = transformed_feature

    # --------------------------------------------------------
    # Remove transformer prefix
    # --------------------------------------------------------

    if "__" in feature:

        feature = feature.split(
            "__",
            1,
        )[1]

    return feature


# ============================================================
# AGGREGATE ORIGINAL FEATURE IMPORTANCE
# ============================================================

def aggregate_feature_importance(
    importance_df: pd.DataFrame,
    original_features: list,
) -> pd.DataFrame:
    """
    Aggregate transformed feature importance back to the
    original dataframe feature level.

    This is important for categorical variables because one
    original categorical column may produce multiple
    one-hot encoded columns.
    """

    records = []

    transformed_features = (
        importance_df[
            "transformed_feature"
        ]
        .tolist()
    )

    transformed_importances = (
        importance_df[
            "importance"
        ]
        .tolist()
    )

    # --------------------------------------------------------
    # Match every transformed feature to exactly one original
    # feature.
    #
    # We use the known original feature names rather than
    # splitting on underscores.
    # --------------------------------------------------------

    for original_feature in original_features:

        total_importance = 0.0

        # Numeric feature example:
        #
        # numeric__amt_income_total
        #
        # Categorical example:
        #
        # categorical__credit_active_ACTIVE
        #
        # We therefore remove the transformer prefix first.
        #

        for (
            transformed_feature,
            importance,
        ) in zip(
            transformed_features,
            transformed_importances,
        ):

            cleaned_name = (
                get_original_feature_name(
                    transformed_feature
                )
            )

            # Exact numeric feature match.
            if cleaned_name == original_feature:

                total_importance += float(
                    importance
                )

                continue

            # One-hot encoded feature:
            #
            # credit_active_ACTIVE
            #
            # should map to:
            #
            # credit_active
            #
            prefix = (
                original_feature + "_"
            )

            if cleaned_name.startswith(
                prefix
            ):

                total_importance += float(
                    importance
                )

        records.append(
            {
                "feature": original_feature,
                "importance": total_importance,
            }
        )

    result = pd.DataFrame(
        records
    )

    result = (
        result
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return result


# ============================================================
# EVALUATE FEATURE COUNT
# ============================================================

def evaluate_feature_count(
    feature_count: int,
    original_importance_df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
) -> dict:
    """
    Train and evaluate a model using the top N original
    features.
    """

    selected_features = (
        original_importance_df
        .head(feature_count)
        ["feature"]
        .tolist()
    )

    X_train_selected = (
        X_train[
            selected_features
        ]
    )

    X_valid_selected = (
        X_valid[
            selected_features
        ]
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

    metrics[
        "feature_count"
    ] = feature_count

    return metrics


# ============================================================
# SELECT BEST FEATURE COUNT
# ============================================================

def select_best_feature_count(
    results: list,
) -> int:
    """
    Select the final feature count.

    Primary metric:
        PR-AUC

    Secondary metric:
        ROC-AUC

    If a smaller feature set is within PR_AUC_TOLERANCE of the
    best PR-AUC, prefer the smaller feature set.

    A maximum feature limit is also applied.
    """

    if not results:

        raise ValueError(
            "No feature-count evaluation results available."
        )

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Only consider feature sets within the configured maximum.
    # --------------------------------------------------------

    eligible = results_df[
        results_df["feature_count"]
        <= MAX_RECOMMENDED_FEATURES
    ].copy()

    if eligible.empty:

        eligible = results_df.copy()

    # --------------------------------------------------------
    # Find highest PR-AUC.
    # --------------------------------------------------------

    best_pr_auc = (
        eligible["pr_auc"]
        .max()
    )

    # --------------------------------------------------------
    # Keep feature counts that are very close to the best.
    # --------------------------------------------------------

    near_best = eligible[
        eligible["pr_auc"]
        >= (
            best_pr_auc -
            PR_AUC_TOLERANCE
        )
    ].copy()

    # --------------------------------------------------------
    # Prefer fewer features among near-equal models.
    #
    # If feature count is equal, higher ROC-AUC wins.
    # --------------------------------------------------------

    near_best = (
        near_best
        .sort_values(
            [
                "feature_count",
                "roc_auc",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    best_feature_count = int(
        near_best.iloc[0][
            "feature_count"
        ]
    )

    return best_feature_count


# ============================================================
# SAVE JSON
# ============================================================

def save_json_report(
    report: dict,
) -> None:
    """
    Save feature-selection results as JSON.
    """

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


# ============================================================
# SAVE TXT REPORT
# ============================================================

def save_txt_report(
    recommended_features: list,
    results: list,
    importance_df: pd.DataFrame,
    selected_feature_count: int,
) -> None:
    """
    Save human-readable feature-selection report.
    """

    RECOMMENDED_FEATURES_TXT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
            "=" * 70 + "\n\n"
        )

        file.write(
            f"Recommended feature count: "
            f"{selected_feature_count}\n\n"
        )

        file.write(
            "RECOMMENDED FEATURES\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        for index, feature in enumerate(
            recommended_features,
            start=1,
        ):

            file.write(
                f"{index:02d}. {feature}\n"
            )

        file.write(
            "\n\nFEATURE IMPORTANCE RANKING\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        for index, row in (
            enumerate(
                importance_df
                .head(75)
                .to_dict("records"),
                start=1,
            )
        ):

            file.write(
                f"{index:02d}. "
                f"{row['feature']} : "
                f"{row['importance']:.8f}\n"
            )

        file.write(
            "\n\nFEATURE COUNT EVALUATION\n"
        )

        file.write(
            "=" * 70 + "\n"
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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "HOME CREDIT FEATURE SELECTION"
    )
    print("=" * 70)

    # ========================================================
    # LOAD
    # ========================================================

    df = load_dataset()

    print(
        f"\nDataset shape: "
        f"{df.shape}"
    )

    # ========================================================
    # VALIDATE
    # ========================================================

    validate_dataset(
        df
    )

    # ========================================================
    # IDENTIFY FEATURES
    # ========================================================

    feature_columns = [
        column
        for column in df.columns
        if column not in [
            ID_COLUMN,
            TARGET_COLUMN,
        ]
    ]

    if not feature_columns:

        raise ValueError(
            "No usable features found."
        )

    print(
        f"\nAvailable original features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Display newly engineered bank-style features
    # --------------------------------------------------------

    bank_features = [
        "monthly_gross_income",
        "loan_annuity_to_monthly_income_ratio",
        "bureau_debt_to_annual_income_ratio",
        "bureau_overdue_to_annual_income_ratio",
        "bureau_credit_to_annual_income_ratio",
        "active_bureau_account_ratio",
        "credit_card_balance_to_monthly_income_ratio",
    ]

    available_bank_features = [
        feature
        for feature in bank_features
        if feature in feature_columns
    ]

    print(
        "\nBank-style engineered features available:"
    )

    for feature in available_bank_features:

        print(
            f"  [OK] {feature}"
        )

    missing_bank_features = [
        feature
        for feature in bank_features
        if feature not in feature_columns
    ]

    if missing_bank_features:

        print(
            "\nWARNING: Some bank-style features "
            "are missing:"
        )

        for feature in missing_bank_features:

            print(
                f"  [MISSING] {feature}"
            )

    # ========================================================
    # CREATE X / Y
    # ========================================================

    X = df[
        feature_columns
    ].copy()

    y = df[
        TARGET_COLUMN
    ].copy()

    # ========================================================
    # CLEAN INFINITE VALUES
    # ========================================================

    X = clean_features(
        X
    )

    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    print(
        "\nCreating stratified train/validation split..."
    )

    (
        X_train,
        X_valid,
        y_train,
        y_valid,
    ) = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(
        f"Training rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(X_valid):,}"
    )

    # ========================================================
    # TRAIN IMPORTANCE MODEL
    # ========================================================

    (
        importance_df,
        _,
        _,
    ) = train_importance_model(
        X_train,
        y_train,
    )

    # ========================================================
    # AGGREGATE ORIGINAL FEATURE IMPORTANCE
    # ========================================================

    print(
        "\nAggregating feature importance "
        "to original features..."
    )

    original_importance_df = (
        aggregate_feature_importance(
            importance_df,
            feature_columns,
        )
    )

    # --------------------------------------------------------
    # Validate importance results
    # --------------------------------------------------------

    if len(
        original_importance_df
    ) != len(feature_columns):

        raise ValueError(
            "Original feature importance count does not "
            "match available feature count."
        )

    print(
        "\nTop 20 features:"
    )

    print(
        original_importance_df
        .head(20)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # EVALUATE FEATURE COUNTS
    # ========================================================

    results = []

    print(
        "\n" + "=" * 70
    )

    print(
        "FEATURE COUNT EVALUATION"
    )

    print(
        "=" * 70
    )

    for requested_count in (
        EVALUATION_FEATURE_COUNTS
    ):

        # ----------------------------------------------------
        # Skip if greater than available feature count.
        # ----------------------------------------------------

        if (
            requested_count
            > len(feature_columns)
        ):

            print(
                f"\nSkipping Top "
                f"{requested_count}: "
                f"only "
                f"{len(feature_columns)} "
                f"features available."
            )

            continue

        print(
            "\n" + "-" * 70
        )

        print(
            f"Evaluating TOP "
            f"{requested_count} FEATURES"
        )

        print(
            "-" * 70
        )

        metrics = (
            evaluate_feature_count(
                requested_count,
                original_importance_df,
                X_train,
                X_valid,
                y_train,
                y_valid,
            )
        )

        results.append(
            metrics
        )

        print(
            f"ROC-AUC : "
            f"{metrics['roc_auc']:.6f}"
        )

        print(
            f"PR-AUC  : "
            f"{metrics['pr_auc']:.6f}"
        )

        print(
            f"Precision: "
            f"{metrics['precision']:.6f}"
        )

        print(
            f"Recall  : "
            f"{metrics['recall']:.6f}"
        )

        print(
            f"F1      : "
            f"{metrics['f1']:.6f}"
        )

    if not results:

        raise ValueError(
            "No feature-count evaluations were completed."
        )

    # ========================================================
    # SELECT BEST FEATURE COUNT
    # ========================================================

    selected_feature_count = (
        select_best_feature_count(
            results
        )
    )

    recommended_features = (
        original_importance_df
        .head(selected_feature_count)
        ["feature"]
        .tolist()
    )

    if (
        len(recommended_features)
        != selected_feature_count
    ):

        raise ValueError(
            "Final feature count does not match "
            "selected feature count."
        )

    # ========================================================
    # FIND SELECTED BANK FEATURES
    # ========================================================

    selected_bank_features = [
        feature
        for feature in recommended_features
        if feature in bank_features
    ]

    # ========================================================
    # BUILD REPORT
    # ========================================================

    report = {
        "selection_method":
            "XGBoost feature importance",

        "dataset_shape": [
            int(df.shape[0]),
            int(df.shape[1]),
        ],

        "available_feature_count":
            int(len(feature_columns)),

        "recommended_feature_count":
            int(selected_feature_count),

        "selection_metric":
            "PR-AUC",

        "pr_auc_tolerance":
            PR_AUC_TOLERANCE,

        "max_recommended_features":
            MAX_RECOMMENDED_FEATURES,

        "recommended_features":
            recommended_features,

        "selected_bank_style_features":
            selected_bank_features,

        "feature_importance":
            original_importance_df
            .to_dict(
                orient="records"
            ),

        "evaluation_results":
            results,

        "random_state":
            RANDOM_STATE,

        "validation_size":
            VALIDATION_SIZE,
    }

    # ========================================================
    # SAVE JSON
    # ========================================================

    save_json_report(
        report
    )

    # ========================================================
    # SAVE TXT
    # ========================================================

    save_txt_report(
        recommended_features,
        results,
        original_importance_df,
        selected_feature_count,
    )

    # ========================================================
    # DISPLAY FINAL FEATURES
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL RECOMMENDED FEATURE SET"
    )

    print(
        "=" * 70
    )

    print(
        f"\nRecommended feature count: "
        f"{selected_feature_count}"
    )

    print(
        "\nFeatures:"
    )

    for index, feature in enumerate(
        recommended_features,
        start=1,
    ):

        marker = (
            " [BANK-RISK]"
            if feature in bank_features
            else ""
        )

        print(
            f"{index:02d}. "
            f"{feature}"
            f"{marker}"
        )

    # ========================================================
    # DISPLAY EVALUATION SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FEATURE COUNT COMPARISON"
    )

    print(
        "=" * 70
    )

    results_df = (
        pd.DataFrame(
            results
        )
        .sort_values(
            "feature_count"
        )
    )

    print(
        results_df[
            [
                "feature_count",
                "roc_auc",
                "pr_auc",
                "precision",
                "recall",
                "f1",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FEATURE SELECTION COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        "\nSaved JSON:"
    )

    print(
        RECOMMENDED_FEATURES_JSON
    )

    print(
        "\nSaved TXT:"
    )

    print(
        RECOMMENDED_FEATURES_TXT
    )

    print(
        "\nNext step:"
    )

    print(
        "Run train_model.py using the generated "
        "recommended_features.json."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
