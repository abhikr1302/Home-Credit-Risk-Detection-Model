import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


DATA_PATH = Path(
    "data/processed/model_features_train.parquet"
)

MODEL_DIRECTORY = Path("models")
REPORT_DIRECTORY = Path("reports")

MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def main() -> None:
    data = pd.read_parquet(DATA_PATH)

    identifiers = data["sk_id_curr"].copy()
    target = data["target"].astype(int)

    features = data.drop(
        columns=[
            "target",
            "dataset_type",
            "sk_id_curr",
        ],
        errors="ignore",
    )

    numeric_columns = features.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = features.select_dtypes(
        exclude="number"
    ).columns.tolist()

    print("Rows:", len(features))
    print("Numeric columns:", len(numeric_columns))
    print("Categorical columns:", len(categorical_columns))

    (
        train_features,
        validation_features,
        train_target,
        validation_target,
        _,
        validation_identifiers,
    ) = train_test_split(
        features,
        target,
        identifiers,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
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
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ]
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model_pipeline.fit(
        train_features,
        train_target,
    )

    validation_probability = (
        model_pipeline.predict_proba(
            validation_features
        )[:, 1]
    )

    validation_prediction = (
        validation_probability >= 0.50
    ).astype(int)

    metrics = {
        "roc_auc": float(
            roc_auc_score(
                validation_target,
                validation_probability,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                validation_target,
                validation_probability,
            )
        ),
        "precision": float(
            precision_score(
                validation_target,
                validation_prediction,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                validation_target,
                validation_prediction,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                validation_target,
                validation_prediction,
                zero_division=0,
            )
        ),
        "confusion_matrix": (
            confusion_matrix(
                validation_target,
                validation_prediction,
            ).tolist()
        ),
    }

    print("\nMetrics:")
    print(json.dumps(metrics, indent=4))

    print("\nClassification report:")
    print(
        classification_report(
            validation_target,
            validation_prediction,
            zero_division=0,
        )
    )

    validation_results = pd.DataFrame(
        {
            "sk_id_curr": validation_identifiers,
            "actual_target": validation_target,
            "predicted_probability": (
                validation_probability
            ),
            "predicted_class": validation_prediction,
        }
    )

    validation_results.to_csv(
        REPORT_DIRECTORY /
        "baseline_validation_predictions.csv",
        index=False,
    )

    with open(
        REPORT_DIRECTORY /
        "baseline_metrics.json",
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(
            metrics,
            metrics_file,
            indent=4,
        )

    joblib.dump(
        model_pipeline,
        MODEL_DIRECTORY /
        "baseline_logistic_regression.joblib",
    )

    print("\nBaseline model saved successfully.")


if __name__ == "__main__":
    main()