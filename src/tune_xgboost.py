import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


DATA_PATH = Path(
    "data/processed/model_features_pos_cash_train.parquet"
)

REPORT_DIRECTORY = Path("reports")
REPORT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_STATE = 42


def main() -> None:
    print("Loading training data...")

    data = pd.read_parquet(DATA_PATH)

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
    print(
        "Numeric columns:",
        len(numeric_columns),
    )
    print(
        "Categorical columns:",
        len(categorical_columns),
    )

    (
        train_features,
        validation_features,
        train_target,
        validation_target,
    ) = train_test_split(
        features,
        target,
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
                    handle_unknown="ignore"
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

    # Controlled configurations.
    configurations = [
        {
            "name": "baseline_100_depth5",
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.10,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "min_child_weight": 1,
            "gamma": 0,
        },
        {
            "name": "depth4_200",
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1,
            "gamma": 0,
        },
        {
            "name": "depth5_200",
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1,
            "gamma": 0,
        },
        {
            "name": "depth6_200",
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1,
            "gamma": 0,
        },
        {
            "name": "depth5_regularized",
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 5,
            "gamma": 0.05,
        },
        {
            "name": "depth6_regularized",
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 5,
            "gamma": 0.05,
        },
    ]

    negative_count = (
        train_target == 0
    ).sum()

    positive_count = (
        train_target == 1
    ).sum()

    scale_pos_weight = (
        negative_count / positive_count
    )

    results = []

    for index, config in enumerate(
        configurations,
        start=1,
    ):
        print(
            f"\n[{index}/{len(configurations)}] "
            f"Training: {config['name']}"
        )

        model = XGBClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            subsample=config["subsample"],
            colsample_bytree=config[
                "colsample_bytree"
            ],
            min_child_weight=config[
                "min_child_weight"
            ],
            gamma=config["gamma"],
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            verbosity=0,
            eval_metric="auc",
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

        pipeline.fit(
            train_features,
            train_target,
        )

        probabilities = pipeline.predict_proba(
            validation_features
        )[:, 1]

        roc_auc = roc_auc_score(
            validation_target,
            probabilities,
        )

        pr_auc = average_precision_score(
            validation_target,
            probabilities,
        )

        result = {
            "name": config["name"],
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            **config,
        }

        results.append(result)

        print(
            f"ROC-AUC: {roc_auc:.6f}"
        )

        print(
            f"PR-AUC:  {pr_auc:.6f}"
        )

    results.sort(
        key=lambda x: x["roc_auc"],
        reverse=True,
    )

    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING RESULTS")
    print("=" * 60)

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{result['name']} | "
            f"ROC-AUC={result['roc_auc']:.6f} | "
            f"PR-AUC={result['pr_auc']:.6f}"
        )

    output_path = (
        REPORT_DIRECTORY
        / "xgboost_tuning_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    print(
        f"\n✓ Tuning results saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
