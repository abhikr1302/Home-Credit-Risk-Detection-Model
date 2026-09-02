import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


DATA_PATH = Path(
    "data/processed/model_features_advanced_train.parquet"
)

MODEL_DIRECTORY = Path("models")
REPORT_DIRECTORY = Path("reports")

MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def main():

    print("Loading advanced training data...")

    data = pd.read_parquet(DATA_PATH)

    target = data["target"].astype(int)
    identifiers = data["sk_id_curr"].copy()

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
        len(numeric_columns)
    )
    print(
        "Categorical columns:",
        len(categorical_columns)
    )

    (
        X_train,
        X_valid,
        y_train,
        y_valid,
        _,
        valid_ids,
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

    print("\nFitting preprocessing...")

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_valid_processed = preprocessor.transform(
        X_valid
    )

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = (
        negative_count / positive_count
    )

    print(
        "scale_pos_weight:",
        scale_pos_weight
    )

    experiments = [

        {
            "name": "lr_003_depth6",
            "n_estimators": 2000,
            "max_depth": 6,
            "learning_rate": 0.03,
            "min_child_weight": 5,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "gamma": 0.05,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },

        {
            "name": "lr_003_depth7",
            "n_estimators": 2000,
            "max_depth": 7,
            "learning_rate": 0.03,
            "min_child_weight": 5,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "gamma": 0.05,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },

        {
            "name": "lr_005_depth6",
            "n_estimators": 1500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_weight": 5,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "gamma": 0.05,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },

        {
            "name": "regularized",
            "n_estimators": 2000,
            "max_depth": 6,
            "learning_rate": 0.03,
            "min_child_weight": 8,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "gamma": 0.10,
            "reg_alpha": 0.10,
            "reg_lambda": 2.0,
        },

        {
            "name": "depth7_regularized",
            "n_estimators": 2000,
            "max_depth": 7,
            "learning_rate": 0.03,
            "min_child_weight": 8,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "gamma": 0.10,
            "reg_alpha": 0.10,
            "reg_lambda": 2.0,
        },
    ]

    results = []

    best_auc = -1
    best_model = None
    best_config = None

    for index, config in enumerate(
        experiments,
        start=1,
    ):

        print(
            f"\n[{index}/{len(experiments)}] "
            f"Training {config['name']}..."
        )

        model = XGBClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            min_child_weight=config[
                "min_child_weight"
            ],
            subsample=config["subsample"],
            colsample_bytree=config[
                "colsample_bytree"
            ],
            gamma=config["gamma"],
            reg_alpha=config["reg_alpha"],
            reg_lambda=config["reg_lambda"],
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="auc",
            early_stopping_rounds=100,
        )

        model.fit(
            X_train_processed,
            y_train,
            eval_set=[
                (
                    X_valid_processed,
                    y_valid,
                )
            ],
            verbose=False,
        )

        probability = model.predict_proba(
            X_valid_processed
        )[:, 1]

        roc_auc = roc_auc_score(
            y_valid,
            probability,
        )

        pr_auc = average_precision_score(
            y_valid,
            probability,
        )

        best_iteration = (
            getattr(
                model,
                "best_iteration",
                None,
            )
        )

        result = {
            "name": config["name"],
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "best_iteration": (
                int(best_iteration)
                if best_iteration is not None
                else None
            ),
            "parameters": config,
        }

        results.append(result)

        print(
            f"ROC-AUC: {roc_auc:.6f}"
        )

        print(
            f"PR-AUC:  {pr_auc:.6f}"
        )

        print(
            f"Best iteration: "
            f"{best_iteration}"
        )

        if roc_auc > best_auc:

            best_auc = roc_auc
            best_model = model
            best_config = result

    print(
        "\n"
        + "=" * 60
    )

    print(
        "EARLY STOPPING RESULTS"
    )

    print(
        "=" * 60
    )

    results.sort(
        key=lambda x: x["roc_auc"],
        reverse=True,
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"{rank}. "
            f"{result['name']} | "
            f"ROC-AUC="
            f"{result['roc_auc']:.6f} | "
            f"PR-AUC="
            f"{result['pr_auc']:.6f} | "
            f"Best iteration="
            f"{result['best_iteration']}"
        )

    final_model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                best_model,
            ),
        ]
    )

    model_path = (
        MODEL_DIRECTORY
        / "xgboost_early_stopping_model.joblib"
    )

    joblib.dump(
        final_model,
        model_path,
    )

    report_path = (
        REPORT_DIRECTORY
        / "xgboost_early_stopping_results.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "best_model": best_config,
                "all_results": results,
            },
            file,
            indent=4,
        )

    print(
        "\n✓ Best model saved to:",
        model_path,
    )

    print(
        "✓ Results saved to:",
        report_path,
    )


if __name__ == "__main__":
    main()
