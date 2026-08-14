from pathlib import Path

import joblib
import pandas as pd


TRAIN_PATH = Path(
    "data/processed/model_features_train.parquet"
)

TEST_PATH = Path(
    "data/processed/model_features_test.parquet"
)

MODEL_PATH = Path(
    "models/xgboost_model.joblib"
)

FINAL_MODEL_PATH = Path(
    "models/final_credit_risk_model.joblib"
)

SUBMISSION_PATH = Path(
    "reports/home_credit_submission.csv"
)


def prepare_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    return data.drop(
        columns=[
            "sk_id_curr",
            "target",
            "dataset_type",
        ],
        errors="ignore",
    )


def main() -> None:
    train_data = pd.read_parquet(TRAIN_PATH)
    test_data = pd.read_parquet(TEST_PATH)

    train_features = prepare_features(train_data)
    test_features = prepare_features(test_data)

    train_target = train_data["target"].astype(int)
    test_identifiers = test_data["sk_id_curr"]

    model_pipeline = joblib.load(MODEL_PATH)

    print("Retraining selected model on all training data...")

    model_pipeline.fit(
        train_features,
        train_target,
    )

    test_probability = (
        model_pipeline.predict_proba(
            test_features
        )[:, 1]
    )

    submission = pd.DataFrame(
        {
            "SK_ID_CURR": test_identifiers.astype(int),
            "TARGET": test_probability,
        }
    )

    submission.to_csv(
        SUBMISSION_PATH,
        index=False,
    )

    joblib.dump(
        model_pipeline,
        FINAL_MODEL_PATH,
    )

    print("Submission shape:", submission.shape)
    print(submission.head())
    print("Saved:", SUBMISSION_PATH)
    print("Saved:", FINAL_MODEL_PATH)


if __name__ == "__main__":
    main()