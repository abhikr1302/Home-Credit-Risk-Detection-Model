from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


DATA_PATH = Path(
    "data/processed/model_features_train.parquet"
)

MODEL_PATH = Path(
    "models/final_credit_risk_model.joblib"
)

REPORT_DIRECTORY = Path("reports")
FIGURE_DIRECTORY = REPORT_DIRECTORY / "figures"

REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 2000
RANDOM_STATE = 42


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
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH.resolve()}"
        )

    data = pd.read_parquet(DATA_PATH)
    features = prepare_features(data)

    sample_size = min(
        SAMPLE_SIZE,
        len(features),
    )

    feature_sample = features.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    )

    model_pipeline = joblib.load(MODEL_PATH)

    preprocessor = model_pipeline.named_steps[
        "preprocessor"
    ]

    model = model_pipeline.named_steps["model"]

    transformed_sample = preprocessor.transform(
        feature_sample
    )

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    print("Calculating SHAP values...")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        transformed_sample
    )

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    mean_absolute_shap = np.abs(
        shap_values
    ).mean(axis=0)

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_absolute_shap": (
                mean_absolute_shap
            ),
        }
    ).sort_values(
        by="mean_absolute_shap",
        ascending=False,
    )

    importance.to_csv(
        REPORT_DIRECTORY /
        "shap_feature_importance.csv",
        index=False,
    )

    top_features = importance.head(20).sort_values(
        by="mean_absolute_shap",
        ascending=True,
    )

    plt.figure(figsize=(10, 8))

    plt.barh(
        top_features["feature"],
        top_features["mean_absolute_shap"],
    )

    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Feature")
    plt.title(
        "Top 20 Credit Risk Features"
    )
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIRECTORY /
        "shap_feature_importance.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print("\nTop 20 features:")
    print(
        importance.head(20).to_string(
            index=False
        )
    )

    print("\nSaved feature importance results.")


if __name__ == "__main__":
    main()