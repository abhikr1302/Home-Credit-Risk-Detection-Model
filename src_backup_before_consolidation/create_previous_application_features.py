from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

APPLICATION_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "model_features_bureau_train.parquet"
)

PREVIOUS_PATH = (
    PROJECT_ROOT / "data" / "raw"
    / "previous_application.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "model_features_history_train.parquet"
)


def create_previous_application_features() -> None:
    """Add previous-application history features."""

    print(f"Loading current features from: {APPLICATION_PATH}")
    app = pd.read_parquet(APPLICATION_PATH)

    print(f"Loading previous applications from: {PREVIOUS_PATH}")
    previous = pd.read_csv(PREVIOUS_PATH)

    previous.columns = previous.columns.str.lower()

    print(f"Previous application shape: {previous.shape}")

    # Prevent division-by-zero.
    previous["credit_application_ratio"] = (
        previous["amt_credit"]
        / previous["amt_application"].replace(0, pd.NA)
    )

    # Applicant-level historical application features.
    previous_features = (
        previous.groupby("sk_id_curr")
        .agg(
            previous_total_applications=(
                "sk_id_prev",
                "count",
            ),
            previous_approved_count=(
                "name_contract_status",
                lambda x: (x == "Approved").sum(),
            ),
            previous_refused_count=(
                "name_contract_status",
                lambda x: (x == "Refused").sum(),
            ),
            previous_canceled_count=(
                "name_contract_status",
                lambda x: (x == "Canceled").sum(),
            ),
            previous_unused_offer_count=(
                "name_contract_status",
                lambda x: (x == "Unused offer").sum(),
            ),
            previous_avg_application_amount=(
                "amt_application",
                "mean",
            ),
            previous_avg_credit_amount=(
                "amt_credit",
                "mean",
            ),
            previous_max_credit_amount=(
                "amt_credit",
                "max",
            ),
            previous_avg_annuity=(
                "amt_annuity",
                "mean",
            ),
            previous_avg_down_payment=(
                "amt_down_payment",
                "mean",
            ),
            previous_avg_credit_application_ratio=(
                "credit_application_ratio",
                "mean",
            ),
            previous_avg_installments=(
                "cnt_payment",
                "mean",
            ),
            previous_contract_type_count=(
                "name_contract_type",
                "nunique",
            ),
            previous_client_type_count=(
                "name_client_type",
                "nunique",
            ),
        )
        .reset_index()
    )

    # Derived behavioral rates.
    previous_features["previous_approval_rate"] = (
        previous_features["previous_approved_count"]
        / previous_features["previous_total_applications"]
    )

    previous_features["previous_refusal_rate"] = (
        previous_features["previous_refused_count"]
        / previous_features["previous_total_applications"]
    )

    # Replace invalid values.
    previous_features = previous_features.replace(
        [float("inf"), float("-inf")],
        0,
    )

    numeric_cols = previous_features.select_dtypes(
        include=["number"]
    ).columns

    previous_features[numeric_cols] = (
        previous_features[numeric_cols].fillna(0)
    )

    print(
        f"Previous-application feature shape: "
        f"{previous_features.shape}"
    )

    # Merge historical features into the application + Bureau dataset.
    result = app.merge(
        previous_features,
        on="sk_id_curr",
        how="left",
    )

    feature_columns = [
        col
        for col in previous_features.columns
        if col != "sk_id_curr"
    ]

    result[feature_columns] = (
        result[feature_columns].fillna(0)
    )

    print(f"Final feature shape: {result.shape}")

    print(
        "Missing values after previous-application "
        "feature engineering:",
        result.isna().sum().sum(),
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"✓ Previous-application features saved to: "
        f"{OUTPUT_PATH}"
    )

    print(f"Total rows: {len(result):,}")
    print(f"Total columns: {len(result.columns)}")


if __name__ == "__main__":
    create_previous_application_features()
