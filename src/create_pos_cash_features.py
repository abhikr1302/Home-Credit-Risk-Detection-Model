from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CURRENT_FEATURES_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "model_features_credit_card_train.parquet"
)

POS_CASH_PATH = (
    PROJECT_ROOT / "data" / "raw"
    / "POS_CASH_balance.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "model_features_pos_cash_train.parquet"
)


def create_pos_cash_features() -> None:
    """Create applicant-level POS/CASH behavioral features."""

    print(
        f"Loading current features from: "
        f"{CURRENT_FEATURES_PATH}"
    )

    app = pd.read_parquet(CURRENT_FEATURES_PATH)

    print(
        f"Loading POS/CASH data from: "
        f"{POS_CASH_PATH}"
    )

    pos = pd.read_csv(POS_CASH_PATH)

    pos.columns = pos.columns.str.lower()

    print(f"POS/CASH data shape: {pos.shape}")

    # Delinquency indicators.
    pos["is_dpd"] = (
        pos["sk_dpd"] > 0
    ).astype(int)

    pos["is_dpd_30"] = (
        pos["sk_dpd"] > 30
    ).astype(int)

    pos["is_dpd_def"] = (
        pos["sk_dpd_def"] > 0
    ).astype(int)

    # Active / completed contract indicators.
    pos["is_active"] = (
        pos["name_contract_status"] == "Active"
    ).astype(int)

    pos["is_completed"] = (
        pos["name_contract_status"] == "Completed"
    ).astype(int)

    # Installment progress.
    pos["installment_progress"] = (
        1
        - (
            pos["cnt_instalment_future"]
            / pos["cnt_instalment"].replace(0, pd.NA)
        )
    )

    # Remaining installment ratio.
    pos["remaining_installment_ratio"] = (
        pos["cnt_instalment_future"]
        / pos["cnt_instalment"].replace(0, pd.NA)
    )

    pos_features = (
        pos.groupby("sk_id_curr")
        .agg(
            pos_cash_history_months=(
                "months_balance",
                "nunique",
            ),
            pos_cash_loan_count=(
                "sk_id_prev",
                "nunique",
            ),
            pos_cash_avg_installments=(
                "cnt_instalment",
                "mean",
            ),
            pos_cash_max_installments=(
                "cnt_instalment",
                "max",
            ),
            pos_cash_avg_future_installments=(
                "cnt_instalment_future",
                "mean",
            ),
            pos_cash_max_future_installments=(
                "cnt_instalment_future",
                "max",
            ),
            pos_cash_avg_installment_progress=(
                "installment_progress",
                "mean",
            ),
            pos_cash_avg_remaining_ratio=(
                "remaining_installment_ratio",
                "mean",
            ),
            pos_cash_avg_dpd=(
                "sk_dpd",
                "mean",
            ),
            pos_cash_max_dpd=(
                "sk_dpd",
                "max",
            ),
            pos_cash_dpd_count=(
                "is_dpd",
                "sum",
            ),
            pos_cash_dpd_30_count=(
                "is_dpd_30",
                "sum",
            ),
            pos_cash_dpd_def_count=(
                "is_dpd_def",
                "sum",
            ),
            pos_cash_dpd_rate=(
                "is_dpd",
                "mean",
            ),
            pos_cash_dpd_def_rate=(
                "is_dpd_def",
                "mean",
            ),
            pos_cash_active_count=(
                "is_active",
                "sum",
            ),
            pos_cash_completed_count=(
                "is_completed",
                "sum",
            ),
        )
        .reset_index()
    )

    pos_features = pos_features.replace(
        [float("inf"), float("-inf")],
        0,
    )

    numeric_cols = pos_features.select_dtypes(
        include=["number"]
    ).columns

    pos_features[numeric_cols] = (
        pos_features[numeric_cols].fillna(0)
    )

    print(
        f"POS/CASH feature shape: "
        f"{pos_features.shape}"
    )

    result = app.merge(
        pos_features,
        on="sk_id_curr",
        how="left",
    )

    feature_columns = [
        col
        for col in pos_features.columns
        if col != "sk_id_curr"
    ]

    result[feature_columns] = (
        result[feature_columns].fillna(0)
    )

    print(
        f"Final feature shape: {result.shape}"
    )

    print(
        "Missing values after POS/CASH "
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
        f"✓ POS/CASH features saved to: "
        f"{OUTPUT_PATH}"
    )

    print(f"Total rows: {len(result):,}")
    print(f"Total columns: {len(result.columns)}")


if __name__ == "__main__":
    create_pos_cash_features()
