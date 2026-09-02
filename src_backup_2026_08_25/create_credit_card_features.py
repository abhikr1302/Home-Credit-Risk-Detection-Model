from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

APPLICATION_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "model_features_repayment_train.parquet"
)

CREDIT_CARD_PATH = (
    PROJECT_ROOT / "data" / "raw"
    / "credit_card_balance.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "model_features_credit_card_train.parquet"
)


def create_credit_card_features() -> None:
    """Create applicant-level credit-card behavior features."""

    print(f"Loading current features from: {APPLICATION_PATH}")
    app = pd.read_parquet(APPLICATION_PATH)

    print(f"Loading credit-card data from: {CREDIT_CARD_PATH}")
    cc = pd.read_csv(CREDIT_CARD_PATH)

    cc.columns = cc.columns.str.lower()

    print(f"Credit-card data shape: {cc.shape}")

    # Credit utilization.
    cc["credit_utilization"] = (
        cc["amt_balance"]
        / cc["amt_credit_limit_actual"].replace(0, pd.NA)
    )

    # Payment relative to balance.
    cc["payment_balance_ratio"] = (
        cc["amt_payment_total_current"]
        / cc["amt_balance"].replace(0, pd.NA)
    )

    # Delinquency indicators.
    cc["is_dpd"] = (cc["sk_dpd"] > 0).astype(int)
    cc["is_dpd_30"] = (cc["sk_dpd"] > 30).astype(int)
    cc["is_dpd_def"] = (cc["sk_dpd_def"] > 0).astype(int)

    # Active contract indicator.
    cc["is_active"] = (
        cc["name_contract_status"] == "Active"
    ).astype(int)

    credit_features = (
        cc.groupby("sk_id_curr")
        .agg(
            credit_card_history_months=(
                "months_balance",
                "nunique",
            ),
            credit_card_count=(
                "sk_id_prev",
                "nunique",
            ),
            credit_card_avg_balance=(
                "amt_balance",
                "mean",
            ),
            credit_card_max_balance=(
                "amt_balance",
                "max",
            ),
            credit_card_total_balance=(
                "amt_balance",
                "sum",
            ),
            credit_card_avg_limit=(
                "amt_credit_limit_actual",
                "mean",
            ),
            credit_card_max_limit=(
                "amt_credit_limit_actual",
                "max",
            ),
            credit_card_avg_utilization=(
                "credit_utilization",
                "mean",
            ),
            credit_card_max_utilization=(
                "credit_utilization",
                "max",
            ),
            credit_card_avg_payment=(
                "amt_payment_total_current",
                "mean",
            ),
            credit_card_total_payment=(
                "amt_payment_total_current",
                "sum",
            ),
            credit_card_avg_payment_balance_ratio=(
                "payment_balance_ratio",
                "mean",
            ),
            credit_card_total_drawings=(
                "amt_drawings_current",
                "sum",
            ),
            credit_card_total_atm_drawings=(
                "amt_drawings_atm_current",
                "sum",
            ),
            credit_card_total_pos_drawings=(
                "amt_drawings_pos_current",
                "sum",
            ),
            credit_card_avg_dpd=(
                "sk_dpd",
                "mean",
            ),
            credit_card_max_dpd=(
                "sk_dpd",
                "max",
            ),
            credit_card_dpd_count=(
                "is_dpd",
                "sum",
            ),
            credit_card_dpd_30_count=(
                "is_dpd_30",
                "sum",
            ),
            credit_card_dpd_def_count=(
                "is_dpd_def",
                "sum",
            ),
            credit_card_dpd_rate=(
                "is_dpd",
                "mean",
            ),
            credit_card_active_count=(
                "is_active",
                "sum",
            ),
            credit_card_max_installments_mature=(
                "cnt_instalment_mature_cum",
                "max",
            ),
        )
        .reset_index()
    )

    credit_features = credit_features.replace(
        [float("inf"), float("-inf")],
        0,
    )

    numeric_cols = credit_features.select_dtypes(
        include=["number"]
    ).columns

    credit_features[numeric_cols] = (
        credit_features[numeric_cols].fillna(0)
    )

    print(
        f"Credit-card feature shape: "
        f"{credit_features.shape}"
    )

    result = app.merge(
        credit_features,
        on="sk_id_curr",
        how="left",
    )

    feature_columns = [
        col
        for col in credit_features.columns
        if col != "sk_id_curr"
    ]

    # Applicants without credit-card history receive zeros.
    result[feature_columns] = (
        result[feature_columns].fillna(0)
    )

    print(f"Final feature shape: {result.shape}")

    print(
        "Missing values after credit-card "
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
        f"✓ Credit-card features saved to: "
        f"{OUTPUT_PATH}"
    )

    print(f"Total rows: {len(result):,}")
    print(f"Total columns: {len(result.columns)}")


if __name__ == "__main__":
    create_credit_card_features()
