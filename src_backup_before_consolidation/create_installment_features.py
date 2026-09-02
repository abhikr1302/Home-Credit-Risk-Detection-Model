from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

APPLICATION_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "model_features_history_train.parquet"
)

INSTALLMENTS_PATH = (
    PROJECT_ROOT / "data" / "raw"
    / "installments_payments.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "model_features_repayment_train.parquet"
)


def create_installment_features() -> None:
    """Add applicant-level installment repayment behavior features."""

    print(f"Loading current features from: {APPLICATION_PATH}")
    app = pd.read_parquet(APPLICATION_PATH)

    print(f"Loading installment payments from: {INSTALLMENTS_PATH}")
    installments = pd.read_csv(INSTALLMENTS_PATH)

    installments.columns = installments.columns.str.lower()

    print(f"Installment data shape: {installments.shape}")

    # Payment delay:
    # positive = paid after scheduled date
    # zero = paid on scheduled date
    # negative = paid before scheduled date
    installments["payment_delay_days"] = (
        installments["days_entry_payment"]
        - installments["days_instalment"]
    )

    # Payment ratio relative to scheduled installment.
    installments["payment_ratio"] = (
        installments["amt_payment"]
        / installments["amt_instalment"].replace(0, pd.NA)
    )

    # Amount underpaid relative to scheduled installment.
    installments["underpayment_amount"] = (
        installments["amt_instalment"]
        - installments["amt_payment"]
    )

    # Late-payment indicator.
    installments["is_late"] = (
        installments["payment_delay_days"] > 0
    ).astype(int)

    # On-time or early indicator.
    installments["is_on_time_or_early"] = (
        installments["payment_delay_days"] <= 0
    ).astype(int)

    repayment_features = (
        installments.groupby("sk_id_curr")
        .agg(
            installment_total_records=(
                "sk_id_prev",
                "count",
            ),
            installment_previous_loans=(
                "sk_id_prev",
                "nunique",
            ),
            installment_avg_payment_delay=(
                "payment_delay_days",
                "mean",
            ),
            installment_max_payment_delay=(
                "payment_delay_days",
                "max",
            ),
            installment_min_payment_delay=(
                "payment_delay_days",
                "min",
            ),
            installment_late_payment_count=(
                "is_late",
                "sum",
            ),
            installment_late_payment_rate=(
                "is_late",
                "mean",
            ),
            installment_on_time_or_early_rate=(
                "is_on_time_or_early",
                "mean",
            ),
            installment_total_amount=(
                "amt_instalment",
                "sum",
            ),
            installment_total_paid=(
                "amt_payment",
                "sum",
            ),
            installment_avg_amount=(
                "amt_instalment",
                "mean",
            ),
            installment_avg_payment=(
                "amt_payment",
                "mean",
            ),
            installment_avg_payment_ratio=(
                "payment_ratio",
                "mean",
            ),
            installment_min_payment_ratio=(
                "payment_ratio",
                "min",
            ),
            installment_max_payment_ratio=(
                "payment_ratio",
                "max",
            ),
            installment_total_underpayment=(
                "underpayment_amount",
                "sum",
            ),
        )
        .reset_index()
    )

    repayment_features = repayment_features.replace(
        [float("inf"), float("-inf")],
        0,
    )

    numeric_cols = repayment_features.select_dtypes(
        include=["number"]
    ).columns

    repayment_features[numeric_cols] = (
        repayment_features[numeric_cols].fillna(0)
    )

    print(
        f"Repayment feature shape: "
        f"{repayment_features.shape}"
    )

    result = app.merge(
        repayment_features,
        on="sk_id_curr",
        how="left",
    )

    feature_columns = [
        col
        for col in repayment_features.columns
        if col != "sk_id_curr"
    ]

    result[feature_columns] = (
        result[feature_columns].fillna(0)
    )

    print(f"Final feature shape: {result.shape}")

    print(
        "Missing values after repayment feature engineering:",
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
        f"✓ Repayment features saved to: "
        f"{OUTPUT_PATH}"
    )

    print(f"Total rows: {len(result):,}")
    print(f"Total columns: {len(result.columns)}")


if __name__ == "__main__":
    create_installment_features()
