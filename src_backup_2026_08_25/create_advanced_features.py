from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_features_pos_cash_train.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_features_advanced_train.parquet"
)


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    denominator = denominator.replace(
        [0, np.inf, -np.inf],
        np.nan,
    )

    return (
        numerator
        / denominator
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )


def create_advanced_features() -> None:

    print(
        f"Loading current features from: "
        f"{INPUT_PATH}"
    )

    df = pd.read_parquet(INPUT_PATH)

    print(
        f"Original shape: {df.shape}"
    )

    # ---------------------------------------------------------
    # External risk score features
    # ---------------------------------------------------------

    external_sources = [
        "ext_source_1",
        "ext_source_2",
        "ext_source_3",
    ]

    available_sources = [
        c
        for c in external_sources
        if c in df.columns
    ]

    if available_sources:

        df["ext_source_mean"] = (
            df[available_sources]
            .mean(axis=1)
        )

        df["ext_source_std"] = (
            df[available_sources]
            .std(axis=1)
        )

        df["ext_source_min"] = (
            df[available_sources]
            .min(axis=1)
        )

        df["ext_source_max"] = (
            df[available_sources]
            .max(axis=1)
        )

        df["ext_source_range"] = (
            df["ext_source_max"]
            - df["ext_source_min"]
        )

    if {
        "ext_source_1",
        "ext_source_2",
    }.issubset(df.columns):

        df["ext_source_1_2_diff"] = (
            df["ext_source_1"]
            - df["ext_source_2"]
        )

    if {
        "ext_source_2",
        "ext_source_3",
    }.issubset(df.columns):

        df["ext_source_2_3_diff"] = (
            df["ext_source_2"]
            - df["ext_source_3"]
        )

    if {
        "ext_source_1",
        "ext_source_3",
    }.issubset(df.columns):

        df["ext_source_1_3_diff"] = (
            df["ext_source_1"]
            - df["ext_source_3"]
        )

    # ---------------------------------------------------------
    # Affordability features
    # ---------------------------------------------------------

    if {
        "amt_credit",
        "amt_income_total",
    }.issubset(df.columns):

        df["credit_to_income_ratio"] = safe_divide(
            df["amt_credit"],
            df["amt_income_total"],
        )

    if {
        "amt_annuity",
        "amt_income_total",
    }.issubset(df.columns):

        df["annuity_to_income_ratio"] = safe_divide(
            df["amt_annuity"],
            df["amt_income_total"],
        )

    if {
        "amt_credit",
        "amt_annuity",
    }.issubset(df.columns):

        df["credit_to_annuity_ratio"] = safe_divide(
            df["amt_credit"],
            df["amt_annuity"],
        )

    if {
        "amt_goods_price",
        "amt_income_total",
    }.issubset(df.columns):

        df["goods_price_to_income_ratio"] = safe_divide(
            df["amt_goods_price"],
            df["amt_income_total"],
        )

    # ---------------------------------------------------------
    # Age and employment features
    # ---------------------------------------------------------

    if "days_birth" in df.columns:

        df["age_years"] = (
            df["days_birth"].abs()
            / 365.25
        )

    if "days_employed" in df.columns:

        df["employment_years"] = (
            df["days_employed"].abs()
            / 365.25
        )

    if {
        "days_birth",
        "days_employed",
    }.issubset(df.columns):

        age_days = df["days_birth"].abs()

        employment_days = (
            df["days_employed"].abs()
        )

        df["employment_to_age_ratio"] = (
            safe_divide(
                employment_days,
                age_days,
            )
        )

    # ---------------------------------------------------------
    # Installment repayment features
    # ---------------------------------------------------------

    if {
        "installment_total_paid",
        "installment_total_underpayment",
    }.issubset(df.columns):

        total_paid = (
            df["installment_total_paid"]
        )

        underpayment = (
            df["installment_total_underpayment"]
        )

        df["installment_payment_coverage"] = (
            safe_divide(
                total_paid,
                total_paid + underpayment,
            )
        )

    if {
        "installment_late_payment_rate",
        "installment_max_payment_delay",
    }.issubset(df.columns):

        df["installment_delay_per_payment"] = (
            df["installment_late_payment_rate"]
            * df["installment_max_payment_delay"]
        )

    if {
        "installment_total_underpayment",
        "installment_total_paid",
    }.issubset(df.columns):

        df["installment_underpayment_rate"] = (
            safe_divide(
                df["installment_total_underpayment"],
                df["installment_total_underpayment"]
                + df["installment_total_paid"],
            )
        )

    # ---------------------------------------------------------
    # Credit-card features
    # ---------------------------------------------------------

    if {
        "credit_card_avg_balance",
        "credit_card_avg_credit_limit",
    }.issubset(df.columns):

        df["credit_card_balance_to_limit"] = (
            safe_divide(
                df["credit_card_avg_balance"],
                df["credit_card_avg_credit_limit"],
            )
        )

    if {
        "credit_card_avg_payment",
        "credit_card_avg_balance",
    }.issubset(df.columns):

        df["credit_card_payment_to_balance"] = (
            safe_divide(
                df["credit_card_avg_payment"],
                df["credit_card_avg_balance"],
            )
        )

    if {
        "credit_card_avg_utilization",
        "credit_card_max_utilization",
    }.issubset(df.columns):

        df["credit_card_utilization_gap"] = (
            df["credit_card_max_utilization"]
            - df["credit_card_avg_utilization"]
        )

    # ---------------------------------------------------------
    # Risk interaction features
    # ---------------------------------------------------------

    if {
        "ext_source_mean",
        "credit_card_avg_utilization",
    }.issubset(df.columns):

        df["external_score_x_utilization"] = (
            df["ext_source_mean"]
            * df["credit_card_avg_utilization"]
        )

    if {
        "ext_source_mean",
        "installment_late_payment_rate",
    }.issubset(df.columns):

        df["external_score_x_late_rate"] = (
            df["ext_source_mean"]
            * df["installment_late_payment_rate"]
        )

    if {
        "ext_source_mean",
        "previous_refusal_rate",
    }.issubset(df.columns):

        df["external_score_x_refusal_rate"] = (
            df["ext_source_mean"]
            * df["previous_refusal_rate"]
        )

    print(
        f"Advanced feature shape: {df.shape}"
    )

    # ---------------------------------------------------------
    # Fill missing values
    # ---------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    df[numeric_columns] = (
        df[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(
            df[numeric_columns].median()
        )
    )

    categorical_columns = df.select_dtypes(
        exclude="number"
    ).columns

    df[categorical_columns] = (
        df[categorical_columns]
        .fillna("Unknown")
    )

    print(
        "Missing values after advanced "
        "feature engineering:",
        df.isna().sum().sum(),
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"✓ Advanced features saved to: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Total rows: {len(df):,}"
    )

    print(
        f"Total columns: {len(df.columns)}"
    )


if __name__ == "__main__":
    create_advanced_features()
