"""
Consolidated feature engineering for Home Credit Risk Detection.

This module creates one applicant-level feature dataset by combining:

1. Application data
2. Bureau history
3. Previous applications
4. Installment payments
5. Credit card balance
6. POS/CASH balance

Output:

    data/processed/model_features_train.parquet
    data/processed/model_features_test.parquet
"""

import numpy as np
import pandas as pd

from config import (
    APPLICATION_TRAIN_PATH,
    APPLICATION_TEST_PATH,
    BUREAU_PATH,
    PREVIOUS_APPLICATION_PATH,
    INSTALLMENTS_PAYMENTS_PATH,
    CREDIT_CARD_BALANCE_PATH,
    POS_CASH_BALANCE_PATH,
    MODEL_TRAIN_FEATURES_PATH,
    MODEL_TEST_FEATURES_PATH,
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def load_csv(path):
    """
    Load CSV and normalize column names.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
    )

    return df


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Safe division.

    Zero denominators are converted to NaN.
    """

    denominator = denominator.replace(0, np.nan)

    result = numerator / denominator

    return result.replace(
        [np.inf, -np.inf],
        np.nan,
    )


# ============================================================
# APPLICATION FEATURES
# ============================================================

def create_application_features(
    application_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create applicant-level application features.
    """

    df = application_df.copy()

    # --------------------------------------------------------
    # Age
    # --------------------------------------------------------

    df["age_years"] = (
        (-df["days_birth"]) / 365.25
    )

    # --------------------------------------------------------
    # Employment duration
    # --------------------------------------------------------

    df["employment_years"] = (
        (-df["days_employed"]) / 365.25
    )

    # Home Credit sometimes contains anomalous employment
    # values. Treat extremely large values as missing.
    df.loc[
        df["employment_years"] > 100,
        "employment_years"
    ] = np.nan

    # --------------------------------------------------------
    # Credit / income
    # --------------------------------------------------------

    df["credit_to_income_ratio"] = safe_divide(
        df["amt_credit"],
        df["amt_income_total"],
    )

    # --------------------------------------------------------
    # Annuity / income
    # --------------------------------------------------------

    df["annuity_to_income_ratio"] = safe_divide(
        df["amt_annuity"],
        df["amt_income_total"],
    )

    # --------------------------------------------------------
    # Income per family member
    # --------------------------------------------------------

    df["income_per_family_member"] = safe_divide(
        df["amt_income_total"],
        df["cnt_fam_members"],
    )

    # --------------------------------------------------------
    # External source average
    # --------------------------------------------------------

    ext_columns = [
        "ext_source_1",
        "ext_source_2",
        "ext_source_3",
    ]

    existing_ext_columns = [
        col
        for col in ext_columns
        if col in df.columns
    ]

    df["ext_source_mean"] = (
        df[existing_ext_columns]
        .mean(axis=1)
    )

    # --------------------------------------------------------
    # Selected application columns
    # --------------------------------------------------------

    application_columns = [
        "sk_id_curr",
        "target",
        "name_contract_type",
        "code_gender",
        "flag_own_car",
        "flag_own_realty",
        "cnt_children",
        "cnt_fam_members",
        "amt_income_total",
        "amt_credit",
        "amt_annuity",
        "amt_goods_price",
        "days_birth",
        "days_employed",
        "name_income_type",
        "name_education_type",
        "name_family_status",
        "name_housing_type",
        "name_type_suite",
        "region_rating_client",
        "region_rating_client_w_city",
        "ext_source_1",
        "ext_source_2",
        "ext_source_3",
        "obs_30_cnt_social_circle",
        "def_30_cnt_social_circle",
        "obs_60_cnt_social_circle",
        "def_60_cnt_social_circle",
        "days_last_phone_change",
        "flag_document_3",
        "age_years",
        "employment_years",
        "credit_to_income_ratio",
        "annuity_to_income_ratio",
        "income_per_family_member",
        "ext_source_mean",
    ]

    existing_columns = [
        col
        for col in application_columns
        if col in df.columns
    ]

    return df[existing_columns].copy()


# ============================================================
# BUREAU FEATURES
# ============================================================

def create_bureau_features(
    bureau_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create applicant-level Bureau features.
    """

    df = bureau_df.copy()

    # --------------------------------------------------------
    # Ensure required numeric columns exist
    # --------------------------------------------------------

    numeric_columns = [
        "credit_active",
        "amt_credit_sum",
        "amt_credit_sum_debt",
        "amt_credit_sum_overdue",
        "days_credit",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    # --------------------------------------------------------
    # Active / closed accounts
    # --------------------------------------------------------

    df["active_account_flag"] = (
        df["credit_active"]
        .astype(str)
        .str.upper()
        .eq("ACTIVE")
        .astype(int)
    )

    df["closed_account_flag"] = (
        df["credit_active"]
        .astype(str)
        .str.upper()
        .eq("CLOSED")
        .astype(int)
    )

    # --------------------------------------------------------
    # Aggregation
    # --------------------------------------------------------

    grouped = (
        df.groupby("sk_id_curr")
        .agg(
            bureau_total_accounts=(
                "sk_id_bureau",
                "count",
            ),
            bureau_active_accounts=(
                "active_account_flag",
                "sum",
            ),
            bureau_closed_accounts=(
                "closed_account_flag",
                "sum",
            ),
            bureau_total_credit=(
                "amt_credit_sum",
                "sum",
            ),
            bureau_total_debt=(
                "amt_credit_sum_debt",
                "sum",
            ),
            bureau_total_overdue=(
                "amt_credit_sum_overdue",
                "sum",
            ),
            bureau_max_overdue=(
                "amt_credit_sum_overdue",
                "max",
            ),
            bureau_avg_days_credit=(
                "days_credit",
                "mean",
            ),
        )
        .reset_index()
    )

    return grouped


# ============================================================
# PREVIOUS APPLICATION FEATURES
# ============================================================

def create_previous_application_features(
    previous_df: pd.DataFrame,
    application_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create applicant-level previous application features.
    """

    df = previous_df.copy()

    # --------------------------------------------------------
    # Approval / refusal flags
    # --------------------------------------------------------

    status = (
        df["name_contract_status"]
        .astype(str)
        .str.upper()
    )

    df["approved_flag"] = (
        status == "APPROVED"
    ).astype(int)

    df["refused_flag"] = (
        status == "REFUSED"
    ).astype(int)

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_columns = [
        "amt_application",
        "amt_credit",
        "amt_annuity",
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    # --------------------------------------------------------
    # Credit / application ratio
    # --------------------------------------------------------

    df["credit_application_ratio"] = safe_divide(
        df["amt_credit"],
        df["amt_application"],
    )

    # --------------------------------------------------------
    # Current applicant income
    # --------------------------------------------------------

    income = (
        application_df[
            [
                "sk_id_curr",
                "amt_income_total",
            ]
        ]
        .drop_duplicates(
            "sk_id_curr"
        )
    )

    df = df.merge(
        income,
        on="sk_id_curr",
        how="left",
    )

    df["credit_to_current_income_ratio"] = safe_divide(
        df["amt_credit"],
        df["amt_income_total"],
    )

    # --------------------------------------------------------
    # Aggregation
    # --------------------------------------------------------

    grouped = (
        df.groupby("sk_id_curr")
        .agg(
            previous_total_applications=(
                "sk_id_prev",
                "count",
            ),
            previous_approved_count=(
                "approved_flag",
                "sum",
            ),
            previous_refused_count=(
                "refused_flag",
                "sum",
            ),
            previous_avg_application_amount=(
                "amt_application",
                "mean",
            ),
            previous_avg_credit_amount=(
                "amt_credit",
                "mean",
            ),
            previous_avg_credit_application_ratio=(
                "credit_application_ratio",
                "mean",
            ),
            previous_avg_annuity=(
                "amt_annuity",
                "mean",
            ),
            previous_avg_credit_to_current_income_ratio=(
                "credit_to_current_income_ratio",
                "mean",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Approval / refusal rates
    # --------------------------------------------------------

    grouped["previous_approval_rate"] = safe_divide(
        grouped["previous_approved_count"],
        grouped["previous_total_applications"],
    )

    grouped["previous_refusal_rate"] = safe_divide(
        grouped["previous_refused_count"],
        grouped["previous_total_applications"],
    )

    return grouped


# ============================================================
# INSTALLMENT FEATURES
# ============================================================

def create_installment_features(
    installments_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create repayment behavior features.
    """

    df = installments_df.copy()

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_columns = [
        "days_instalment",
        "days_entry_payment",
        "amt_instalment",
        "amt_payment",
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    # --------------------------------------------------------
    # Payment delay
    # --------------------------------------------------------

    df["payment_delay"] = (
        df["days_entry_payment"]
        - df["days_instalment"]
    )

    # --------------------------------------------------------
    # Late payment
    # --------------------------------------------------------

    df["late_payment_flag"] = (
        df["payment_delay"] > 0
    ).astype(int)

    # --------------------------------------------------------
    # Payment ratio
    # --------------------------------------------------------

    df["payment_ratio"] = safe_divide(
        df["amt_payment"],
        df["amt_instalment"],
    )

    # --------------------------------------------------------
    # Underpayment
    # --------------------------------------------------------

    df["underpayment"] = (
        df["amt_instalment"]
        - df["amt_payment"]
    ).clip(lower=0)

    # --------------------------------------------------------
    # Aggregation
    # --------------------------------------------------------

    grouped = (
        df.groupby("sk_id_curr")
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
                "payment_delay",
                "mean",
            ),
            installment_max_payment_delay=(
                "payment_delay",
                "max",
            ),
            installment_late_payment_count=(
                "late_payment_flag",
                "sum",
            ),
            installment_avg_payment_ratio=(
                "payment_ratio",
                "mean",
            ),
            installment_total_underpayment=(
                "underpayment",
                "sum",
            ),
        )
        .reset_index()
    )

    grouped["installment_late_payment_rate"] = safe_divide(
        grouped["installment_late_payment_count"],
        grouped["installment_total_records"],
    )

    return grouped


# ============================================================
# CREDIT CARD FEATURES
# ============================================================

def create_credit_card_features(
    credit_card_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create applicant-level credit card behavior features.
    """

    df = credit_card_df.copy()

    numeric_columns = [
        "amt_balance",
        "amt_credit_limit_actual",
        "amt_total_receivable",
        "sk_dpd",
        "sk_dpd_def",
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    # --------------------------------------------------------
    # Utilization
    # --------------------------------------------------------

    df["utilization"] = safe_divide(
        df["amt_balance"],
        df["amt_credit_limit_actual"],
    )

    # --------------------------------------------------------
    # DPD
    # --------------------------------------------------------

    df["dpd_flag"] = (
        df["sk_dpd"] > 0
    ).astype(int)

    df["dpd_30_flag"] = (
        df["sk_dpd"] >= 30
    ).astype(int)

    # --------------------------------------------------------
    # Aggregation
    # --------------------------------------------------------

    grouped = (
        df.groupby("sk_id_curr")
        .agg(
            credit_card_count=(
                "sk_id_prev",
                "nunique",
            ),
            credit_card_avg_balance=(
                "amt_balance",
                "mean",
            ),
            credit_card_avg_limit=(
                "amt_credit_limit_actual",
                "mean",
            ),
            credit_card_avg_utilization=(
                "utilization",
                "mean",
            ),
            credit_card_max_utilization=(
                "utilization",
                "max",
            ),
            credit_card_avg_dpd=(
                "sk_dpd",
                "mean",
            ),
            credit_card_dpd_count=(
                "dpd_flag",
                "sum",
            ),
            credit_card_dpd_30_count=(
                "dpd_30_flag",
                "sum",
            ),
        )
        .reset_index()
    )

    grouped["credit_card_dpd_rate"] = safe_divide(
        grouped["credit_card_dpd_count"],
        grouped["credit_card_count"],
    )

    # Keep the final agreed features only.
    grouped = grouped[
        [
            "sk_id_curr",
            "credit_card_count",
            "credit_card_avg_balance",
            "credit_card_avg_limit",
            "credit_card_avg_utilization",
            "credit_card_max_utilization",
            "credit_card_avg_dpd",
            "credit_card_dpd_rate",
            "credit_card_dpd_30_count",
        ]
    ]

    return grouped


# ============================================================
# POS CASH FEATURES
# ============================================================

def create_pos_cash_features(
    pos_cash_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create applicant-level POS/CASH features.
    """

    df = pos_cash_df.copy()

    numeric_columns = [
        "sk_dpd",
        "sk_dpd_def",
        "months_balance",
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    # --------------------------------------------------------
    # DPD flags
    # --------------------------------------------------------

    df["dpd_flag"] = (
        df["sk_dpd"] > 0
    ).astype(int)

    df["dpd_30_flag"] = (
        df["sk_dpd"] >= 30
    ).astype(int)

    # --------------------------------------------------------
    # Active / completed
    # --------------------------------------------------------

    status = (
        df["name_contract_status"]
        .astype(str)
        .str.upper()
    )

    df["active_flag"] = (
        status == "ACTIVE"
    ).astype(int)

    df["completed_flag"] = (
        status == "COMPLETED"
    ).astype(int)

    # --------------------------------------------------------
    # Aggregation
    # --------------------------------------------------------

    grouped = (
        df.groupby("sk_id_curr")
        .agg(
            pos_cash_loan_count=(
                "sk_id_prev",
                "nunique",
            ),
            pos_cash_history_months=(
                "months_balance",
                "count",
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
                "dpd_flag",
                "sum",
            ),
            pos_cash_dpd_30_count=(
                "dpd_30_flag",
                "sum",
            ),
            pos_cash_active_count=(
                "active_flag",
                "sum",
            ),
            pos_cash_completed_count=(
                "completed_flag",
                "sum",
            ),
        )
        .reset_index()
    )

    grouped["pos_cash_dpd_rate"] = safe_divide(
        grouped["pos_cash_dpd_count"],
        grouped["pos_cash_history_months"],
    )

    # --------------------------------------------------------
    # Final features
    # --------------------------------------------------------

    grouped = grouped[
        [
            "sk_id_curr",
            "pos_cash_loan_count",
            "pos_cash_history_months",
            "pos_cash_avg_dpd",
            "pos_cash_max_dpd",
            "pos_cash_dpd_rate",
            "pos_cash_dpd_30_count",
            "pos_cash_active_count",
            "pos_cash_completed_count",
        ]
    ]

    return grouped


# ============================================================
# COMPLETE DATASET BUILDER
# ============================================================

def build_feature_dataset(
    application_df: pd.DataFrame,
    bureau_df: pd.DataFrame,
    previous_df: pd.DataFrame,
    installments_df: pd.DataFrame,
    credit_card_df: pd.DataFrame,
    pos_cash_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the complete applicant-level feature dataset.
    """

    print("Creating application features...")

    features = create_application_features(
        application_df
    )

    print("Creating Bureau features...")

    bureau_features = create_bureau_features(
        bureau_df
    )

    features = features.merge(
        bureau_features,
        on="sk_id_curr",
        how="left",
    )

    print("Creating previous application features...")

    previous_features = (
        create_previous_application_features(
            previous_df,
            application_df,
        )
    )

    features = features.merge(
        previous_features,
        on="sk_id_curr",
        how="left",
    )

    print("Creating installment features...")

    installment_features = (
        create_installment_features(
            installments_df
        )
    )

    features = features.merge(
        installment_features,
        on="sk_id_curr",
        how="left",
    )

    print("Creating credit card features...")

    credit_card_features = (
        create_credit_card_features(
            credit_card_df
        )
    )

    features = features.merge(
        credit_card_features,
        on="sk_id_curr",
        how="left",
    )

    print("Creating POS/CASH features...")

    pos_cash_features = (
        create_pos_cash_features(
            pos_cash_df
        )
    )

    features = features.merge(
        pos_cash_features,
        on="sk_id_curr",
        how="left",
    )

    # --------------------------------------------------------
    # Replace infinities with NaN
    # --------------------------------------------------------

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # --------------------------------------------------------
    # Validate unique applicants
    # --------------------------------------------------------

    if features["sk_id_curr"].duplicated().any():

        duplicate_count = (
            features["sk_id_curr"]
            .duplicated()
            .sum()
        )

        raise ValueError(
            f"Duplicate applicants detected: "
            f"{duplicate_count}"
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    features = features.sort_values(
        "sk_id_curr"
    ).reset_index(drop=True)

    return features


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HOME CREDIT FEATURE ENGINEERING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load raw data
    # --------------------------------------------------------

    print("\nLoading application train...")
    application_train = load_csv(
        APPLICATION_TRAIN_PATH
    )

    print("Loading application test...")
    application_test = load_csv(
        APPLICATION_TEST_PATH
    )

    print("Loading Bureau...")
    bureau = load_csv(
        BUREAU_PATH
    )

    print("Loading previous applications...")
    previous_application = load_csv(
        PREVIOUS_APPLICATION_PATH
    )

    print("Loading installments...")
    installments = load_csv(
        INSTALLMENTS_PAYMENTS_PATH
    )

    print("Loading credit card...")
    credit_card = load_csv(
        CREDIT_CARD_BALANCE_PATH
    )

    print("Loading POS/CASH...")
    pos_cash = load_csv(
        POS_CASH_BALANCE_PATH
    )

    # --------------------------------------------------------
    # Build train
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BUILDING TRAIN FEATURES")
    print("=" * 70)

    train_features = build_feature_dataset(
        application_train,
        bureau,
        previous_application,
        installments,
        credit_card,
        pos_cash,
    )

    # --------------------------------------------------------
    # Build test
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BUILDING TEST FEATURES")
    print("=" * 70)

    test_features = build_feature_dataset(
        application_test,
        bureau,
        previous_application,
        installments,
        credit_card,
        pos_cash,
    )

    # --------------------------------------------------------
    # Ensure test doesn't contain target
    # --------------------------------------------------------

    if "target" in test_features.columns:
        test_features = test_features.drop(
            columns=["target"]
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    MODEL_TRAIN_FEATURES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_TEST_FEATURES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_features.to_parquet(
        MODEL_TRAIN_FEATURES_PATH,
        index=False,
    )

    test_features.to_parquet(
        MODEL_TEST_FEATURES_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 70)

    print(
        f"Train shape: "
        f"{train_features.shape}"
    )

    print(
        f"Test shape: "
        f"{test_features.shape}"
    )

    print(
        f"Train output: "
        f"{MODEL_TRAIN_FEATURES_PATH}"
    )

    print(
        f"Test output: "
        f"{MODEL_TEST_FEATURES_PATH}"
    )


if __name__ == "__main__":
    main()