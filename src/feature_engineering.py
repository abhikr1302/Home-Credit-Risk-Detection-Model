
import numpy as np
import pandas as pd

from src.config import (
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
    """Load CSV and normalize column names."""

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

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
    """Safely divide two Series while handling zero denominators."""

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

    Focus:
    - Applicant profile
    - Income
    - Requested credit
    - Loan burden
    - External credit indicators
    - Social/document indicators
    - Basic demographic ratios

    Goods / purchase price is intentionally excluded.
    """

    df = application_df.copy()

    # --------------------------------------------------------
    # Age
    # --------------------------------------------------------

    df["age_years"] = (
        -df["days_birth"] / 365.25
    )

    # --------------------------------------------------------
    # Employment duration
    # --------------------------------------------------------

    df["employment_years"] = (
        -df["days_employed"] / 365.25
    )

    # Home Credit contains anomalous employment values.
    df.loc[
        (df["employment_years"] > 100) | (df["employment_years"] < 0),
        "employment_years",
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
    # Employment / age
    # --------------------------------------------------------

    df["employment_to_age_ratio"] = safe_divide(
        df["employment_years"],
        df["age_years"],
    )

    # --------------------------------------------------------
    # Children / family size
    # --------------------------------------------------------

    df["children_to_family_ratio"] = safe_divide(
        df["cnt_children"],
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

    if existing_ext_columns:
        df["ext_source_mean"] = (
            df[existing_ext_columns].mean(axis=1)
        )
    else:
        df["ext_source_mean"] = np.nan

    # --------------------------------------------------------
    # Social-circle default rates
    # --------------------------------------------------------

    if {
        "obs_30_cnt_social_circle",
        "def_30_cnt_social_circle",
    }.issubset(df.columns):

        df["social_30_def_rate"] = safe_divide(
            df["def_30_cnt_social_circle"],
            df["obs_30_cnt_social_circle"],
        )

    else:
        df["social_30_def_rate"] = np.nan

    if {
        "obs_60_cnt_social_circle",
        "def_60_cnt_social_circle",
    }.issubset(df.columns):

        df["social_60_def_rate"] = safe_divide(
            df["def_60_cnt_social_circle"],
            df["obs_60_cnt_social_circle"],
        )

    else:
        df["social_60_def_rate"] = np.nan

    # --------------------------------------------------------
    # Selected application columns
    # --------------------------------------------------------

    application_columns = [

        "sk_id_curr",
        "target",

        # Applicant profile
        "name_contract_type",
        "code_gender",
        "flag_own_car",
        "flag_own_realty",
        "cnt_children",
        "cnt_fam_members",

        # Financial information
        "amt_income_total",
        "amt_credit",
        "amt_annuity",

        # Age / employment
        "days_birth",
        "days_employed",
        "age_years",
        "employment_years",

        # Applicant characteristics
        "name_income_type",
        "name_education_type",
        "name_family_status",
        "name_housing_type",
        "name_type_suite",

        # Regional indicators
        "region_rating_client",
        "region_rating_client_w_city",

        # External credit indicators
        "ext_source_1",
        "ext_source_2",
        "ext_source_3",
        "ext_source_mean",

        # Social indicators
        "obs_30_cnt_social_circle",
        "def_30_cnt_social_circle",
        "obs_60_cnt_social_circle",
        "def_60_cnt_social_circle",

        # Other documented indicators
        "days_last_phone_change",
        "flag_document_3",

        # Existing derived ratios
        "credit_to_income_ratio",
        "annuity_to_income_ratio",
        "income_per_family_member",

        # New backend-only profile features
        "employment_to_age_ratio",
        "children_to_family_ratio",
        "social_30_def_rate",
        "social_60_def_rate",
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

    Retained:
    - Total historical accounts
    - Active / closed accounts
    - Total credit exposure
    - Total debt
    - Total overdue
    - Debt / credit exposure
    - Overdue / debt exposure
    - Average credit/debt per account
    - Active credit/debt exposure

    credit_active remains categorical and is never converted
    to numeric.
    """

    df = bureau_df.copy()

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_columns = [
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
    # Account status
    # --------------------------------------------------------

    status = (
        df["credit_active"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["active_account_flag"] = (
        status == "ACTIVE"
    ).astype(int)

    df["closed_account_flag"] = (
        status == "CLOSED"
    ).astype(int)

    # --------------------------------------------------------
    # Active credit / debt exposure
    # --------------------------------------------------------

    df["active_credit_amount"] = (
        df["amt_credit_sum"]
        .where(
            df["active_account_flag"] == 1,
            0,
        )
    )

    df["active_debt_amount"] = (
        df["amt_credit_sum_debt"]
        .where(
            df["active_account_flag"] == 1,
            0,
        )
    )

    # --------------------------------------------------------
    # Aggregate Bureau information
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

            bureau_active_credit=(
                "active_credit_amount",
                "sum",
            ),

            bureau_active_debt=(
                "active_debt_amount",
                "sum",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Account ratios
    # --------------------------------------------------------

    grouped["bureau_closed_account_ratio"] = safe_divide(
        grouped["bureau_closed_accounts"],
        grouped["bureau_total_accounts"],
    )

    # --------------------------------------------------------
    # Debt / credit ratio
    # --------------------------------------------------------

    grouped["bureau_debt_to_credit_ratio"] = safe_divide(
        grouped["bureau_total_debt"],
        grouped["bureau_total_credit"],
    )

    # --------------------------------------------------------
    # Overdue / debt ratio
    # --------------------------------------------------------

    grouped["bureau_overdue_to_debt_ratio"] = safe_divide(
        grouped["bureau_total_overdue"],
        grouped["bureau_total_debt"],
    )

    # --------------------------------------------------------
    # Average credit per account
    # --------------------------------------------------------

    grouped["bureau_avg_credit_per_account"] = safe_divide(
        grouped["bureau_total_credit"],
        grouped["bureau_total_accounts"],
    )

    # --------------------------------------------------------
    # Average debt per account
    # --------------------------------------------------------

    grouped["bureau_avg_debt_per_account"] = safe_divide(
        grouped["bureau_total_debt"],
        grouped["bureau_total_accounts"],
    )

    # --------------------------------------------------------
    # Active credit exposure ratio
    # --------------------------------------------------------

    grouped["bureau_active_credit_ratio"] = safe_divide(
        grouped["bureau_active_credit"],
        grouped["bureau_total_credit"],
    )

    # --------------------------------------------------------
    # Active debt exposure ratio
    # --------------------------------------------------------

    grouped["bureau_active_debt_ratio"] = safe_divide(
        grouped["bureau_active_debt"],
        grouped["bureau_total_debt"],
    )

    return grouped


# ============================================================
# PREVIOUS APPLICATION FEATURES
# ============================================================

def create_previous_application_features(
    previous_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create historical application outcome features.

    Retained:
    - Total previous applications
    - Approved count
    - Refused count
    - Approval rate
    - Refusal rate
    - Refused / approved ratio
    - Historical approval/refusal indicators

    Removed:
    - Average previous loan amount
    - Average previous EMI
    - Previous credit/application ratio
    - Previous credit-to-income ratio
    """

    df = previous_df.copy()

    # --------------------------------------------------------
    # Application status
    # --------------------------------------------------------

    status = (
        df["name_contract_status"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["approved_flag"] = (
        status == "APPROVED"
    ).astype(int)

    df["refused_flag"] = (
        status == "REFUSED"
    ).astype(int)

    # --------------------------------------------------------
    # Aggregate
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
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Approval rate
    # --------------------------------------------------------

    grouped["previous_approval_rate"] = safe_divide(
        grouped["previous_approved_count"],
        grouped["previous_total_applications"],
    )

    # --------------------------------------------------------
    # Refusal rate
    # --------------------------------------------------------

    grouped["previous_refusal_rate"] = safe_divide(
        grouped["previous_refused_count"],
        grouped["previous_total_applications"],
    )

    # --------------------------------------------------------
    # Refused / approved ratio
    # --------------------------------------------------------

    grouped["previous_refused_to_approved_ratio"] = safe_divide(
        grouped["previous_refused_count"],
        grouped["previous_approved_count"],
    )

    # --------------------------------------------------------
    # Historical outcome indicators
    # --------------------------------------------------------

    grouped["previous_has_refused"] = (
        grouped["previous_refused_count"] > 0
    ).astype(int)

    grouped["previous_has_approved"] = (
        grouped["previous_approved_count"] > 0
    ).astype(int)

    # --------------------------------------------------------
    # Application activity
    # --------------------------------------------------------

    grouped["previous_application_outcome_activity"] = (
        grouped["previous_total_applications"]
        .clip(upper=10)
    )

    return grouped


# ============================================================
# CREDIT CARD FEATURES
# ============================================================

def create_credit_card_features(
    credit_card_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create limited credit-card exposure features.

    Retained:
    - Number of credit cards
    - Average credit limit
    - Average utilization
    - Maximum utilization
    - High-utilization card ratio

    Removed:
    - Average balance
    - DPD features
    - 30+ DPD features
    - Detailed repayment behavior
    """

    df = credit_card_df.copy()

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "amt_credit_limit_actual",
        "amt_balance",
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    # --------------------------------------------------------
    # Credit utilization
    # --------------------------------------------------------

    df["utilization"] = safe_divide(
        df["amt_balance"],
        df["amt_credit_limit_actual"],
    )

    # Avoid extreme numerical anomalies.
    df["utilization"] = (
        df["utilization"]
        .clip(lower=0, upper=5)
    )

    # --------------------------------------------------------
    # High utilization flag
    # --------------------------------------------------------

    df["high_utilization_flag"] = (
        df["utilization"] >= 0.80
    ).astype(int)

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    grouped = (
        df.groupby("sk_id_curr")
        .agg(

            credit_card_count=(
                "sk_id_prev",
                "nunique",
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

            credit_card_high_utilization_ratio=(
                "high_utilization_flag",
                "mean",
            ),
        )
        .reset_index()
    )

    return grouped


# ============================================================
# POS/CASH FEATURES
# ============================================================

def create_pos_cash_features(
    pos_cash_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a minimal POS/CASH history feature.

    Only the number of historical POS/CASH loans is retained.

    Detailed:
    - DPD
    - 30+ DPD
    - active/completed behavior
    - repayment behavior

    are intentionally excluded.
    """

    df = pos_cash_df.copy()

    grouped = (
        df.groupby("sk_id_curr")
        .agg(
            pos_cash_loan_count=(
                "sk_id_prev",
                "nunique",
            ),
        )
        .reset_index()
    )

    return grouped


# ============================================================
# BANK-STYLE DERIVED FEATURES
# ============================================================

def add_bank_style_features(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add business-friendly credit-risk ratios.

    Existing:
    - Monthly gross income
    - Loan annuity / monthly income
    - Bureau debt / annual income
    - Bureau overdue / annual income
    - Bureau credit / annual income
    - Active bureau account ratio

    Additional:
    - Closed bureau account ratio
    - Bureau debt / credit ratio
    - Bureau overdue / debt ratio
    - Average bureau credit/debt
    - Active credit/debt exposure
    - Credit-card utilization indicators
    """

    # --------------------------------------------------------
    # Income
    # --------------------------------------------------------

    income = (
        features["amt_income_total"]
        .replace(0, np.nan)
    )

    features["monthly_gross_income"] = (
        income / 12.0
    )

    monthly_income = (
        features["monthly_gross_income"]
        .replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Current loan burden
    # --------------------------------------------------------

    features[
        "loan_annuity_to_monthly_income_ratio"
    ] = safe_divide(
        features["amt_annuity"],
        monthly_income,
    )

    # --------------------------------------------------------
    # Bureau debt burden
    # --------------------------------------------------------

    features[
        "bureau_debt_to_annual_income_ratio"
    ] = safe_divide(
        features["bureau_total_debt"],
        income,
    )

    # --------------------------------------------------------
    # Bureau overdue burden
    # --------------------------------------------------------

    features[
        "bureau_overdue_to_annual_income_ratio"
    ] = safe_divide(
        features["bureau_total_overdue"],
        income,
    )

    # --------------------------------------------------------
    # Bureau credit exposure
    # --------------------------------------------------------

    features[
        "bureau_credit_to_annual_income_ratio"
    ] = safe_divide(
        features["bureau_total_credit"],
        income,
    )

    # --------------------------------------------------------
    # Active bureau account ratio
    # --------------------------------------------------------

    bureau_accounts = (
        features["bureau_total_accounts"]
        .replace(0, np.nan)
    )

    features[
        "active_bureau_account_ratio"
    ] = safe_divide(
        features["bureau_active_accounts"],
        bureau_accounts,
    )

    return features


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
    Build complete applicant-level feature dataset.

    Installments and POS/CASH are accepted for compatibility
    with the existing pipeline.

    Detailed repayment/POS behavioral features remain excluded.
    """

    print("Creating application features...")

    features = create_application_features(
        application_df
    )

    # --------------------------------------------------------
    # Bureau
    # --------------------------------------------------------

    print("Creating Bureau features...")

    bureau_features = create_bureau_features(
        bureau_df
    )

    features = features.merge(
        bureau_features,
        on="sk_id_curr",
        how="left",
    )

    # --------------------------------------------------------
    # Previous applications
    # --------------------------------------------------------

    print("Creating previous application features...")

    previous_features = (
        create_previous_application_features(
            previous_df
        )
    )

    features = features.merge(
        previous_features,
        on="sk_id_curr",
        how="left",
    )

    # --------------------------------------------------------
    # Installments
    # --------------------------------------------------------

    print(
        "Skipping detailed installment repayment features..."
    )

    # Intentionally excluded.

    # --------------------------------------------------------
    # Credit card
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # POS/CASH
    # --------------------------------------------------------

    print("Creating minimal POS/CASH features...")

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
    # Bank-style features
    # --------------------------------------------------------

    print("Adding bank-style derived features...")

    features = add_bank_style_features(
        features
    )

    # --------------------------------------------------------
    # Replace infinities
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

    features = (
        features
        .sort_values("sk_id_curr")
        .reset_index(drop=True)
    )

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

    # ========================================================
    # BUILD TRAIN FEATURES
    # ========================================================

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

    # ========================================================
    # BUILD TEST FEATURES
    # ========================================================

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

    # ========================================================
    # Validate train/test schema
    # ========================================================

    train_feature_columns = set(
        train_features.columns
    ) - {"target"}

    test_feature_columns = set(
        test_features.columns
    )

    if train_feature_columns != test_feature_columns:

        missing_from_test = sorted(
            train_feature_columns
            - test_feature_columns
        )

        extra_in_test = sorted(
            test_feature_columns
            - train_feature_columns
        )

        raise ValueError(
            "Train/test feature mismatch.\n"
            f"Missing from test: {missing_from_test}\n"
            f"Extra in test: {extra_in_test}"
        )

    # ========================================================
    # Validate target
    # ========================================================

    if "target" not in train_features.columns:

        raise ValueError(
            "Target column is missing from training data."
        )

    # ========================================================
    # Save
    # ========================================================

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

    # ========================================================
    # Summary
    # ========================================================

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
        f"Number of model features: "
        f"{len(train_feature_columns)}"
    )

    print(
        f"Train output: "
        f"{MODEL_TRAIN_FEATURES_PATH}"
    )

    print(
        f"Test output: "
        f"{MODEL_TEST_FEATURES_PATH}"
    )

    # --------------------------------------------------------
    # New feature summary
    # --------------------------------------------------------

    new_features = [
        "employment_to_age_ratio",
        "children_to_family_ratio",
        "social_30_def_rate",
        "social_60_def_rate",
        "bureau_closed_account_ratio",
        "bureau_debt_to_credit_ratio",
        "bureau_overdue_to_debt_ratio",
        "bureau_avg_credit_per_account",
        "bureau_avg_debt_per_account",
        "bureau_active_credit_ratio",
        "bureau_active_debt_ratio",
        "previous_refused_to_approved_ratio",
        "previous_has_refused",
        "previous_has_approved",
        "previous_application_outcome_activity",
        "credit_card_max_utilization",
        "credit_card_high_utilization_ratio",
        "monthly_gross_income",
        "loan_annuity_to_monthly_income_ratio",
        "bureau_debt_to_annual_income_ratio",
        "bureau_overdue_to_annual_income_ratio",
        "bureau_credit_to_annual_income_ratio",
        "active_bureau_account_ratio",
    ]

    available_new_features = [
        col
        for col in new_features
        if col in train_features.columns
    ]

    print(
        "\nNew backend features added:"
    )

    for feature in available_new_features:
        print(f"  + {feature}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
