from pathlib import Path

import pandas as pd


def create_application_features(
    application: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare base application features."""

    df = application.copy()

    df.columns = df.columns.str.lower()

    if "dataset_type" not in df.columns:
        df["dataset_type"] = "train"

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns

    df[numeric_columns] = df[numeric_columns].fillna(
        df[numeric_columns].median()
    )

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns

    df[categorical_columns] = df[categorical_columns].fillna(
        "Unknown"
    )

    return df


def create_bureau_features(
    app: pd.DataFrame,
    bureau: pd.DataFrame,
) -> pd.DataFrame:
    """Create applicant-level Bureau credit-history features."""

    bureau = bureau.copy()
    bureau.columns = bureau.columns.str.lower()

    print(f"Bureau shape: {bureau.shape}")

    bureau_features = (
        bureau.groupby("sk_id_curr")
        .agg(
            bureau_total_accounts=("sk_id_bureau", "count"),
            bureau_active_accounts=(
                "credit_active",
                lambda x: (x == "active").sum(),
            ),
            bureau_closed_accounts=(
                "credit_active",
                lambda x: (x == "closed").sum(),
            ),
            bureau_total_credit=("amt_credit_sum", "sum"),
            bureau_avg_credit=("amt_credit_sum", "mean"),
            bureau_total_debt=("amt_credit_sum_debt", "sum"),
            bureau_avg_debt=("amt_credit_sum_debt", "mean"),
            bureau_total_overdue=("amt_credit_sum_overdue", "sum"),
            bureau_max_overdue=("amt_credit_sum_overdue", "max"),
            bureau_max_credit_overdue=(
                "amt_credit_max_overdue",
                "max",
            ),
            bureau_total_prolonged=(
                "cnt_credit_prolong",
                "sum",
            ),
            bureau_avg_days_credit=(
                "days_credit",
                "mean",
            ),
            bureau_min_days_credit=(
                "days_credit",
                "min",
            ),
            bureau_max_days_overdue=(
                "credit_day_overdue",
                "max",
            ),
            bureau_avg_days_overdue=(
                "credit_day_overdue",
                "mean",
            ),
            bureau_credit_type_count=(
                "credit_type",
                "nunique",
            ),
        )
        .reset_index()
    )

    feature_columns = [
        column
        for column in bureau_features.columns
        if column != "sk_id_curr"
    ]

    bureau_features[feature_columns] = (
        bureau_features[feature_columns].fillna(0)
    )

    print(
        f"Bureau feature shape: {bureau_features.shape}"
    )

    result = app.merge(
        bureau_features,
        on="sk_id_curr",
        how="left",
    )

    result[feature_columns] = (
        result[feature_columns].fillna(0)
    )

    return result


def create_previous_application_features(
    app: pd.DataFrame,
    previous: pd.DataFrame,
) -> pd.DataFrame:
    """Create applicant-level previous-application features."""

    previous = previous.copy()
    previous.columns = previous.columns.str.lower()

    print(
        f"Previous application shape: "
        f"{previous.shape}"
    )

    # Prevent division by zero.
    previous["credit_application_ratio"] = (
        previous["amt_credit"]
        / previous["amt_application"].replace(0, pd.NA)
    )

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

    previous_features["previous_approval_rate"] = (
        previous_features["previous_approved_count"]
        / previous_features["previous_total_applications"]
    )

    previous_features["previous_refusal_rate"] = (
        previous_features["previous_refused_count"]
        / previous_features["previous_total_applications"]
    )

    previous_features = previous_features.replace(
        [float("inf"), float("-inf")],
        0,
    )

    numeric_columns = previous_features.select_dtypes(
        include=["number"]
    ).columns

    previous_features[numeric_columns] = (
        previous_features[numeric_columns].fillna(0)
    )

    print(
        "Previous-application feature shape:",
        previous_features.shape,
    )

    result = app.merge(
        previous_features,
        on="sk_id_curr",
        how="left",
    )

    feature_columns = [
        column
        for column in previous_features.columns
        if column != "sk_id_curr"
    ]

    result[feature_columns] = (
        result[feature_columns].fillna(0)
    )

    print(
        "Final feature shape:",
        result.shape,
    )

    print(
        "Missing values after previous-application "
        "feature engineering:",
        result.isna().sum().sum(),
    )

    return result
