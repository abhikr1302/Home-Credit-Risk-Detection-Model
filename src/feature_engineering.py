"""Build the 30 production features from the five relevant Kaggle tables.

Run from the repository root: python -m src.feature_engineering
"""

import numpy as np
import pandas as pd

from src.config import (
    MODEL_TEST_FEATURES_PATH,
    MODEL_TRAIN_FEATURES_PATH,
    PRODUCTION_FEATURES,
    RAW_DATA_DIR,
)


def load_csv(filename, columns):
    path = RAW_DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Download {filename} into {RAW_DATA_DIR} first.")
    data = pd.read_csv(path, usecols=lambda name: name.lower() in columns)
    data.columns = data.columns.str.lower()
    return data


def safe_divide(numerator, denominator):
    return (numerator / denominator.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )


def create_application_features(data):
    data = data.copy()
    data["age_years"] = -data["days_birth"] / 365.25
    years = -data["days_employed"] / 365.25
    data["employment_years"] = years.where(years.between(0, 100))
    scores = [f"ext_source_{i}" for i in range(1, 4)]
    data["ext_source_mean"] = data[scores].mean(axis=1)
    return data


def create_bureau_features(data):
    data = data.assign(
        active=data["credit_active"].str.upper().eq("ACTIVE").astype(int)
    )
    return (
        data.groupby("sk_id_curr")
        .agg(
            bureau_total_accounts=("sk_id_bureau", "count"),
            bureau_active_accounts=("active", "sum"),
            bureau_total_debt=("amt_credit_sum_debt", "sum"),
        )
        .reset_index()
    )


def create_previous_application_features(data):
    status = data["name_contract_status"].str.upper()
    data = data.assign(approved=status.eq("APPROVED"), refused=status.eq("REFUSED"))
    grouped = (
        data.groupby("sk_id_curr")
        .agg(
            previous_total_applications=("sk_id_prev", "count"),
            previous_approved_count=("approved", "sum"),
            previous_refused_count=("refused", "sum"),
        )
        .reset_index()
    )
    for name in ["approved", "refused"]:
        rate = "approval" if name == "approved" else "refusal"
        grouped[f"previous_{rate}_rate"] = safe_divide(
            grouped[f"previous_{name}_count"], grouped["previous_total_applications"]
        )
    return grouped


def create_credit_card_features(data):
    data = data.assign(
        utilization=safe_divide(
            data["amt_balance"], data["amt_credit_limit_actual"]
        ).clip(0, 5)
    )
    return (
        data.groupby("sk_id_curr")
        .agg(credit_card_avg_utilization=("utilization", "mean"))
        .reset_index()
    )


def create_pos_cash_features(data):
    return (
        data.groupby("sk_id_curr")
        .agg(pos_cash_loan_count=("sk_id_prev", "nunique"))
        .reset_index()
    )


def build_feature_dataset(application, history):
    features = create_application_features(application)
    for table in history:
        features = features.merge(
            table, on="sk_id_curr", how="left", validate="many_to_one"
        )
    features["active_bureau_account_ratio"] = safe_divide(
        features["bureau_active_accounts"], features["bureau_total_accounts"]
    )
    features["bureau_debt_to_annual_income_ratio"] = safe_divide(
        features["bureau_total_debt"], features["amt_income_total"]
    )
    if features["sk_id_curr"].duplicated().any():
        raise ValueError("Each applicant must have a unique sk_id_curr.")
    columns = ["sk_id_curr"] + PRODUCTION_FEATURES
    if "target" in features:
        columns.append("target")
    return (
        features[columns]
        .replace([np.inf, -np.inf], np.nan)
        .sort_values("sk_id_curr")
        .reset_index(drop=True)
    )


def main():
    history = [
        create_bureau_features(
            load_csv(
                "bureau.csv",
                {
                    "sk_id_curr",
                    "sk_id_bureau",
                    "credit_active",
                    "amt_credit_sum_debt",
                },
            )
        ),
        create_previous_application_features(
            load_csv(
                "previous_application.csv",
                {
                    "sk_id_curr",
                    "sk_id_prev",
                    "name_contract_status",
                },
            )
        ),
        create_credit_card_features(
            load_csv(
                "credit_card_balance.csv",
                {
                    "sk_id_curr",
                    "amt_balance",
                    "amt_credit_limit_actual",
                },
            )
        ),
        create_pos_cash_features(
            load_csv("POS_CASH_balance.csv", {"sk_id_curr", "sk_id_prev"})
        ),
    ]
    application_columns = set(PRODUCTION_FEATURES) | {
        "sk_id_curr",
        "target",
        "days_employed",
        "amt_income_total",
    }
    for filename, output in [
        ("application_train.csv", MODEL_TRAIN_FEATURES_PATH),
        ("application_test.csv", MODEL_TEST_FEATURES_PATH),
    ]:
        features = build_feature_dataset(
            load_csv(filename, application_columns), history
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(output, index=False)
        print(f"Saved {len(features):,} applicants and 30 model features to {output}")


if __name__ == "__main__":
    main()
