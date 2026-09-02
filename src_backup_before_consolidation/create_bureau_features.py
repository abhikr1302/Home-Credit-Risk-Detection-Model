from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

APPLICATION_PATH = (
    PROJECT_ROOT / "data" / "processed" / "model_features_train.parquet"
)
BUREAU_PATH = PROJECT_ROOT / "data" / "raw" / "bureau.csv"
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "model_features_bureau_train.parquet"
)


def create_bureau_features() -> None:
    """Add applicant-level Bureau credit-history features."""

    print(f"Loading application features from: {APPLICATION_PATH}")
    app = pd.read_parquet(APPLICATION_PATH)

    print(f"Loading Bureau data from: {BUREAU_PATH}")
    bureau = pd.read_csv(BUREAU_PATH)

    bureau.columns = bureau.columns.str.lower()

    print(f"Bureau shape: {bureau.shape}")

    # Aggregate historical Bureau records at applicant level.
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
            bureau_max_credit_overdue=("amt_credit_max_overdue", "max"),
            bureau_total_prolonged=("cnt_credit_prolong", "sum"),
            bureau_avg_days_credit=("days_credit", "mean"),
            bureau_min_days_credit=("days_credit", "min"),
            bureau_max_days_overdue=("credit_day_overdue", "max"),
            bureau_avg_days_overdue=("credit_day_overdue", "mean"),
            bureau_credit_type_count=("credit_type", "nunique"),
        )
        .reset_index()
    )

    # Replace aggregation NaNs before merging.
    numeric_cols = bureau_features.select_dtypes(include=["number"]).columns
    bureau_features[numeric_cols] = bureau_features[numeric_cols].fillna(0)

    print(f"Bureau feature shape: {bureau_features.shape}")

    # Merge historical features into application-level data.
    result = app.merge(
        bureau_features,
        on="sk_id_curr",
        how="left",
    )

    # Applicants without Bureau history receive zero for these features.
    bureau_feature_cols = [
        col
        for col in bureau_features.columns
        if col != "sk_id_curr"
    ]

    result[bureau_feature_cols] = result[bureau_feature_cols].fillna(0)

    print(f"Final feature shape: {result.shape}")
    print(
        "Missing values after Bureau feature engineering:",
        result.isna().sum().sum(),
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)

    print(f"✓ Bureau features saved to: {OUTPUT_PATH}")
    print(f"Total rows: {len(result):,}")
    print(f"Total columns: {len(result.columns)}")


if __name__ == "__main__":
    create_bureau_features()
