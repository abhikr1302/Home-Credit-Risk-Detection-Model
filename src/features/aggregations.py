from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"


def add_bureau_features(app: pd.DataFrame) -> pd.DataFrame:
    print("Creating Bureau features...")

    path = RAW_DATA / "bureau.csv"
    bureau = pd.read_csv(path)

    bureau.columns = bureau.columns.str.lower()

    print(f"Bureau shape: {bureau.shape}")

    features = (
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
            bureau_total_prolonged=("cnt_credit_prolong", "sum"),
            bureau_avg_days_credit=("days_credit", "mean"),
            bureau_min_days_credit=("days_credit", "min"),
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

    numeric = features.select_dtypes(include="number").columns
    features[numeric] = features[numeric].fillna(0)

    result = app.merge(
        features,
        on="sk_id_curr",
        how="left",
    )

    feature_cols = [
        c for c in features.columns
        if c != "sk_id_curr"
    ]

    result[feature_cols] = result[feature_cols].fillna(0)

    print(f"Bureau features added: {len(feature_cols)}")

    return result


def add_previous_application_features(
    app: pd.DataFrame,
) -> pd.DataFrame:

    print("Creating Previous Application features...")

    path = RAW_DATA / "previous_application.csv"
    previous = pd.read_csv(path)

    previous.columns = previous.columns.str.lower()

    print(f"Previous application shape: {previous.shape}")

    previous["credit_application_ratio"] = (
        previous["amt_credit"]
        / previous["amt_application"].replace(0, pd.NA)
    )

    features = (
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

    features["previous_approval_rate"] = (
        features["previous_approved_count"]
        / features["previous_total_applications"]
    )

    features["previous_refusal_rate"] = (
        features["previous_refused_count"]
        / features["previous_total_applications"]
    )

    features = features.replace(
        [float("inf"), float("-inf")],
        0,
    )

    numeric = features.select_dtypes(include="number").columns
    features[numeric] = features[numeric].fillna(0)

    result = app.merge(
        features,
        on="sk_id_curr",
        how="left",
    )

    feature_cols = [
        c for c in features.columns
        if c != "sk_id_curr"
    ]

    result[feature_cols] = result[feature_cols].fillna(0)

    print(
        f"Previous Application features added: "
        f"{len(feature_cols)}"
    )

    return result


def add_installment_features(
    app: pd.DataFrame,
) -> pd.DataFrame:

    print("Creating Installment features...")

    path = RAW_DATA / "installments_payments.csv"
    installments = pd.read_csv(path)
    installments.columns = installments.columns.str.lower()

    print(f"Installment shape: {installments.shape}")

    installments["payment_delay"] = (
        installments["days_entry_payment"]
        - installments["days_installment"]
    )

    installments["payment_ratio"] = (
        installments["amt_payment"]
        / installments["amt_installment"].replace(0, pd.NA)
    )

    installments["underpayment"] = (
        installments["amt_installment"]
        - installments["amt_payment"]
    )

    features = (
        installments.groupby("sk_id_curr")
        .agg(
            installment_count=(
                "sk_id_prev",
                "count",
            ),
            installment_total_paid=(
                "amt_payment",
                "sum",
            ),
            installment_avg_payment=(
                "amt_payment",
                "mean",
            ),
            installment_total_amount=(
                "amt_installment",
                "sum",
            ),
            installment_avg_amount=(
                "amt_installment",
                "mean",
            ),
            installment_avg_payment_ratio=(
                "payment_ratio",
                "mean",
            ),
            installment_late_payment_rate=(
                "payment_delay",
                lambda x: (x > 0).mean(),
            ),
            installment_on_time_or_early_rate=(
                "payment_delay",
                lambda x: (x <= 0).mean(),
            ),
            installment_max_payment_delay=(
                "payment_delay",
                "max",
            ),
            installment_total_underpayment=(
                "underpayment",
                "sum",
            ),
        )
        .reset_index()
    )

    features = features.replace(
        [float("inf"), float("-inf")],
        0,
    )

    numeric = features.select_dtypes(include="number").columns
    features[numeric] = features[numeric].fillna(0)

    result = app.merge(
        features,
        on="sk_id_curr",
        how="left",
    )

    feature_cols = [
        c for c in features.columns
        if c != "sk_id_curr"
    ]

    result[feature_cols] = result[feature_cols].fillna(0)

    print(
        f"Installment features added: "
        f"{len(feature_cols)}"
    )

    return result


def add_credit_card_features(
    app: pd.DataFrame,
) -> pd.DataFrame:

    print("Creating Credit Card features...")

    path = RAW_DATA / "credit_card_balance.csv"
    cc = pd.read_csv(path)
    cc.columns = cc.columns.str.lower()

    print(f"Credit Card shape: {cc.shape}")

    cc["utilization"] = (
        cc["amt_balance"]
        / cc["amt_credit_limit_actual"].replace(0, pd.NA)
    )

    features = (
        cc.groupby("sk_id_curr")
        .agg(
            credit_card_active_count=(
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
            credit_card_avg_utilization=(
                "utilization",
                "mean",
            ),
            credit_card_max_utilization=(
                "utilization",
                "max",
            ),
            credit_card_total_payment=(
                "amt_payment_total_current",
                "sum",
            ),
            credit_card_avg_payment=(
                "amt_payment_total_current",
                "mean",
            ),
            credit_card_max_dpd=(
                "sk_dpd",
                "max",
            ),
            credit_card_dpd_count=(
                "sk_dpd",
                lambda x: (x > 0).sum(),
            ),
        )
        .reset_index()
    )

    features = features.replace(
        [float("inf"), float("-inf")],
        0,
    )

    numeric = features.select_dtypes(include="number").columns
    features[numeric] = features[numeric].fillna(0)

    result = app.merge(
        features,
        on="sk_id_curr",
        how="left",
    )

    feature_cols = [
        c for c in features.columns
        if c != "sk_id_curr"
    ]

    result[feature_cols] = result[feature_cols].fillna(0)

    print(
        f"Credit Card features added: "
        f"{len(feature_cols)}"
    )

    return result


def add_pos_cash_features(
    app: pd.DataFrame,
) -> pd.DataFrame:

    print("Creating POS/CASH features...")

    path = RAW_DATA / "POS_CASH_balance.csv"
    pos = pd.read_csv(path)

    pos.columns = pos.columns.str.lower()

    print(f"POS/CASH shape: {pos.shape}")

    pos["remaining_ratio"] = (
        pos["cnt_instalment_future"]
        / pos["cnt_instalment"].replace(0, pd.NA)
    )

    features = (
        pos.groupby("sk_id_curr")
        .agg(
            pos_cash_loan_count=(
                "sk_id_prev",
                "nunique",
            ),
            pos_cash_history_months=(
                "months_balance",
                "count",
            ),
            pos_cash_max_installments=(
                "cnt_instalment",
                "max",
            ),
            pos_cash_max_future_installments=(
                "cnt_instalment_future",
                "max",
            ),
            pos_cash_avg_future_installments=(
                "cnt_instalment_future",
                "mean",
            ),
            pos_cash_avg_remaining_ratio=(
                "remaining_ratio",
                "mean",
            ),
            pos_cash_max_dpd=(
                "sk_dpd",
                "max",
            ),
            pos_cash_dpd_count=(
                "sk_dpd",
                lambda x: (x > 0).sum(),
            ),
            pos_cash_dpd_def_count=(
                "sk_dpd_def",
                lambda x: (x > 0).sum(),
            ),
        )
        .reset_index()
    )

    features["pos_cash_dpd_rate"] = (
        features["pos_cash_dpd_count"]
        / features["pos_cash_history_months"]
    )

    features["pos_cash_dpd_def_rate"] = (
        features["pos_cash_dpd_def_count"]
        / features["pos_cash_history_months"]
    )

    features = features.replace(
        [float("inf"), float("-inf")],
        0,
    )

    numeric = features.select_dtypes(include="number").columns
    features[numeric] = features[numeric].fillna(0)

    result = app.merge(
        features,
        on="sk_id_curr",
        how="left",
    )

    feature_cols = [
        c for c in features.columns
        if c != "sk_id_curr"
    ]

    result[feature_cols] = result[feature_cols].fillna(0)

    print(
        f"POS/CASH features added: "
        f"{len(feature_cols)}"
    )

    return result


def create_all_features() -> pd.DataFrame:

    application_path = (
        PROCESSED_DATA / "model_features_train.parquet"
    )

    print("Loading application features...")
    app = pd.read_parquet(application_path)

    print(f"Initial shape: {app.shape}")

    app = add_bureau_features(app)
    app = add_previous_application_features(app)
    app = add_installment_features(app)
    app = add_credit_card_features(app)
    app = add_pos_cash_features(app)

    print("=" * 60)
    print("FINAL FEATURE DATASET")
    print("=" * 60)
    print(f"Rows: {len(app):,}")
    print(f"Columns: {len(app.columns)}")
    print(
        f"Missing values: "
        f"{app.isna().sum().sum():,}"
    )

    output = (
        PROCESSED_DATA
        / "model_features_final_train.parquet"
    )

    app.to_parquet(
        output,
        index=False,
    )

    print(f"Saved: {output}")

    return app


if __name__ == "__main__":
    create_all_features()
