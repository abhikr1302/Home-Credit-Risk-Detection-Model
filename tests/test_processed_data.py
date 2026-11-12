from pathlib import Path

import pandas as pd
import pytest


DATA_PATH = Path(
    Path(__file__).resolve().parents[1] / "data/processed/model_features_train.parquet"
)


pytestmark = pytest.mark.skipif(not DATA_PATH.exists(), reason="Local Kaggle dataset is not available.")


def test_processed_dataset_exists():
    assert DATA_PATH.exists()


def test_processed_dataset_structure():
    data = pd.read_parquet(DATA_PATH)

    required_columns = {
        "sk_id_curr",
        "target",
        "amt_income_total",
        "amt_credit",
        "amt_annuity",
        "bureau_total_accounts",
        "bureau_total_debt",
        "previous_total_applications",
        "credit_card_count",
        "pos_cash_loan_count",
        "monthly_gross_income",
        "loan_annuity_to_monthly_income_ratio",
        "bureau_debt_to_annual_income_ratio",
        "bureau_overdue_to_annual_income_ratio",
        "bureau_credit_to_annual_income_ratio",
        "active_bureau_account_ratio",
    }

    assert required_columns.issubset(data.columns)
    assert not data.empty
    assert data["sk_id_curr"].is_unique
    assert data["target"].notna().all()
    assert set(data["target"].unique()).issubset({0, 1})