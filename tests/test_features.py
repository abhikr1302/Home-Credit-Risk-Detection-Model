
import numpy as np
import pandas as pd
import pytest

from src.config import MODEL_TRAIN_FEATURES_PATH, PRODUCTION_FEATURES
from src.feature_engineering import (
    build_feature_dataset,
    create_application_features,
    create_bureau_features,
    create_credit_card_features,
    create_pos_cash_features,
    create_previous_application_features,
    safe_divide,
)


def test_division_keeps_missing_denominators_missing():
    result = safe_divide(pd.Series([2, 1, 0]), pd.Series([1, 0, 0]))
    assert result.iloc[0] == 2
    assert result.iloc[1:].isna().all()


def test_application_features_handle_employment_sentinel():
    data = pd.DataFrame(
        {
            "days_birth": [-3652.5],
            "days_employed": [365243],
            "ext_source_1": [0.2],
            "ext_source_2": [np.nan],
            "ext_source_3": [0.6],
        }
    )
    row = create_application_features(data).iloc[0]
    assert row.age_years == 10
    assert pd.isna(row.employment_years)
    assert row.ext_source_mean == pytest.approx(0.4)


def test_history_tables_produce_the_production_contract():
    application = pd.DataFrame({name: [0.5] for name in PRODUCTION_FEATURES})
    application = application.drop(
        columns=[
            "previous_approval_rate",
            "previous_refusal_rate",
            "previous_refused_count",
            "credit_card_avg_utilization",
            "pos_cash_loan_count",
            "active_bureau_account_ratio",
            "bureau_debt_to_annual_income_ratio",
        ]
    )
    application["sk_id_curr"] = 1
    application["target"] = 0
    application["days_birth"] = -3652.5
    application["days_employed"] = -365.25
    application["amt_income_total"] = 200
    bureau = create_bureau_features(
        pd.DataFrame(
            {
                "sk_id_curr": [1, 1],
                "sk_id_bureau": [10, 11],
                "credit_active": ["Active", "Closed"],
                "amt_credit_sum_debt": [40, 0],
            }
        )
    )
    previous = create_previous_application_features(
        pd.DataFrame(
            {
                "sk_id_curr": [1, 1],
                "sk_id_prev": [10, 11],
                "name_contract_status": ["Approved", "Refused"],
            }
        )
    )
    cards = create_credit_card_features(
        pd.DataFrame(
            {
                "sk_id_curr": [1],
                "amt_balance": [25],
                "amt_credit_limit_actual": [100],
            }
        )
    )
    pos = create_pos_cash_features(
        pd.DataFrame({"sk_id_curr": [1, 1], "sk_id_prev": [1, 1]})
    )
    result = build_feature_dataset(application, [bureau, previous, cards, pos])
    assert list(result.columns) == ["sk_id_curr"] + PRODUCTION_FEATURES + ["target"]
    row = result.iloc[0]
    assert row.active_bureau_account_ratio == 0.5
    assert row.bureau_debt_to_annual_income_ratio == 0.2
    assert row.previous_approval_rate == 0.5
    assert row.previous_refusal_rate == 0.5
    assert row.credit_card_avg_utilization == 0.25
    assert row.pos_cash_loan_count == 1


@pytest.mark.skipif(
    not MODEL_TRAIN_FEATURES_PATH.exists(), reason="Local Kaggle data is unavailable."
)
def test_processed_dataset_has_valid_production_features():
    data = pd.read_parquet(MODEL_TRAIN_FEATURES_PATH)
    assert set(PRODUCTION_FEATURES).issubset(data.columns)
    assert not data.empty
    assert data.sk_id_curr.is_unique
    assert data.target.notna().all()
    assert set(data.target.unique()) == {0, 1}
