import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import create_application_features, create_bureau_features, safe_divide
from src.feature_selection import aggregate_feature_importance
from src.model_contract import PRODUCTION_FEATURES
from src import model_training, train_model
from src.config import MODEL_PATH, RECOMMENDED_FEATURES_JSON


def test_module_imports_and_production_training_contract():
    assert model_training.PRODUCTION_FEATURES == PRODUCTION_FEATURES
    assert train_model.MODEL_PATH != MODEL_PATH
    assert train_model.load_recommended_features(RECOMMENDED_FEATURES_JSON) == PRODUCTION_FEATURES


def test_division_keeps_missing_denominators_missing():
    result = safe_divide(pd.Series([2, 1, 0]), pd.Series([1, 0, 0]))
    assert result.iloc[0] == 2
    assert result.iloc[1:].isna().all()


def test_application_features_handle_employment_sentinel():
    data = pd.DataFrame({
        'sk_id_curr': [1], 'days_birth': [-3652.5], 'days_employed': [365243],
        'amt_credit': [100], 'amt_income_total': [200], 'amt_annuity': [10],
        'cnt_fam_members': [2], 'cnt_children': [0],
        'ext_source_1': [0.2], 'ext_source_2': [np.nan], 'ext_source_3': [0.6],
    })
    row = create_application_features(data).iloc[0]
    assert row.age_years == 10
    assert pd.isna(row.employment_years)
    assert row.ext_source_mean == pytest.approx(0.4)


def test_bureau_statuses_are_aggregated_by_applicant():
    data = pd.DataFrame({
        'sk_id_curr': [1, 1], 'sk_id_bureau': [10, 11],
        'credit_active': ['Active', 'Closed'], 'amt_credit_sum': [100, 50],
        'amt_credit_sum_debt': [40, 0], 'amt_credit_sum_overdue': [5, 0],
    })
    row = create_bureau_features(data).iloc[0]
    assert row.bureau_total_accounts == 2
    assert row.bureau_active_accounts == 1
    assert row.bureau_active_credit == 100
    assert row.bureau_total_debt == 40


def test_importance_does_not_double_count_prefixes():
    data = pd.DataFrame({
        'transformed_feature': ['numeric__income', 'numeric__income_ratio', 'categorical__family_status_Married'],
        'importance': [0.2, 0.3, 0.5],
    })
    result = aggregate_feature_importance(data, ['income', 'income_ratio', 'family', 'family_status']).set_index('feature')
    assert result.loc['income', 'importance'] == pytest.approx(0.2)
    assert result.loc['family', 'importance'] == 0
    assert result.importance.sum() == pytest.approx(1)
