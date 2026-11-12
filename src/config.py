"""Paths and the shared 30-feature model contract."""

import math
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data/raw"
MODEL_TRAIN_FEATURES_PATH = PROJECT_ROOT / "data/processed/model_features_train.parquet"
MODEL_TEST_FEATURES_PATH = PROJECT_ROOT / "data/processed/model_features_test.parquet"
FINAL_MODEL_METRICS_PATH = PROJECT_ROOT / "reports/final_model_metrics.json"
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/xgboost_home_credit_pipeline.joblib"))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = PROJECT_ROOT / MODEL_PATH

CLASSIFICATION_THRESHOLD = float(os.getenv("CLASSIFICATION_THRESHOLD", "0.60"))
if not math.isfinite(CLASSIFICATION_THRESHOLD) or not 0 < CLASSIFICATION_THRESHOLD < 1:
    raise ValueError("CLASSIFICATION_THRESHOLD must be finite and between 0 and 1.")

PRODUCTION_FEATURES = [
    "ext_source_mean",
    "name_education_type",
    "code_gender",
    "name_income_type",
    "name_housing_type",
    "flag_own_car",
    "name_type_suite",
    "name_family_status",
    "name_contract_type",
    "flag_document_3",
    "credit_card_avg_utilization",
    "ext_source_3",
    "region_rating_client",
    "previous_refusal_rate",
    "flag_own_realty",
    "def_60_cnt_social_circle",
    "age_years",
    "pos_cash_loan_count",
    "employment_years",
    "active_bureau_account_ratio",
    "region_rating_client_w_city",
    "ext_source_1",
    "ext_source_2",
    "def_30_cnt_social_circle",
    "previous_approval_rate",
    "amt_credit",
    "previous_refused_count",
    "days_birth",
    "amt_annuity",
    "bureau_debt_to_annual_income_ratio",
]
