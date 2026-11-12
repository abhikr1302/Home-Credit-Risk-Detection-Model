import math
import os
from pathlib import Path


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


# ============================================================
# RAW DATA FILES
# ============================================================

APPLICATION_TRAIN_PATH = RAW_DATA_DIR / "application_train.csv"
APPLICATION_TEST_PATH = RAW_DATA_DIR / "application_test.csv"

BUREAU_PATH = RAW_DATA_DIR / "bureau.csv"
BUREAU_BALANCE_PATH = RAW_DATA_DIR / "bureau_balance.csv"

PREVIOUS_APPLICATION_PATH = RAW_DATA_DIR / "previous_application.csv"

INSTALLMENTS_PAYMENTS_PATH = RAW_DATA_DIR / "installments_payments.csv"

CREDIT_CARD_BALANCE_PATH = RAW_DATA_DIR / "credit_card_balance.csv"

POS_CASH_BALANCE_PATH = RAW_DATA_DIR / "POS_CASH_balance.csv"


# ============================================================
# PROCESSED DATA
# ============================================================

MODEL_TRAIN_FEATURES_PATH = (
    PROCESSED_DATA_DIR / "model_features_train.parquet"
)

MODEL_TEST_FEATURES_PATH = (
    PROCESSED_DATA_DIR / "model_features_test.parquet"
)


# ============================================================
# FEATURE SELECTION REPORTS
# ============================================================

RECOMMENDED_FEATURES_JSON = (
    REPORTS_DIR / "recommended_features.json"
)

RECOMMENDED_FEATURES_TXT = (
    REPORTS_DIR / "recommended_features.txt"
)


# ============================================================
# MODEL ARTIFACTS
# ============================================================

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODELS_DIR / "xgboost_home_credit_pipeline.joblib")))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = PROJECT_ROOT / MODEL_PATH

CLASSIFICATION_THRESHOLD = float(os.getenv("CLASSIFICATION_THRESHOLD", "0.60"))
if not math.isfinite(CLASSIFICATION_THRESHOLD) or not 0 < CLASSIFICATION_THRESHOLD < 1:
    raise ValueError("CLASSIFICATION_THRESHOLD must be finite and between 0 and 1.")

FINAL_MODEL_METRICS_PATH = (
    REPORTS_DIR / "final_model_metrics.json"
)

FINAL_MODEL_FEATURES_PATH = (
    REPORTS_DIR / "final_model_features.json"
)
