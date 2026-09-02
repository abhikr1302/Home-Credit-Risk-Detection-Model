from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"


APPLICATION_TRAIN = RAW_DIR / "application_train.csv"
APPLICATION_TEST = RAW_DIR / "application_test.csv"

BUREAU = RAW_DIR / "bureau.csv"
PREVIOUS_APPLICATION = RAW_DIR / "previous_application.csv"
INSTALLMENTS = RAW_DIR / "installments_payments.csv"
CREDIT_CARD = RAW_DIR / "credit_card_balance.csv"
POS_CASH = RAW_DIR / "POS_CASH_balance.csv"


TRAIN_FEATURES = (
    PROCESSED_DIR / "model_features_advanced_train.parquet"
)

TEST_FEATURES = (
    PROCESSED_DIR / "model_features_advanced_test.parquet"
)

MODEL_PATH = MODEL_DIR / "home_credit_xgboost.joblib"

METRICS_PATH = REPORT_DIR / "model_metrics.json"

RANDOM_STATE = 42


for directory in [
    PROCESSED_DIR,
    MODEL_DIR,
    REPORT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
