import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

def convert_csv_to_parquet(file_name: str) -> None:
    source = RAW_DIR / file_name
    destination = INTERIM_DIR / file_name.replace(".csv", ".parquet")

    df = pd.read_csv(source)
    df.columns = df.columns.str.lower()
    df.to_parquet(destination, index=False)

if __name__ == "__main__":
    files = [
        "application_train.csv",
        "application_test.csv",
        "bureau.csv",
        "bureau_balance.csv",
        "previous_application.csv",
        "installments_payments.csv",
        "credit_card_balance.csv",
        "POS_CASH_balance.csv",
    ]

    for file_name in files:
        convert_csv_to_parquet(file_name)