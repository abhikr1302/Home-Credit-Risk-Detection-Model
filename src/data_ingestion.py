"""
Data ingestion module.

Converts raw Home Credit CSV files into Parquet files.

Raw files are expected under:
    data/raw/

Output files are written to:
    data/interim/
"""

from pathlib import Path
import pandas as pd

from config import (
    APPLICATION_TRAIN_PATH,
    APPLICATION_TEST_PATH,
    BUREAU_PATH,
    BUREAU_BALANCE_PATH,
    PREVIOUS_APPLICATION_PATH,
    INSTALLMENTS_PAYMENTS_PATH,
    CREDIT_CARD_BALANCE_PATH,
    POS_CASH_BALANCE_PATH,
    INTERIM_DATA_DIR,
)


# ============================================================
# HELPERS
# ============================================================

def load_csv(path: Path) -> pd.DataFrame:
    """
    Load a CSV file and normalize column names.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    print(f"Loading: {path.name}")

    df = pd.read_csv(path)

    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()

    print(
        f"Loaded {path.name}: "
        f"{df.shape[0]:,} rows x {df.shape[1]:,} columns"
    )

    return df


def save_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save DataFrame as Parquet.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        output_path,
        index=False,
    )

    print(f"Saved: {output_path}")


# ============================================================
# MAIN INGESTION
# ============================================================

def ingest_data() -> None:
    """
    Convert all raw CSV files into Parquet.
    """

    datasets = {
        "application_train": APPLICATION_TRAIN_PATH,
        "application_test": APPLICATION_TEST_PATH,
        "bureau": BUREAU_PATH,
        "bureau_balance": BUREAU_BALANCE_PATH,
        "previous_application": PREVIOUS_APPLICATION_PATH,
        "installments_payments": INSTALLMENTS_PAYMENTS_PATH,
        "credit_card_balance": CREDIT_CARD_BALANCE_PATH,
        "pos_cash_balance": POS_CASH_BALANCE_PATH,
    }

    for dataset_name, input_path in datasets.items():

        df = load_csv(input_path)

        output_path = (
            INTERIM_DATA_DIR /
            f"{dataset_name}.parquet"
        )

        save_parquet(
            df,
            output_path,
        )

    print("\nData ingestion completed successfully.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    ingest_data()