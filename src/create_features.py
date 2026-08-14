"""
Feature engineering script to create model_features_train.parquet.
Loads application_train.csv and performs basic feature engineering.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "application_train.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_features_train.parquet"


def create_features() -> None:
    """Create and save engineered features from raw data."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw data not found: {RAW_DATA_PATH.resolve()}"
        )

    print(f"Loading raw data from: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)

    # Standardize column names
    df.columns = df.columns.str.lower()

    print(f"Original shape: {df.shape}")

    # Basic feature engineering
    # Fill missing values for numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Fill missing values for object columns
    object_cols = df.select_dtypes(include=["object"]).columns
    df[object_cols] = df[object_cols].fillna("Unknown")

    print(f"Processed shape: {df.shape}")

    # Save processed features
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DATA_PATH, index=False)

    print(f"✓ Features saved to: {PROCESSED_DATA_PATH}")
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Missing values after processing: {df.isna().sum().sum()}")


if __name__ == "__main__":
    create_features()
