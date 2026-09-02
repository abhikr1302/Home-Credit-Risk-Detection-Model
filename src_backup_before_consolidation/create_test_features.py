"""
Feature engineering script to create model_features_test.parquet from test data.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_TEST_PATH = PROJECT_ROOT / "data" / "raw" / "application_test.csv"
PROCESSED_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "model_features_test.parquet"


def create_test_features() -> None:
    """Create and save engineered features from raw test data."""

    if not RAW_TEST_PATH.exists():
        raise FileNotFoundError(
            f"Raw test data not found: {RAW_TEST_PATH.resolve()}"
        )

    print(f"Loading raw test data from: {RAW_TEST_PATH}")
    df = pd.read_csv(RAW_TEST_PATH)

    # Standardize column names
    df.columns = df.columns.str.lower()

    print(f"Original shape: {df.shape}")

    # Add dataset type identifier
    df["dataset_type"] = "test"

    # Basic feature engineering
    # Fill missing values for numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Fill missing values for object columns
    object_cols = df.select_dtypes(include=["object"]).columns
    df[object_cols] = df[object_cols].fillna("Unknown")

    print(f"Processed shape: {df.shape}")

    # Save processed features
    PROCESSED_TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_TEST_PATH, index=False)

    print(f"✓ Test features saved to: {PROCESSED_TEST_PATH}")
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Missing values after processing: {df.isna().sum().sum()}")


if __name__ == "__main__":
    create_test_features()
