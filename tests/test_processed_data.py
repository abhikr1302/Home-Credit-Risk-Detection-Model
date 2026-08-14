from pathlib import Path

import pandas as pd


DATA_PATH = Path(
    "data/processed/model_features_train.parquet"
)


def test_processed_dataset_exists():
    assert DATA_PATH.exists()


def test_processed_dataset_structure():
    data = pd.read_parquet(DATA_PATH)

    required_columns = {
        "sk_id_curr",
        "target",
        "dataset_type",
    }

    assert required_columns.issubset(data.columns)
    assert not data.empty
    assert data["sk_id_curr"].is_unique
    assert data["target"].notna().all()
    assert set(data["target"].unique()).issubset(
        {0, 1}
    )