from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
file_path = PROJECT_ROOT / "data" / "raw" / "application_train.csv"

if not file_path.exists():
    raise FileNotFoundError(
        f"File not found: {file_path.resolve()}"
    )

sample = pd.read_csv(
    file_path,
    nrows=5,
    low_memory=False,
)

print("File:", file_path)
print("Number of columns:", len(sample.columns))
print("\nFirst 10 column names:")

for column in sample.columns[:10]:
    print(column)

print("\nSample data:")
print(sample.head())