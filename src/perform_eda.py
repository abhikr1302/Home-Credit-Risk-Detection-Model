from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_PATH = Path(
    "data/processed/model_features_train.parquet"
)

FIGURE_DIRECTORY = Path("reports/figures")
FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {DATA_PATH.resolve()}"
        )

    data = pd.read_parquet(DATA_PATH)

    print("Dataset shape:", data.shape)
    print("\nTarget distribution:")
    print(data["target"].value_counts(dropna=False))

    print("\nTarget percentage:")
    print(
        data["target"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nColumn types:")
    print(data.dtypes.value_counts())

    missing_values = (
        data.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .rename("missing_percentage")
        .reset_index()
        .rename(columns={"index": "column_name"})
    )

    missing_values.to_csv(
        "reports/missing_values.csv",
        index=False,
    )

    print("\nTop 20 columns by missing percentage:")
    print(missing_values.head(20))

    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=data,
        x="target",
        hue="target",
        legend=False,
    )

    plt.title("Credit Default Target Distribution")
    plt.xlabel("Target")
    plt.ylabel("Applicant Count")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIRECTORY / "target_distribution.png",
        dpi=200,
    )

    plt.close()

    numeric_data = data.select_dtypes(
        include="number"
    )

    correlations = (
        numeric_data
        .corr(numeric_only=True)["target"]
        .drop("target")
        .sort_values(
            key=abs,
            ascending=False,
        )
    )

    correlations.to_csv(
        "reports/target_correlations.csv",
        header=["correlation_with_target"],
    )

    print("\nTop correlations with target:")
    print(correlations.head(20))

    print("\nEDA completed.")


if __name__ == "__main__":
    main()