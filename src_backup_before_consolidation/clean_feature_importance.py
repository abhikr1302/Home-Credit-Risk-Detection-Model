from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "reports/shap_feature_importance.csv"
)

OUTPUT_PATH = Path(
    "reports/shap_feature_importance_clean.csv"
)


def main() -> None:
    importance = pd.read_csv(INPUT_PATH)

    importance["feature"] = (
        importance["feature"]
        .str.replace(
            "numeric__",
            "",
            regex=False,
        )
        .str.replace(
            "categorical__",
            "",
            regex=False,
        )
    )

    importance.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(importance.head(20).to_string(index=False))


if __name__ == "__main__":
    main()