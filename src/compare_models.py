import json
from pathlib import Path

import pandas as pd


REPORT_DIRECTORY = Path("reports")

MODEL_REPORTS = {
    "Logistic Regression": (
        REPORT_DIRECTORY / "baseline_metrics.json"
    ),
    "XGBoost": (
        REPORT_DIRECTORY / "xgboost_metrics.json"
    ),
}


def main() -> None:
    comparison_rows = []

    for model_name, report_path in MODEL_REPORTS.items():
        with open(
            report_path,
            "r",
            encoding="utf-8",
        ) as report_file:
            metrics = json.load(report_file)

        comparison_rows.append(
            {
                "model": model_name,
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
            }
        )

    comparison = pd.DataFrame(
        comparison_rows
    ).sort_values(
        by=["roc_auc", "pr_auc"],
        ascending=False,
    )

    comparison.to_csv(
        REPORT_DIRECTORY / "model_comparison.csv",
        index=False,
    )

    print(comparison.to_string(index=False))
    print(
        "\nRecommended model:",
        comparison.iloc[0]["model"],
    )


if __name__ == "__main__":
    main()