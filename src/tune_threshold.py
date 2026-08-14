import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)


PREDICTION_PATH = Path(
    "reports/baseline_validation_predictions.csv"
)

OUTPUT_DIRECTORY = Path("reports")
FIGURE_DIRECTORY = OUTPUT_DIRECTORY / "figures"

FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)


def main() -> None:
    predictions = pd.read_csv(PREDICTION_PATH)

    actual = predictions["actual_target"]
    probabilities = predictions["predicted_probability"]

    results = []

    for threshold in np.arange(0.05, 0.96, 0.01):
        predicted = (
            probabilities >= threshold
        ).astype(int)

        results.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": precision_score(
                    actual,
                    predicted,
                    zero_division=0,
                ),
                "recall": recall_score(
                    actual,
                    predicted,
                    zero_division=0,
                ),
                "f1_score": f1_score(
                    actual,
                    predicted,
                    zero_division=0,
                ),
            }
        )

    results_data = pd.DataFrame(results)

    best_f1_row = results_data.loc[
        results_data["f1_score"].idxmax()
    ]

    minimum_recall = 0.70

    eligible_results = results_data[
        results_data["recall"] >= minimum_recall
    ]

    if not eligible_results.empty:
        best_recall_row = eligible_results.loc[
            eligible_results["precision"].idxmax()
        ]
    else:
        best_recall_row = best_f1_row

    # ensure thresholds are plain python floats (not pandas scalars/Series)
    best_f1_threshold = float(best_f1_row.at["threshold"])
    recommended_threshold = float(best_recall_row.at["threshold"])

    threshold_summary = {
        "best_f1_threshold": best_f1_threshold,
        "best_f1_score": float(
            best_f1_row["f1_score"]
        ),
        "precision_at_best_f1": float(
            best_f1_row["precision"]
        ),
        "recall_at_best_f1": float(
            best_f1_row["recall"]
        ),
        "recommended_credit_risk_threshold": recommended_threshold,
        "recall_requirement": minimum_recall,
        "precision_at_recommended_threshold": float(
            best_recall_row["precision"]
        ),
        "recall_at_recommended_threshold": float(
            best_recall_row["recall"]
        ),
    }

    results_data.to_csv(
        OUTPUT_DIRECTORY / "threshold_results.csv",
        index=False,
    )

    with open(
        OUTPUT_DIRECTORY / "threshold_summary.json",
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            threshold_summary,
            output_file,
            indent=4,
        )

    plt.figure(figsize=(9, 6))

    plt.plot(
        results_data["threshold"],
        results_data["precision"],
        label="Precision",
    )

    plt.plot(
        results_data["threshold"],
        results_data["recall"],
        label="Recall",
    )

    plt.plot(
        results_data["threshold"],
        results_data["f1_score"],
        label="F1-score",
    )

    plt.axvline(
        best_f1_threshold,
        color="black",
        linestyle="--",
        label="Best F1 threshold",
    )

    plt.xlabel("Classification Threshold")
    plt.ylabel("Metric Score")
    plt.title("Classification Threshold Analysis")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIRECTORY / "threshold_analysis.png",
        dpi=200,
    )

    plt.close()

    print(json.dumps(threshold_summary, indent=4))


if __name__ == "__main__":
    main()