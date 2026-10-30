from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    project_folder = Path(__file__).parent
    data_file = project_folder / "data" / "creditcard.csv"
    data = pd.read_csv(data_file)

    if "Class" not in data.columns:
        raise ValueError("The dataset does not contain the expected 'Class' column.")

    features = data.drop(columns=["Class"])
    target = data["Class"]

    # First reserve 20% for testing; do not use it to fit or assess this baseline.
    X_train_validation, _, y_train_validation, _ = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    # Reserve 25% of the remaining 80% for validation (20% of all data).
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=0.25,
        random_state=42,
        stratify=y_train_validation,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(class_weight="balanced", max_iter=1000),
            ),
        ]
    )
    model.fit(X_train, y_train)

    validation_probabilities = model.predict_proba(X_validation)[:, 1]
    average_precision = average_precision_score(
        y_validation, validation_probabilities
    )

    thresholds = [value / 100 for value in range(1, 100)]
    comparison_rows = []
    for threshold in thresholds:
        validation_predictions = (validation_probabilities >= threshold).astype(int)
        matrix = confusion_matrix(
            y_validation, validation_predictions, labels=[0, 1]
        )
        comparison_rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y_validation, validation_predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_validation, validation_predictions, zero_division=0
                ),
                "f1_score": f1_score(
                    y_validation, validation_predictions, zero_division=0
                ),
                "confusion_matrix": str(matrix.tolist()),
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    best_row = comparison.loc[comparison["f1_score"].idxmax()]

    reports_folder = project_folder / "reports"
    reports_folder.mkdir(parents=True, exist_ok=True)
    comparison_file = reports_folder / "threshold_comparison.csv"
    comparison.to_csv(comparison_file, index=False, float_format="%.4f")

    print(
        "Validation threshold comparison "
        "(thresholds 0.01-0.99; test data has not been evaluated):"
    )
    print(
        comparison.to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1_score": "{:.4f}".format,
            },
        )
    )
    print(
        "\nBest threshold on the validation grid "
        f"(highest F1-score): {best_row['threshold']:.2f}"
    )
    print(
        f"At this threshold: precision={best_row['precision']:.4f}, "
        f"recall={best_row['recall']:.4f}, F1-score={best_row['f1_score']:.4f}"
    )
    print(f"Average Precision (PR-AUC summary): {average_precision:.4f}")
    print("Confusion matrix rows = actual [normal, fraud]; columns = predicted.")
    print(f"Comparison saved to: {comparison_file}")


if __name__ == "__main__":
    main()
