from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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


THRESHOLDS = [value / 100 for value in range(1, 100)]


def evaluate_on_validation(model_name, model, X_validation, y_validation):
    probabilities = model.predict_proba(X_validation)[:, 1]
    average_precision = average_precision_score(y_validation, probabilities)

    rows = []
    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "model": model_name,
                "threshold": threshold,
                "precision": precision_score(
                    y_validation, predictions, zero_division=0
                ),
                "recall": recall_score(y_validation, predictions, zero_division=0),
                "f1_score": f1_score(y_validation, predictions, zero_division=0),
                "confusion_matrix": confusion_matrix(
                    y_validation, predictions, labels=[0, 1]
                ).tolist(),
                "average_precision": average_precision,
            }
        )

    threshold_results = pd.DataFrame(rows)
    best_result = threshold_results.loc[threshold_results["f1_score"].idxmax()]
    return best_result


def main():
    project_folder = Path(__file__).parent
    data_file = project_folder / "data" / "creditcard.csv"
    data = pd.read_csv(data_file)

    if "Class" not in data.columns:
        raise ValueError("The dataset does not contain the expected 'Class' column.")

    features = data.drop(columns=["Class"])
    target = data["Class"]

    # Keep the split identical to train_model.py. The reserved test portion is unused.
    X_train_validation, _, y_train_validation, _ = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=0.25,
        random_state=42,
        stratify=y_train_validation,
    )

    logistic_regression = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(class_weight="balanced", max_iter=1000),
            ),
        ]
    )
    random_forest = RandomForestClassifier(
        random_state=42,
        n_estimators=100,
        max_depth=12,
        class_weight="balanced_subsample",
        n_jobs=2,
    )

    logistic_regression.fit(X_train, y_train)
    random_forest.fit(X_train, y_train)

    best_results = [
        evaluate_on_validation(
            "Logistic Regression",
            logistic_regression,
            X_validation,
            y_validation,
        ),
        evaluate_on_validation(
            "Random Forest", random_forest, X_validation, y_validation
        ),
    ]
    comparison = pd.DataFrame(best_results).reset_index(drop=True)

    reports_folder = project_folder / "reports"
    reports_folder.mkdir(parents=True, exist_ok=True)
    comparison_file = reports_folder / "model_comparison.csv"
    comparison.to_csv(comparison_file, index=False, float_format="%.4f")

    display = comparison.copy()
    display["confusion_matrix"] = display["confusion_matrix"].map(str)
    print("Validation comparison at each model's best F1 threshold:")
    print(
        display[
            [
                "model",
                "threshold",
                "precision",
                "recall",
                "f1_score",
                "average_precision",
                "confusion_matrix",
            ]
        ].to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1_score": "{:.4f}".format,
                "average_precision": "{:.4f}".format,
            },
        )
    )

    best_f1_model = comparison.loc[comparison["f1_score"].idxmax(), "model"]
    print(f"\nValidation par sab se zyada F1-score: {best_f1_model}")
    print(
        "Roman Urdu: Threshold barhane se aam tor par fraud ka ghalat alarm kam "
        "ho sakta hai (precision barh sakti hai), lekin kuch asal fraud miss "
        "ho sakte hain (recall kam ho sakta hai). Har model ke liye 0.01 se "
        "0.99 tak validation thresholds check kiye gaye; table mein us model "
        "ka threshold hai jis par F1-score sab se zyada tha. Average Precision "
        "probabilities ke overall ranking ko summarize karti hai. Test data "
        "bilkul evaluate nahi hua."
    )
    print(f"\nComparison saved to: {comparison_file}")


if __name__ == "__main__":
    main()
