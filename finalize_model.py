import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


SELECTED_THRESHOLD = 0.35


def main():
    project_folder = Path(__file__).parent
    data_file = project_folder / "data" / "creditcard.csv"
    data = pd.read_csv(data_file)

    if "Class" not in data.columns:
        raise ValueError("The dataset does not contain the expected 'Class' column.")

    features = data.drop(columns=["Class"])
    target = data["Class"]

    # Match compare_models.py exactly; the test partition stays unused until scoring.
    X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )
    X_train, _, y_train, _ = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=0.25,
        random_state=42,
        stratify=y_train_validation,
    )

    model = RandomForestClassifier(
        random_state=42,
        n_estimators=100,
        max_depth=12,
        class_weight="balanced_subsample",
        n_jobs=2,
    )
    model.fit(X_train, y_train)

    # This is the single final evaluation on the held-out test partition.
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= SELECTED_THRESHOLD).astype(int)
    precision = precision_score(y_test, test_predictions, zero_division=0)
    recall = recall_score(y_test, test_predictions, zero_division=0)
    f1 = f1_score(y_test, test_predictions, zero_division=0)
    matrix = confusion_matrix(y_test, test_predictions, labels=[0, 1])
    average_precision = average_precision_score(y_test, test_probabilities)

    models_folder = project_folder / "models"
    reports_folder = project_folder / "reports"
    models_folder.mkdir(parents=True, exist_ok=True)
    reports_folder.mkdir(parents=True, exist_ok=True)

    model_file = models_folder / "fraud_model.joblib"
    metadata_file = models_folder / "model_metadata.json"
    report_file = reports_folder / "final_test_results.txt"

    joblib.dump(model, model_file)
    metadata = {
        "threshold": SELECTED_THRESHOLD,
        "feature_names": list(features.columns),
        "target_column": "Class",
        "class_meanings": {"0": "normal", "1": "fraud"},
        "model_type": "RandomForestClassifier",
    }
    metadata_file.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    report = "\n".join(
        [
            "FINAL RANDOM FOREST TEST RESULTS",
            "=================================",
            f"Selected decision threshold: {SELECTED_THRESHOLD:.2f}",
            f"Test transactions: {len(y_test):,}",
            f"Precision: {precision:.4f}",
            f"Recall: {recall:.4f}",
            f"F1-score: {f1:.4f}",
            "Confusion matrix (rows = actual [normal, fraud], columns = predicted):",
            str(matrix),
            f"Average Precision: {average_precision:.4f}",
            "",
            "Roman Urdu mein:",
            "Precision batati hai ke model ne jin transactions ko fraud kaha, "
            "un mein se kitni waqai fraud thin. Recall batati hai ke asal fraud "
            "transactions mein se model ne kitni pakrin. F1-score precision "
            "aur recall ka mila-jula score hai. Confusion matrix sahi aur ghalat "
            "predictions dikhati hai. Average Precision fraud probabilities ki "
            "ranking ka summary hai. Yeh final scores alag test data par hain; "
            "in scores se threshold ya model settings nahi badli gayin.",
            "",
            f"Saved model: {model_file}",
            f"Saved metadata: {metadata_file}",
        ]
    ) + "\n"
    report_file.write_text(report, encoding="utf-8")

    print(report, end="")
    print(f"Test results saved to: {report_file}")


if __name__ == "__main__":
    main()
