import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


NORMAL_COLOR = "#2563EB"
FRAUD_COLOR = "#DC2626"
GRID_COLOR = "#E5E7EB"


def main():
    project_folder = Path(__file__).parent
    data_file = project_folder / "data" / "creditcard.csv"
    data = pd.read_csv(data_file)

    if "Class" not in data.columns:
        raise ValueError("The dataset does not contain the expected 'Class' column.")

    metadata = json.loads(
        (project_folder / "models" / "model_metadata.json").read_text(encoding="utf-8")
    )
    model = joblib.load(project_folder / "models" / "fraud_model.joblib")
    threshold = metadata["threshold"]
    feature_names = metadata["feature_names"]

    features = data.drop(columns=["Class"])
    target = data["Class"]

    # Rebuild the same held-out test partition as finalize_model.py. The saved
    # model is only scored here; nothing is retrained or re-tuned.
    _, X_test, _, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )
    probabilities = model.predict_proba(X_test[feature_names])[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    average_precision = average_precision_score(y_test, probabilities)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Average Precision: {average_precision:.4f}")
    print(f"Confusion matrix:\n{matrix}")

    charts_folder = project_folder / "charts"
    charts_folder.mkdir(exist_ok=True)

    curve_precision, curve_recall, _ = precision_recall_curve(y_test, probabilities)
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot(curve_recall, curve_precision, color=NORMAL_COLOR, linewidth=2)
    axis.scatter(
        [recall],
        [precision],
        s=90,
        color=FRAUD_COLOR,
        edgecolor="white",
        linewidth=2,
        zorder=3,
    )
    axis.annotate(
        f"Selected threshold {threshold:.2f}\n"
        f"Precision {precision:.2f} · Recall {recall:.2f}",
        xy=(recall, precision),
        xytext=(-14, -14),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=10,
    )
    axis.set_xlim(0, 1.02)
    axis.set_ylim(0, 1.05)
    axis.set_title(
        f"Precision-Recall Curve on Test Data (Average Precision {average_precision:.2f})",
        fontsize=13,
        pad=14,
    )
    axis.set_xlabel("Recall (share of fraud caught)")
    axis.set_ylabel("Precision (share of flagged transactions that are fraud)")
    axis.grid(color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(charts_folder / "precision_recall_curve.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    row_share = matrix / matrix.sum(axis=1, keepdims=True)
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    axis.imshow(row_share, cmap="Blues", vmin=0, vmax=1)
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}\n{row_share[row, column]:.2%}",
                ha="center",
                va="center",
                fontsize=13,
                color="white" if row_share[row, column] > 0.5 else "#111827",
            )
    axis.set_xticks([0, 1], labels=["Normal", "Fraud"])
    axis.set_yticks([0, 1], labels=["Normal", "Fraud"])
    # White gaps between the cells.
    axis.set_xticks([0.5], minor=True)
    axis.set_yticks([0.5], minor=True)
    axis.grid(which="minor", color="white", linewidth=4)
    axis.tick_params(which="both", length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_title(
        f"Confusion Matrix on Test Data (threshold {threshold:.2f})",
        fontsize=13,
        pad=14,
    )
    figure.tight_layout()
    figure.savefig(charts_folder / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    importances = (
        pd.Series(model.feature_importances_, index=feature_names)
        .nlargest(10)
        .sort_values()
    )
    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.barh(importances.index, importances.to_numpy(), color=NORMAL_COLOR, height=0.6)
    axis.xaxis.set_major_locator(MultipleLocator(0.05))
    axis.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axis.set_title("Top 10 Features in the Random Forest", fontsize=15, pad=14)
    axis.set_xlabel("Share of total feature importance")
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(charts_folder / "feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"\nCharts saved to: {charts_folder}")


if __name__ == "__main__":
    main()
