from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def main():
    project_folder = Path(__file__).parent
    data_file = project_folder / "data" / "creditcard.csv"
    data = pd.read_csv(data_file)

    if "Class" not in data.columns:
        raise ValueError("The dataset does not contain the expected 'Class' column.")

    transaction_counts = data["Class"].value_counts()
    normal_count = int(transaction_counts.get(0, 0))
    fraud_count = int(transaction_counts.get(1, 0))
    total_count = len(data)
    fraud_percentage = (fraud_count / total_count * 100) if total_count else 0

    print(f"Total transactions: {total_count:,}")
    print(f"Normal transactions: {normal_count:,}")
    print(f"Fraud transactions: {fraud_count:,}")
    print(f"Fraud percentage: {fraud_percentage:.3f}%")

    labels = ["Normal", "Fraud"]
    counts = [normal_count, fraud_count]
    colors = ["#2563EB", "#DC2626"]

    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(labels, counts, color=colors, width=0.58)

    axis.set_title("Normal vs. Fraudulent Transactions", fontsize=15, pad=18)
    axis.set_ylabel("Number of transactions")
    axis.set_ylim(0, max(counts, default=0) * 1.12)
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    axis.bar_label(bars, labels=[f"{count:,}" for count in counts], padding=5, fontsize=11)
    axis.text(
        0.5,
        0.96,
        "Highly imbalanced dataset: fraud transactions are rare.",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="#4B5563",
    )

    figure.tight_layout()
    charts_folder = project_folder / "charts"
    charts_folder.mkdir(exist_ok=True)
    chart_file = charts_folder / "fraud_distribution.png"
    plt.savefig(chart_file, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"\nChart saved to: {chart_file}")


if __name__ == "__main__":
    main()
