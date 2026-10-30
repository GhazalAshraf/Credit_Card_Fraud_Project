from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def main():
    project_folder = Path(__file__).parent
    data_file = project_folder / "data" / "creditcard.csv"
    data = pd.read_csv(data_file)

    required_columns = {"Amount", "Class", "Time"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))
        raise ValueError(f"The dataset is missing required columns: {missing_names}")

    charts_folder = project_folder / "charts"
    charts_folder.mkdir(exist_ok=True)

    total_transactions = len(data)
    fraud_transactions = int((data["Class"] == 1).sum())
    fraud_percentage = (
        fraud_transactions / total_transactions * 100 if total_transactions else 0
    )
    average_amount = data["Amount"].mean()
    maximum_amount = data["Amount"].max()

    print(f"Total transactions: {total_transactions:,}")
    print(f"Fraud transactions: {fraud_transactions:,}")
    print(f"Fraud percentage: {fraud_percentage:.3f}%")
    print(f"Average transaction amount: {average_amount:.2f}")
    print(f"Maximum transaction amount: {maximum_amount:.2f}")

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(data["Amount"], bins=50, color="#2563EB", edgecolor="white")
    axis.set_title("Transaction Amount Distribution", fontsize=15, pad=14)
    axis.set_xlabel("Transaction amount")
    axis.set_ylabel("Number of transactions")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(charts_folder / "amount_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    class_counts = data["Class"].value_counts().reindex([0, 1], fill_value=0)
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.pie(
        class_counts,
        labels=["Normal", "Fraud"],
        colors=["#2563EB", "#DC2626"],
        autopct="%1.2f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    axis.set_title("Normal vs. Fraudulent Transactions", fontsize=15, pad=14)
    figure.tight_layout()
    figure.savefig(charts_folder / "class_pie_chart.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(data["Time"], bins=50, color="#0F766E", edgecolor="white")
    axis.set_title("Transaction Time Distribution", fontsize=15, pad=14)
    axis.set_xlabel("Time (seconds from the first recorded transaction)")
    axis.set_ylabel("Number of transactions")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(charts_folder / "time_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
