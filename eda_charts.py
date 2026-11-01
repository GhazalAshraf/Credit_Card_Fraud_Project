from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
import numpy as np
import pandas as pd


NORMAL_COLOR = "#2563EB"
FRAUD_COLOR = "#DC2626"
GRID_COLOR = "#E5E7EB"


def style_axis(axis):
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


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

    # Amounts are heavily skewed, so compare the classes on a log scale.
    # Zero amounts are counted in the lowest bin.
    bins = np.logspace(np.log10(0.5), np.log10(maximum_amount + 1), 40)
    figure, axis = plt.subplots(figsize=(9, 5))
    for class_value, label, color in [
        (0, "Normal", NORMAL_COLOR),
        (1, "Fraud", FRAUD_COLOR),
    ]:
        amounts = data.loc[data["Class"] == class_value, "Amount"].clip(lower=0.5)
        if amounts.empty:
            continue
        axis.hist(
            amounts,
            bins=bins,
            weights=np.full(len(amounts), 1 / len(amounts)),
            histtype="step",
            linewidth=2,
            color=color,
            label=label,
        )
    axis.set_xscale("log")
    axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axis.set_title("Transaction Amounts: Normal vs. Fraud", fontsize=15, pad=14)
    axis.set_xlabel("Transaction amount (log scale)")
    axis.set_ylabel("Share of transactions in each class")
    axis.legend(frameon=False)
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(charts_folder / "amount_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    hours = (data["Time"] // 3600).astype(int)
    hourly = data.groupby(hours)["Class"].agg(["size", "mean"])
    peak_hour = hourly["mean"].idxmax()
    peak_rate = hourly.loc[peak_hour, "mean"]
    print(f"Highest hourly fraud rate: {peak_rate:.2%} in hour {peak_hour}")

    figure, (count_axis, rate_axis) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True
    )
    count_axis.bar(hourly.index, hourly["size"], color=NORMAL_COLOR, width=0.8)
    count_axis.set_title("Transactions per hour", loc="left", fontsize=12)
    count_axis.set_ylabel("Transactions")
    count_axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    style_axis(count_axis)

    rate_axis.bar(hourly.index, hourly["mean"], color=FRAUD_COLOR, width=0.8)
    rate_axis.set_title("Fraud rate per hour", loc="left", fontsize=12)
    rate_axis.set_ylabel("Fraud rate")
    rate_axis.set_xlabel("Hours since the first recorded transaction")
    rate_axis.set_ylim(0, peak_rate * 1.2)
    rate_axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    rate_axis.annotate(
        f"{peak_rate:.2%}",
        xy=(peak_hour, peak_rate),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    style_axis(rate_axis)

    figure.suptitle("Transactions and Fraud Rate by Hour", fontsize=15)
    figure.tight_layout()
    figure.savefig(charts_folder / "time_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
