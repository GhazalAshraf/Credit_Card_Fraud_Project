from pathlib import Path

import pandas as pd


def main():
    data_file = Path(__file__).parent / "data" / "creditcard.csv"
    data = pd.read_csv(data_file)

    print("Dataset size:")
    print(f"Rows: {data.shape[0]}")
    print(f"Columns: {data.shape[1]}")

    print("\nFirst 5 rows:")
    print(data.head())

    print("\nMissing values in each column:")
    print(data.isnull().sum())

    if "Class" not in data.columns:
        raise ValueError("The dataset does not contain the expected 'Class' column.")

    transaction_counts = data["Class"].value_counts()
    normal_count = transaction_counts.get(0, 0)
    fraud_count = transaction_counts.get(1, 0)

    print("\nTransaction counts:")
    print(f"Normal transactions (Class 0): {normal_count}")
    print(f"Fraud transactions (Class 1): {fraud_count}")


if __name__ == "__main__":
    main()
