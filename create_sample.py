from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    project_folder = Path(__file__).parent
    data_file = project_folder / "data" / "creditcard.csv"
    data = pd.read_csv(data_file)

    if "Class" not in data.columns:
        raise ValueError("The dataset does not contain the expected 'Class' column.")

    # Take the sample from the same held-out test partition as finalize_model.py,
    # so the dashboard demo uses transactions the model did not see in training.
    _, test_data = train_test_split(
        data,
        test_size=0.20,
        random_state=42,
        stratify=data["Class"],
    )
    # Every held-out fraud case plus the same number of normal transactions.
    fraud_rows = test_data.loc[test_data["Class"] == 1]
    normal_rows = test_data.loc[test_data["Class"] == 0].sample(
        n=len(fraud_rows), random_state=42
    )
    sample = pd.concat([normal_rows, fraud_rows]).drop(columns=["Class"])

    samples_folder = project_folder / "samples"
    samples_folder.mkdir(parents=True, exist_ok=True)
    sample_file = samples_folder / "test_transactions.csv"
    sample.to_csv(sample_file, index=False)
    print(
        f"Saved {len(sample)} held-out test transactions "
        f"({len(fraud_rows)} fraud, {len(normal_rows)} normal) to: {sample_file}"
    )


if __name__ == "__main__":
    main()
