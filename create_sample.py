from pathlib import Path

import pandas as pd


def main():
    project_folder = Path(__file__).parent
    data_file = project_folder / "data" / "creditcard.csv"
    data = pd.read_csv(data_file)

    if "Class" not in data.columns:
        raise ValueError("The dataset does not contain the expected 'Class' column.")

    normal_rows = data.loc[data["Class"] == 0].sample(n=5, random_state=42)
    fraud_rows = data.loc[data["Class"] == 1].sample(n=5, random_state=42)
    sample = pd.concat([normal_rows, fraud_rows]).drop(columns=["Class"])

    samples_folder = project_folder / "samples"
    samples_folder.mkdir(parents=True, exist_ok=True)
    sample_file = samples_folder / "test_transactions.csv"
    sample.to_csv(sample_file, index=False)
    print(f"Saved 10 real sample transactions to: {sample_file}")


if __name__ == "__main__":
    main()
