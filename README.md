# Credit Card Fraud Detection & Analytics Dashboard

A local data analytics and machine learning portfolio project for exploring
credit card transactions, analyzing fraud patterns with SQL, comparing
classification models, and reviewing transaction risk estimates in a Streamlit
dashboard.

## Project Overview

The project combines exploratory data analysis, SQL-based summaries, supervised
machine learning, and a local dashboard. The saved Random Forest model can score
uploaded transaction CSV files and label each row as Normal or Fraud using a
validation-selected decision threshold.

The dashboard's predictions are statistical risk estimates. They are not proof
that a transaction is fraudulent and should not be treated as an automated
financial decision.

## Business Problem

Card fraud is difficult to identify because fraudulent transactions are rare
compared with normal transactions. A useful fraud detection workflow must
identify as many fraud cases as possible while also considering false alarms.
This project explores that challenge and presents model performance using
precision, recall, F1-score, Average Precision, and a confusion matrix.

## Dataset

Place the dataset at:

```text
data/creditcard.csv
```

The project uses the
[Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
The dataset contains **284,807 transactions**, including **284,315 normal
transactions** and **492 fraud cases**. The `Class` column is the target:
`0` means normal and `1` means fraud. Features include `Time`, `Amount`, and
anonymized variables `V1` through `V28`.

The dataset is highly imbalanced: fraud represents approximately **0.173%** of
all transactions. Keep the original CSV in `data/`; the analysis and prediction
scripts read it but do not modify it.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib
- SQLite through Python's built-in `sqlite3` library
- scikit-learn
- joblib
- Streamlit

## Exploratory Data Analysis

The analysis scripts inspect dataset dimensions, sample rows, missing values,
class counts, and summary statistics. The charts are saved in `charts/`:

- `fraud_distribution.png` compares normal and fraud transaction counts.
- `amount_distribution.png` shows the transaction amount distribution.
- `class_pie_chart.png` shows the class proportions.
- `time_distribution.png` shows the transaction time distribution.

`fraud_analysis.py` reports transaction counts and the fraud percentage.
`eda_charts.py` generates the amount, class, and time charts.

## SQL Analysis

`sql_analysis.py` loads the CSV into an in-memory SQLite table named
`transactions` and runs SQL queries defined in the Python script. It also saves
a copy of those queries to `sql/fraud_queries.sql`. The queries calculate:

- Total, normal, and fraud transaction counts and the overall fraud percentage.
- Average transaction amount for each class.
- Fraud counts and fraud rates in four transaction amount ranges.

The calculated report is saved to `reports/sql_summary.txt`. In this dataset,
the report shows an average amount of 88.29 for normal transactions and 122.21
for fraud transactions. The highest of the four reported range-level fraud
rates is 0.369% for transactions of 500 and above. These are descriptive
summaries, not evidence that amount alone causes or predicts fraud.

## Machine Learning Models

The project trains and compares two models:

- **Logistic Regression** uses a scikit-learn Pipeline with
  `StandardScaler` and `LogisticRegression(class_weight="balanced",
  max_iter=1000)`.
- **Random Forest** uses 100 trees, `max_depth=12`,
  `class_weight="balanced_subsample"`, `random_state=42`, and `n_jobs=2`.
  Scaling is not applied to Random Forest.

The data is split reproducibly and stratified: 60% for training, 20% for
validation, and 20% held out for final testing. Stratification preserves the
rare fraud class across the splits. The validation data is used for model
comparison and decision-threshold selection. The final test partition is kept
out of those decisions.

`compare_models.py` compares the models on validation data and searches
thresholds from 0.01 to 0.99 for the highest validation F1-score. The saved
Random Forest uses the selected threshold **0.35**. The model and metadata are
stored in `models/fraud_model.joblib` and `models/model_metadata.json`.

## Model Evaluation

The final held-out test results recorded in
`reports/final_test_results.txt`, at decision threshold **0.35**, are:

| Metric | Held-out test result |
|---|---:|
| Precision | 0.8039 |
| Recall | 0.8367 |
| F1-score | 0.8200 |
| Average Precision | 0.8288 |

The test confusion matrix, with rows as actual classes `[normal, fraud]` and
columns as predicted classes, is:

```text
[[56844    20]
 [   16    82]]
```

These metrics are from the held-out test partition and were calculated once
after selecting the model and threshold with validation data. They are not
results from the sample CSV upload demonstration.

The validation model comparison is recorded in
`reports/model_comparison.csv`. For Random Forest, the validation-selected
threshold was 0.35 and the validation F1-score was 0.8280. Validation results
are separate from the final test results above.

## Dashboard Features

Run the local Streamlit app to access:

- An Overview page with total, normal, and fraud transaction counts and fraud
  percentage.
- The four saved EDA charts and a short explanation of class imbalance.
- Model Performance metrics read from the saved final test report.
- A Fraud Prediction page for uploading transaction CSV files.
- Per-row fraud risk scores, predicted classes, prediction totals, and a
  downloadable results CSV.

For prediction uploads, the CSV must contain the feature columns listed in
`models/model_metadata.json`. The app uses that metadata to restore the
expected feature order and threshold. An optional `Class` column is discarded
as an input feature. Extra columns are rejected. Risk scores are predictions,
not confirmed fraud labels.

### Try the Sample Upload

The sample is created from **real rows in the supplied dataset**, not generated
fake data. The `create_sample.py` script selects five normal and five fraud
transactions with `random_state=42`, removes `Class`, and saves the input
features to `samples/test_transactions.csv`.

1. If the sample file does not exist, run `python create_sample.py` from the
   project folder.
2. Start the dashboard using the command in Local Setup Instructions.
3. Choose **Fraud Prediction** in the sidebar and upload
   `samples/test_transactions.csv`.
4. Review the predictions and use **Download predictions as CSV** to save them.

This sample is only for checking the upload and prediction workflow. Its rows
come from the original dataset, so its predictions are a demonstration and
must not be presented as an independent or unbiased accuracy evaluation.

## Local Setup Instructions

1. Install Python and open this project folder in VS Code.
2. Ensure the Kaggle dataset file is located at `data/creditcard.csv`.
3. Open **Terminal → New Terminal** in VS Code. Confirm that the terminal is
   showing the project folder (the folder containing `app.py`).
4. Install the listed third-party packages into the Python installation you
   intend to use:

   ```text
   python -m pip install -r requirements.txt
   ```

   This project setup does not require a virtual environment. Python's built-in
   `sqlite3`, `pathlib`, `json`, and other standard-library modules are not listed
   in `requirements.txt` because they do not need a separate package install.
5. Start the dashboard from the project folder:

   ```text
   python -m streamlit run app.py
   ```

Streamlit opens the local dashboard in a browser. In the sidebar, select
**Fraud Prediction**, upload a CSV with the model's transaction feature columns,
review the risk scores and predicted classes, then select **Download predictions
as CSV** to save the results. The optional `Class` column is not used as an
input.

To create the project's 10-row upload demonstration file if it is missing, run
this command from the project folder:

```text
python create_sample.py
```

Then upload `samples/test_transactions.csv` on the Fraud Prediction page. This
file contains five normal and five fraud rows selected from the source dataset.
It is for checking the upload/prediction workflow only; it is not an independent
evaluation and must not be used to report model accuracy. The held-out test
metrics in `reports/final_test_results.txt` are a separate evaluation.

To run other analysis scripts from the project folder, use `python` followed by
the script name, for example `python sql_analysis.py` or
`python compare_models.py`. The model training/evaluation scripts can overwrite
their generated model and report files when run.

`requirements.txt` lists package names without version pins. Exact compatible
versions and a Python version range have not been verified or locked for this
project, so installing the requirements may produce different environments on
different machines. The saved scikit-learn model may also depend on compatible
package versions when loading.

The project is intended to run locally; these instructions do not deploy or
publish it.

## Project Structure

```text
Credit_Card_Fraud_Project/
├── app.py
├── compare_models.py
├── create_sample.py
├── eda_charts.py
├── explore_data.py
├── finalize_model.py
├── fraud_analysis.py
├── requirements.txt
├── sql_analysis.py
├── train_model.py
├── charts/
│   ├── amount_distribution.png
│   ├── class_pie_chart.png
│   ├── fraud_distribution.png
│   └── time_distribution.png
├── data/
│   └── creditcard.csv
├── models/
│   ├── fraud_model.joblib
│   └── model_metadata.json
├── reports/
│   ├── final_test_results.txt
│   ├── model_comparison.csv
│   ├── sql_summary.txt
│   └── threshold_comparison.csv
├── samples/
│   └── test_transactions.csv
└── sql/
    └── fraud_queries.sql
```

## Key Insights

- The dataset has a severe class imbalance: 492 fraud cases among 284,807
  transactions.
- On the held-out test set at threshold 0.35, the Random Forest achieved
  precision 0.8039 and recall 0.8367.
- The SQL summary reports different average transaction amounts for the two
  classes, but amount-range fraud rates remain below 0.4% in each reported
  range.
- The decision threshold represents a trade-off: changing it can affect
  precision and recall in different directions.

## Limitations

- The transaction features `V1` through `V28` are anonymized, which limits
  domain interpretation.
- The dataset is highly imbalanced, and performance on this historical dataset
  may not generalize to other time periods, institutions, or fraud patterns.
- The sample upload file is selected from the same source dataset and is only a
  functional demonstration.
- A predicted risk score does not confirm fraud. Real-world use would need
  appropriate human review, monitoring, and operational safeguards.
- The reported test scores describe this particular held-out split; they do
  not guarantee future performance.

## Future Improvements

Possible future work includes:

- Add a time-aware evaluation to investigate performance on newer transactions.
- Track model performance and recalibrate the decision threshold as data
  changes.
- Add model interpretability and clearer explanations for individual risk
  estimates.
- Expand dashboard filtering and summary visualizations.
- Add automated tests for data validation and prediction behavior.
