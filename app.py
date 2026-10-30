import json
from pathlib import Path
import pickle
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_FOLDER = Path(__file__).parent
DATA_FILE = PROJECT_FOLDER / "data" / "creditcard.csv"
CHARTS = [
    ("Fraud transaction counts", "fraud_distribution.png"),
    ("Transaction amount distribution", "amount_distribution.png"),
    ("Normal and fraud class distribution", "class_pie_chart.png"),
    ("Transaction time distribution", "time_distribution.png"),
]
RESULTS_FILE = PROJECT_FOLDER / "reports" / "final_test_results.txt"
MODEL_FILE = PROJECT_FOLDER / "models" / "fraud_model.joblib"
MODEL_METADATA_FILE = PROJECT_FOLDER / "models" / "model_metadata.json"


def read_metric(report_text, metric_name):
    match = re.search(
        rf"^{re.escape(metric_name)}:\s*(.+)$", report_text, flags=re.MULTILINE
    )
    return match.group(1).strip() if match else None


def load_prediction_model():
    if not MODEL_FILE.is_file():
        st.error("Saved model not found: `models/fraud_model.joblib`.")
        return None
    if not MODEL_METADATA_FILE.is_file():
        st.error("Model metadata not found: `models/model_metadata.json`.")
        return None

    try:
        metadata = json.loads(MODEL_METADATA_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        st.error(f"Could not read the model metadata: {error}")
        return None

    feature_names = metadata.get("feature_names")
    threshold = metadata.get("threshold")
    if (
        not isinstance(feature_names, list)
        or not feature_names
        or not all(isinstance(name, str) for name in feature_names)
        or len(set(feature_names)) != len(feature_names)
    ):
        st.error("The model metadata has an invalid feature_names list.")
        return None
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0 <= threshold <= 1
    ):
        st.error("The model metadata has an invalid decision threshold.")
        return None

    try:
        model = joblib.load(MODEL_FILE)
    except (OSError, EOFError, ValueError, ImportError, pickle.UnpicklingError) as error:
        st.error(f"Could not load the saved model: {error}")
        return None

    if not callable(getattr(model, "predict_proba", None)):
        st.error("The saved model cannot calculate prediction probabilities.")
        return None
    return model, feature_names, float(threshold)


def show_fraud_prediction():
    st.subheader("Fraud Prediction")
    st.write(
        "Upload a CSV containing the same transaction features used to train "
        "the saved model. The file is analyzed locally by this app."
    )
    st.warning(
        "Predictions are statistical risk estimates, not proof that a "
        "transaction is fraudulent."
    )

    uploaded_file = st.file_uploader(
        "Choose a transaction CSV file", type=["csv"], key="fraud_prediction_csv"
    )
    if uploaded_file is None:
        st.info("Choose a CSV file to begin. The required feature names are read from the saved model metadata.")
        return

    loaded_model = load_prediction_model()
    if loaded_model is None:
        return
    model, feature_names, threshold = loaded_model
    st.caption(f"Transactions with a fraud risk score of {threshold:.2f} or higher are flagged as fraud.")

    try:
        uploaded_data = pd.read_csv(uploaded_file)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        st.error(f"Could not read the uploaded CSV: {error}")
        return

    if uploaded_data.empty:
        st.error("The uploaded CSV contains no transaction rows.")
        return
    if uploaded_data.columns.has_duplicates:
        st.error("The uploaded CSV contains duplicate column names. Please use unique column names.")
        return

    input_data = uploaded_data.drop(columns=["Class"], errors="ignore")
    missing_columns = [name for name in feature_names if name not in input_data.columns]
    extra_columns = [name for name in input_data.columns if name not in feature_names]
    if missing_columns:
        st.error("Required feature columns are missing: " + ", ".join(missing_columns))
        return
    if extra_columns:
        st.error(
            "Unexpected columns found: "
            + ", ".join(extra_columns)
            + ". Only model features and an optional `Class` column are accepted."
        )
        return

    if list(input_data.columns) != feature_names:
        st.info(
            "The feature columns were not in the model's expected order. "
            "They have been rearranged to match the saved metadata."
        )
    input_data = input_data.loc[:, feature_names]

    converted_data = input_data.copy()
    invalid_values = []
    missing_values = []
    non_finite_values = []
    for column in feature_names:
        original_column = input_data[column]
        numeric_column = pd.to_numeric(original_column, errors="coerce")
        bad_numeric = original_column.notna() & numeric_column.isna()
        if bad_numeric.any():
            invalid_values.append((column, bad_numeric))
        missing_mask = original_column.isna()
        if missing_mask.any():
            missing_values.append((column, missing_mask))
        finite_mask = numeric_column.notna() & ~np.isfinite(
            numeric_column.fillna(0).to_numpy(dtype=float)
        )
        if finite_mask.any():
            non_finite_values.append((column, pd.Series(finite_mask, index=input_data.index)))
        converted_data[column] = numeric_column

    if missing_values:
        details = ", ".join(
            f"{column} ({int(mask.sum())} missing)"
            for column, mask in missing_values
        )
        st.error(f"Missing values found in required features: {details}. Please fix the CSV.")
        return
    if invalid_values:
        details = ", ".join(
            f"{column} (CSV row(s) {', '.join(str(index + 2) for index in mask[mask].index[:5])})"
            for column, mask in invalid_values
        )
        st.error(f"Some feature values are not numeric: {details}. Please fix the CSV.")
        return
    if non_finite_values:
        details = ", ".join(column for column, _ in non_finite_values)
        st.error(f"Infinite values found in required features: {details}. Please fix the CSV.")
        return

    try:
        fraud_risk_scores = model.predict_proba(converted_data)[:, 1]
    except (ValueError, TypeError) as error:
        st.error(f"The model could not process these feature values: {error}")
        return

    results = uploaded_data.drop(columns=["Class"], errors="ignore").copy()
    results["fraud_risk_score"] = fraud_risk_scores
    results["predicted_class"] = np.where(
        fraud_risk_scores >= threshold, "Fraud", "Normal"
    )

    fraud_count = int((fraud_risk_scores >= threshold).sum())
    normal_count = len(results) - fraud_count
    st.caption(f"Saved model: `{MODEL_FILE.relative_to(PROJECT_FOLDER)}`")
    kpi_columns = st.columns(3)
    kpi_columns[0].metric("Uploaded transactions", f"{len(results):,}")
    kpi_columns[1].metric("Predicted normal", f"{normal_count:,}")
    kpi_columns[2].metric("Predicted fraud", f"{fraud_count:,}")

    st.subheader("Prediction Results")
    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True,
        column_config={
            "fraud_risk_score": st.column_config.NumberColumn(
                "Fraud risk score", format="percent"
            )
        },
    )
    st.download_button(
        "Download predictions as CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="fraud_predictions.csv",
        mime="text/csv",
    )


def show_model_performance():
    st.subheader("Model Performance")
    st.caption("Final Random Forest results, read from the saved test report.")

    if not RESULTS_FILE.is_file():
        st.warning(
            "Final test results are not available yet. Expected file: "
            f"`{RESULTS_FILE.relative_to(PROJECT_FOLDER)}`"
        )
        return

    report_text = RESULTS_FILE.read_text(encoding="utf-8")
    metric_names = [
        "Precision",
        "Recall",
        "F1-score",
        "Average Precision",
    ]
    metrics = {name: read_metric(report_text, name) for name in metric_names}
    available_metrics = [(name, value) for name, value in metrics.items() if value]

    if available_metrics:
        columns = st.columns(len(available_metrics))
        for column, (name, value) in zip(columns, available_metrics):
            column.metric(name, value)
    else:
        st.warning("The saved test report does not contain the expected metric values.")

    threshold = read_metric(report_text, "Selected decision threshold")
    if threshold:
        st.write(f"**Selected decision threshold:** {threshold}")

    confusion_start = report_text.find(
        "Confusion matrix (rows = actual [normal, fraud], columns = predicted):"
    )
    if confusion_start >= 0:
        confusion_lines = report_text[confusion_start:].splitlines()
        matrix_lines = confusion_lines[1:3]
        if matrix_lines:
            st.write("**Confusion matrix** — rows are actual classes; columns are predictions.")
            st.code("\n".join(matrix_lines), language="text")


def main():
    st.set_page_config(
        page_title="Credit Card Fraud Analytics",
        page_icon="💳",
        layout="wide",
    )
    st.title("Credit Card Fraud Detection & Analytics")
    st.write("Explore transaction patterns and review the saved model test results.")

    with st.sidebar:
        st.header("Navigation")
        selected_page = st.radio(
            "Page",
            ["Overview", "Fraud Prediction"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Local project dashboard")

    if selected_page == "Fraud Prediction":
        show_fraud_prediction()
        return

    if not DATA_FILE.is_file():
        st.error(
            "Dataset file not found. Make sure `data/creditcard.csv` exists "
            "inside the project folder."
        )
        return

    try:
        data = pd.read_csv(DATA_FILE)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Could not read the dataset: {error}")
        return

    if "Class" not in data.columns:
        st.error("The dataset is missing the required `Class` column.")
        return

    total_transactions = len(data)
    fraud_transactions = int((data["Class"] == 1).sum())
    normal_transactions = int((data["Class"] == 0).sum())
    fraud_percentage = (
        fraud_transactions / total_transactions * 100 if total_transactions else 0
    )

    st.subheader("Overview")
    kpi_columns = st.columns(4)
    kpi_columns[0].metric("Total transactions", f"{total_transactions:,}")
    kpi_columns[1].metric("Normal transactions", f"{normal_transactions:,}")
    kpi_columns[2].metric("Fraud transactions", f"{fraud_transactions:,}")
    kpi_columns[3].metric("Fraud percentage", f"{fraud_percentage:.3f}%")

    st.info(
        "This dataset is highly imbalanced: normal transactions greatly "
        "outnumber fraud transactions. Class 0 means a normal transaction; "
        "Class 1 means a fraudulent transaction."
    )

    st.subheader("Exploratory Data Charts")
    for row_start in range(0, len(CHARTS), 2):
        columns = st.columns(2)
        for column, (title, filename) in zip(
            columns, CHARTS[row_start : row_start + 2]
        ):
            chart_file = PROJECT_FOLDER / "charts" / filename
            with column:
                st.markdown(f"**{title}**")
                if chart_file.is_file():
                    st.image(str(chart_file), use_container_width=True)
                else:
                    st.warning(
                        f"Chart not found: `charts/{filename}`. "
                        "Create the chart first, then refresh this page."
                    )

    st.divider()
    show_model_performance()


if __name__ == "__main__":
    main()
