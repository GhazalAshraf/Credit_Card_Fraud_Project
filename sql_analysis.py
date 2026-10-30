from pathlib import Path
import sqlite3
import textwrap

import pandas as pd


SQL_QUERIES = {
    "Total, normal, fraud, and fraud percentage": """
        SELECT
            COUNT(*) AS total_transactions,
            COALESCE(SUM(CASE WHEN Class = 0 THEN 1 ELSE 0 END), 0)
                AS normal_transactions,
            COALESCE(SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END), 0)
                AS fraud_transactions,
            ROUND(
                100.0 * SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END)
                / NULLIF(COUNT(*), 0),
                3
            ) AS fraud_percentage
        FROM transactions;
    """,
    "Average transaction amount for each class": """
        SELECT
            Class AS transaction_class,
            COUNT(*) AS transaction_count,
            ROUND(AVG(Amount), 2) AS average_amount
        FROM transactions
        GROUP BY Class
        ORDER BY Class;
    """,
    "Fraud counts and rates by transaction amount range": """
        SELECT
            CASE
                WHEN Amount < 50 THEN 'Under 50'
                WHEN Amount < 100 THEN '50 to under 100'
                WHEN Amount < 500 THEN '100 to under 500'
                ELSE '500 and above'
            END AS amount_range,
            COUNT(*) AS total_transactions,
            SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
            ROUND(
                100.0 * SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END)
                / NULLIF(COUNT(*), 0),
                3
            ) AS fraud_rate_percentage
        FROM transactions
        WHERE Amount IS NOT NULL
        GROUP BY
            CASE
                WHEN Amount < 50 THEN 'Under 50'
                WHEN Amount < 100 THEN '50 to under 100'
                WHEN Amount < 500 THEN '100 to under 500'
                ELSE '500 and above'
            END
        ORDER BY MIN(Amount);
    """,
}


def format_count(value):
    return f"{int(value):,}"


def build_report(connection):
    summary = connection.execute(
        SQL_QUERIES["Total, normal, fraud, and fraud percentage"]
    ).fetchone()
    average_rows = connection.execute(
        SQL_QUERIES["Average transaction amount for each class"]
    ).fetchall()
    range_rows = connection.execute(
        SQL_QUERIES["Fraud counts and rates by transaction amount range"]
    ).fetchall()

    total, normal, fraud, fraud_percentage = summary
    lines = [
        "SQL FRAUD ANALYSIS SUMMARY",
        "==========================",
        "",
        "Overall transaction counts (Class 0 = normal, Class 1 = fraud):",
        f"Total transactions: {format_count(total)}",
        f"Normal transactions: {format_count(normal)}",
        f"Fraud transactions: {format_count(fraud)}",
        f"Fraud percentage: {fraud_percentage if fraud_percentage is not None else 0:.3f}%",
        "",
        "Average transaction amount by class:",
    ]

    for transaction_class, transaction_count, average_amount in average_rows:
        class_name = "Normal" if transaction_class == 0 else "Fraud"
        average_text = f"{average_amount:.2f}" if average_amount is not None else "N/A"
        lines.append(
            f"{class_name} (Class {transaction_class}): "
            f"{format_count(transaction_count)} transactions; average amount {average_text}"
        )

    lines.extend(
        [
            "",
            "Fraud by transaction amount range:",
            "Range | All transactions | Fraud transactions | Fraud rate",
        ]
    )
    for amount_range, count, fraud_count, fraud_rate in range_rows:
        rate_text = f"{fraud_rate:.3f}%" if fraud_rate is not None else "N/A"
        lines.append(
            f"{amount_range} | {format_count(count)} | "
            f"{format_count(fraud_count)} | {rate_text}"
        )

    lines.extend(
        [
            "",
            "Asaan Roman Urdu mein:",
            "COUNT(*) tamam transactions ko ginta hai. CASE WHEN Class = 0 ya 1 "
            "normal aur fraud transactions ko alag ginta hai. Fraud percentage "
            "fraud transactions ko total transactions se divide karke 100 se "
            "multiply karti hai.",
            "AVG(Amount) har class ki ausat transaction raqam nikalta hai. "
            "Amount ranges wali query har range mein total aur fraud transactions "
            "ginti hai, phir us range ka fraud rate percentage nikalti hai.",
            "Yeh report CSV se SQL ke zariye chalaye gaye asal calculations dikhati hai.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    project_folder = Path(__file__).parent
    data_file = project_folder / "data" / "creditcard.csv"
    data = pd.read_csv(data_file)

    required_columns = {"Class", "Amount"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))
        raise ValueError(f"The dataset is missing required columns: {missing_names}")

    sql_folder = project_folder / "sql"
    reports_folder = project_folder / "reports"
    sql_folder.mkdir(parents=True, exist_ok=True)
    reports_folder.mkdir(parents=True, exist_ok=True)

    query_text = "\n\n".join(
        f"-- {title}\n{textwrap.dedent(query).strip()}"
        for title, query in SQL_QUERIES.items()
    )
    (sql_folder / "fraud_queries.sql").write_text(query_text + "\n", encoding="utf-8")

    with sqlite3.connect(":memory:") as connection:
        data.to_sql("transactions", connection, if_exists="replace", index=False)
        report = build_report(connection)

    report_file = reports_folder / "sql_summary.txt"
    report_file.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"\nReport saved to: {report_file}")
    print(f"SQL queries saved to: {sql_folder / 'fraud_queries.sql'}")


if __name__ == "__main__":
    main()
