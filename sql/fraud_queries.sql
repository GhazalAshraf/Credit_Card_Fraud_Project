-- Total, normal, fraud, and fraud percentage
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

-- Average transaction amount for each class
SELECT
    Class AS transaction_class,
    COUNT(*) AS transaction_count,
    ROUND(AVG(Amount), 2) AS average_amount
FROM transactions
GROUP BY Class
ORDER BY Class;

-- Fraud counts and rates by transaction amount range
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
