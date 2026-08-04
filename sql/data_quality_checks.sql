--Validate the target

SELECT
    target,
    COUNT(*) AS applicant_count,
    ROUND(
        100.0 * COUNT(*) /
        SUM(COUNT(*)) OVER (),
        2
    ) AS percentage
FROM raw.application_train
GROUP BY target
ORDER BY target;

--Check invalid financial values

SELECT
    COUNT(*) FILTER (
        WHERE amt_income_total IS NULL
    ) AS missing_income,

    COUNT(*) FILTER (
        WHERE amt_income_total <= 0
    ) AS non_positive_income,

    COUNT(*) FILTER (
        WHERE amt_credit IS NULL
    ) AS missing_credit,

    COUNT(*) FILTER (
        WHERE amt_credit <= 0
    ) AS non_positive_credit,

    COUNT(*) FILTER (
        WHERE amt_annuity IS NULL
    ) AS missing_annuity,

    COUNT(*) FILTER (
        WHERE amt_annuity <= 0
    ) AS non_positive_annuity

FROM raw.application_train;

--Check related applicant coverage

WITH all_applicants AS (
    SELECT sk_id_curr
    FROM raw.application_train

    UNION

    SELECT sk_id_curr
    FROM raw.application_test
)
SELECT
    COUNT(*) AS unmatched_bureau_rows
FROM raw.bureau bureau
LEFT JOIN all_applicants applicants
    ON bureau.sk_id_curr = applicants.sk_id_curr
WHERE applicants.sk_id_curr IS NULL;

--Check previous applications

WITH all_applicants AS (
    SELECT sk_id_curr
    FROM raw.application_train

    UNION

    SELECT sk_id_curr
    FROM raw.application_test
)
SELECT
    COUNT(*) AS unmatched_previous_rows
FROM raw.previous_application previous
LEFT JOIN all_applicants applicants
    ON previous.sk_id_curr = applicants.sk_id_curr
WHERE applicants.sk_id_curr IS NULL;

--Check bureau-balance relationships

SELECT COUNT(*) AS unmatched_bureau_balance_rows
FROM raw.bureau_balance balance
LEFT JOIN raw.bureau bureau
    ON balance.sk_id_bureau = bureau.sk_id_bureau
WHERE bureau.sk_id_bureau IS NULL;

--Check previous-application relationships

SELECT COUNT(*) AS unmatched_installment_rows
FROM raw.installments_payments installment
LEFT JOIN raw.previous_application previous
    ON installment.sk_id_prev = previous.sk_id_prev
WHERE previous.sk_id_prev IS NULL;