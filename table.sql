DROP TABLE IF EXISTS "Element" CASCADE;

DROP TABLE IF EXISTS "Step" CASCADE;

DROP TABLE IF EXISTS "Thread" CASCADE;

DROP TABLE IF EXISTS "User" CASCADE;

SELECT
    *
FROM
    profiles
LIMIT
    1;

SELECT
    *
FROM
    financial_profiles
LIMIT
    1;

SELECT
    *
FROM
    financial_income
LIMIT
    1;

SELECT
    *
FROM
    financial_expenses
LIMIT
    1;

SELECT
    *
FROM
    tax_calculations
LIMIT
    1;

ALTER TABLE chat_sessions
DROP COLUMN extra_metadata;