CREATE UNIQUE INDEX IF NOT EXISTS
    idx_application_train_curr
ON raw.application_train (sk_id_curr);

CREATE UNIQUE INDEX IF NOT EXISTS
    idx_application_test_curr
ON raw.application_test (sk_id_curr);

CREATE INDEX IF NOT EXISTS
    idx_bureau_curr
ON raw.bureau (sk_id_curr);

CREATE UNIQUE INDEX IF NOT EXISTS
    idx_bureau_id
ON raw.bureau (sk_id_bureau);

CREATE INDEX IF NOT EXISTS
    idx_bureau_balance_id
ON raw.bureau_balance (sk_id_bureau);

CREATE UNIQUE INDEX IF NOT EXISTS
    idx_previous_application_prev
ON raw.previous_application (sk_id_prev);

CREATE INDEX IF NOT EXISTS
    idx_previous_application_curr
ON raw.previous_application (sk_id_curr);

CREATE INDEX IF NOT EXISTS
    idx_installments_curr
ON raw.installments_payments (sk_id_curr);

CREATE INDEX IF NOT EXISTS
    idx_installments_prev
ON raw.installments_payments (sk_id_prev);

CREATE INDEX IF NOT EXISTS
    idx_credit_card_curr
ON raw.credit_card_balance (sk_id_curr);

CREATE INDEX IF NOT EXISTS
    idx_credit_card_prev
ON raw.credit_card_balance (sk_id_prev);

CREATE INDEX IF NOT EXISTS
    idx_pos_cash_curr
ON raw.pos_cash_balance (sk_id_curr);

CREATE INDEX IF NOT EXISTS
    idx_pos_cash_prev
ON raw.pos_cash_balance (sk_id_prev);