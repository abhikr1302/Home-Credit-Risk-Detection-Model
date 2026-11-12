# Home Credit Risk Detection Model

XGBoost credit-risk prediction with a FastAPI backend and Streamlit UI.
Production uses 30 features and a saved preprocessing pipeline.

## Setup

Use Python 3.11 or newer. Run commands from the repository root in an activated
virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-ci.txt
```

The scikit-learn and XGBoost versions are pinned to match the model artifacts.
Load joblib files only from trusted sources.

## Dataset

This project uses the Home Credit Default Risk dataset.

The dataset is not included in this repository because of its size.

1. Download the dataset from Kaggle:
   https://www.kaggle.com/competitions/home-credit-default-risk/data
2. Extract the CSV files.
3. Place them inside:

   data/raw/

Expected files include:

- application_train.csv
- application_test.csv
- bureau.csv
- bureau_balance.csv
- previous_application.csv
- installments_payments.csv
- credit_card_balance.csv
- POS_CASH_balance.csv

## Training

```powershell
python -m src.data_ingestion
python -m src.feature_engineering
python -m src.model_training
```

Ingestion is optional for feature engineering, which reads raw CSVs directly.
Production training uses the fixed contract in `src/model_contract.py`, saves
`models/xgboost_home_credit_pipeline.joblib`, and updates the final model reports.
Data and model binaries are excluded from Git, so a fresh clone needs training
or a compatible artifact before serving predictions.

`python -m src.feature_selection` re-evaluates feature recommendations.
`python -m src.train_model` compares baseline and enhanced models and saves
experimental artifacts separately from production. The historically named
`python -m src.compare_30_vs_50` runs an additional-feature ablation experiment;
the actual feature counts depend on its inputs.

## Running

Start the banking API:

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

In a second terminal using the same environment:

```powershell
python -m streamlit run app.py
```

`python -m streamlit run ui/streamlit_app.py` opens the same screen.
API documentation is at `http://127.0.0.1:8000/docs`.
`GET /health` returns 503 when the model cannot load or its features differ from
the production contract. `POST /predict` accepts banking inputs and returns
`risk_probability`, `risk_level`, `decision`, financial indicators and flags.

Example request:

```json
{
  "age": 35,
  "annual_income": 600000,
  "loan_amount": 300000,
  "monthly_emi": 7500,
  "external_score_1": 0.7,
  "external_score_2": 0.7,
  "external_score_3": 0.7
}
```

External scores must be normalized to 0–1. The UI exposes these and other model
inputs in an expander; defaults of 0.5 are placeholders, not retrieved scores.
Financial inputs must use training-data units. The sample UI's rupee labels do
not perform currency conversion.

The legacy API, `python -m uvicorn app.main:app --port 8000`, accepts
`{"applicant_id": 123, "features": {...}}` with engineered model features.
It is a separate integration contract. The Streamlit UI requires `src.api:app`;
run only one API on a given port.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_PATH` | `models/xgboost_home_credit_pipeline.joblib` | Model artifact; relative paths resolve against the repository |
| `CLASSIFICATION_THRESHOLD` | `0.60` | Decision cutoff, strictly between 0 and 1 |
| `API_URL` | `http://127.0.0.1:8000` | Streamlit's backend URL |

## Testing

```powershell
python -m pytest -v
```

Automated tests in `tests/` cover both API contracts, validation, features,
unavailable models, invalid model outputs, and Streamlit submission and display.
Tests requiring local data/model files skip on clean clones.
Root-level `test_api.py`, `test_model_cases.py` and `test_feature_sensitivity.py`
are manual HTTP probes requiring a running banking API.

## Docker

Train or supply the production artifact, then run:

```powershell
docker compose up --build api
```

Compose mounts `models/` read-only. Optional PostgreSQL needs `POSTGRES_DB`,
`POSTGRES_USER`, `POSTGRES_PASSWORD` and `POSTGRES_PORT` in a local `.env`.
It is not needed for inference. SQL scripts target an existing `raw` schema;
they do not create or populate its tables.

## Interpretation

Saved metrics use a fixed stratified validation split, not an independent
deployment evaluation. Class-weighted scores have not been demonstrated to be
calibrated default probabilities. Results support review and do not automatically
approve or reject loans. Demographic features remain in the model and need
assessment before any real lending deployment.
