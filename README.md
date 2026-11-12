# Home Credit Risk Detection

A 30-feature XGBoost model with one FastAPI backend and one Streamlit screen.

## Setup

Use Python 3.11+ and run commands from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train

Download the [Home Credit dataset](https://www.kaggle.com/competitions/home-credit-default-risk/data)
and extract these six CSV files into `data/raw/`:

- `application_train.csv` and `application_test.csv`
- `bureau.csv` and `previous_application.csv`
- `credit_card_balance.csv` and `POS_CASH_balance.csv`

```powershell
python -m src.feature_engineering
python -m src.train_model
```

This saves processed data, `models/xgboost_home_credit_pipeline.joblib`, and
`reports/final_model_metrics.json`. Data and model binaries stay out of Git.
On a fresh clone, train first or supply a compatible trusted model artifact.

## Run

Start the API, then the UI in a second terminal using the same environment:

```powershell
python -m uvicorn src.api:app --port 8000
python -m streamlit run app.py
```

Open the UI at `http://localhost:8501` or API docs at `http://localhost:8000/docs`.
For Docker, supply the trained model and run `docker compose up --build`.

## Project layout

```text
app.py                     Streamlit UI
src/
  config.py                Paths, threshold and feature list
  feature_engineering.py   Raw CSVs → 30 model features
  train_model.py           Train, evaluate and save
  api.py                   Validate inputs and predict
tests/                     API, feature, training and UI checks
reports/                   Latest model metrics
```

## Test

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Checks needing local data or the saved model skip when those files are absent.

Optional environment variables: `MODEL_PATH` selects the model artifact,
`CLASSIFICATION_THRESHOLD` sets the decision cutoff (default `0.60`), and
`API_URL` sets the UI's backend URL (default `http://127.0.0.1:8000`).

Use verified external scores normalized to 0–1; UI defaults of 0.5 are
placeholders. Financial amounts must use the training data's units. The model
includes demographic inputs and produces uncalibrated scores for review, not
automatic lending decisions. Reported metrics come from a fixed validation split.
