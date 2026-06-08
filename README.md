# ArXiv Citation Trajectory

Predicts the 24-month citation impact of arXiv papers using a full ML pipeline — from data collection through a REST API and interactive dashboard.

**Stack:** Python · FastAPI · Streamlit · scikit-learn · SQLite · pytest

---

## Overview

| Phase | What it does |
|-------|-------------|
| **1 — Data pipeline** | Fetches ~5,000 papers (2019–2022) from arXiv API; enriches with 12-month and 24-month citation counts from OpenAlex |
| **2 — Feature engineering** | Extracts text (TF-IDF), metadata, and abstract quality signals; handles nulls, standardises |
| **3 — ML models** | Trains LASSO, Ridge, ElasticNet, Gradient Boosting; compares on held-out test set with calibrated confidence intervals |
| **4 — Dashboard & API** | Streamlit dashboard for interactive analysis; FastAPI REST endpoint for single and batch predictions |

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Collect data (~40 min, rate-limited)
python pipeline.py

# 2. Build features and train models
python feature_engineering.py
python model_results.py

# 3. Launch dashboard
streamlit run streamlit_app.py

# 4. Launch REST API
python api.py          # → http://localhost:8000
                       # → http://localhost:8000/docs  (interactive docs)
```

---

## REST API

```
GET  /health            Health check
GET  /model/info        Model metadata and feature list
POST /predict           Single-paper prediction
POST /predict/batch     Batch prediction (up to 100 papers)
```

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "Attention Is All You Need", "abstract": "...", "category": "cs.LG", "num_authors": 8}'
```

Returns predicted 24-month citation count with 95% confidence interval.

---

## Model performance

Evaluated on a 20% holdout set (log-scale targets):

| Model | Test R² | Test MAE |
|-------|---------|----------|
| LASSO | 0.010 | 0.844 |
| Ridge | −0.041 | 0.866 |
| ElasticNet | 0.009 | 0.845 |
| Gradient Boosting | −0.049 | 0.878 |

Confidence intervals are empirically calibrated (95% nominal → 95.3% empirical coverage).

See [`ADVANCED_ANALYSIS_REPORT.md`](ADVANCED_ANALYSIS_REPORT.md) for full metrics: citation velocity, author-count effects, keyword performance, and stratified breakdowns.

---

## Project structure

```
pipeline.py               Data collection (arXiv + OpenAlex)
feature_engineering.py    Feature extraction and preprocessing
model_results.py          Model training, evaluation, comparison
advanced_analysis.py      Stratified metrics, confidence calibration
metrics_deep_dive.py      Granular performance breakdowns
api.py                    FastAPI REST server
streamlit_app.py          Interactive Streamlit dashboard
streamlit_app_enhanced.py Dashboard with SHAP-like explanations and batch analysis
demo.py                   Quick single-prediction demo
demo_full_pipeline.py     End-to-end pipeline demo
tests/                    pytest test suite
```

---

## Database schema

SQLite (`citations.db`), two tables:

**papers** — `arxiv_id`, `title`, `abstract`, `authors`, `category`, `submitted_date`

**citations** — `arxiv_id`, `citations_12mo`, `citations_24mo`, `fetched_date`

PostgreSQL is also supported; set `DB_URL` env var to a Postgres connection string.

---

## Testing

```bash
pytest tests/ -v
```

Coverage includes pipeline unit tests, feature engineering, and model evaluation checks.

---

## Configuration

Key variables in `pipeline.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_PATH` | `citations.db` | SQLite database path |
| `categories` | cs.LG, cs.AI, econ.GN, stat.ML | arXiv categories |
| `start_date` / `end_date` | 2019-01-01 / 2022-12-31 | Date range |
| `target_count` | 5000 | Papers to fetch |
