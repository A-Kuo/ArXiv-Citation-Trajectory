# ArXiv Citation Trajectory Prediction

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikitlearn) ![FastAPI](https://img.shields.io/badge/FastAPI-serving-teal?logo=fastapi) ![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red?logo=streamlit) ![Status](https://img.shields.io/badge/Status-Active-brightgreen)

> **Predicts the citation trajectory of ArXiv ML papers from metadata available on day one — with honestly calibrated uncertainty instead of overconfident point estimates.**

## Problem

Which papers will matter is obvious in hindsight and hard on submission day. Citation prediction is genuinely difficult: text and metadata carry weak signal, and most published models hide that behind cherry-picked test splits. This project takes the opposite stance — build the full pipeline (data → features → models → API → dashboard), report holdout results as they are, and make the uncertainty intervals trustworthy enough to act on.

## Approach

An end-to-end pipeline: papers from the ArXiv API (cs.LG, cs.AI, stat.ML, econ.GN, 2019–2022), citation counts at 12/24 months from OpenAlex, engineered text/metadata features (readability via textstat, author counts, category, timing), then regularized linear models and gradient boosting on log-citations with residual-based conformal-style prediction intervals.

```
ArXiv API (~5,000 papers) ──┐
                            ├─► SQLite/PostgreSQL store ─► Feature engineering
OpenAlex citations (12/24mo)┘        (resumable, rate-limited)      │
                                                                    ▼
                          Streamlit dashboard ◄── LASSO / Ridge / ElasticNet / GBM
                          FastAPI /predict  ◄──── + calibrated prediction intervals
```

## Results

**Interval calibration (the headline result)** — nominal coverage matches empirical coverage almost exactly on the 20% holdout:

| Nominal coverage | Empirical coverage |
|------------------|--------------------|
| 68% | 69.5% |
| 90% | 89.8% |
| 95% | 95.3% |
| 99% | 99.2% |

**Point prediction (reported honestly)** — day-one metadata alone is a weak predictor, and the numbers say so:

| Model | Test R² (log scale) | Test MAE | Test RMSE |
|-------|--------------------:|---------:|----------:|
| LASSO | 0.010 | 0.844 | 1.055 |
| ElasticNet | 0.009 | 0.845 | 1.055 |
| Ridge | -0.041 | 0.866 | 1.082 |
| GBM | -0.049 | 0.878 | 1.085 |

_Source: [ADVANCED_ANALYSIS_REPORT.md](ADVANCED_ANALYSIS_REPORT.md), 20% holdout. Additional findings: citation velocity decays after month 12 (0.65× ratio), and 145 "fast-track" papers reach the top quartile within 12 months — motivating the trajectory-classification reframing below._

## Tech Stack

- **ML**: scikit-learn (LASSO/Ridge/ElasticNet), XGBoost, LightGBM
- **Data**: pandas, NumPy, SQLAlchemy (SQLite → PostgreSQL), textstat, pyarrow
- **Serving**: FastAPI + pydantic (`/predict`, batch, health checks), Streamlit + Plotly dashboard
- **Quality**: pytest suite (feature engineering, pipeline, model evaluation), data-quality reporting built into ingestion

## Quick Start

```bash
git clone https://github.com/A-Kuo/ArXiv-Citation-Trajectory.git
cd ArXiv-Citation-Trajectory
pip install -r requirements.txt

python pipeline.py            # fetch papers + citations (resumable, rate-limited)
python advanced_analysis.py   # holdout evaluation + calibration report
python api.py                 # REST API at http://localhost:8000/docs
streamlit run streamlit_app.py
```

## Project Structure

```
├── pipeline.py                 # ArXiv + OpenAlex ingestion (resume-safe, rate-limited)
├── feature_engineering.py      # text/metadata feature construction
├── advanced_analysis.py        # holdout eval, calibration, velocity analysis
├── model_results.py            # model comparison utilities
├── api.py                      # FastAPI prediction service
├── streamlit_app.py            # interactive dashboard
├── tests/                      # pytest suite (features, pipeline, evaluation)
└── ADVANCED_ANALYSIS_REPORT.md # full results writeup
```

## ML Details

**Why log-scale regression with conformal-style intervals?** Citation counts are heavy-tailed; modeling log(1+citations) with residual-std intervals gives calibrated bands that survive the holdout (see table above). Calibration was validated at four nominal levels rather than assumed.

**Why the near-zero R² is a feature of the report, not a bug of the project.** Metadata-only signal on day one is weak — a result consistent with the citation-prediction literature. The pipeline was built to measure that honestly (train/test R² comparison for overfitting diagnosis) rather than inflate it. The velocity analysis points to the more tractable reframing now in progress: classifying *trajectory shape* (fast-track vs. slow-burn, using the 145 identified fast-track papers) and adding abstract embeddings + temporal citation-graph features, where the signal actually lives.

**Rejected along the way:** unregularized linear models (unstable on correlated text features) and deeper GBMs (negative holdout R² — memorization, not signal).

## Status

🔧 Active development. Ingestion, evaluation, API, and dashboard are functional; current work is abstract-embedding features and trajectory-shape classification on the fast-track cohort.

## Author

**Austin Kuo** | [GitHub](https://github.com/A-Kuo) | ML Engineer & Data Engineer
