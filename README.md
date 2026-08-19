# ArXiv Citation Trajectory

Two things live in this repo, sharing the same arXiv paper dataset:

1. **Citation prediction** — predicts the 24-month citation impact of arXiv papers using a full ML pipeline, from data collection through a REST API and interactive dashboard.
2. **Paper Q&A (RAG)** — hybrid-search retrieval + LLM-generated, cited answers to natural-language questions over the same indexed papers.

**Stack:** Python · FastAPI · Streamlit · scikit-learn · SQLite · pytest · rank-bm25 · Claude API

---

## Overview

| Phase | What it does |
|-------|-------------|
| **1 — Data pipeline** | Fetches ~5,000 papers (2019–2022) from arXiv API; enriches with 12-month and 24-month citation counts from OpenAlex |
| **2 — Feature engineering** | Extracts text (TF-IDF), metadata, and abstract quality signals; handles nulls, standardises |
| **3 — ML models** | Trains LASSO, Ridge, ElasticNet, Gradient Boosting; compares on held-out test set with calibrated confidence intervals |
| **4 — Dashboard & API** | Streamlit dashboard for interactive analysis; FastAPI REST endpoint for single and batch predictions |
| **5 — RAG paper Q&A** | Chunks + indexes the same papers (BM25 + vector, RRF-fused hybrid search); Claude generates grounded, cited answers |

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

# 5. (Optional) Build the RAG index and ask questions
python rag_indexing.py                       # builds rag_index/ from the papers table
export ANTHROPIC_API_KEY=sk-ant-...          # required for /rag/ask generation
streamlit run streamlit_app_enhanced.py      # → "📚 Paper Q&A" page
```

---

## REST API

```
GET  /health            Health check
GET  /model/info        Model metadata and feature list
POST /predict           Single-paper prediction
POST /predict/batch     Batch prediction (up to 100 papers)

GET  /rag/status         RAG index health/metadata
POST /rag/search         Hybrid (BM25 + vector) search, no LLM call
POST /rag/ask            Hybrid search + Claude-generated cited answer
```

Example — citation prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "Attention Is All You Need", "abstract": "...", "category": "cs.LG", "num_authors": 8}'
```

Returns predicted 24-month citation count with 95% confidence interval.

Example — paper Q&A:

```bash
curl -X POST http://localhost:8000/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What approaches have been proposed for transformer attention?", "top_k": 5}'
```

Returns a generated answer with inline `[N]` citations and the source excerpts they refer to. `/rag/ask` requires `ANTHROPIC_API_KEY`; `/rag/search` does not (retrieval only, no generation).

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
feature_schema.py         Shared FeatureBuilder (keeps training/serving features aligned)
model_results.py          Model training, evaluation, comparison
advanced_analysis.py      Stratified metrics, confidence calibration
metrics_deep_dive.py      Granular performance breakdowns
api.py                    FastAPI REST server (citation prediction + RAG endpoints)
streamlit_app.py          Interactive Streamlit dashboard
streamlit_app_enhanced.py Dashboard with SHAP-like explanations, batch analysis, and Paper Q&A
demo.py                   Quick single-prediction demo
demo_full_pipeline.py     End-to-end pipeline demo
rag_embeddings.py         Embedding backends (sentence-transformers, offline TF-IDF+SVD fallback)
rag_indexing.py           Chunking + hybrid index build/save/load
rag_retrieval.py          BM25 + vector search, Reciprocal Rank Fusion
rag_generation.py         Claude-based grounded answer generation
tests/                    pytest test suite
```

---

## RAG paper Q&A

Retrieval-augmented Q&A over the same `papers` table the citation-prediction pipeline uses — no separate ingestion step, no full-text PDF parsing (title + abstract only, for now).

```
question ─▶ hybrid_search() ─┬─▶ BM25Okapi (rank-bm25)         ─┐
                              └─▶ cosine similarity (dense vec) ─┴─▶ Reciprocal Rank Fusion ─▶ top_k chunks
                                                                                                   │
                                                                                                   ▼
                                                                                   generate_answer() → Claude API
                                                                                   (cited, grounded answer)
```

**Embeddings:** `rag_embeddings.py` tries `sentence-transformers` first (`all-MiniLM-L6-v2`) and falls back automatically to an offline TF-IDF+SVD projection if the model hub is unreachable or the package isn't installed — the rest of the pipeline (chunking, BM25, RRF, generation) behaves identically either way, so the whole system runs with zero network dependency when needed, and upgrades to real semantic embeddings transparently wherever HF access is available.

**Build the index:**

```bash
python rag_indexing.py    # writes rag_index/ (parquet + npy + pickle artifacts)
```

**Generation** requires `ANTHROPIC_API_KEY`; retrieval (`/rag/search`, or the Streamlit page before clicking "Ask") does not.

This is additive — it does not touch `feature_engineering.py`, `feature_schema.py`, `model_results.py`, or the `/predict` endpoints. The two systems share only the `papers` table.

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

Coverage includes pipeline unit tests, feature engineering, model evaluation checks, feature-schema alignment (train/serve consistency), and the RAG module (chunking, RRF fusion math, hybrid retrieval correctness, generation with a mocked LLM client, API endpoint wiring). All RAG tests run fully offline — no network calls, no API key required.

---

## Configuration

Key variables in `pipeline.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_PATH` | `citations.db` | SQLite database path |
| `categories` | cs.LG, cs.AI, econ.GN, stat.ML | arXiv categories |
| `start_date` / `end_date` | 2019-01-01 / 2022-12-31 | Date range |
| `target_count` | 5000 | Papers to fetch |
