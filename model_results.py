#!/usr/bin/env python3
"""
Model analysis for ArXiv citation prediction.

Runs LASSO and Random Forest on log1p(citations_24mo),
compares models, analyses confounders, and writes MODEL_REPORT.md.

Usage:
    python model_results.py                     # expects features.parquet + targets.parquet
    python model_results.py --demo              # generates synthetic data first
"""

import argparse
import json
import logging
import re
import sqlite3
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

RANDOM_STATE = 42
N_FOLDS = 5


# ---------------------------------------------------------------------------
# Synthetic data generator (used in --demo mode)
# ---------------------------------------------------------------------------

def _make_synthetic_db(db_path: str, n_papers: int = 600) -> None:
    """Write a synthetic citations.db with realistic citation distributions."""
    rng = np.random.default_rng(RANDOM_STATE)

    categories = ["cs.LG", "cs.AI", "stat.ML", "econ.GN"]
    keywords = ["survey", "benchmark", "state-of-the-art", "novel", "we propose"]
    keyword_boost = {"survey": 1.4, "benchmark": 1.3, "state-of-the-art": 1.2,
                     "novel": 1.05, "we propose": 1.0}

    word_pool = [
        "learning", "neural", "network", "deep", "model", "training",
        "optimization", "gradient", "attention", "transformer", "graph",
        "reinforcement", "generative", "representation", "classification",
        "regression", "inference", "bayesian", "causal", "distribution",
        "language", "vision", "image", "text", "policy", "reward",
        "data", "dataset", "baseline", "evaluation", "empirical",
        "theoretical", "convergence", "variance", "bias",
    ]
    bigrams = [
        "machine learning", "deep learning", "neural network", "gradient descent",
        "attention mechanism", "transfer learning", "few shot", "zero shot",
        "language model", "image classification", "random forest",
    ]

    papers = []
    citations = []

    for i in range(n_papers):
        arxiv_id = f"{2019 + i // 150:04d}.{i:05d}"
        cat = rng.choice(categories, p=[0.40, 0.30, 0.20, 0.10])
        year = int(2019 + i // 150)
        month = int(rng.integers(1, 13))
        day = int(rng.integers(1, 29))
        submitted = f"{year}-{month:02d}-{day:02d}"

        n_authors = int(rng.integers(1, 9))

        # Decide which keywords appear in this paper
        kws_present = [k for k in keywords if rng.random() < 0.15]
        kw_phrase = " ".join(kws_present)

        # Abstract: mix word pool + bigrams + keywords
        n_words = int(rng.integers(60, 200))
        body_words = list(rng.choice(word_pool, n_words, replace=True))
        if rng.random() < 0.3:
            body_words += list(rng.choice(bigrams, int(rng.integers(1, 4)), replace=True))
        n_eq = int(rng.integers(0, 8))
        eq_str = " $ " * n_eq
        n_cite_refs = int(rng.integers(0, 5))
        cite_str = " ".join(f"[{j+1}]" for j in range(n_cite_refs))
        abstract = " ".join(body_words) + " " + eq_str + " " + cite_str
        if kws_present:
            abstract = kw_phrase + " " + abstract

        title_words = list(rng.choice(word_pool, int(rng.integers(3, 10)), replace=True))
        title = " ".join(title_words).capitalize()

        authors = ", ".join(f"Author{rng.integers(100,999)}" for _ in range(n_authors))

        # ------------------------------------------------------------------
        # Citation model: log-normal base, with interpretable signals
        # ------------------------------------------------------------------
        log_mu = 2.5  # ~exp(2.5) ≈ 12 citations baseline
        # Category norms (ML papers cited more than econ)
        cat_effect = {"cs.LG": 0.5, "cs.AI": 0.3, "stat.ML": 0.2, "econ.GN": -0.4}
        log_mu += cat_effect[cat]
        # More authors → slightly higher citation rate (established teams)
        log_mu += 0.08 * n_authors
        # Year effect: older papers had more time to accumulate (even within 24mo window)
        log_mu += 0.1 * (year - 2019)
        # Keyword boosts
        for kw in kws_present:
            log_mu += np.log(keyword_boost[kw])
        # Equation richness: negative for readability-challenged papers
        log_mu -= 0.03 * n_eq
        # Noise
        log_mu += rng.normal(0, 0.8)

        c24 = max(0, int(rng.lognormal(log_mu, 0.5)))
        c12 = max(0, int(c24 * rng.uniform(0.3, 0.7)))

        papers.append((arxiv_id, title, abstract, authors, cat, submitted))
        citations.append((arxiv_id, c12, c24, "2024-01-01"))

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS papers")
    cur.execute("DROP TABLE IF EXISTS citations")
    cur.execute("""CREATE TABLE papers (
        arxiv_id TEXT PRIMARY KEY, title TEXT, abstract TEXT,
        authors TEXT, category TEXT, submitted_date TEXT)""")
    cur.execute("""CREATE TABLE citations (
        arxiv_id TEXT PRIMARY KEY, citations_12mo INTEGER,
        citations_24mo INTEGER, fetched_date TEXT)""")
    cur.executemany("INSERT INTO papers VALUES (?,?,?,?,?,?)", papers)
    cur.executemany("INSERT INTO citations VALUES (?,?,?,?)", citations)
    conn.commit()
    conn.close()
    log.info(f"Synthetic DB written: {db_path} ({n_papers} papers)")


def _make_parquets(db_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run feature engineering and return (features, targets)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from feature_engineering import FeatureEngineer
    import feature_engineering as fe_module

    original = fe_module.DB_PATH
    fe_module.DB_PATH = db_path
    try:
        eng = FeatureEngineer(
            db_path=db_path,
            cutoff_date=datetime(2026, 1, 1),  # everything qualifies in demo
        )
        features, targets, arxiv_ids = eng.run()
    finally:
        fe_module.DB_PATH = original

    features.to_parquet("features.parquet", index=False)
    targets.to_parquet("targets.parquet", index=False)
    arxiv_ids.to_parquet("arxiv_ids.parquet", index=False)
    log.info(f"Wrote features.parquet {features.shape}, targets.parquet {targets.shape}")
    return features, targets


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(demo: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if demo:
        log.info("Demo mode: generating synthetic database and parquets")
        _make_synthetic_db("demo_citations.db")
        return _make_parquets("demo_citations.db")

    fp = Path("features.parquet")
    tp = Path("targets.parquet")
    if not fp.exists() or not tp.exists():
        raise FileNotFoundError(
            "features.parquet or targets.parquet not found. "
            "Run `python feature_engineering.py` first, or use --demo."
        )
    features = pd.read_parquet(fp)
    targets = pd.read_parquet(tp)
    log.info(f"Loaded features {features.shape}, targets {targets.shape}")
    return features, targets


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def preprocess(features: pd.DataFrame, targets: pd.DataFrame
               ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Impute, scale, return (X_scaled, y, feature_names)."""
    y = targets["citations_24mo_log"].values.astype(float)

    # Fill any NaNs (e.g. flesch on empty strings)
    X = features.fillna(0.0)
    col_names = list(X.columns)
    X_arr = X.values.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)
    return X_scaled, y, col_names


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def baseline_stats(y: np.ndarray) -> dict:
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_pred_all = np.empty_like(y)
    for train_idx, test_idx in kf.split(y):
        fold_mean = y[train_idx].mean()
        y_pred_all[test_idx] = fold_mean

    r2 = r2_score(y, y_pred_all)
    mae = mean_absolute_error(y, y_pred_all)
    log.info(f"Baseline — R²={r2:.4f}  MAE={mae:.4f}")
    return {"name": "Baseline (predict mean)", "r2_cv": r2, "mae_cv": mae,
            "notes": "Constant prediction: fold training mean"}


# ---------------------------------------------------------------------------
# LASSO
# ---------------------------------------------------------------------------

def run_lasso(X: np.ndarray, y: np.ndarray, col_names: list[str]) -> dict:
    log.info("Running LassoCV (5-fold, 50 alphas) …")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    lasso_cv = LassoCV(
        cv=kf,
        alphas=50,
        max_iter=10_000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    lasso_cv.fit(X, y)
    best_alpha = lasso_cv.alpha_
    log.info(f"Best alpha: {best_alpha:.6f}")

    # Refit final model and compute CV metrics on the chosen alpha
    lasso_final = lasso_cv  # already fit on all data
    coefs = pd.Series(lasso_final.coef_, index=col_names)
    n_nonzero = (coefs != 0).sum()
    log.info(f"Non-zero coefficients: {n_nonzero}")

    # Cross-validated R² and MAE with the chosen alpha
    from sklearn.linear_model import Lasso
    lasso_fixed = Lasso(alpha=best_alpha, max_iter=10_000)
    cv_r2 = cross_val_score(
        lasso_fixed, X, y, cv=kf, scoring="r2", n_jobs=-1
    ).mean()
    cv_mae = -cross_val_score(
        lasso_fixed, X, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1
    ).mean()

    log.info(f"LASSO — R²={cv_r2:.4f}  MAE={cv_mae:.4f}")

    # Signed top predictors — strictly filter by sign so tables never cross-contaminate
    top_pos = coefs[coefs > 0].sort_values(ascending=False)
    top_neg = coefs[coefs < 0].sort_values()  # most negative first

    return {
        "name": "LASSO",
        "r2_cv": cv_r2,
        "mae_cv": cv_mae,
        "notes": f"alpha={best_alpha:.4f}, {n_nonzero} non-zero features",
        "best_alpha": best_alpha,
        "n_nonzero": int(n_nonzero),
        "top_positive": top_pos,
        "top_negative": top_neg,
        "all_coefs": coefs,
    }


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def run_random_forest(X: np.ndarray, y: np.ndarray, col_names: list[str]) -> dict:
    log.info("Training Random Forest (200 trees, max_depth=8, oob=True) …")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        oob_score=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X, y)
    oob_r2 = rf.oob_score_
    log.info(f"OOB R²: {oob_r2:.4f}")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_r2 = cross_val_score(
        RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1
        ),
        X, y, cv=kf, scoring="r2", n_jobs=1,
    ).mean()
    cv_mae = -cross_val_score(
        RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1
        ),
        X, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=1,
    ).mean()

    importances = pd.Series(rf.feature_importances_, index=col_names).sort_values(
        ascending=False
    )

    log.info(f"RF — R²(CV)={cv_r2:.4f}  OOB R²={oob_r2:.4f}  MAE={cv_mae:.4f}")

    return {
        "name": "Random Forest",
        "r2_cv": cv_r2,
        "mae_cv": cv_mae,
        "notes": f"200 trees, max_depth=8, OOB R²={oob_r2:.4f}",
        "oob_r2": oob_r2,
        "importances": importances,
    }


# ---------------------------------------------------------------------------
# Confounding variable analysis
# ---------------------------------------------------------------------------

def confounding_analysis(
    X: np.ndarray,
    y: np.ndarray,
    col_names: list[str],
    lasso_result: dict,
) -> dict:
    """Drop author_count and refit LASSO; compare R²."""
    log.info("Confounding analysis: author_count ablation …")

    try:
        author_idx = col_names.index("author_count")
    except ValueError:
        log.warning("author_count not found in features; skipping confounding analysis")
        return {"r2_full": lasso_result["r2_cv"], "r2_no_author": None,
                "delta_r2": None, "interpretation": "author_count not present in features"}

    mask = [i for i in range(len(col_names)) if i != author_idx]
    X_no_author = X[:, mask]

    from sklearn.linear_model import Lasso
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    lasso_no_author = LassoCV(cv=kf, alphas=50, max_iter=10_000,
                               random_state=RANDOM_STATE, n_jobs=-1)
    lasso_no_author.fit(X_no_author, y)

    r2_no_author = cross_val_score(
        Lasso(alpha=lasso_no_author.alpha_, max_iter=10_000),
        X_no_author, y, cv=kf, scoring="r2", n_jobs=-1,
    ).mean()

    r2_full = lasso_result["r2_cv"]
    delta = r2_full - r2_no_author
    pct_change = (delta / abs(r2_full) * 100) if r2_full != 0 else 0.0

    log.info(f"R² full={r2_full:.4f}  no-author={r2_no_author:.4f}  Δ={delta:.4f} ({pct_change:.1f}%)")

    if abs(pct_change) < 5:
        interpretation = (
            "Dropping author_count changes R² by less than 5%, suggesting "
            "team size is NOT a major confounder in this model. "
            "Writing quality signals (TF-IDF, readability) retain independent predictive value."
        )
    elif pct_change > 15:
        interpretation = (
            f"Dropping author_count reduces R² by {pct_change:.1f}%, indicating "
            "team size is a substantial confounder. Claims about writing quality "
            "should control for author count, as established teams may both "
            "write differently AND have larger citation networks."
        )
    else:
        interpretation = (
            f"Dropping author_count reduces R² by {pct_change:.1f}%, a moderate "
            "effect. Author count is a partial confounder; writing quality signals "
            "retain some independent predictive value but causal claims require caution."
        )

    return {
        "r2_full": r2_full,
        "r2_no_author": r2_no_author,
        "delta_r2": delta,
        "pct_change": pct_change,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _fmt_coef_table(series: pd.Series, n: int = 20, label: str = "") -> str:
    rows = [f"| {label} | Feature | Coefficient |",
            "|--------|---------|-------------|"]
    for rank, (feat, coef) in enumerate(series.head(n).items(), 1):
        clean = feat.replace("tfidf_", "").replace("has_", "has:").replace("cat_", "cat:")
        rows.append(f"| {rank} | `{clean}` | {coef:+.4f} |")
    return "\n".join(rows)


def _fmt_importance_table(series: pd.Series, n: int = 30) -> str:
    rows = ["| Rank | Feature | Importance |",
            "|------|---------|------------|"]
    for rank, (feat, imp) in enumerate(series.head(n).items(), 1):
        clean = feat.replace("tfidf_", "").replace("has_", "has:").replace("cat_", "cat:")
        rows.append(f"| {rank} | `{clean}` | {imp:.5f} |")
    return "\n".join(rows)


def generate_report(
    baseline: dict,
    lasso: dict,
    rf: dict,
    confound: dict,
    n_samples: int,
    n_features: int,
) -> str:
    ts = datetime.now().isoformat(timespec="seconds")
    r = []

    r.append("# MODEL REPORT — ArXiv Citation Prediction\n")
    r.append(f"Generated: {ts}  \n")
    r.append(f"Dataset: **{n_samples} papers**, **{n_features} features**  \n")
    r.append(f"Target: `log1p(citations_24mo)` (inverse: `expm1(pred)` gives citation count)\n\n")

    # ------------------------------------------------------------------
    r.append("---\n\n## 1. Model Comparison\n\n")
    r.append("| Model | CV R² | CV MAE (log scale) | Notes |\n")
    r.append("|-------|-------|--------------------|-------|\n")
    for m in [baseline, lasso, rf]:
        r2_str = f"{m['r2_cv']:.4f}"
        mae_str = f"{m['mae_cv']:.4f}"
        r.append(f"| {m['name']} | {r2_str} | {mae_str} | {m['notes']} |\n")
    r.append("\n> MAE is in log1p units. Multiply by ~2.3 to get rough factor on citation scale.\n\n")

    # ------------------------------------------------------------------
    r.append("---\n\n## 2. LASSO — Interpretable Predictors\n\n")
    r.append(f"**Best alpha (LassoCV, 5-fold):** `{lasso['best_alpha']:.6f}`  \n")
    r.append(f"**Non-zero features:** {lasso['n_nonzero']} of {n_features}  \n")
    r.append(f"**CV R²:** {lasso['r2_cv']:.4f}  \n\n")

    r.append("### 2.1 Top 20 Positive Predictors (more citations)\n\n")
    r.append(_fmt_coef_table(lasso["top_positive"], n=20, label="+"))
    r.append("\n\n")

    r.append("### 2.2 Top 20 Negative Predictors (fewer citations)\n\n")
    r.append(_fmt_coef_table(lasso["top_negative"], n=20, label="−"))
    r.append("\n\n")

    # -- LASSO word interpretation (only features with the correct sign)
    pos_tfidf = [(f, v) for f, v in lasso["top_positive"].items() if f.startswith("tfidf_")][:5]
    neg_tfidf = [(f, v) for f, v in lasso["top_negative"].items() if f.startswith("tfidf_")][:5]
    r.append("### 2.3 Abstract Word Interpretation\n\n")
    if pos_tfidf:
        r.append("**Words predicting MORE citations** (positive LASSO weight):\n")
        for feat, coef in pos_tfidf:
            r.append(f"- `{feat.replace('tfidf_','')}` (coef={coef:+.4f})\n")
        r.append("\n")
    else:
        r.append("**Words predicting MORE citations:** No TF-IDF terms survived LASSO regularisation "
                 "with a positive coefficient — structural/metadata features dominate.\n\n")
    if neg_tfidf:
        r.append("**Words predicting FEWER citations** (negative LASSO weight):\n")
        for feat, coef in neg_tfidf:
            r.append(f"- `{feat.replace('tfidf_','')}` (coef={coef:+.4f})\n")
        r.append("\n")
    else:
        r.append("**Words predicting FEWER citations:** No TF-IDF terms survived LASSO regularisation "
                 "with a negative coefficient.\n\n")

    # ------------------------------------------------------------------
    r.append("---\n\n## 3. Random Forest — Feature Importances\n\n")
    r.append(f"**OOB R²:** {rf['oob_r2']:.4f}  \n")
    r.append(f"**CV R²:** {rf['r2_cv']:.4f}  \n\n")
    r.append("### Top 30 Features by Importance\n\n")
    r.append(_fmt_importance_table(rf["importances"], n=30))
    r.append("\n\n")

    # ------------------------------------------------------------------
    r.append("---\n\n## 4. Confounding Variable Analysis — Author Count\n\n")
    r.append("**Question:** Does LASSO's predictive power drop significantly ")
    r.append("when `author_count` is removed? If so, writing quality signals ")
    r.append("may be proxying for team prestige rather than paper quality itself.\n\n")
    r.append("| Configuration | CV R² |\n")
    r.append("|---------------|-------|\n")
    r.append(f"| Full model (all features) | {confound['r2_full']:.4f} |\n")
    if confound["r2_no_author"] is not None:
        r.append(f"| Without `author_count` | {confound['r2_no_author']:.4f} |\n")
        r.append(f"| Δ R² | {confound['delta_r2']:+.4f} ({confound['pct_change']:+.1f}%) |\n")
    r.append("\n")
    r.append(f"**Interpretation:** {confound['interpretation']}\n\n")
    r.append("**What this means for the writing-quality question:**\n")
    r.append("Even if LASSO picks up TF-IDF and readability signals, those signals ")
    r.append("may reflect *who* writes the paper (established groups use domain jargon ")
    r.append("consistently) rather than *how well* it is written. Without randomising ")
    r.append("authorship, the causal claim 'better writing → more citations' cannot ")
    r.append("be confirmed from observational data alone.\n\n")

    # ------------------------------------------------------------------
    r.append("---\n\n## 5. Honest Limitations\n\n")

    r.append("### 5.1 Self-Citation Inflation\n")
    r.append("OpenAlex citation counts include self-citations, which inflate ")
    r.append("counts for prolific authors disproportionately. A paper by a ")
    r.append("10-person group where each member cites it in their next 3 papers ")
    r.append("gains 30 citations regardless of community interest. ")
    r.append("**Effect on models:** `author_count` will absorb some of this signal, ")
    r.append("making it a proxy for self-citation capacity as much as team prestige. ")
    r.append("Mitigation: filter to cross-citations only (requires per-citing-paper ")
    r.append("author-overlap check, not done here).\n\n")

    r.append("### 5.2 Category-Specific Citation Norms\n")
    r.append("ML/AI papers (cs.LG, cs.AI) cite at dramatically higher rates than ")
    r.append("economics papers (econ.GN). A median cs.LG paper may receive 5× the ")
    r.append("citations of a median econ.GN paper of identical quality. ")
    r.append("One-hot category encoding partially controls for this, but:\n")
    r.append("- LASSO/RF learn category offsets from training data; if category ")
    r.append("  composition shifts, predictions degrade.\n")
    r.append("- Cross-category comparisons of coefficients are invalid without ")
    r.append("  within-category standardisation.\n")
    r.append("- Subcategory effects (cs.LG/NLP vs cs.LG/RL) are not captured.\n\n")

    r.append("### 5.3 h-Index Unavailability and Causal Claims\n")
    r.append("We lack author h-index data. This matters because:\n")
    r.append("- High-h authors attract citations partly due to *reputation*, not paper content.\n")
    r.append("- TF-IDF signals from senior authors' preferred vocabulary will be ")
    r.append("  confounded with their citation gravity.\n")
    r.append("- Any feature correlated with author seniority (author count, abstract ")
    r.append("  sophistication, equation density) may be picking up reputation effects.\n")
    r.append("**Consequence:** R² improvements from text features **cannot** be ")
    r.append("interpreted as causal evidence that writing quality drives citations. ")
    r.append("They are predictive associations only. Establishing causality would ")
    r.append("require a design that holds author identity fixed (e.g. comparing ")
    r.append("revised vs original versions of the same paper).\n\n")

    r.append("### 5.4 Additional Limitations\n")
    r.append("- **Citation window bias:** Papers from 2019 had longer to accumulate ")
    r.append("citations than 2022 papers, even within the 24-month window, if ")
    r.append("OpenAlex's indexing lagged publication by weeks or months.\n")
    r.append("- **Abstract ≠ paper quality:** TF-IDF on the abstract misses the ")
    r.append("technical contribution in the body. A well-written abstract of a weak ")
    r.append("paper and a poorly-written abstract of a strong paper are not ")
    r.append("distinguishable here.\n")
    r.append("- **Flesch score mismatch:** Flesch reading ease was designed for ")
    r.append("general prose; technical abstracts with formulae routinely score ")
    r.append("near zero, compressing variation in the useful range.\n")
    r.append("- **Single-split targets:** P75 for `top_quartile` is computed on ")
    r.append("the full dataset, not per-fold, introducing mild leakage into ")
    r.append("binary classification tasks.\n\n")

    # ------------------------------------------------------------------
    r.append("---\n\n## 6. Reproducibility\n\n")
    r.append("```bash\n")
    r.append("# Full pipeline from scratch\n")
    r.append("pip install -r requirements.txt\n")
    r.append("python pipeline.py                  # Phase 1: fetch ArXiv + OpenAlex\n")
    r.append("python feature_engineering.py       # Phase 2: build parquets\n")
    r.append("python model_results.py             # Phase 3: models + report\n\n")
    r.append("# Or run self-contained demo\n")
    r.append("python model_results.py --demo\n")
    r.append("```\n\n")
    r.append(f"Random seed: `{RANDOM_STATE}` (set for LassoCV, RF, KFold)\n")

    return "".join(r)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true",
                        help="Generate synthetic data and run analysis")
    args = parser.parse_args()

    # 1. Load data
    features, targets = load_data(demo=args.demo)
    n_samples, n_features = features.shape
    log.info(f"Dataset: {n_samples} samples × {n_features} features")

    # 2. Pre-process
    X, y, col_names = preprocess(features, targets)

    # 3. Models
    baseline = baseline_stats(y)
    lasso = run_lasso(X, y, col_names)
    rf = run_random_forest(X, y, col_names)
    confound = confounding_analysis(X, y, col_names, lasso)

    # 4. Print comparison table
    print("\n" + "=" * 64)
    print("MODEL COMPARISON")
    print("=" * 64)
    header = f"{'Model':<30} {'R²(CV)':>8} {'MAE(CV)':>10}"
    print(header)
    print("-" * 64)
    for m in [baseline, lasso, rf]:
        print(f"{m['name']:<30} {m['r2_cv']:>8.4f} {m['mae_cv']:>10.4f}")
    print("=" * 64)

    print("\nLASSO TOP 10 POSITIVE PREDICTORS:")
    for feat, coef in lasso["top_positive"].head(10).items():
        clean = feat.replace("tfidf_", "[tfidf] ").replace("has_", "[kw] ").replace("cat_", "[cat] ")
        print(f"  {coef:+.4f}  {clean}")

    print("\nLASSO TOP 10 NEGATIVE PREDICTORS:")
    for feat, coef in lasso["top_negative"].head(10).items():
        clean = feat.replace("tfidf_", "[tfidf] ").replace("has_", "[kw] ").replace("cat_", "[cat] ")
        print(f"  {coef:+.4f}  {clean}")

    print("\nRANDOM FOREST TOP 10 FEATURES:")
    for feat, imp in rf["importances"].head(10).items():
        clean = feat.replace("tfidf_", "[tfidf] ").replace("has_", "[kw] ").replace("cat_", "[cat] ")
        print(f"  {imp:.5f}  {clean}")

    print(f"\nCONFOUNDING (author_count ablation):")
    print(f"  R² with author_count:    {confound['r2_full']:.4f}")
    if confound["r2_no_author"] is not None:
        print(f"  R² without author_count: {confound['r2_no_author']:.4f}")
        print(f"  Δ R²:                    {confound['delta_r2']:+.4f} ({confound['pct_change']:+.1f}%)")
    print(f"\n  {confound['interpretation']}\n")

    # 5. Write report
    report_text = generate_report(
        baseline=baseline,
        lasso=lasso,
        rf=rf,
        confound=confound,
        n_samples=n_samples,
        n_features=n_features,
    )
    Path("MODEL_REPORT.md").write_text(report_text)
    log.info("MODEL_REPORT.md written")

    # 6. Save numeric results as JSON for downstream use
    results = {
        "generated": datetime.now().isoformat(),
        "n_samples": n_samples,
        "n_features": n_features,
        "baseline": {k: v for k, v in baseline.items() if isinstance(v, (int, float, str))},
        "lasso": {k: v for k, v in lasso.items() if isinstance(v, (int, float, str))},
        "rf": {k: v for k, v in rf.items() if isinstance(v, (int, float, str))},
        "confounding": {k: v for k, v in confound.items() if isinstance(v, (int, float, str, type(None)))},
    }
    Path("model_results_summary.json").write_text(json.dumps(results, indent=2))
    log.info("model_results_summary.json written")


if __name__ == "__main__":
    main()
