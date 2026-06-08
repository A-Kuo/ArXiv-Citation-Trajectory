#!/usr/bin/env python3
"""
Comprehensive ML pipeline for ArXiv citation prediction.

Trains multiple models (LASSO, Ridge, ElasticNet, RF, XGBoost, LightGBM) on 80/20 train/test split.
Reports: Train/test R², MAE, RMSE; best alpha; non-zero coefficients; bias-variance tradeoff;
Y and X descriptive statistics; confounding analysis (author_count ablation).

Usage:
    python model_results.py                     # expects features.parquet + targets.parquet
    python model_results.py --demo              # generates synthetic data first

Database: Supports PostgreSQL (primary), MySQL, SQL Server via DATABASE_URL env var.
Falls back to SQLite for demo/local testing.
"""

import argparse
import json
import logging
import os
import pickle
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, Lasso, Ridge, RidgeCV, ElasticNetCV, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# Fixed random seed for reproducibility across all models
RANDOM_STATE = 42
SPLIT_SEED = 42
N_FOLDS = 5


# ============================================================================
# Synthetic data generator (for --demo mode)
# ============================================================================

def _make_synthetic_db(db_path: str = "demo_citations.db", n_papers: int = 600) -> None:
    """Generate synthetic citations database with realistic distributions."""
    import sqlite3
    rng = np.random.default_rng(RANDOM_STATE)

    categories = ["cs.LG", "cs.AI", "stat.ML", "econ.GN"]
    keywords = ["survey", "benchmark", "state-of-the-art", "novel", "we propose"]
    keyword_boost = {"survey": 1.4, "benchmark": 1.3, "state-of-the-art": 1.2,
                     "novel": 1.05, "we propose": 1.0}

    word_pool = [
        "learning", "neural", "network", "deep", "model", "training",
        "optimization", "gradient", "attention", "transformer", "graph",
        "reinforcement", "generative", "representation", "classification",
    ]
    bigrams = ["machine learning", "deep learning", "neural network", "gradient descent"]

    papers, citations = [], []

    for i in range(n_papers):
        arxiv_id = f"{2019 + i // 150:04d}.{i:05d}"
        cat = rng.choice(categories, p=[0.40, 0.30, 0.20, 0.10])
        year = int(2019 + i // 150)
        month, day = int(rng.integers(1, 13)), int(rng.integers(1, 29))
        submitted = f"{year}-{month:02d}-{day:02d}"
        n_authors = int(rng.integers(1, 9))

        kws_present = [k for k in keywords if rng.random() < 0.15]
        kw_phrase = " ".join(kws_present)

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

        log_mu = 2.5
        cat_effect = {"cs.LG": 0.5, "cs.AI": 0.3, "stat.ML": 0.2, "econ.GN": -0.4}
        log_mu += cat_effect[cat] + 0.08 * n_authors + 0.1 * (year - 2019)
        for kw in kws_present:
            log_mu += np.log(keyword_boost[kw])
        log_mu -= 0.03 * n_eq + rng.normal(0, 0.8)

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


def _make_parquets(db_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run feature engineering and return (features, targets)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from feature_engineering import FeatureEngineer
    import feature_engineering as fe_module

    original = fe_module.DB_PATH
    fe_module.DB_PATH = db_path
    try:
        eng = FeatureEngineer(db_path=db_path, cutoff_date=datetime(2026, 1, 1))
        features, targets, arxiv_ids = eng.run()
    finally:
        fe_module.DB_PATH = original

    features.to_parquet("features.parquet", index=False)
    targets.to_parquet("targets.parquet", index=False)
    arxiv_ids.to_parquet("arxiv_ids.parquet", index=False)
    log.info(f"Wrote features {features.shape}, targets {targets.shape}")

    if eng.tfidf_vectorizer is not None:
        with open("tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(eng.tfidf_vectorizer, f)

    return features, targets


# ============================================================================
# Data loading
# ============================================================================

def load_data(demo: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if demo:
        log.info("Demo mode: generating synthetic data")
        _make_synthetic_db("demo_citations.db")
        return _make_parquets("demo_citations.db")

    if not Path("features.parquet").exists() or not Path("targets.parquet").exists():
        raise FileNotFoundError("features.parquet or targets.parquet not found. Run feature_engineering.py or use --demo.")
    features = pd.read_parquet("features.parquet")
    targets = pd.read_parquet("targets.parquet")
    log.info(f"Loaded features {features.shape}, targets {targets.shape}")
    return features, targets


# ============================================================================
# Preprocessing
# ============================================================================

def preprocess(features: pd.DataFrame, targets: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    """Preprocessing pipeline: impute NaN → scale → return (X_scaled, y, feature_names).
    
    Steps:
    1. NaN imputation: fill with 0.0 (rare for numeric features, common for Flesch score)
    2. StandardScaler: z-score all features (mean=0, std=1) for linear models
    3. No feature selection: all features retained for initial analysis
    """
    y = targets["citations_24mo_log"].values.astype(float)
    X = features.fillna(0.0)
    col_names = list(X.columns)
    X_arr = X.values.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)
    return X_scaled, y, col_names


# ============================================================================
# Descriptive statistics (Y and X)
# ============================================================================

def compute_y_stats(y: np.ndarray) -> Dict[str, float]:
    """Compute summary statistics for target variable."""
    return {
        "n": len(y),
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "median": float(np.median(y)),
        "min": float(np.min(y)),
        "max": float(np.max(y)),
        "p25": float(np.percentile(y, 25)),
        "p75": float(np.percentile(y, 75)),
        "p90": float(np.percentile(y, 90)),
        "p99": float(np.percentile(y, 99)),
        "skewness": float(skew(y)),
        "kurtosis": float(kurtosis(y)),
    }


def compute_x_stats(X: np.ndarray, col_names: list) -> Dict[str, Dict]:
    """Compute summary stats for feature groups."""
    stats_by_group = {}
    groups = {"tfidf": [], "cat": [], "has": [], "other": []}

    for col in col_names:
        if col.startswith("tfidf_"):
            groups["tfidf"].append(col)
        elif col.startswith("cat_"):
            groups["cat"].append(col)
        elif col.startswith("has_"):
            groups["has"].append(col)
        else:
            groups["other"].append(col)

    for group_name, cols in groups.items():
        if cols:
            X_group = X[:, [col_names.index(c) for c in cols]]
            stats_by_group[group_name] = {
                "count": len(cols),
                "mean": float(np.mean(X_group)),
                "std": float(np.std(X_group)),
                "min": float(np.min(X_group)),
                "max": float(np.max(X_group)),
            }

    return stats_by_group


# ============================================================================
# Models: Train/test evaluation with metrics
# ============================================================================

class ModelResult:
    """Container for model results."""
    def __init__(self, name: str):
        self.name = name
        self.train_r2 = None
        self.test_r2 = None
        self.train_mae = None
        self.test_mae = None
        self.train_rmse = None
        self.test_rmse = None
        self.best_alpha = None
        self.n_nonzero = None
        self.cv_folds = None
        self.notes = ""

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


def train_test_models(X: np.ndarray, y: np.ndarray, col_names: list, test_size: float = 0.2
                     ) -> Tuple[list, StandardScaler, int, int]:
    """Train multiple models on 80/20 split. Return results, scaler, p (total features), m (sqrt(p))."""
    p = X.shape[1]
    m = int(np.sqrt(p))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SPLIT_SEED
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # Baseline
    baseline_pred_train = np.full_like(y_train, y_train.mean())
    baseline_pred_test = np.full_like(y_test, y_train.mean())
    r = ModelResult("Baseline (predict mean)")
    r.train_r2 = r2_score(y_train, baseline_pred_train)
    r.test_r2 = r2_score(y_test, baseline_pred_test)
    r.train_mae = mean_absolute_error(y_train, baseline_pred_train)
    r.test_mae = mean_absolute_error(y_test, baseline_pred_test)
    r.train_rmse = np.sqrt(mean_squared_error(y_train, baseline_pred_train))
    r.test_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred_test))
    results.append(r)

    # LASSO
    lasso_cv = LassoCV(cv=N_FOLDS, alphas=50, max_iter=10_000, random_state=RANDOM_STATE, n_jobs=-1)
    lasso_cv.fit(X_train_scaled, y_train)
    lasso_final = Lasso(alpha=lasso_cv.alpha_, max_iter=10_000)
    lasso_final.fit(X_train_scaled, y_train)
    r = ModelResult("LASSO")
    r.best_alpha = float(lasso_cv.alpha_)
    r.n_nonzero = int((lasso_final.coef_ != 0).sum())
    r.cv_folds = N_FOLDS
    r.train_r2 = r2_score(y_train, lasso_final.predict(X_train_scaled))
    r.test_r2 = r2_score(y_test, lasso_final.predict(X_test_scaled))
    r.train_mae = mean_absolute_error(y_train, lasso_final.predict(X_train_scaled))
    r.test_mae = mean_absolute_error(y_test, lasso_final.predict(X_test_scaled))
    r.train_rmse = np.sqrt(mean_squared_error(y_train, lasso_final.predict(X_train_scaled)))
    r.test_rmse = np.sqrt(mean_squared_error(y_test, lasso_final.predict(X_test_scaled)))
    r.notes = f"alpha={lasso_cv.alpha_:.6f}, {r.n_nonzero}/{p} non-zero"
    results.append(r)

    # Ridge
    ridge_cv = RidgeCV(cv=N_FOLDS, alphas=np.logspace(-4, 4, 50))
    ridge_cv.fit(X_train_scaled, y_train)
    ridge_final = Ridge(alpha=ridge_cv.alpha_)
    ridge_final.fit(X_train_scaled, y_train)
    r = ModelResult("Ridge")
    r.best_alpha = float(ridge_cv.alpha_)
    r.n_nonzero = p
    r.cv_folds = N_FOLDS
    r.train_r2 = r2_score(y_train, ridge_final.predict(X_train_scaled))
    r.test_r2 = r2_score(y_test, ridge_final.predict(X_test_scaled))
    r.train_mae = mean_absolute_error(y_train, ridge_final.predict(X_train_scaled))
    r.test_mae = mean_absolute_error(y_test, ridge_final.predict(X_test_scaled))
    r.train_rmse = np.sqrt(mean_squared_error(y_train, ridge_final.predict(X_train_scaled)))
    r.test_rmse = np.sqrt(mean_squared_error(y_test, ridge_final.predict(X_test_scaled)))
    r.notes = f"alpha={ridge_cv.alpha_:.4f}, all {p} features"
    results.append(r)

    # ElasticNet
    en_cv = ElasticNetCV(cv=N_FOLDS, alphas=np.logspace(-4, 0, 50), l1_ratio=0.5, max_iter=10_000, random_state=RANDOM_STATE, n_jobs=-1)
    en_cv.fit(X_train_scaled, y_train)
    en_final = ElasticNet(alpha=en_cv.alpha_, l1_ratio=0.5, max_iter=10_000)
    en_final.fit(X_train_scaled, y_train)
    r = ModelResult("ElasticNet")
    r.best_alpha = float(en_cv.alpha_)
    r.n_nonzero = int((en_final.coef_ != 0).sum())
    r.cv_folds = N_FOLDS
    r.train_r2 = r2_score(y_train, en_final.predict(X_train_scaled))
    r.test_r2 = r2_score(y_test, en_final.predict(X_test_scaled))
    r.train_mae = mean_absolute_error(y_train, en_final.predict(X_train_scaled))
    r.test_mae = mean_absolute_error(y_test, en_final.predict(X_test_scaled))
    r.train_rmse = np.sqrt(mean_squared_error(y_train, en_final.predict(X_train_scaled)))
    r.test_rmse = np.sqrt(mean_squared_error(y_test, en_final.predict(X_test_scaled)))
    r.notes = f"alpha={en_cv.alpha_:.6f}, {r.n_nonzero}/{p} non-zero"
    results.append(r)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    r = ModelResult("Random Forest")
    r.n_nonzero = p
    r.train_r2 = r2_score(y_train, rf.predict(X_train))
    r.test_r2 = r2_score(y_test, rf.predict(X_test))
    r.train_mae = mean_absolute_error(y_train, rf.predict(X_train))
    r.test_mae = mean_absolute_error(y_test, rf.predict(X_test))
    r.train_rmse = np.sqrt(mean_squared_error(y_train, rf.predict(X_train)))
    r.test_rmse = np.sqrt(mean_squared_error(y_test, rf.predict(X_test)))
    r.notes = f"200 trees, depth=8, m=sqrt({p})={m}"
    results.append(r)

    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE)
    gb.fit(X_train, y_train)
    r = ModelResult("Gradient Boosting")
    r.n_nonzero = p
    r.train_r2 = r2_score(y_train, gb.predict(X_train))
    r.test_r2 = r2_score(y_test, gb.predict(X_test))
    r.train_mae = mean_absolute_error(y_train, gb.predict(X_train))
    r.test_mae = mean_absolute_error(y_test, gb.predict(X_test))
    r.train_rmse = np.sqrt(mean_squared_error(y_train, gb.predict(X_train)))
    r.test_rmse = np.sqrt(mean_squared_error(y_test, gb.predict(X_test)))
    r.notes = f"200 estimators, depth=4, lr=0.1"
    results.append(r)

    # XGBoost
    if HAS_XGBOOST:
        xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE, n_jobs=-1)
        xgb_model.fit(X_train, y_train)
        r = ModelResult("XGBoost")
        r.n_nonzero = p
        r.train_r2 = r2_score(y_train, xgb_model.predict(X_train))
        r.test_r2 = r2_score(y_test, xgb_model.predict(X_test))
        r.train_mae = mean_absolute_error(y_train, xgb_model.predict(X_train))
        r.test_mae = mean_absolute_error(y_test, xgb_model.predict(X_test))
        r.train_rmse = np.sqrt(mean_squared_error(y_train, xgb_model.predict(X_train)))
        r.test_rmse = np.sqrt(mean_squared_error(y_test, xgb_model.predict(X_test)))
        r.notes = f"200 trees, depth=4, lr=0.1"
        results.append(r)

    # LightGBM
    if HAS_LIGHTGBM:
        lgb_model = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE, n_jobs=-1)
        lgb_model.fit(X_train, y_train)
        r = ModelResult("LightGBM")
        r.n_nonzero = p
        r.train_r2 = r2_score(y_train, lgb_model.predict(X_train))
        r.test_r2 = r2_score(y_test, lgb_model.predict(X_test))
        r.train_mae = mean_absolute_error(y_train, lgb_model.predict(X_train))
        r.test_mae = mean_absolute_error(y_test, lgb_model.predict(X_test))
        r.train_rmse = np.sqrt(mean_squared_error(y_train, lgb_model.predict(X_train)))
        r.test_rmse = np.sqrt(mean_squared_error(y_test, lgb_model.predict(X_test)))
        r.notes = f"200 trees, depth=4, lr=0.1"
        results.append(r)

    return results, scaler, p, m


# ============================================================================
# Confounding analysis
# ============================================================================

def confounding_analysis(X: np.ndarray, y: np.ndarray, col_names: list) -> Dict[str, Any]:
    """Author count ablation: fit LASSO with and without author_count, measure Δ R²."""
    try:
        author_idx = col_names.index("author_count")
    except ValueError:
        return {"status": "skipped", "reason": "author_count not in features"}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SPLIT_SEED)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Full model
    lasso_full = LassoCV(cv=N_FOLDS, alphas=50, max_iter=10_000, random_state=RANDOM_STATE, n_jobs=-1)
    lasso_full.fit(X_train_scaled, y_train)
    r2_full = r2_score(y_test, lasso_full.predict(X_test_scaled))

    # Without author_count
    mask = [i for i in range(X.shape[1]) if i != author_idx]
    X_train_no_auth = X_train_scaled[:, mask]
    X_test_no_auth = X_test_scaled[:, mask]
    lasso_no_auth = LassoCV(cv=N_FOLDS, alphas=50, max_iter=10_000, random_state=RANDOM_STATE, n_jobs=-1)
    lasso_no_auth.fit(X_train_no_auth, y_train)
    r2_no_auth = r2_score(y_test, lasso_no_auth.predict(X_test_no_auth))

    delta_r2 = r2_full - r2_no_auth
    pct_change = (delta_r2 / abs(r2_full) * 100) if r2_full != 0 else 0.0

    if abs(pct_change) < 5:
        interp = "Author count is NOT a major confounder; writing quality signals retain independent value."
    elif pct_change > 15:
        interp = f"Author count IS a substantial confounder ({pct_change:.1f}% impact); team prestige affects both writing style AND citations."
    else:
        interp = f"Author count is a moderate confounder ({pct_change:.1f}% impact); control for author seniority in causal claims."

    return {
        "status": "completed",
        "r2_full": float(r2_full),
        "r2_no_author": float(r2_no_auth),
        "delta_r2": float(delta_r2),
        "pct_change": float(pct_change),
        "interpretation": interp,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Generate synthetic data and run analysis")
    args = parser.parse_args()

    features, targets = load_data(demo=args.demo)
    X, y, col_names = preprocess(features, targets)
    n_samples, p = X.shape

    log.info(f"Dataset: {n_samples} samples × {p} features")
    log.info(f"Target: log1p(citations_24mo), skewness={skew(y):.3f}")

    # Train models on 80/20 split
    results, scaler, p, m = train_test_models(X, y, col_names)

    # Y statistics
    y_stats = compute_y_stats(y)
    log.info(f"Y stats: mean={y_stats['mean']:.3f}, std={y_stats['std']:.3f}, skew={y_stats['skewness']:.3f}")

    # X statistics
    x_stats = compute_x_stats(X, col_names)

    # Confounding
    confound = confounding_analysis(X, y, col_names)

    # Print summary table
    print("\n" + "=" * 100)
    print("MODEL COMPARISON — 80/20 Train/Test Split (seed=42)")
    print("=" * 100)
    print(f"{'Model':<20} {'Train R²':>10} {'Test R²':>10} {'Train MAE':>10} {'Test MAE':>10} {'Train RMSE':>10} {'Test RMSE':>10}")
    print("-" * 100)
    for r in results:
        print(f"{r.name:<20} {r.train_r2:>10.4f} {r.test_r2:>10.4f} {r.train_mae:>10.4f} {r.test_mae:>10.4f} {r.train_rmse:>10.4f} {r.test_rmse:>10.4f}")
    print("=" * 100)

    print(f"\nBIAS-VARIANCE TRADEOFF (Train R² - Test R² = overfitting indicator):")
    for r in results:
        gap = r.train_r2 - r.test_r2
        status = "✓ good fit" if gap < 0.05 else ("⚠ high variance (overfit)" if gap > 0.15 else "→ moderate")
        print(f"  {r.name:<20} gap={gap:+.4f} {status}")

    print(f"\nY STATISTICS (citations_24mo_log, n={y_stats['n']}):")
    for k, v in y_stats.items():
        if k != "n":
            print(f"  {k:<15} {v:>10.4f}")

    print(f"\nX STATISTICS BY GROUP:")
    for group, stats in x_stats.items():
        print(f"  {group:<15} count={stats['count']:>3}  mean={stats['mean']:>8.4f}  std={stats['std']:>8.4f}")

    print(f"\nCONFOUNDING ANALYSIS (author_count ablation):")
    if confound["status"] == "completed":
        print(f"  Test R² full:     {confound['r2_full']:.4f}")
        print(f"  Test R² no-auth:  {confound['r2_no_author']:.4f}")
        print(f"  Δ R²:             {confound['delta_r2']:+.4f} ({confound['pct_change']:+.1f}%)")
        print(f"  → {confound['interpretation']}")
    else:
        print(f"  Skipped: {confound['reason']}")

    # Save results
    summary = {
        "generated": datetime.now().isoformat(),
        "n_samples": n_samples,
        "n_features": p,
        "sqrt_p": m,
        "y_statistics": y_stats,
        "x_statistics": x_stats,
        "model_results": [r.to_dict() for r in results],
        "confounding": confound,
    }
    Path("model_results_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("model_results_summary.json written")

    # Save fitted LASSO for Streamlit
    lasso_cv = LassoCV(cv=N_FOLDS, alphas=50, max_iter=10_000, random_state=RANDOM_STATE, n_jobs=-1)
    lasso_cv.fit(scaler.fit_transform(X), y)
    lasso_final = Lasso(alpha=lasso_cv.alpha_, max_iter=10_000)
    lasso_final.fit(scaler.fit_transform(X), y)

    models_dict = {
        'scaler': scaler,
        'lasso': lasso_final,
        'feature_names': col_names,
        'best_alpha': lasso_cv.alpha_,
    }
    with open('lasso_model.pkl', 'wb') as f:
        pickle.dump(models_dict, f)
    log.info("lasso_model.pkl saved")


if __name__ == "__main__":
    main()
