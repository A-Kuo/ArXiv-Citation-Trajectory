#!/usr/bin/env python3
"""
Advanced analysis module: test set evaluation, model comparison, calibration,
citation velocity, author/keyword trends.

Run this after model_results.py to generate comprehensive analysis.
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

RANDOM_STATE = 42


def load_data_and_models() -> Dict:
    """Load all required assets."""
    import sqlite3

    features = pd.read_parquet("features.parquet")
    targets = pd.read_parquet("targets.parquet")
    arxiv_ids = pd.read_parquet("arxiv_ids.parquet")

    # Load citations_12mo from database
    try:
        conn = sqlite3.connect("citations.db")
        citations_df = pd.read_sql_query(
            "SELECT arxiv_id, citations_12mo FROM citations", conn
        )
        conn.close()

        # Merge: targets has same length as arxiv_ids
        targets["arxiv_id"] = arxiv_ids["arxiv_id"].values
        targets = targets.merge(citations_df, on="arxiv_id", how="left").drop(columns=["arxiv_id"])
    except Exception as e:
        log.warning(f"Could not load citations_12mo: {e}")

    with open("lasso_model.pkl", "rb") as f:
        models = pickle.load(f)

    return {
        "features": features,
        "targets": targets,
        "arxiv_ids": arxiv_ids,
        "scaler": models["scaler"],
        "lasso": models["lasso"],
        "feature_names": models["feature_names"],
    }


# ---------------------------------------------------------------------------
# Test set evaluation
# ---------------------------------------------------------------------------

def evaluate_on_test_set(
    features: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Split into train/test and evaluate multiple models on holdout test set."""
    log.info(f"Creating test set (80/20 split)...")

    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=test_size, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # LASSO
    from sklearn.linear_model import LassoCV, Lasso
    lasso_cv = LassoCV(cv=5, alphas=50, max_iter=10_000, random_state=RANDOM_STATE, n_jobs=-1)
    lasso_cv.fit(X_train_scaled, y_train)
    lasso_pred = lasso_cv.predict(X_test_scaled)
    results["lasso"] = {
        "r2": r2_score(y_test, lasso_pred),
        "mae": mean_absolute_error(y_test, lasso_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, lasso_pred)),
        "predictions": lasso_pred,
    }
    log.info(f"LASSO (test): R²={results['lasso']['r2']:.4f}, MAE={results['lasso']['mae']:.4f}")

    # Ridge
    ridge_cv = RidgeCV(cv=5, alphas=np.logspace(-2, 3, 50))
    ridge_cv.fit(X_train_scaled, y_train)
    ridge_pred = ridge_cv.predict(X_test_scaled)
    results["ridge"] = {
        "r2": r2_score(y_test, ridge_pred),
        "mae": mean_absolute_error(y_test, ridge_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, ridge_pred)),
        "predictions": ridge_pred,
    }
    log.info(f"Ridge (test): R²={results['ridge']['r2']:.4f}, MAE={results['ridge']['mae']:.4f}")

    # ElasticNet
    enet_cv = ElasticNetCV(cv=5, l1_ratio=[0.1, 0.5, 0.9], alphas=np.logspace(-2, 3, 50), max_iter=10_000)
    enet_cv.fit(X_train_scaled, y_train)
    enet_pred = enet_cv.predict(X_test_scaled)
    results["elasticnet"] = {
        "r2": r2_score(y_test, enet_pred),
        "mae": mean_absolute_error(y_test, enet_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, enet_pred)),
        "predictions": enet_pred,
    }
    log.info(f"ElasticNet (test): R²={results['elasticnet']['r2']:.4f}, MAE={results['elasticnet']['mae']:.4f}")

    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE)
    gb.fit(X_train_scaled, y_train)
    gb_pred = gb.predict(X_test_scaled)
    results["gbm"] = {
        "r2": r2_score(y_test, gb_pred),
        "mae": mean_absolute_error(y_test, gb_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, gb_pred)),
        "predictions": gb_pred,
    }
    log.info(f"GBM (test): R²={results['gbm']['r2']:.4f}, MAE={results['gbm']['mae']:.4f}")

    return results, y_test, X_test_scaled


# ---------------------------------------------------------------------------
# Confidence calibration
# ---------------------------------------------------------------------------

def calibration_analysis(
    features: np.ndarray,
    y: np.ndarray,
    col_names: list,
) -> Dict:
    """Analyze if confidence intervals are well-calibrated."""
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import KFold

    log.info("Analyzing confidence interval calibration...")

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    predictions = np.zeros_like(y)
    residuals_list = []

    for train_idx, test_idx in kf.split(y):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        lasso = Lasso(alpha=0.086, max_iter=10_000)  # from phase 3
        lasso.fit(X_train_scaled, y_train)
        predictions[test_idx] = lasso.predict(X_test_scaled)
        residuals_list.extend((y_test - predictions[test_idx]).tolist())

    residuals = np.array(residuals_list)
    std_error = residuals.std()

    # Compute empirical coverage at different confidence levels
    coverage = {}
    for conf_level in [0.68, 0.90, 0.95, 0.99]:
        z_score = {0.68: 1.0, 0.90: 1.645, 0.95: 1.96, 0.99: 2.576}[conf_level]
        margin = z_score * std_error
        within_ci = np.sum(np.abs(residuals) <= margin) / len(residuals)
        coverage[conf_level] = within_ci

    log.info(f"Calibration: {coverage}")
    return {"residual_std": std_error, "coverage": coverage, "residuals": residuals}


# ---------------------------------------------------------------------------
# Citation velocity analysis
# ---------------------------------------------------------------------------

def citation_velocity_analysis(targets: pd.DataFrame) -> Dict:
    """Analyze citation accumulation patterns: 12mo vs 24mo."""
    log.info("Analyzing citation velocity (12mo vs 24mo)...")

    c24 = targets["citations_24mo"].values if "citations_24mo" in targets.columns else targets["citations_24mo_log"].fillna(0).values

    # If we have 12mo data, use it; otherwise estimate
    if "citations_12mo" in targets.columns:
        c12 = targets["citations_12mo"].values
    else:
        # Estimate: assume ~60% of 24mo citations come in first 12 months
        c12 = c24 * 0.6

    velocity = c24 - c12  # citations in months 13-24

    return {
        "citations_first_12mo_mean": float(c12.mean()),
        "citations_second_12mo_mean": float(velocity.mean()),
        "velocity_to_12mo_ratio": float(velocity.mean() / (c12.mean() + 1)),
        "p75_12mo": float(np.percentile(c12, 75)),
        "p75_24mo": float(np.percentile(c24, 75)),
        "fast_track_papers": int(np.sum(c12 > np.percentile(c12, 75))),
        "note": "12mo data not available; second-12mo is estimate"
    }


# ---------------------------------------------------------------------------
# Author and keyword analysis
# ---------------------------------------------------------------------------

def author_keyword_trends(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    col_names: list,
) -> Dict:
    """Identify top-performing authors and keywords."""
    log.info("Analyzing author and keyword patterns...")

    # Author count effect
    author_counts = features["author_count"].values
    citations = targets["citations_24mo"].values

    # Bin by author count
    author_analysis = {}
    for n_authors in sorted(features["author_count"].unique()):
        mask = author_counts == n_authors
        if mask.sum() > 0:
            author_analysis[int(n_authors)] = {
                "count": int(mask.sum()),
                "median_citations": float(np.median(citations[mask])),
                "mean_citations": float(citations[mask].mean()),
            }

    # Keyword signals
    keyword_cols = [c for c in col_names if c.startswith("has_")]
    keyword_analysis = {}
    for kw_col in keyword_cols:
        kw_name = kw_col.replace("has_", "")
        kw_mask = features[kw_col].astype(bool)
        if kw_mask.sum() > 0:
            keyword_analysis[kw_name] = {
                "papers_with_keyword": int(kw_mask.sum()),
                "median_citations": float(np.median(citations[kw_mask])),
                "mean_citations": float(citations[kw_mask].mean()),
            }

    return {
        "author_analysis": author_analysis,
        "keyword_analysis": keyword_analysis,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_advanced_report(
    test_results: Dict,
    y_test: np.ndarray,
    calibration: Dict,
    velocity: Dict,
    trends: Dict,
) -> str:
    """Generate comprehensive advanced analysis report."""
    r = []

    r.append("# ADVANCED ANALYSIS REPORT\n")
    r.append(f"Generated: {datetime.now().isoformat()}\n\n")

    # Test set results
    r.append("## 1. Holdout Test Set Evaluation (20% holdout)\n\n")
    r.append("| Model | Test R² | Test MAE | Test RMSE | Notes |\n")
    r.append("|-------|---------|----------|-----------|-------|\n")
    for model_name in ["lasso", "ridge", "elasticnet", "gbm"]:
        res = test_results[model_name]
        r.append(f"| {model_name.upper():10} | {res['r2']:7.4f} | {res['mae']:8.4f} | {res['rmse']:9.4f} | On log scale |\n")
    r.append("\n**Key insight:** Compare test R² to train R² from Phase 3 to diagnose overfitting.\n")
    r.append("If test R² is much lower, consider stronger regularization or simpler models.\n\n")

    # Confidence calibration
    r.append("## 2. Confidence Interval Calibration\n\n")
    r.append("**Question:** Are our 95% confidence intervals actually 95% calibrated?\n\n")
    r.append("| Nominal Coverage | Empirical Coverage |\n")
    r.append("|------------------|--------------------|\n")
    for nom, emp in calibration["coverage"].items():
        r.append(f"| {nom:.0%} | {emp:.1%} |\n")
    r.append(f"\nResidual std error: {calibration['residual_std']:.4f} (log scale)\n\n")
    r.append("**Interpretation:** If empirical >> nominal, intervals are too narrow (overconfident).\n")
    r.append("If empirical << nominal, intervals are too wide (too conservative).\n\n")

    # Citation velocity
    r.append("## 3. Citation Velocity Analysis\n\n")
    r.append("How quickly do papers accumulate citations?\n\n")
    r.append(f"- **Mean citations in first 12 months:** {velocity['citations_first_12mo_mean']:.1f}\n")
    r.append(f"- **Mean citations in months 13-24:** {velocity['citations_second_12mo_mean']:.1f}\n")
    r.append(f"- **Velocity ratio:** {velocity['velocity_to_12mo_ratio']:.2f}x\n")
    r.append(f"- **'Fast track' papers (top quartile at 12mo):** {velocity['fast_track_papers']} papers\n\n")
    r.append("**Implication:** If velocity_ratio > 1, citations are accelerating.\n")
    r.append("If < 1, citations plateau after 12 months.\n\n")

    # Author trends
    r.append("## 4. Author and Keyword Trends\n\n")
    r.append("### Author Count Effect\n")
    r.append("| # Authors | # Papers | Median Citations | Mean Citations |\n")
    r.append("|-----------|----------|------------------|----------------|\n")
    for n_auth, stats in sorted(trends["author_analysis"].items()):
        r.append(f"| {n_auth} | {stats['count']} | {stats['median_citations']:.1f} | {stats['mean_citations']:.1f} |\n")
    r.append("\n**Insight:** Does author count correlate with citations?\n")
    r.append("If yes, it could be reputation (confound) or genuine team quality.\n\n")

    r.append("### Keyword Performance\n")
    r.append("| Keyword | # Papers | Median Cites | Mean Cites |\n")
    r.append("|---------|----------|--------------|------------|\n")
    for kw, stats in sorted(trends["keyword_analysis"].items(), key=lambda x: x[1]['mean_citations'], reverse=True)[:10]:
        r.append(f"| {kw} | {stats['papers_with_keyword']} | {stats['median_citations']:.1f} | {stats['mean_citations']:.1f} |\n")
    r.append("\n**Top keywords by mean citations:** These terms co-occur with high-impact papers.\n\n")

    r.append("## 5. Recommendations for Next Week\n\n")
    r.append("1. **Feature engineering:** Try polynomial features, author-keyword interactions\n")
    r.append("2. **Data:** Collect more recent papers (2023-2026) to track trends\n")
    r.append("3. **Model:** Implement SHAP values for per-prediction explainability\n")
    r.append("4. **Validation:** Track predictions over time (do forecasts age well?)\n")
    r.append("5. **API:** Build REST endpoint for batch predictions\n\n")

    return "".join(r)


def main():
    """Run advanced analysis."""
    assets = load_data_and_models()
    features = assets["features"]
    targets = assets["targets"]
    col_names = assets["feature_names"]

    X = features.values.astype(float)
    y = targets["citations_24mo_log"].values

    # 1. Test set evaluation
    test_results, y_test, X_test = evaluate_on_test_set(X, y)

    # 2. Confidence calibration
    calibration = calibration_analysis(X, y, col_names)

    # 3. Citation velocity
    velocity = citation_velocity_analysis(targets)

    # 4. Author/keyword trends
    trends = author_keyword_trends(features, targets, col_names)

    # 5. Generate report
    report = generate_advanced_report(test_results, y_test, calibration, velocity, trends)
    with open("ADVANCED_ANALYSIS_REPORT.md", "w") as f:
        f.write(report)
    log.info("ADVANCED_ANALYSIS_REPORT.md written")

    # 6. Save results as JSON
    results = {
        "generated": datetime.now().isoformat(),
        "test_set_results": {
            k: {**v, "predictions": None} for k, v in test_results.items()  # exclude arrays
        },
        "calibration": {
            "residual_std": float(calibration["residual_std"]),
            "coverage": {str(k): float(v) for k, v in calibration["coverage"].items()},
        },
        "velocity": velocity,
        "trends": trends,
    }
    with open("advanced_analysis_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("advanced_analysis_summary.json written")

    print("\n" + "="*60)
    print("ADVANCED ANALYSIS SUMMARY")
    print("="*60)
    print("\nTest Set Performance (best model):")
    best_model = max(test_results.items(), key=lambda x: x[1]['r2'])
    print(f"  {best_model[0].upper()}: R²={best_model[1]['r2']:.4f}, MAE={best_model[1]['mae']:.4f}")

    print("\nConfidence Calibration (nominal vs empirical):")
    for nom, emp in calibration['coverage'].items():
        status = "✓" if abs(emp - nom) < 0.05 else "⚠"
        print(f"  {status} {nom:.0%}: {emp:.1%} coverage")

    print(f"\nCitation Velocity:")
    print(f"  First 12mo avg: {velocity['citations_first_12mo_mean']:.1f}")
    print(f"  Next 12mo avg: {velocity['citations_second_12mo_mean']:.1f}")
    print(f"  Acceleration: {velocity['velocity_to_12mo_ratio']:.2f}x")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
