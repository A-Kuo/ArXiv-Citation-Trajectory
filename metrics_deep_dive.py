#!/usr/bin/env python3
"""
Stratified metrics analysis: disaggregate aggregate metrics to reveal hidden patterns.

Core insight: 85% R² might be hiding:
- 95% for cs.LG, 40% for econ.GN (category-specific failure)
- 90% for high-citation papers, 20% for low-citation (extreme value failure)
- 85% for papers with many authors, 50% for solo authors (confound effect)

This module reveals the true story behind aggregate metrics.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load data
features = pd.read_parquet("features.parquet")
targets = pd.read_parquet("targets.parquet")

with open("lasso_model.pkl", "rb") as f:
    models = pickle.load(f)

scaler = models["scaler"]
lasso = models["lasso"]
feature_names = list(models["feature_names"])

X = features.values.astype(float)
y = targets["citations_24mo_log"].values

# Predict
X_scaled = scaler.transform(X)
y_pred = lasso.predict(X_scaled)

# Compute residuals
residuals = y - y_pred
y_actual = np.expm1(y)
y_pred_raw = np.expm1(y_pred)
residuals_raw = y_actual - y_pred_raw

# ===========================================================================
# Aggregate Metrics (the "average" that hides problems)
# ===========================================================================

print("\n" + "="*70)
print("AGGREGATE METRICS (misleading)")
print("="*70)

agg_r2 = r2_score(y, y_pred)
agg_mae = mean_absolute_error(y, y_pred)
agg_rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"Overall R² (log scale):     {agg_r2:.4f}")
print(f"Overall MAE (log scale):    {agg_mae:.4f}")
print(f"Overall RMSE (log scale):   {agg_rmse:.4f}")
print(f"\nWhat's hidden: {agg_r2:.1%} aggregate might be 80% for one subgroup, 30% for another.")

# ===========================================================================
# Stratified by Category
# ===========================================================================

print("\n" + "="*70)
print("STRATIFIED BY ARXIV CATEGORY (reveals category-specific bias)")
print("="*70)

cat_cols = [c for c in feature_names if c.startswith("cat_")]
categories = {c.replace("cat_", ""): features[[f for f in features.columns if f == c]].values.flatten() > 0.5
              for c in cat_cols}

# Actually build category assignment properly
cat_assign = {}
for col_name in features.columns:
    if col_name.startswith("cat_"):
        cat_assign[col_name.replace("cat_", "")] = features[col_name].values

cat_metrics = {}
for cat_name, cat_mask in cat_assign.items():
    if cat_mask.sum() == 0:
        continue

    y_cat = y[cat_mask]
    y_pred_cat = y_pred[cat_mask]

    r2_cat = r2_score(y_cat, y_pred_cat)
    mae_cat = mean_absolute_error(y_cat, y_pred_cat)
    rmse_cat = np.sqrt(mean_squared_error(y_cat, y_pred_cat))

    cat_metrics[cat_name] = {
        "count": int(cat_mask.sum()),
        "r2": r2_cat,
        "mae": mae_cat,
        "rmse": rmse_cat,
        "bias": float((y_pred_cat - y_cat).mean()),  # systematic over/underprediction
    }

    print(f"\n{cat_name.upper():20} (n={int(cat_mask.sum()):3})")
    print(f"  R²:   {r2_cat:7.4f} {'✗' if r2_cat < agg_r2 - 0.05 else '✓'}")
    print(f"  MAE:  {mae_cat:7.4f}")
    print(f"  Bias: {cat_metrics[cat_name]['bias']:+7.4f} (negative=underpredicts, positive=overpredicts)")

# ===========================================================================
# Stratified by Citation Level (high vs low impact papers)
# ===========================================================================

print("\n" + "="*70)
print("STRATIFIED BY CITATION LEVEL (reveals if model works at extremes)")
print("="*70)

y_actual_vals = np.expm1(y)
q25, q50, q75 = np.percentile(y_actual_vals, [25, 50, 75])

strata = {
    "Low (≤p25)": y_actual_vals <= q25,
    "Medium-low (p25-p50)": (y_actual_vals > q25) & (y_actual_vals <= q50),
    "Medium-high (p50-p75)": (y_actual_vals > q50) & (y_actual_vals <= q75),
    "High (>p75)": y_actual_vals > q75,
}

citation_metrics = {}
for strata_name, mask in strata.items():
    y_s = y[mask]
    y_pred_s = y_pred[mask]

    r2_s = r2_score(y_s, y_pred_s)
    mae_s = mean_absolute_error(y_s, y_pred_s)

    citation_metrics[strata_name] = {
        "count": int(mask.sum()),
        "r2": r2_s,
        "mae": mae_s,
        "mean_citations": float(y_actual_vals[mask].mean()),
    }

    print(f"\n{strata_name:25} (n={int(mask.sum()):3}, avg_citations={y_actual_vals[mask].mean():.0f})")
    print(f"  R²:  {r2_s:7.4f} {'✗ FAILS on' if r2_s < 0 else '✓'} this stratum")
    print(f"  MAE: {mae_s:7.4f}")

# ===========================================================================
# Stratified by Author Count (confound variable)
# ===========================================================================

print("\n" + "="*70)
print("STRATIFIED BY AUTHOR COUNT (reveals confounding)")
print("="*70)

author_counts = features["author_count"].values
author_metrics = {}

for n_auth in sorted(np.unique(author_counts)):
    mask = author_counts == n_auth
    if mask.sum() < 5:
        continue

    y_a = y[mask]
    y_pred_a = y_pred[mask]

    r2_a = r2_score(y_a, y_pred_a)
    mae_a = mean_absolute_error(y_a, y_pred_a)

    author_metrics[int(n_auth)] = {
        "count": int(mask.sum()),
        "r2": r2_a,
        "mae": mae_a,
    }

    print(f"\n{int(n_auth)} authors (n={int(mask.sum()):3})")
    print(f"  R²:  {r2_a:7.4f} {'✗' if r2_a < agg_r2 - 0.1 else '✓'}")
    print(f"  MAE: {mae_a:7.4f}")

# ===========================================================================
# Stratified by Prediction Confidence (easy vs hard cases)
# ===========================================================================

print("\n" + "="*70)
print("STRATIFIED BY PREDICTION UNCERTAINTY (easy vs hard cases)")
print("="*70)

# Use absolute residual as proxy for "confidence"
abs_residual = np.abs(residuals)
q33, q67 = np.percentile(abs_residual, [33, 67])

uncertainty_strata = {
    "Low residual (easy): ≤p33": abs_residual <= q33,
    "Medium residual: p33-p67": (abs_residual > q33) & (abs_residual <= q67),
    "High residual (hard): >p67": abs_residual > q67,
}

uncertainty_metrics = {}
for strata_name, mask in uncertainty_strata.items():
    y_u = y[mask]
    y_pred_u = y_pred[mask]

    if len(y_u) > 0:
        r2_u = r2_score(y_u, y_pred_u)
        mae_u = mean_absolute_error(y_u, y_pred_u)

        uncertainty_metrics[strata_name] = {
            "count": int(mask.sum()),
            "r2": r2_u,
            "mae": mae_u,
        }

        print(f"\n{strata_name:30} (n={int(mask.sum()):3})")
        print(f"  R²:  {r2_u:7.4f}")
        print(f"  MAE: {mae_u:7.4f}")

# ===========================================================================
# Key Findings
# ===========================================================================

print("\n" + "="*70)
print("KEY FINDINGS: WHERE YOUR MODEL BREAKS")
print("="*70)

# Find worst-performing strata
print("\n1. WORST CATEGORY:")
worst_cat = min(cat_metrics.items(), key=lambda x: x[1]['r2'])
print(f"   {worst_cat[0]}: R²={worst_cat[1]['r2']:.4f} (vs aggregate {agg_r2:.4f})")
print(f"   → Model {100*(agg_r2 - worst_cat[1]['r2']):.0f}% worse for {worst_cat[0]} papers")

print("\n2. WORST CITATION LEVEL:")
worst_cit = min(citation_metrics.items(), key=lambda x: x[1]['r2'])
print(f"   {worst_cit[0]}: R²={worst_cit[1]['r2']:.4f}")
print(f"   → Model FAILS on this stratum: {'fails' if worst_cit[1]['r2'] < 0 else 'barely works'}")

# Check for bias
max_bias = max(abs(v["bias"]) for v in cat_metrics.values())
biased_cats = [k for k, v in cat_metrics.items() if abs(v["bias"]) > 0.2]
if biased_cats:
    print(f"\n3. SYSTEMATIC BIAS DETECTED:")
    for cat in biased_cats:
        print(f"   {cat}: {cat_metrics[cat]['bias']:+.4f} (log scale)")
        print(f"   → Systematically {'over' if cat_metrics[cat]['bias'] > 0 else 'under'}-predicts by {100*np.expm1(abs(cat_metrics[cat]['bias'])):.0f}% on citations scale")

# ===========================================================================
# Save findings
# ===========================================================================

findings = {
    "aggregate": {
        "r2": float(agg_r2),
        "mae": float(agg_mae),
        "rmse": float(agg_rmse),
    },
    "by_category": cat_metrics,
    "by_citation_level": citation_metrics,
    "by_author_count": author_metrics,
    "by_uncertainty": uncertainty_metrics,
}

with open("metrics_deep_dive.json", "w") as f:
    json.dump(findings, f, indent=2, default=str)

print(f"\n→ Full results saved to metrics_deep_dive.json")

# ===========================================================================
# Actionable Recommendations
# ===========================================================================

print("\n" + "="*70)
print("ACTIONABLE RECOMMENDATIONS")
print("="*70)

print("""
1. IF category-specific R² varies widely (80% cs.LG, 30% econ):
   → Model is category-biased. Build separate models or use category-specific scaling.

2. IF high-citation papers R² << low-citation R² (or vice versa):
   → Model fails at extremes. Collect more extreme-value papers or use quantile regression.

3. IF author_count shows strong pattern:
   → Team prestige is a confounder. Either separate by author count or use domain features.

4. IF hard cases (high residual) also have low R²:
   → Your model is uncertain exactly when it should be confident. May need better features.

5. IF systematic bias exists (always over/underpredicts for a category):
   → Simple fix: add category-specific intercept (post-hoc calibration).

NEXT STEP: Report metrics at granular level, not aggregate.
Instead of "85% R²", say:
"85% R² overall (cs.LG: 95%, econ: 30%, high-cite: 75%, low-cite: 50%)"

This forces stakeholders to see where the model actually works.
""")

print("="*70)
