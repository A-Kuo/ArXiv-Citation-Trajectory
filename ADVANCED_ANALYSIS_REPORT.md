# ADVANCED ANALYSIS REPORT
Generated: 2026-04-22T11:36:37.718993

## 1. Holdout Test Set Evaluation (20% holdout)

| Model | Test R² | Test MAE | Test RMSE | Notes |
|-------|---------|----------|-----------|-------|
| LASSO      |  0.0096 |   0.8440 |    1.0548 | On log scale |
| RIDGE      | -0.0414 |   0.8664 |    1.0816 | On log scale |
| ELASTICNET |  0.0091 |   0.8447 |    1.0550 | On log scale |
| GBM        | -0.0485 |   0.8777 |    1.0853 | On log scale |

**Key insight:** Compare test R² to train R² from Phase 3 to diagnose overfitting.
If test R² is much lower, consider stronger regularization or simpler models.

## 2. Confidence Interval Calibration

**Question:** Are our 95% confidence intervals actually 95% calibrated?

| Nominal Coverage | Empirical Coverage |
|------------------|--------------------|
| 68% | 69.5% |
| 90% | 89.8% |
| 95% | 95.3% |
| 99% | 99.2% |

Residual std error: 0.9615 (log scale)

**Interpretation:** If empirical >> nominal, intervals are too narrow (overconfident).
If empirical << nominal, intervals are too wide (too conservative).

## 3. Citation Velocity Analysis

How quickly do papers accumulate citations?

- **Mean citations in first 12 months:** 30.1
- **Mean citations in months 13-24:** 20.1
- **Velocity ratio:** 0.65x
- **'Fast track' papers (top quartile at 12mo):** 145 papers

**Implication:** If velocity_ratio > 1, citations are accelerating.
If < 1, citations plateau after 12 months.

## 4. Author and Keyword Trends

### Author Count Effect
| # Authors | # Papers | Median Citations | Mean Citations |
|-----------|----------|------------------|----------------|
| 1 | 70 | 26.0 | 43.8 |
| 2 | 76 | 21.0 | 31.9 |
| 3 | 87 | 25.0 | 48.3 |
| 4 | 87 | 23.0 | 32.9 |
| 5 | 67 | 39.0 | 57.7 |
| 6 | 79 | 34.0 | 66.5 |
| 7 | 62 | 38.5 | 52.7 |
| 8 | 72 | 42.0 | 71.7 |

**Insight:** Does author count correlate with citations?
If yes, it could be reputation (confound) or genuine team quality.

### Keyword Performance
| Keyword | # Papers | Median Cites | Mean Cites |
|---------|----------|--------------|------------|
| survey | 86 | 38.0 | 64.5 |
| we_propose | 90 | 39.0 | 55.5 |
| state_of_the_art | 77 | 41.0 | 54.8 |
| novel | 76 | 25.5 | 50.3 |
| benchmark | 89 | 27.0 | 47.2 |

**Top keywords by mean citations:** These terms co-occur with high-impact papers.

## 5. Recommendations for Next Week

1. **Feature engineering:** Try polynomial features, author-keyword interactions
2. **Data:** Collect more recent papers (2023-2026) to track trends
3. **Model:** Implement SHAP values for per-prediction explainability
4. **Validation:** Track predictions over time (do forecasts age well?)
5. **API:** Build REST endpoint for batch predictions

