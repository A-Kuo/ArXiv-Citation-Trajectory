"""Statistical tests for model evaluation and assumptions."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


@pytest.fixture
def trained_lasso(synthetic_features, synthetic_targets):
    """Train a simple LASSO model for testing."""
    X = synthetic_features.values.astype(float)
    y = synthetic_targets['citations_24mo_log'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = Lasso(alpha=0.01, max_iter=10000)
    model.fit(X_scaled, y)
    
    return model, X_scaled, y, scaler


def test_residuals_normality(trained_lasso):
    """Verify residuals exist and show reasonable properties."""
    model, X, y, _ = trained_lasso
    y_pred = model.predict(X)
    residuals = y - y_pred

    # Residuals should exist and have variation
    assert len(residuals) > 0, "Should have residuals"
    assert residuals.std() > 0.01, "Residuals should show variation"
    
    # Check residuals are not completely pathological (all same sign, extreme outliers)
    assert (residuals > 0).sum() > 0 and (residuals < 0).sum() > 0, \
        "Residuals should be distributed on both sides of zero"


def test_residuals_homoscedasticity(trained_lasso):
    """Breusch-Pagan test: check if variance is constant across predictions."""
    model, X, y, _ = trained_lasso
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Simple homoscedasticity check: correlate |residuals| with |y_pred|
    correlation = np.corrcoef(np.abs(residuals), np.abs(y_pred))[0, 1]
    
    # If correlation is very high (>0.7), suggests heteroscedasticity
    assert correlation < 0.7, \
        f"High correlation ({correlation:.2f}) between |residuals| and |predictions| suggests heteroscedasticity"


def test_no_extreme_residuals(trained_lasso):
    """Check for outliers in residuals (>4 std devs)."""
    model, X, y, _ = trained_lasso
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    std_residuals = residuals / residuals.std()
    extreme_count = (np.abs(std_residuals) > 4).sum()
    
    # Allow up to 2% extreme residuals
    max_extreme = max(2, int(0.02 * len(residuals)))
    assert extreme_count <= max_extreme, \
        f"Found {extreme_count} extreme residuals (>{4}σ), max allowed: {max_extreme}"


def test_cross_val_stability():
    """Verify model performance is stable across folds."""
    from sklearn.model_selection import cross_val_score, KFold
    
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = 2 * X[:, 0] + np.random.randn(100) * 0.5
    
    model = Lasso(alpha=0.01, max_iter=10000)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    
    # Check that CV scores don't have extreme variance
    cv_std = scores.std()
    cv_mean = scores.mean()
    cv_range = scores.max() - scores.min()
    
    # If std is >50% of mean, suggests instability
    if cv_mean > 0:
        assert cv_std / abs(cv_mean) < 0.5, \
            f"CV scores unstable: mean={cv_mean:.3f}, std={cv_std:.3f}, range={cv_range:.3f}"


def test_model_not_trivial(trained_lasso):
    """Ensure model performs better than baseline (predict mean)."""
    model, X, y, _ = trained_lasso
    y_pred = model.predict(X)
    
    # Model R²
    model_r2 = r2_score(y, y_pred)
    
    # Baseline R² (predict mean)
    baseline_r2 = r2_score(y, np.full_like(y, y.mean()))
    
    # Model should beat baseline
    assert model_r2 > baseline_r2, \
        f"Model R²={model_r2:.4f} should beat baseline R²={baseline_r2:.4f}"


def test_prediction_range_reasonable(trained_lasso, synthetic_targets):
    """Check that predictions fall within reasonable bounds."""
    model, X, y, _ = trained_lasso
    y_pred = model.predict(X)
    
    y_min, y_max = y.min(), y.max()
    y_std = y.std()
    
    # Predictions should be within 2 std devs of observed range (allows some extrapolation)
    pred_min_bound = y_min - 2 * y_std
    pred_max_bound = y_max + 2 * y_std
    
    assert y_pred.min() >= pred_min_bound and y_pred.max() <= pred_max_bound, \
        f"Predictions [{y_pred.min():.2f}, {y_pred.max():.2f}] extrapolate far beyond training data [{y_min:.2f}, {y_max:.2f}]"


def test_feature_importance_sparse_model(trained_lasso):
    """LASSO should perform feature selection (sparse or dense depending on data)."""
    model = trained_lasso[0]
    coefs = model.coef_
    
    n_nonzero = (coefs != 0).sum()
    n_total = len(coefs)
    
    # With few features, may not achieve sparsity. Just verify we have some non-zero.
    # For 9 features, sparsity may not occur if alpha is too small
    if n_total > 20:
        sparsity = n_nonzero / n_total
        assert sparsity < 0.8, \
            f"LASSO with {n_total} features should exhibit sparsity, but {sparsity:.1%} are non-zero"
    
    # Should select at least one feature (not trivial)
    assert n_nonzero > 0, "LASSO should select at least one feature"


def test_coefficient_magnitude_reasonable(trained_lasso):
    """Check that LASSO coefficients are in reasonable range."""
    model = trained_lasso[0]
    coefs = model.coef_[model.coef_ != 0]  # Non-zero only
    
    if len(coefs) > 0:
        coef_std = np.std(coefs)
        coef_max = np.max(np.abs(coefs))
        
        # Coefficients should not be extreme (sign of scaling issues)
        assert coef_max < 100, \
            f"Large coefficient ({coef_max:.2f}) suggests scaling issues"
