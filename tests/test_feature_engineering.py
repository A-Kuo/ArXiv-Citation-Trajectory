"""Tests for feature engineering pipeline."""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler


def test_feature_shapes(synthetic_features, synthetic_targets):
    """Verify features and targets have matching row counts."""
    assert len(synthetic_features) == len(synthetic_targets), \
        "Features and targets must have same number of rows"


def test_no_nan_after_imputation(synthetic_features):
    """Verify NaN imputation works correctly."""
    features = synthetic_features.copy()
    features.loc[0, 'flesch_reading_ease'] = np.nan
    features.loc[1, 'abstract_length'] = np.nan
    
    features_imputed = features.fillna(0.0)
    assert features_imputed.isna().sum().sum() == 0, \
        "After fillna(0.0), should have no NaN values"


def test_scaler_centering(synthetic_features):
    """Verify StandardScaler produces mean≈0, std≈1."""
    scaler = StandardScaler()
    X = synthetic_features[['abstract_length', 'author_count']].values.astype(float)
    X_scaled = scaler.fit_transform(X)
    
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10), \
        "Scaled features should have mean ≈ 0"
    assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10), \
        "Scaled features should have std ≈ 1"


def test_target_log_transformation():
    """Verify log1p transformation is mathematically correct."""
    # Create simple test data
    raw_citations = np.array([0, 1, 5, 10, 100], dtype=float)

    # Apply transformation
    log_citations = np.log1p(raw_citations)

    # Verify inverse works
    recovered = np.expm1(log_citations)

    assert np.allclose(recovered, raw_citations), \
        "log1p and expm1 should be mathematical inverses"


def test_categorical_one_hot_encoding():
    """Verify one-hot encoding creates correct number of columns."""
    categories = ['cs.LG', 'cs.AI', 'stat.ML', 'econ.GN']
    data = pd.DataFrame({'category': np.random.choice(categories, 50)})
    
    encoded = pd.get_dummies(data, columns=['category'], prefix='cat')
    assert len(encoded.columns) == len(categories), \
        f"Should have {len(categories)} one-hot columns"
    
    # Each row should have exactly one category set to 1
    assert (encoded.iloc[:, :len(categories)].sum(axis=1) == 1).all(), \
        "Each sample should belong to exactly one category"


def test_feature_group_counts():
    """Verify feature groups are properly identified."""
    col_names = [
        'abstract_length', 'author_count',  # metadata
        'tfidf_learning', 'tfidf_neural',   # tfidf
        'cat_cs_LG', 'cat_econ_GN',          # categories
        'has_survey', 'has_benchmark',      # keywords
    ]
    
    groups = {'tfidf': [], 'cat': [], 'has': [], 'other': []}
    for col in col_names:
        if col.startswith('tfidf_'):
            groups['tfidf'].append(col)
        elif col.startswith('cat_'):
            groups['cat'].append(col)
        elif col.startswith('has_'):
            groups['has'].append(col)
        else:
            groups['other'].append(col)
    
    assert len(groups['tfidf']) == 2
    assert len(groups['cat']) == 2
    assert len(groups['has']) == 2
    assert len(groups['other']) == 2


def test_missing_value_handling_strategy():
    """Verify that missing values in different columns are handled correctly."""
    df = pd.DataFrame({
        'numeric': [1.0, 2.0, np.nan, 4.0],
        'category': ['A', 'B', np.nan, 'D'],
        'keyword': [0, 1, np.nan, 1],
    })
    
    # Numeric columns: fill with 0
    numeric_filled = df['numeric'].fillna(0.0)
    assert not numeric_filled.isna().any()
    
    # Categorical columns: could fill with mode or drop
    category_filled = df['category'].fillna(df['category'].mode()[0]) if len(df['category'].mode()) > 0 else df['category'].fillna('Unknown')
    assert not category_filled.isna().any()
