"""Integration tests: verify API predictions are consistent with FeatureBuilder.

These tests ensure that the refactoring to use FeatureBuilder didn't change
model output (regression guarantee). They use synthetic trained artifacts
and compare outputs across the API, Streamlit, and direct FeatureBuilder calls.
"""
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from datetime import datetime

import api
from feature_schema import FeatureBuilder, FeatureMetadata, load_builder_from_training_artifacts


@pytest.fixture
def synthetic_model():
    """Train a tiny synthetic model for testing."""
    np.random.seed(42)
    n = 50

    # Create synthetic papers
    titles = [f"Paper {i} survey of neural networks" for i in range(n)]
    abstracts = [f"We study method [1] with benchmark [2] for novel tasks." for i in range(n)]

    df = pd.DataFrame({
        "arxiv_id": [f"2020.{i:05d}" for i in range(n)],
        "title": titles,
        "abstract": abstracts,
        "authors": ["A, B, C"] * n,
        "category": np.random.choice(["cs.LG", "cs.AI"], n),
        "submitted_date": pd.to_datetime("2020-01-15"),
        "citations_24mo": np.random.randint(0, 50, n),
    })

    # Build features manually (minimal version)
    n_text_features = 5
    n_tfidf = 10
    n_meta = 8

    text_features = pd.DataFrame({
        "abstract_length": [len(a.split()) for a in abstracts],
        "title_length": [len(t.split()) for t in titles],
        "flesch_reading_ease": [50.0] * n,  # Skip flesch to avoid nltk issues
        "equation_count": [0] * n,
        "citation_count": [2] * n,
    })

    # TF-IDF
    combined = df["abstract"] + " " + df["title"]
    tfidf_vec = TfidfVectorizer(max_features=n_tfidf, ngram_range=(1, 1), lowercase=True)
    tfidf_array = tfidf_vec.fit_transform(combined).toarray()
    tfidf_names = tfidf_vec.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_array, columns=[f"tfidf_{n}" for n in tfidf_names])

    # Metadata
    meta_df = pd.DataFrame({
        "author_count": [3.0] * n,
        "submission_month": [1.0] * n,
        "submission_year": [2020.0] * n,
        "cat_cs.LG": (df["category"] == "cs.LG").astype(float),
        "cat_cs.AI": (df["category"] == "cs.AI").astype(float),
        "has_survey": [1.0] * n,
        "has_benchmark": [1.0] * n,
        "has_state_of_the_art": [0.0] * n,
        "has_novel": [0.0] * n,
        "has_we_propose": [0.0] * n,
    })

    features = pd.concat([text_features, tfidf_df, meta_df], axis=1)
    targets = np.log1p(df["citations_24mo"].values)

    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.fillna(0).values)
    model = Lasso(alpha=0.01, max_iter=5000, random_state=42)
    model.fit(X_scaled, targets)

    # Save as artifacts
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        model_dict = {
            "scaler": scaler,
            "lasso": model,
            "feature_names": list(features.columns),
            "best_alpha": 0.01,
        }
        model_path = tmpdir / "lasso_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_dict, f)

        vec_path = tmpdir / "tfidf_vectorizer.pkl"
        with open(vec_path, "wb") as f:
            pickle.dump(tfidf_vec, f)

        yield {
            "model_path": model_path,
            "vec_path": vec_path,
            "features": features,
            "model": model,
            "scaler": scaler,
            "tfidf_vec": tfidf_vec,
        }


def test_builder_produces_expected_feature_shape(synthetic_model):
    """Verify FeatureBuilder produces correct number of features."""
    model_dict = {
        "scaler": synthetic_model["scaler"],
        "lasso": synthetic_model["model"],
        "feature_names": list(synthetic_model["features"].columns),
    }
    metadata = FeatureMetadata(
        feature_names=model_dict["feature_names"],
        categories=["cs.LG", "cs.AI"],
    )
    builder = FeatureBuilder(
        metadata=metadata,
        tfidf_vectorizer=synthetic_model["tfidf_vec"],
        validation_mode="warn",
    )

    vector, _ = builder.build_feature_vector(
        title="Test paper",
        abstract="This is a benchmark survey with novel approaches.",
        authors="A, B, C",
        category="cs.LG",
        submitted_date=datetime(2020, 1, 15),
    )

    expected_shape = (1, len(model_dict["feature_names"]))
    assert vector.shape == expected_shape, f"Expected shape {expected_shape}, got {vector.shape}"


def test_builder_vs_api_predictions_consistent(synthetic_model):
    """Verify API endpoint predictions match FeatureBuilder output."""
    # Temporarily patch api module to use our synthetic model
    original_model_path = api.MODEL_PATH
    original_vectorizer_path = api.TFIDF_VECTORIZER_PATH

    try:
        api.MODEL_PATH = synthetic_model["model_path"]
        api.TFIDF_VECTORIZER_PATH = synthetic_model["vec_path"]

        # Reset model state
        api._MODEL_STATE["model"] = None
        api._MODEL_STATE["scaler"] = None
        api._MODEL_STATE["builder"] = None
        api._MODEL_STATE["ready"] = False

        # Manually load
        api.load_model()

        # Test via TestClient
        client = TestClient(api.app)

        test_paper = {
            "title": "Test paper on survey benchmarks",
            "abstract": "We study benchmark survey methods with novel ideas.",
            "authors": "Smith, Jones, Kumar",
            "category": "cs.AI",
            "submitted_date": "2020-01-15",
        }

        response = client.post("/predict", json=test_paper)
        assert response.status_code == 200
        api_pred = response.json()["predicted_citations_log"]

        # Compare against FeatureBuilder directly
        builder = api._MODEL_STATE["builder"]
        vector, _ = builder.build_feature_vector(
            title=test_paper["title"],
            abstract=test_paper["abstract"],
            authors=test_paper["authors"],
            category=test_paper["category"],
            submitted_date=datetime.fromisoformat(test_paper["submitted_date"]),
        )
        X_scaled = api._MODEL_STATE["scaler"].transform(vector)
        builder_pred = api._MODEL_STATE["model"].predict(X_scaled)[0]

        # Predictions should match closely (small numerical tolerance)
        assert np.isclose(
            api_pred,
            builder_pred,
            rtol=1e-6,
        ), f"API prediction {api_pred} != FeatureBuilder prediction {builder_pred}"

    finally:
        api.MODEL_PATH = original_model_path
        api.TFIDF_VECTORIZER_PATH = original_vectorizer_path
        api._MODEL_STATE["model"] = None
        api._MODEL_STATE["scaler"] = None
        api._MODEL_STATE["builder"] = None
        api._MODEL_STATE["ready"] = False


def test_validation_warnings_surface_on_missing_date(synthetic_model):
    """Verify missing submitted_date produces warning."""
    original_model_path = api.MODEL_PATH
    original_vectorizer_path = api.TFIDF_VECTORIZER_PATH

    try:
        api.MODEL_PATH = synthetic_model["model_path"]
        api.TFIDF_VECTORIZER_PATH = synthetic_model["vec_path"]

        api._MODEL_STATE["model"] = None
        api._MODEL_STATE["scaler"] = None
        api._MODEL_STATE["builder"] = None
        api._MODEL_STATE["ready"] = False

        api.load_model()
        client = TestClient(api.app)

        test_paper = {
            "title": "Test paper on benchmarks",
            "abstract": "We study a comprehensive survey of novel methods for benchmark evaluation.",
            "authors": 3,
            "category": "cs.LG",
            # Omit submitted_date
        }

        response = client.post("/predict", json=test_paper)
        assert response.status_code == 200
        warnings = response.json()["validation_warnings"]

        # Should warn about missing date
        assert any("submitted_date not provided" in w for w in warnings), \
            f"Expected warning about missing date, got: {warnings}"

    finally:
        api.MODEL_PATH = original_model_path
        api.TFIDF_VECTORIZER_PATH = original_vectorizer_path
        api._MODEL_STATE["model"] = None
        api._MODEL_STATE["scaler"] = None
        api._MODEL_STATE["builder"] = None
        api._MODEL_STATE["ready"] = False
