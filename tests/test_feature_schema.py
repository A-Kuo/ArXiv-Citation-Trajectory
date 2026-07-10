"""Tests for the shared FeatureBuilder (feature_schema.py).

These exercise FeatureBuilder in isolation (no pickled model/vectorizer
required) to lock in the exact bugs that used to exist in api.py's
extract_features_from_paper() and streamlit_app_enhanced.py's
predict_from_abstract().
"""
from datetime import datetime

import numpy as np
import pytest

from feature_schema import FeatureBuilder, FeatureMetadata


FEATURE_NAMES = [
    "abstract_length", "title_length", "flesch_reading_ease",
    "equation_count", "citation_count",
    "author_count", "submission_month", "submission_year",
    "cat_cs.LG", "cat_cs.AI",
    "has_survey", "has_benchmark", "has_state_of_the_art", "has_novel", "has_we_propose",
]


@pytest.fixture
def builder():
    metadata = FeatureMetadata(feature_names=FEATURE_NAMES, categories=["cs.LG", "cs.AI"])
    return FeatureBuilder(metadata=metadata, tfidf_vectorizer=None, validation_mode="warn")


def test_citation_count_extracted(builder):
    """Regression test: api.py used to omit citation_count entirely."""
    vector, _ = builder.build_feature_vector(
        title="A paper",
        abstract="prior work [1] and [2] showed this",
        submitted_date=datetime(2021, 3, 1),
    )
    idx = FEATURE_NAMES.index("citation_count")
    assert vector[0, idx] == 2


def test_all_five_keywords_detected(builder):
    """Regression test: api.py only checked 3 of 5 keywords."""
    abstract = "This is a survey and benchmark of state-of-the-art methods. We propose a novel approach."
    vector, _ = builder.build_feature_vector(
        title="Title", abstract=abstract, submitted_date=datetime(2021, 1, 1),
    )
    for kw_feature in ["has_survey", "has_benchmark", "has_state_of_the_art", "has_novel", "has_we_propose"]:
        idx = FEATURE_NAMES.index(kw_feature)
        assert vector[0, idx] == 1.0, f"{kw_feature} should be 1"


def test_author_count_parses_string_and_int(builder):
    vector_str, _ = builder.build_feature_vector(
        title="T", abstract="Some abstract text here.", authors="A, B, C",
        submitted_date=datetime(2021, 1, 1),
    )
    vector_int, _ = builder.build_feature_vector(
        title="T", abstract="Some abstract text here.", authors=3,
        submitted_date=datetime(2021, 1, 1),
    )
    idx = FEATURE_NAMES.index("author_count")
    assert vector_str[0, idx] == 3
    assert vector_int[0, idx] == 3


def test_submitted_date_used_when_provided(builder):
    """Regression test: api.py always used datetime.now() instead of the paper's date."""
    vector, log = builder.build_feature_vector(
        title="T", abstract="Some abstract text here.", submitted_date=datetime(2018, 7, 15),
    )
    month_idx = FEATURE_NAMES.index("submission_month")
    year_idx = FEATURE_NAMES.index("submission_year")
    assert vector[0, month_idx] == 7
    assert vector[0, year_idx] == 2018
    assert not any("submitted_date not provided" in msg for _, msg in log)


def test_missing_submitted_date_warns(builder):
    _, log = builder.build_feature_vector(
        title="T", abstract="Some abstract text here.", submitted_date=None,
    )
    assert any("submitted_date not provided" in msg for level, msg in log if level == "warning")


def test_unknown_category_warns_and_zeros(builder):
    vector, log = builder.build_feature_vector(
        title="T", abstract="Some abstract text here.",
        category="stat.AP", submitted_date=datetime(2021, 1, 1),
    )
    for cat_feature in ["cat_cs.LG", "cat_cs.AI"]:
        idx = FEATURE_NAMES.index(cat_feature)
        assert vector[0, idx] == 0.0
    assert any("not seen during training" in msg for level, msg in log if level == "warning")


def test_known_category_one_hot(builder):
    vector, _ = builder.build_feature_vector(
        title="T", abstract="Some abstract text here.",
        category="cs.AI", submitted_date=datetime(2021, 1, 1),
    )
    assert vector[0, FEATURE_NAMES.index("cat_cs.AI")] == 1.0
    assert vector[0, FEATURE_NAMES.index("cat_cs.LG")] == 0.0


def test_feature_vector_matches_metadata_order(builder):
    vector, _ = builder.build_feature_vector(
        title="T", abstract="Some abstract text here.", submitted_date=datetime(2021, 1, 1),
    )
    assert vector.shape == (1, len(FEATURE_NAMES))


def test_missing_vectorizer_stubs_tfidf_to_zero():
    metadata = FeatureMetadata(
        feature_names=["abstract_length", "tfidf_learning", "tfidf_neural"],
        categories=[],
    )
    builder = FeatureBuilder(metadata=metadata, tfidf_vectorizer=None, validation_mode="warn")
    vector, log = builder.build_feature_vector(
        title="T", abstract="deep learning neural networks",
    )
    assert vector[0, 1] == 0.0
    assert vector[0, 2] == 0.0
    assert any("TF-IDF vectorizer not loaded" in msg for level, msg in log if level == "warning")


def test_strict_mode_requires_title_and_abstract():
    metadata = FeatureMetadata(feature_names=FEATURE_NAMES, categories=["cs.LG"])
    builder = FeatureBuilder(metadata=metadata, tfidf_vectorizer=None, validation_mode="strict")
    with pytest.raises(ValueError):
        builder.build_feature_vector(title="", abstract="something")
    with pytest.raises(ValueError):
        builder.build_feature_vector(title="Something", abstract="")
