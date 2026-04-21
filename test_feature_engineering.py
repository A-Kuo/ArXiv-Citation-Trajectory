#!/usr/bin/env python3
"""Tests for feature engineering pipeline."""

import sqlite3
import tempfile
import os
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

import feature_engineering


def setup_test_db(db_path: str, num_papers: int = 10, include_nulls: bool = False):
    """Create a test database with configurable data."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE papers (
            arxiv_id TEXT PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            authors TEXT,
            category TEXT,
            submitted_date TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE citations (
            arxiv_id TEXT PRIMARY KEY,
            citations_12mo INTEGER,
            citations_24mo INTEGER,
            fetched_date TEXT
        )
    """)

    base_date = datetime(2020, 1, 1)

    for i in range(num_papers):
        arxiv_id = f"2001.{i:05d}"
        title = f"Test Paper {i}: Novel Deep Learning Approach"
        abstract = (
            f"We propose a state-of-the-art benchmark for machine learning. "
            f"The survey covers recent advances in the field. We analyze [1] existing "
            f"methods and [2] propose novel solutions. This is $important$ work."
        )
        authors = f"Author {i}, Author {i+1}, Author {i+2}"
        category = ["cs.LG", "cs.AI", "stat.ML"][i % 3]
        submitted_date = (base_date + timedelta(days=i*30)).date()

        cursor.execute(
            """INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?)""",
            (arxiv_id, title, abstract, authors, category, str(submitted_date)),
        )

        citations_24mo = i * 10 if not include_nulls or i % 2 == 0 else None
        cursor.execute(
            """INSERT INTO citations VALUES (?, ?, ?, ?)""",
            (arxiv_id, i * 5, citations_24mo, "2024-01-01"),
        )

    conn.commit()
    conn.close()


def test_basic_pipeline():
    """Test basic pipeline execution."""
    test_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db').name
    original_db = feature_engineering.DB_PATH

    try:
        setup_test_db(test_db, num_papers=10)
        feature_engineering.DB_PATH = test_db

        engineer = feature_engineering.FeatureEngineer(db_path=test_db)
        features, targets, arxiv_ids = engineer.run()

        assert features.shape[0] > 0, "No features generated"
        assert targets.shape[0] > 0, "No targets generated"
        assert features.shape[0] == targets.shape[0], "Feature-target mismatch"

        print("✓ Basic pipeline test passed")
        print(f"  Generated {features.shape[0]} samples with {features.shape[1]} features")

    finally:
        feature_engineering.DB_PATH = original_db
        if os.path.exists(test_db):
            os.remove(test_db)


def test_null_handling():
    """Test handling of null citations."""
    test_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db').name
    original_db = feature_engineering.DB_PATH

    try:
        setup_test_db(test_db, num_papers=10, include_nulls=True)
        feature_engineering.DB_PATH = test_db

        engineer = feature_engineering.FeatureEngineer(db_path=test_db)
        features, targets, arxiv_ids = engineer.run()

        assert features.shape[0] == 5, "Null citations should be filtered"
        print("✓ Null handling test passed")
        print(f"  Correctly filtered out null citations: {features.shape[0]}/10 retained")

    finally:
        feature_engineering.DB_PATH = original_db
        if os.path.exists(test_db):
            os.remove(test_db)


def test_feature_types():
    """Test that all expected features are created."""
    test_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db').name
    original_db = feature_engineering.DB_PATH

    try:
        setup_test_db(test_db, num_papers=10)
        feature_engineering.DB_PATH = test_db

        engineer = feature_engineering.FeatureEngineer(db_path=test_db)
        features, targets, arxiv_ids = engineer.run()

        expected_features = [
            "abstract_length", "title_length", "flesch_reading_ease",
            "equation_count", "citation_count", "author_count",
            "submission_month", "submission_year"
        ]

        for feat in expected_features:
            assert feat in features.columns, f"Missing feature: {feat}"

        assert any("cat_" in col for col in features.columns), "Missing category features"
        assert any("has_" in col for col in features.columns), "Missing keyword features"
        if features.shape[0] > 5:
            assert any("tfidf_" in col for col in features.columns), "Missing TF-IDF features"

        print("✓ Feature types test passed")
        print(f"  All expected feature types present")

    finally:
        feature_engineering.DB_PATH = original_db
        if os.path.exists(test_db):
            os.remove(test_db)


def test_target_construction():
    """Test target variable construction."""
    test_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db').name
    original_db = feature_engineering.DB_PATH

    try:
        setup_test_db(test_db, num_papers=10)
        feature_engineering.DB_PATH = test_db

        engineer = feature_engineering.FeatureEngineer(db_path=test_db)
        features, targets, arxiv_ids = engineer.run()

        assert "citations_24mo" in targets.columns
        assert "citations_24mo_log" in targets.columns
        assert "top_quartile" in targets.columns

        assert targets["citations_24mo"].min() >= 0
        assert targets["citations_24mo_log"].min() >= 0
        assert targets["top_quartile"].isin([0, 1]).all()

        print("✓ Target construction test passed")
        print(f"  All target variables correctly constructed")
        print(f"  Citation range: {targets['citations_24mo'].min():.0f}-{targets['citations_24mo'].max():.0f}")

    finally:
        feature_engineering.DB_PATH = original_db
        if os.path.exists(test_db):
            os.remove(test_db)


if __name__ == "__main__":
    print("Running feature engineering tests...\n")
    test_basic_pipeline()
    test_null_handling()
    test_feature_types()
    test_target_construction()
    print("\n" + "="*50)
    print("All feature engineering tests passed!")
    print("="*50)
