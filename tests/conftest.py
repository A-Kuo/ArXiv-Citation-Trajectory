"""Pytest fixtures and shared test utilities."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


@pytest.fixture
def synthetic_features():
    """Generate small synthetic feature matrix for testing."""
    np.random.seed(42)
    n_samples = 100
    features = pd.DataFrame({
        'abstract_length': np.random.randint(50, 200, n_samples),
        'title_length': np.random.randint(5, 20, n_samples),
        'flesch_reading_ease': np.random.uniform(0, 100, n_samples),
        'author_count': np.random.randint(1, 10, n_samples),
        'submission_month': np.random.randint(1, 13, n_samples),
        'submission_year': np.random.choice([2019, 2020, 2021, 2022], n_samples),
        'cat_cs_LG': np.random.choice([0, 1], n_samples),
        'cat_cs_AI': np.random.choice([0, 1], n_samples),
        'has_survey': np.random.choice([0, 1], n_samples),
    })
    return features


@pytest.fixture
def synthetic_targets():
    """Generate small synthetic target for testing."""
    np.random.seed(42)
    targets = pd.DataFrame({
        'citations_24mo': np.random.randint(0, 100, 100),
        'citations_24mo_log': np.log1p(np.random.randint(0, 100, 100)),
        'top_quartile': np.random.choice([0, 1], 100),
    })
    return targets


@pytest.fixture
def synthetic_arxiv_ids():
    """Generate ArXiv IDs for testing."""
    arxiv_ids = pd.DataFrame({
        'arxiv_id': [f"2020.{i:05d}" for i in range(100)],
    })
    return arxiv_ids


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for test outputs."""
    return tmp_path
