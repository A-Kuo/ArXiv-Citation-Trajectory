#!/usr/bin/env python3
"""Shared feature construction for training and inference.

feature_engineering.py builds the training feature matrix directly from a
dataframe (vectorized, via pandas). This module provides an equivalent,
single-paper implementation so that api.py and the Streamlit apps compute
features the exact same way as training — same field names, same formulas,
same TF-IDF text ordering, same keyword list, same category handling.

Import FeatureBuilder wherever a single paper (title/abstract/authors/
category/submitted_date) needs to be turned into a model-ready feature
vector.
"""

import re
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import textstat

# Keep in sync with feature_engineering.py FeatureEngineer.extract_metadata_features
KEYWORDS = ["survey", "benchmark", "state-of-the-art", "novel", "we propose"]


def _keyword_feature_name(keyword: str) -> str:
    return f"has_{keyword.replace('-', '_').replace(' ', '_')}"


@dataclass
class FeatureMetadata:
    """Training-derived schema: what features exist and in what order."""

    feature_names: List[str]
    categories: List[str] = field(default_factory=list)


class FeatureBuilder:
    """Builds a single-paper feature vector matching training-time features.

    validation_mode:
        "warn"   — log issues to an internal list but still produce a vector
        "strict" — raise ValueError on missing required fields
    """

    KEYWORDS = KEYWORDS

    def __init__(
        self,
        metadata: FeatureMetadata,
        tfidf_vectorizer: Optional[Any] = None,
        validation_mode: str = "warn",
    ):
        self.metadata = metadata
        self.tfidf_vectorizer = tfidf_vectorizer
        self.validation_mode = validation_mode
        self._validation_log: List[Tuple[str, str]] = []

    def _warn(self, msg: str):
        self._validation_log.append(("warning", msg))

    def _debug(self, msg: str):
        self._validation_log.append(("debug", msg))

    # ------------------------------------------------------------------
    # Feature groups (mirror feature_engineering.py exactly)
    # ------------------------------------------------------------------

    def extract_text_features(self, title: str, abstract: str) -> Dict[str, float]:
        """abstract_length, title_length, flesch_reading_ease, equation_count,
        citation_count — same fields/formulas as
        FeatureEngineer.extract_text_features.
        """
        features: Dict[str, float] = {}

        features["abstract_length"] = float(len(abstract.split()))
        features["title_length"] = float(len(title.split()))

        try:
            features["flesch_reading_ease"] = float(textstat.flesch_reading_ease(abstract))
        except Exception as e:
            self._warn(f"flesch_reading_ease failed ({e}); defaulting to 0.0")
            features["flesch_reading_ease"] = 0.0

        features["equation_count"] = float(abstract.count("$"))
        features["citation_count"] = float(len(re.findall(r"\[\d+\]", abstract)))

        return features

    def extract_metadata_features(
        self,
        authors: Optional[Union[int, str]],
        submitted_date: Optional[datetime],
    ) -> Dict[str, float]:
        """author_count, submission_month, submission_year."""
        features: Dict[str, float] = {}

        if isinstance(authors, str):
            features["author_count"] = float(len(authors.split(",")))
        elif isinstance(authors, (int, float)):
            features["author_count"] = float(authors)
        else:
            self._warn("authors missing; defaulting author_count to 1")
            features["author_count"] = 1.0

        if submitted_date is None:
            self._warn(
                "submitted_date not provided; using current date for "
                "submission_month/submission_year (introduces train/serve "
                "skew vs. the paper's actual submission date)"
            )
            submitted_date = datetime.now()

        features["submission_month"] = float(submitted_date.month)
        features["submission_year"] = float(submitted_date.year)

        return features

    def extract_category_features(self, category: Optional[str]) -> Dict[str, float]:
        """One-hot cat_<category> columns, using the categories seen at
        training time (self.metadata.categories), not a hardcoded list.
        """
        features: Dict[str, float] = {}

        if category is not None and category not in self.metadata.categories:
            self._warn(
                f"category '{category}' was not seen during training "
                f"(known: {self.metadata.categories}); all cat_* features "
                "will be 0"
            )
            category = None

        for cat in self.metadata.categories:
            features[f"cat_{cat}"] = 1.0 if category == cat else 0.0

        return features

    def extract_keyword_features(self, title: str, abstract: str) -> Dict[str, float]:
        """has_survey / has_benchmark / has_state_of_the_art / has_novel /
        has_we_propose — all 5 keywords, matching
        FeatureEngineer.extract_metadata_features.
        """
        combined = (title + " " + abstract).lower()
        features: Dict[str, float] = {}

        for keyword in self.KEYWORDS:
            fname = _keyword_feature_name(keyword)
            features[fname] = 1.0 if re.search(re.escape(keyword), combined) else 0.0

        return features

    def extract_tfidf_features(self, title: str, abstract: str) -> Dict[str, float]:
        """tfidf_<term> features from the training-fitted vectorizer, using
        the same text order as training: abstract + " " + title.

        Looked up by vocabulary term name (not positional index), so the
        mapping is robust regardless of how the vectorizer's internal
        column order compares to feature_names.
        """
        features: Dict[str, float] = {}
        tfidf_feature_names = [f for f in self.metadata.feature_names if f.startswith("tfidf_")]

        if self.tfidf_vectorizer is None:
            if tfidf_feature_names:
                self._warn("TF-IDF vectorizer not loaded; tfidf_* features will be 0")
            for fname in tfidf_feature_names:
                features[fname] = 0.0
            return features

        combined_text = abstract + " " + title
        try:
            vector = self.tfidf_vectorizer.transform([combined_text]).toarray()[0]
            vocabulary = self.tfidf_vectorizer.vocabulary_
        except Exception as e:
            self._warn(f"TF-IDF transform failed ({e}); tfidf_* features will be 0")
            for fname in tfidf_feature_names:
                features[fname] = 0.0
            return features

        for fname in tfidf_feature_names:
            term = fname[len("tfidf_"):]
            idx = vocabulary.get(term)
            if idx is None:
                features[fname] = 0.0
                self._debug(f"term '{term}' not in TF-IDF vocabulary")
            else:
                features[fname] = float(vector[idx])

        return features

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def build_feature_vector(
        self,
        title: str,
        abstract: str,
        authors: Optional[Union[int, str]] = None,
        category: Optional[str] = None,
        submitted_date: Optional[datetime] = None,
    ) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
        """Build a (1, n_features) vector ordered per self.metadata.feature_names.

        Returns (feature_vector, validation_log). validation_log entries are
        (level, message) tuples; level is "warning" or "debug".
        """
        self._validation_log = []

        if not title:
            if self.validation_mode == "strict":
                raise ValueError("title is required")
            self._warn("title missing/empty")
            title = ""

        if not abstract:
            if self.validation_mode == "strict":
                raise ValueError("abstract is required")
            self._warn("abstract missing/empty")
            abstract = ""

        all_features: Dict[str, float] = {}
        all_features.update(self.extract_text_features(title, abstract))
        all_features.update(self.extract_metadata_features(authors, submitted_date))
        all_features.update(self.extract_category_features(category))
        all_features.update(self.extract_keyword_features(title, abstract))
        all_features.update(self.extract_tfidf_features(title, abstract))

        vector = np.zeros(len(self.metadata.feature_names))
        for i, fname in enumerate(self.metadata.feature_names):
            if fname not in all_features:
                self._debug(f"feature '{fname}' not produced by builder; defaulting to 0.0")
            vector[i] = all_features.get(fname, 0.0)

        return vector.reshape(1, -1), self._validation_log


def load_builder_from_training_artifacts(
    model_path: Path = Path("lasso_model.pkl"),
    vectorizer_path: Path = Path("tfidf_vectorizer.pkl"),
    validation_mode: str = "warn",
) -> FeatureBuilder:
    """Reconstruct a FeatureBuilder from saved training artifacts, so
    inference feature order/categories exactly match what the model was
    trained on.
    """
    with open(model_path, "rb") as f:
        model_dict = pickle.load(f)

    feature_names = model_dict.get("feature_names", [])
    if not feature_names:
        raise ValueError(f"'feature_names' not found in {model_path}")

    categories = sorted(
        fname[len("cat_"):] for fname in feature_names if fname.startswith("cat_")
    )

    tfidf_vectorizer = None
    if vectorizer_path.exists():
        with open(vectorizer_path, "rb") as f:
            tfidf_vectorizer = pickle.load(f)

    metadata = FeatureMetadata(feature_names=list(feature_names), categories=categories)
    return FeatureBuilder(
        metadata=metadata,
        tfidf_vectorizer=tfidf_vectorizer,
        validation_mode=validation_mode,
    )
