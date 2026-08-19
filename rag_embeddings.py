#!/usr/bin/env python3
"""Embedding backends for RAG retrieval.

Provides a pluggable interface so the retrieval pipeline can use real
sentence embeddings when a model registry is reachable, and fall back to a
fully offline TF-IDF + SVD projection when it isn't (e.g. sandboxed CI,
air-gapped deployments). Both backends expose the same encode() contract so
callers never need to know which one is active.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

DEFAULT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TFIDF_SVD_COMPONENTS = 256


class EmbeddingBackend:
    """Interface: encode(texts) -> L2-normalized (n, dim) float32 array."""

    name: str
    dim: int

    def encode(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: Path):
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> "EmbeddingBackend":
        raise NotImplementedError


class SentenceTransformerBackend(EmbeddingBackend):
    """Wraps a sentence-transformers model. Requires network access to the
    model hub on first use (weights are cached locally after that)."""

    def __init__(self, model_name: str = DEFAULT_ST_MODEL):
        from sentence_transformers import SentenceTransformer  # local import: optional dep

        self._model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.name = f"sentence-transformers:{model_name}"
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> np.ndarray:
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return normalize(vectors.astype(np.float32))

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "backend.txt").write_text(f"sentence-transformers\n{self.model_name}\n")

    @classmethod
    def load(cls, path: Path) -> "SentenceTransformerBackend":
        lines = (path / "backend.txt").read_text().splitlines()
        return cls(model_name=lines[1])


class TfidfEmbeddingBackend(EmbeddingBackend):
    """Offline fallback: TF-IDF vectors projected to a dense space via
    TruncatedSVD (latent semantic analysis). Fit once on the corpus at
    index-build time; `.encode()` reuses the fitted vectorizer/projection
    for queries and new documents.
    """

    def __init__(self, n_components: int = TFIDF_SVD_COMPONENTS, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.name = "tfidf-svd"
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._svd: Optional[TruncatedSVD] = None
        self.dim = n_components

    def fit(self, corpus_texts: List[str]):
        self._vectorizer = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), stop_words="english")
        tfidf = self._vectorizer.fit_transform(corpus_texts)

        n_components = min(self.n_components, max(2, min(tfidf.shape) - 1))
        self._svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        self._svd.fit(tfidf)
        self.dim = n_components
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        if self._vectorizer is None or self._svd is None:
            raise RuntimeError("TfidfEmbeddingBackend must be fit() before encode()")
        tfidf = self._vectorizer.transform(texts)
        vectors = self._svd.transform(tfidf).astype(np.float32)
        return normalize(vectors)

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "tfidf_backend.pkl", "wb") as f:
            pickle.dump({
                "vectorizer": self._vectorizer,
                "svd": self._svd,
                "n_components": self.n_components,
                "dim": self.dim,
            }, f)

    @classmethod
    def load(cls, path: Path) -> "TfidfEmbeddingBackend":
        with open(path / "tfidf_backend.pkl", "rb") as f:
            state = pickle.load(f)
        backend = cls(n_components=state["n_components"])
        backend._vectorizer = state["vectorizer"]
        backend._svd = state["svd"]
        backend.dim = state["dim"]
        return backend


def build_embedding_backend(
    corpus_texts: List[str],
    prefer: str = "auto",
    st_model_name: str = DEFAULT_ST_MODEL,
) -> EmbeddingBackend:
    """Build the best available embedding backend.

    prefer="auto": try sentence-transformers, fall back to TF-IDF on any
    failure (missing package, blocked network, model load error).
    prefer="sentence-transformers": require it, raise on failure.
    prefer="tfidf": always use the offline fallback.
    """
    if prefer not in ("auto", "sentence-transformers", "tfidf"):
        raise ValueError(f"Unknown prefer={prefer!r}")

    if prefer in ("auto", "sentence-transformers"):
        try:
            backend = SentenceTransformerBackend(model_name=st_model_name)
            logger.info(f"Using sentence-transformers embedding backend ({st_model_name})")
            return backend
        except Exception as e:
            if prefer == "sentence-transformers":
                raise
            logger.warning(
                f"sentence-transformers backend unavailable ({e}); "
                f"falling back to offline TF-IDF embeddings"
            )

    backend = TfidfEmbeddingBackend()
    backend.fit(corpus_texts)
    logger.info("Using TF-IDF+SVD embedding backend (offline fallback)")
    return backend


def load_embedding_backend(path: Path) -> EmbeddingBackend:
    """Load a previously saved backend by inspecting which artifact is present."""
    path = Path(path)
    if (path / "tfidf_backend.pkl").exists():
        return TfidfEmbeddingBackend.load(path)
    if (path / "backend.txt").exists():
        return SentenceTransformerBackend.load(path)
    raise FileNotFoundError(f"No embedding backend artifacts found in {path}")
