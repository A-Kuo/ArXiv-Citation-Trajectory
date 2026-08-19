#!/usr/bin/env python3
"""Chunking and index construction for ArXiv paper RAG retrieval.

Pulls papers from the same `papers` table used by the citation-trajectory
pipeline (pipeline.py) and builds a hybrid BM25 + dense-vector index over
title+abstract chunks. No full-text PDF ingestion yet — see README for the
scope note; abstracts are typically short enough to index as a single chunk,
but the chunker supports sliding windows for longer text.
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from rag_embeddings import EmbeddingBackend, build_embedding_backend, load_embedding_backend

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = Path("rag_index")
CHUNK_SIZE_WORDS = 150
CHUNK_OVERLAP_WORDS = 30


def chunk_text(text: str, chunk_size_words: int = CHUNK_SIZE_WORDS, overlap_words: int = CHUNK_OVERLAP_WORDS) -> List[str]:
    """Split text into overlapping word-count windows.

    Most arXiv abstracts (~150-300 words) fit in a single chunk; this only
    kicks in for longer text (e.g. future full-text ingestion).
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size_words:
        return [text.strip()]

    step = max(1, chunk_size_words - overlap_words)
    chunks = []
    for start in range(0, len(words), step):
        window = words[start:start + chunk_size_words]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size_words >= len(words):
            break
    return chunks


def _simple_tokenize(text: str) -> List[str]:
    return text.lower().split()


@dataclass
class RetrievedChunk:
    arxiv_id: str
    title: str
    text: str
    chunk_index: int
    bm25_score: float
    vector_score: float
    fused_score: float
    rank: int


class RAGIndex:
    """In-memory hybrid index: BM25 over tokens + dense vectors over the
    same chunk corpus. Persisted as a directory of pickle/npy artifacts.
    """

    def __init__(self, chunks_df: pd.DataFrame, embedding_backend: EmbeddingBackend,
                 embeddings: np.ndarray, bm25: BM25Okapi):
        self.chunks_df = chunks_df.reset_index(drop=True)
        self.embedding_backend = embedding_backend
        self.embeddings = embeddings
        self.bm25 = bm25

    def __len__(self):
        return len(self.chunks_df)

    def save(self, index_dir: Path = DEFAULT_INDEX_DIR):
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        self.chunks_df.to_parquet(index_dir / "chunks.parquet", index=False)
        np.save(index_dir / "embeddings.npy", self.embeddings)
        with open(index_dir / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
        self.embedding_backend.save(index_dir / "embedding_backend")

        logger.info(f"Saved RAG index ({len(self)} chunks) to {index_dir}")

    @classmethod
    def load(cls, index_dir: Path = DEFAULT_INDEX_DIR) -> "RAGIndex":
        index_dir = Path(index_dir)
        chunks_df = pd.read_parquet(index_dir / "chunks.parquet")
        embeddings = np.load(index_dir / "embeddings.npy")
        with open(index_dir / "bm25.pkl", "rb") as f:
            bm25 = pickle.load(f)
        embedding_backend = load_embedding_backend(index_dir / "embedding_backend")

        return cls(chunks_df=chunks_df, embedding_backend=embedding_backend,
                    embeddings=embeddings, bm25=bm25)


def build_chunks_from_papers(papers_df: pd.DataFrame) -> pd.DataFrame:
    """Turn a papers dataframe (arxiv_id, title, abstract, ...) into a
    flat chunk table (arxiv_id, title, chunk_index, text)."""
    rows = []
    for _, paper in papers_df.iterrows():
        combined = f"{paper['title']}\n\n{paper['abstract']}"
        for i, chunk in enumerate(chunk_text(combined)):
            rows.append({
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "chunk_index": i,
                "text": chunk,
            })
    return pd.DataFrame(rows, columns=["arxiv_id", "title", "chunk_index", "text"])


def build_index(papers_df: pd.DataFrame, embedding_prefer: str = "auto") -> RAGIndex:
    """Build a fresh RAGIndex from a papers dataframe."""
    chunks_df = build_chunks_from_papers(papers_df)
    if len(chunks_df) == 0:
        raise ValueError("No chunks produced — papers_df is empty or missing title/abstract")

    texts = chunks_df["text"].tolist()

    embedding_backend = build_embedding_backend(texts, prefer=embedding_prefer)
    embeddings = embedding_backend.encode(texts)

    tokenized = [_simple_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)

    logger.info(f"Built RAG index: {len(chunks_df)} chunks from {papers_df['arxiv_id'].nunique()} papers "
                f"(embedding backend: {embedding_backend.name})")
    return RAGIndex(chunks_df=chunks_df, embedding_backend=embedding_backend, embeddings=embeddings, bm25=bm25)


def build_index_from_db(engine=None, embedding_prefer: str = "auto") -> RAGIndex:
    """Convenience wrapper: pull papers from the `papers` table and build an index."""
    from pipeline import get_engine

    if engine is None:
        engine = get_engine()

    papers_df = pd.read_sql("SELECT arxiv_id, title, abstract FROM papers", engine)
    papers_df = papers_df.dropna(subset=["title", "abstract"])
    papers_df = papers_df[(papers_df["title"].str.strip() != "") & (papers_df["abstract"].str.strip() != "")]

    return build_index(papers_df, embedding_prefer=embedding_prefer)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    index = build_index_from_db()
    index.save()


if __name__ == "__main__":
    main()
