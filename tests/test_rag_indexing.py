"""Tests for rag_indexing.py: chunking and index construction."""
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from rag_indexing import (
    build_chunks_from_papers,
    build_index,
    chunk_text,
    RAGIndex,
)


def test_chunk_text_short_text_single_chunk():
    text = "This is a short abstract about neural networks."
    chunks = chunk_text(text, chunk_size_words=150, overlap_words=30)
    assert len(chunks) == 1
    assert chunks[0] == text.strip()


def test_chunk_text_empty_string():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_long_text_multiple_windows_with_overlap():
    words = [f"word{i}" for i in range(400)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size_words=150, overlap_words=30)

    assert len(chunks) > 1
    # Each chunk (except possibly the last) should have chunk_size_words words
    for chunk in chunks[:-1]:
        assert len(chunk.split()) == 150

    # Verify overlap: last 30 words of chunk[0] == first 30 words of chunk[1]
    chunk0_words = chunks[0].split()
    chunk1_words = chunks[1].split()
    assert chunk0_words[-30:] == chunk1_words[:30]

    # Last chunk should end with the last word of input
    assert chunks[-1].split()[-1] == "word399"


def test_build_chunks_from_papers():
    papers_df = pd.DataFrame({
        "arxiv_id": ["2001.00001", "2001.00002"],
        "title": ["Paper One", "Paper Two"],
        "abstract": ["A short abstract about learning.", "Another short abstract about vision."],
    })
    chunks_df = build_chunks_from_papers(papers_df)

    assert len(chunks_df) == 2  # one chunk per paper (short abstracts)
    assert set(chunks_df.columns) == {"arxiv_id", "title", "chunk_index", "text"}
    assert chunks_df.iloc[0]["arxiv_id"] == "2001.00001"
    assert "Paper One" in chunks_df.iloc[0]["text"]
    assert "abstract about learning" in chunks_df.iloc[0]["text"]


def test_build_index_tfidf_fallback():
    papers_df = pd.DataFrame({
        "arxiv_id": [f"2001.{i:05d}" for i in range(10)],
        "title": [f"Paper {i} on neural networks" for i in range(10)],
        "abstract": [f"We study deep learning method number {i} for image classification." for i in range(10)],
    })
    index = build_index(papers_df, embedding_prefer="tfidf")

    assert len(index) == 10
    assert index.embeddings.shape[0] == 10
    assert index.embedding_backend.name == "tfidf-svd"


def test_build_index_empty_raises():
    papers_df = pd.DataFrame(columns=["arxiv_id", "title", "abstract"])
    with pytest.raises(ValueError):
        build_index(papers_df, embedding_prefer="tfidf")


def test_index_save_load_roundtrip():
    papers_df = pd.DataFrame({
        "arxiv_id": [f"2001.{i:05d}" for i in range(5)],
        "title": [f"Paper {i}" for i in range(5)],
        "abstract": [f"Abstract text number {i} about machine learning topics." for i in range(5)],
    })
    index = build_index(papers_df, embedding_prefer="tfidf")

    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = Path(tmpdir) / "rag_index"
        index.save(index_dir)

        loaded = RAGIndex.load(index_dir)

        assert len(loaded) == len(index)
        assert loaded.embedding_backend.name == index.embedding_backend.name
        assert (loaded.embeddings.shape == index.embeddings.shape)

        # Query encoding should still work post-load
        query_vec = loaded.embedding_backend.encode(["machine learning"])
        assert query_vec.shape[1] == loaded.embeddings.shape[1]
