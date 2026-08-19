"""Tests for rag_retrieval.py: BM25/vector search and RRF fusion."""
import pandas as pd
import pytest

from rag_indexing import build_index
from rag_retrieval import (
    bm25_search,
    hybrid_search,
    reciprocal_rank_fusion,
    vector_search,
)


def test_reciprocal_rank_fusion_basic():
    # Item 0 ranks #1 in list A and #2 in list B -> should score highest
    list_a = [0, 1, 2]
    list_b = [1, 0, 2]
    fused = reciprocal_rank_fusion([list_a, list_b], k=60)

    expected_0 = 1 / 61 + 1 / 62  # rank 1 in A, rank 2 in B
    expected_1 = 1 / 62 + 1 / 61  # rank 2 in A, rank 1 in B
    expected_2 = 1 / 63 + 1 / 63  # rank 3 in both

    assert fused[0] == pytest.approx(expected_0)
    assert fused[1] == pytest.approx(expected_1)
    assert fused[2] == pytest.approx(expected_2)
    # 0 and 1 are symmetric (each #1 in one list, #2 in the other) -> tied
    assert fused[0] == pytest.approx(fused[1])
    assert fused[0] > fused[2]


def test_reciprocal_rank_fusion_item_missing_from_one_list():
    list_a = [0, 1]
    list_b = [1]  # item 0 absent here
    fused = reciprocal_rank_fusion([list_a, list_b], k=60)

    assert fused[0] == pytest.approx(1 / 61)  # only from list_a
    assert fused[1] == pytest.approx(1 / 62 + 1 / 61)  # from both


@pytest.fixture
def toy_index():
    papers_df = pd.DataFrame({
        "arxiv_id": ["2001.00001", "2001.00002", "2001.00003"],
        "title": [
            "Transformer Architectures for Machine Translation",
            "Convolutional Networks for Image Classification",
            "Reinforcement Learning for Robotic Control",
        ],
        "abstract": [
            "We propose a novel transformer architecture using self-attention "
            "mechanisms for neural machine translation between languages.",
            "This paper presents a convolutional neural network approach for "
            "classifying images into categories using deep learning.",
            "We study reinforcement learning algorithms for controlling robotic "
            "arms in continuous action spaces using policy gradients.",
        ],
    })
    return build_index(papers_df, embedding_prefer="tfidf")


def test_bm25_search_finds_relevant_paper(toy_index):
    ranked = bm25_search(toy_index, "transformer attention translation", top_k=3)
    assert len(ranked) > 0
    top_chunk = toy_index.chunks_df.iloc[ranked[0]]
    assert top_chunk["arxiv_id"] == "2001.00001"


def test_vector_search_finds_relevant_paper(toy_index):
    ranked = vector_search(toy_index, "convolutional image classification", top_k=3)
    assert len(ranked) > 0
    top_chunk = toy_index.chunks_df.iloc[ranked[0]]
    assert top_chunk["arxiv_id"] == "2001.00002"


def test_hybrid_search_returns_ranked_results_with_scores(toy_index):
    results = hybrid_search(toy_index, "robotic arm reinforcement learning", top_k=2)

    assert len(results) <= 2
    assert results[0].arxiv_id == "2001.00003"
    assert results[0].rank == 1
    # Scores should be populated (not all zero)
    assert results[0].fused_score > 0
    assert isinstance(results[0].bm25_score, float)
    assert isinstance(results[0].vector_score, float)


def test_hybrid_search_empty_index_returns_empty():
    empty_df = pd.DataFrame({"arxiv_id": [], "title": [], "abstract": []})
    # build_index raises on empty input, so construct an index with 0 chunks differently
    papers_df = pd.DataFrame({
        "arxiv_id": ["x"], "title": ["t"], "abstract": ["some abstract text here"],
    })
    index = build_index(papers_df, embedding_prefer="tfidf")
    index.chunks_df = index.chunks_df.iloc[0:0]
    results = hybrid_search(index, "anything", top_k=5)
    assert results == []
