#!/usr/bin/env python3
"""Hybrid retrieval over a RAGIndex: BM25 keyword search + dense vector
search, merged with Reciprocal Rank Fusion (RRF) — the same combination
strategy used by production hybrid-search RAG systems.
"""

import logging
from typing import Dict, List

import numpy as np

from rag_indexing import RAGIndex, RetrievedChunk, _simple_tokenize

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_K = 20
RRF_K = 60  # standard RRF damping constant


def bm25_search(index: RAGIndex, query: str, top_k: int = DEFAULT_CANDIDATE_K) -> List[int]:
    """Return chunk indices ranked by BM25 score, best first."""
    tokens = _simple_tokenize(query)
    scores = index.bm25.get_scores(tokens)
    ranked = np.argsort(scores)[::-1][:top_k]
    return [int(i) for i in ranked if scores[i] > 0] or [int(i) for i in ranked[:top_k]]


def vector_search(index: RAGIndex, query: str, top_k: int = DEFAULT_CANDIDATE_K) -> List[int]:
    """Return chunk indices ranked by cosine similarity, best first."""
    query_vec = index.embedding_backend.encode([query])[0]
    scores = index.embeddings @ query_vec
    ranked = np.argsort(scores)[::-1][:top_k]
    return [int(i) for i in ranked]


def reciprocal_rank_fusion(rank_lists: List[List[int]], k: int = RRF_K) -> Dict[int, float]:
    """Fuse multiple ranked lists of chunk indices into one score per index.

    RRF score for item i = sum over lists containing i of 1 / (k + rank),
    where rank is the item's 1-indexed position in that list. Items absent
    from a list contribute nothing from it. This rewards items that rank
    well across multiple retrieval methods without needing score
    normalization between BM25 and cosine similarity.
    """
    fused: Dict[int, float] = {}
    for rank_list in rank_lists:
        for rank, idx in enumerate(rank_list, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return fused


def hybrid_search(index: RAGIndex, query: str, top_k: int = DEFAULT_TOP_K,
                   candidate_k: int = DEFAULT_CANDIDATE_K) -> List[RetrievedChunk]:
    """Hybrid BM25 + vector search with RRF fusion.

    Returns the top_k chunks with their component scores attached so
    callers (API, dashboard) can show why a result was retrieved.
    """
    if len(index) == 0:
        return []

    bm25_ranked = bm25_search(index, query, top_k=candidate_k)
    vector_ranked = vector_search(index, query, top_k=candidate_k)

    bm25_rank_of = {idx: rank for rank, idx in enumerate(bm25_ranked, start=1)}
    vector_rank_of = {idx: rank for rank, idx in enumerate(vector_ranked, start=1)}

    tokens = _simple_tokenize(query)
    bm25_raw_scores = index.bm25.get_scores(tokens)
    query_vec = index.embedding_backend.encode([query])[0]
    vector_raw_scores = index.embeddings @ query_vec

    fused = reciprocal_rank_fusion([bm25_ranked, vector_ranked])
    fused_sorted = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    results = []
    for rank, (idx, fused_score) in enumerate(fused_sorted, start=1):
        row = index.chunks_df.iloc[idx]
        results.append(RetrievedChunk(
            arxiv_id=row["arxiv_id"],
            title=row["title"],
            text=row["text"],
            chunk_index=int(row["chunk_index"]),
            bm25_score=float(bm25_raw_scores[idx]),
            vector_score=float(vector_raw_scores[idx]),
            fused_score=float(fused_score),
            rank=rank,
        ))

    logger.info(f"hybrid_search({query!r}): {len(bm25_ranked)} bm25 + {len(vector_ranked)} vector "
                f"candidates -> {len(results)} fused results")
    return results
