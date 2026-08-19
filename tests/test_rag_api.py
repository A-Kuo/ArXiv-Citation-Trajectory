"""Integration tests for the /rag/* FastAPI endpoints.

Builds a small synthetic RAGIndex (TF-IDF backend, fully offline) and
points the API at it directly via _RAG_STATE, so no real index-build or
network calls are needed. Generation is exercised with a monkeypatched
rag_generate_answer to avoid requiring ANTHROPIC_API_KEY.
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api
from rag_indexing import build_index
from rag_generation import RAGAnswer, Source


@pytest.fixture
def client_with_rag_index():
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
    index = build_index(papers_df, embedding_prefer="tfidf")

    original_ready = api._RAG_STATE["ready"]
    original_index = api._RAG_STATE["index"]
    api._RAG_STATE["index"] = index
    api._RAG_STATE["ready"] = True
    api._RAG_STATE["load_time"] = "2024-01-01T00:00:00"

    yield TestClient(api.app)

    api._RAG_STATE["ready"] = original_ready
    api._RAG_STATE["index"] = original_index


@pytest.fixture
def client_without_rag_index():
    original_ready = api._RAG_STATE["ready"]
    original_index = api._RAG_STATE["index"]
    api._RAG_STATE["ready"] = False
    api._RAG_STATE["index"] = None

    yield TestClient(api.app)

    api._RAG_STATE["ready"] = original_ready
    api._RAG_STATE["index"] = original_index


def test_rag_status_reports_not_loaded(client_without_rag_index):
    resp = client_without_rag_index.get("/rag/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["index_loaded"] is False
    assert body["n_chunks"] == 0


def test_rag_status_reports_loaded(client_with_rag_index):
    resp = client_with_rag_index.get("/rag/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["index_loaded"] is True
    assert body["n_chunks"] == 3
    assert body["embedding_backend"] == "tfidf-svd"


def test_rag_search_without_index_returns_503(client_without_rag_index):
    resp = client_without_rag_index.post("/rag/search", json={"question": "What is a transformer?"})
    assert resp.status_code == 503


def test_rag_search_returns_relevant_sources(client_with_rag_index):
    resp = client_with_rag_index.post(
        "/rag/search", json={"question": "transformer self-attention translation", "top_k": 2}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["question"] == "transformer self-attention translation"
    assert len(body["sources"]) <= 2
    assert body["sources"][0]["arxiv_id"] == "2001.00001"
    assert body["sources"][0]["index"] == 1
    assert "fused_score" in body["sources"][0]


def test_rag_search_validates_short_question(client_with_rag_index):
    resp = client_with_rag_index.post("/rag/search", json={"question": "a"})
    assert resp.status_code == 422


def test_rag_ask_without_index_returns_503(client_without_rag_index):
    resp = client_without_rag_index.post("/rag/ask", json={"question": "What is a transformer?"})
    assert resp.status_code == 503


def test_rag_ask_missing_api_key_returns_503(client_with_rag_index, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    resp = client_with_rag_index.post("/rag/ask", json={"question": "What is a transformer?"})
    assert resp.status_code == 503
    assert "ANTHROPIC_API_KEY" in resp.json()["detail"]


def test_rag_ask_with_mocked_generation(client_with_rag_index, monkeypatch):
    def fake_generate_answer(question, chunks, client=None, **kwargs):
        return RAGAnswer(
            question=question,
            answer="Transformers use self-attention [1].",
            sources=[Source(index=1, arxiv_id=c.arxiv_id, title=c.title, excerpt=c.text) for c in chunks],
            model="claude-sonnet-4-5",
        )

    monkeypatch.setattr(api, "rag_generate_answer", fake_generate_answer)

    resp = client_with_rag_index.post(
        "/rag/ask", json={"question": "How do transformers work?", "top_k": 2}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Transformers use self-attention [1]."
    assert body["model"] == "claude-sonnet-4-5"
    assert len(body["sources"]) > 0
