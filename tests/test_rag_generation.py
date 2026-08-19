"""Tests for rag_generation.py: prompt construction and answer generation.

No real Anthropic API calls are made — generate_answer() accepts an
injectable client, so these tests use a fake client that mimics the
anthropic SDK's messages.create() response shape.
"""
import os
from dataclasses import dataclass
from typing import List

import pytest

from rag_indexing import RetrievedChunk
from rag_generation import build_prompt, generate_answer, RAGAnswer


def make_chunk(arxiv_id, title, text, rank=1):
    return RetrievedChunk(
        arxiv_id=arxiv_id, title=title, text=text, chunk_index=0,
        bm25_score=1.0, vector_score=1.0, fused_score=1.0, rank=rank,
    )


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


class FakeMessages:
    def __init__(self, response_text: str):
        self._response_text = response_text
        self.last_call_kwargs = None

    def create(self, **kwargs):
        self.last_call_kwargs = kwargs

        class FakeResponse:
            content = [FakeTextBlock(text=self._response_text)]

        return FakeResponse()


class FakeAnthropicClient:
    def __init__(self, response_text="This is a generated answer [1]."):
        self.messages = FakeMessages(response_text)


def test_build_prompt_includes_question_and_numbered_excerpts():
    chunks = [
        make_chunk("2001.00001", "Paper A", "Abstract text A", rank=1),
        make_chunk("2001.00002", "Paper B", "Abstract text B", rank=2),
    ]
    prompt = build_prompt("What is paper A about?", chunks)

    assert "What is paper A about?" in prompt
    assert "[1]" in prompt
    assert "[2]" in prompt
    assert "2001.00001" in prompt
    assert "Abstract text A" in prompt
    assert "2001.00002" in prompt


def test_build_prompt_no_chunks():
    prompt = build_prompt("Any question", [])
    assert "No relevant excerpts" in prompt
    assert "Any question" in prompt


def test_generate_answer_with_injected_client():
    chunks = [make_chunk("2001.00001", "Paper A", "Abstract text A")]
    client = FakeAnthropicClient(response_text="Paper A studies X [1].")

    result = generate_answer("What does paper A study?", chunks, client=client)

    assert isinstance(result, RAGAnswer)
    assert result.answer == "Paper A studies X [1]."
    assert result.question == "What does paper A study?"
    assert len(result.sources) == 1
    assert result.sources[0].arxiv_id == "2001.00001"
    assert result.sources[0].index == 1

    # Verify the prompt sent to the client included the question
    sent_prompt = client.messages.last_call_kwargs["messages"][0]["content"]
    assert "What does paper A study?" in sent_prompt


def test_generate_answer_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    chunks = [make_chunk("2001.00001", "Paper A", "Abstract text A")]

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        generate_answer("Any question", chunks, client=None)


def test_generate_answer_sources_preserve_order():
    chunks = [
        make_chunk("2001.00001", "Paper A", "Text A", rank=1),
        make_chunk("2001.00002", "Paper B", "Text B", rank=2),
        make_chunk("2001.00003", "Paper C", "Text C", rank=3),
    ]
    client = FakeAnthropicClient()
    result = generate_answer("Q", chunks, client=client)

    assert [s.arxiv_id for s in result.sources] == ["2001.00001", "2001.00002", "2001.00003"]
    assert [s.index for s in result.sources] == [1, 2, 3]
