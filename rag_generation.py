#!/usr/bin/env python3
"""Answer generation for ArXiv paper RAG: takes a question and retrieved
chunks, prompts Claude to produce a grounded answer with inline citations.

Requires ANTHROPIC_API_KEY in the environment (see
https://console.anthropic.com/settings/keys). The Anthropic client is only
constructed lazily, inside generate_answer(), so importing this module or
running retrieval-only code paths never requires a key.
"""

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

from rag_indexing import RetrievedChunk

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a research assistant answering questions about arXiv papers using "
    "only the provided excerpts. Cite sources inline using bracketed numbers "
    "matching the excerpt numbers, e.g. [1], [2]. If the excerpts don't contain "
    "enough information to answer, say so plainly instead of guessing."
)


@dataclass
class Source:
    index: int
    arxiv_id: str
    title: str
    excerpt: str


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: List[Source]
    model: str


def build_prompt(question: str, retrieved_chunks: List[RetrievedChunk]) -> str:
    """Build the user-turn prompt: numbered excerpts followed by the question."""
    if not retrieved_chunks:
        return f"No relevant excerpts were found in the index.\n\nQuestion: {question}"

    excerpt_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        excerpt_blocks.append(
            f"[{i}] arXiv:{chunk.arxiv_id} — {chunk.title}\n{chunk.text}"
        )

    excerpts_text = "\n\n".join(excerpt_blocks)
    return f"Excerpts:\n\n{excerpts_text}\n\nQuestion: {question}"


def _chunks_to_sources(retrieved_chunks: List[RetrievedChunk]) -> List[Source]:
    return [
        Source(index=i, arxiv_id=c.arxiv_id, title=c.title, excerpt=c.text)
        for i, c in enumerate(retrieved_chunks, start=1)
    ]


def generate_answer(
    question: str,
    retrieved_chunks: List[RetrievedChunk],
    client=None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = MAX_TOKENS,
) -> RAGAnswer:
    """Generate a grounded answer from retrieved chunks via the Claude API.

    Args:
        question: the user's natural-language question
        retrieved_chunks: results from rag_retrieval.hybrid_search()
        client: an anthropic.Anthropic-compatible client (injectable for
            testing); if None, one is constructed from ANTHROPIC_API_KEY
        model: model id to use for generation
        max_tokens: generation token cap

    Raises:
        RuntimeError: if client is None and ANTHROPIC_API_KEY is not set
    """
    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Set it in your environment to use "
                "RAG answer generation, e.g.: export ANTHROPIC_API_KEY=sk-ant-..."
            )
        import anthropic
        client = anthropic.Anthropic()

    prompt = build_prompt(question, retrieved_chunks)

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    answer_text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )

    logger.info(f"Generated answer for {question!r} using {len(retrieved_chunks)} sources")

    return RAGAnswer(
        question=question,
        answer=answer_text,
        sources=_chunks_to_sources(retrieved_chunks),
        model=model,
    )
