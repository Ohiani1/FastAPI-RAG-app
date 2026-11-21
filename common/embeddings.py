"""Embedding calls, shared by ingestion (documents) and the API (questions)."""

from __future__ import annotations

import numpy as np
from openai import OpenAI

from common.config import EMBEDDING_MODEL

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()  # reads OPENAI_API_KEY
    return _client


def embed_texts(texts: list[str], batch_size: int = 100) -> tuple[list[np.ndarray], int]:
    """Return one vector per text, plus the total tokens used (for cost tracking)."""
    vectors, tokens = [], 0
    for i in range(0, len(texts), batch_size):
        resp = _get_client().embeddings.create(model=EMBEDDING_MODEL, input=texts[i:i + batch_size])
        vectors.extend(np.array(d.embedding, dtype=np.float32) for d in resp.data)
        tokens += resp.usage.total_tokens
    return vectors, tokens


def embed_query(text: str) -> tuple[np.ndarray, int]:
    vectors, tokens = embed_texts([text])
    return vectors[0], tokens
