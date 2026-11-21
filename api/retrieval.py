"""Vector retrieval over the chunks table. (Hybrid BM25 + reranking come later.)"""

from __future__ import annotations

from dataclasses import dataclass

from common.db import get_pool
from common.embeddings import embed_query


@dataclass
class Retrieved:
    id: str
    page_path: str
    heading_path: str
    url: str
    content: str
    similarity: float   # cosine similarity, 1 = identical


def retrieve(question: str, k: int) -> tuple[list[Retrieved], int]:
    """Return the k nearest chunks and the embedding tokens used."""
    qvec, tokens = embed_query(question)
    with get_pool().connection() as conn:
        rows = conn.execute(
            """
            select id, page_path, heading_path, url, content,
                   1 - (embedding <=> %(q)s) as similarity
            from chunks
            order by embedding <=> %(q)s
            limit %(k)s
            """,
            {"q": qvec, "k": k},
        ).fetchall()
    return [Retrieved(*r) for r in rows], tokens
