"""
The RAG pipeline: retrieve -> build prompt -> call LLM -> extract citations.
Used by both the /ask endpoint and eval/run_eval.py, so the eval tests exactly what users get.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

from api.llm import get_llm
from api.retrieval import Retrieved, retrieve
from common.config import MIN_SIMILARITY, TOP_K

NOT_FOUND = "I couldn't find this in the docs."

SYSTEM_PROMPT = f"""You are a support assistant for the FastAPI web framework.

Answer the user's question using ONLY the numbered documentation excerpts provided.
Rules:
- Cite every claim with the excerpt number in square brackets, like [1] or [2][3].
- Do not use outside knowledge, even if you know the answer. If the excerpts do not
  contain the answer, reply exactly: "{NOT_FOUND}" and nothing else.
- Include short code examples only when they appear in, or follow directly from, the excerpts.
- Be concise and direct."""


@dataclass
class Citation:
    number: int
    heading_path: str
    url: str
    page_path: str


@dataclass
class Answer:
    answer: str
    citations: list[Citation]
    found: bool
    retrieved: list[dict] = field(default_factory=list)   # for debugging and eval
    usage: dict = field(default_factory=dict)
    latency_ms: dict = field(default_factory=dict)


def _format_context(chunks: list[Retrieved]) -> str:
    return "\n\n".join(
        f"[{i}] {c.heading_path}\nSource: {c.url}\n\n{c.content}" for i, c in enumerate(chunks, 1)
    )


def answer_question(question: str, history: list[dict] | None = None, k: int = TOP_K) -> Answer:
    t0 = time.perf_counter()
    chunks, embed_tokens = retrieve(question, k)
    t_retrieve = time.perf_counter()

    retrieved = [{"id": c.id, "page_path": c.page_path, "url": c.url,
                  "similarity": round(c.similarity, 4)} for c in chunks]

    if not chunks or chunks[0].similarity < MIN_SIMILARITY:
        return Answer(NOT_FOUND, [], False, retrieved, {"embedding_tokens": embed_tokens},
                      {"retrieval": round((t_retrieve - t0) * 1000)})

    # History holds earlier turns as plain {"role", "content"} messages; the excerpts
    # are attached only to the latest question.
    messages = [*(history or []), {
        "role": "user",
        "content": f"Documentation excerpts:\n\n{_format_context(chunks)}\n\nQuestion: {question}",
    }]
    resp = get_llm().complete(SYSTEM_PROMPT, messages)
    t_llm = time.perf_counter()

    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", resp.text)})
    citations = [Citation(n, chunks[n - 1].heading_path, chunks[n - 1].url, chunks[n - 1].page_path)
                 for n in cited if 1 <= n <= len(chunks)]
    found = NOT_FOUND.lower().rstrip(".") not in resp.text.lower()

    return Answer(
        answer=resp.text, citations=citations, found=found, retrieved=retrieved,
        usage={"embedding_tokens": embed_tokens, "model": resp.model,
               "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens},
        latency_ms={"retrieval": round((t_retrieve - t0) * 1000),
                    "llm": round((t_llm - t_retrieve) * 1000),
                    "total": round((t_llm - t0) * 1000)},
    )


def to_dict(a: Answer) -> dict:
    return asdict(a)
