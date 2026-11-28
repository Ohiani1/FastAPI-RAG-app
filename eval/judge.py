"""
LLM-as-judge for answer quality.

Answerable questions are scored 1-5 on:
  relevance     - does the answer actually address the question?
  groundedness  - is every claim supported by the excerpts? (catches answers from memory)
  citations     - do the cited excerpt numbers support the sentences they're attached to?
  pass          = relevance >= 4 and groundedness >= 4

Unanswerable questions pass when the answer says the docs don't cover it and doesn't
invent a how-to. Mentioning a related topic the docs DO cover is fine.

Spot-check the judge by hand: read ~10 verdicts per run and note disagreements.
"""

from __future__ import annotations

import json
import re

from api.llm import make_llm
from api.rag import format_context
from api.retrieval import Retrieved
from common.config import JUDGE_MODEL, JUDGE_PROVIDER

ANSWERABLE_PROMPT = """You are grading an answer from a FastAPI documentation assistant.
The assistant was told to answer ONLY from the numbered excerpts and cite them like [1].

Score each from 1 (worst) to 5 (best):
- relevance: Does the answer address what the user asked? A correct "the docs don't cover
  this" when the excerpts truly lack the answer scores 3; a refusal when the excerpts DO
  contain the answer scores 1.
- groundedness: Is every factual claim and code line supported by the excerpts? Any claim
  that comes from outside knowledge, even if true, lowers this score.
- citations: Do the [n] markers point to excerpts that support the attached sentences?
  Score 1 if there are no citations at all.

Reply with JSON only:
{{"relevance": n, "groundedness": n, "citations": n, "reason": "one or two sentences"}}

=== Question
{question}

=== Excerpts the assistant saw
{context}

=== Assistant's answer
{answer}"""

UNANSWERABLE_PROMPT = """You are grading an answer from a FastAPI documentation assistant.
The assistant was told to answer ONLY from the provided documentation excerpts. This
question is NOT answered by the FastAPI docs, so the correct behavior is to say the docs
don't cover it. Briefly pointing to a related topic the excerpts do cover is fine.

declined = true if the answer clearly says the docs don't cover this.
hallucinated = true if the answer gives instructions, APIs, or code that are not in the
excerpts (for example, inventing a FastAPI feature or explaining how to set up the tool).

Reply with JSON only:
{{"declined": true or false, "hallucinated": true or false, "reason": "one sentence"}}

=== Question
{question}

=== Excerpts the assistant saw
{context}

=== Assistant's answer
{answer}"""

_judge = None


def _get_judge():
    global _judge
    if _judge is None:
        _judge = make_llm(JUDGE_PROVIDER, JUDGE_MODEL)
    return _judge


def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def judge(question: str, answer: str, chunks: list[Retrieved], answerable: bool) -> dict:
    prompt = ANSWERABLE_PROMPT if answerable else UNANSWERABLE_PROMPT
    resp = _get_judge().complete("", [{"role": "user", "content": prompt.format(
        question=question, context=format_context(chunks) or "(none)", answer=answer)}],
        max_tokens=300)
    data = _parse(resp.text)
    if not data:
        return {"error": "unparseable judge output", "raw": resp.text[:300], "pass": False}
    if answerable:
        for key in ("relevance", "groundedness", "citations"):
            data[key] = int(data.get(key, 0))
        data["pass"] = data["relevance"] >= 4 and data["groundedness"] >= 4
    else:
        data["pass"] = bool(data.get("declined")) and not data.get("hallucinated")
    return data
