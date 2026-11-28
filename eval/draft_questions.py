"""
Turn eval/candidates.jsonl (from find_candidates.py) into draft eval questions.

For each candidate discussion, the LLM decides whether it's a usable docs question
(not a bug report or a question about someone's private code) and rewrites it as a
short, self-contained question. Expected pages come from the docs links in the thread.

    python -m eval.draft_questions --limit 80

Output: eval/questions_draft.jsonl. REVIEW IT BY HAND before using it:
  - delete questions that are vague, duplicated, or not really answered by the page
  - open each expected page and confirm it answers the question; add a second page if needed
Then append the good ones:  cat eval/questions_draft.jsonl >> eval/questions.jsonl
"""

import argparse
import json
import re
from pathlib import Path

from api.llm import make_llm
from common.config import LLM_MODEL, LLM_PROVIDER

CANDIDATES = Path("eval/candidates.jsonl")
QUESTIONS = Path("eval/questions.jsonl")
OUT = Path("eval/questions_draft.jsonl")

PROMPT = """You are helping build an evaluation set for a FastAPI documentation assistant.

Below is a GitHub Discussion from the FastAPI repo. Decide if it is a good eval question:
KEEP it if it is a "how do I..." or "why does..." question about using FastAPI that the
official docs could answer.
DROP it if it is a bug report, a feature request, mostly about another library, or only
makes sense with the user's own private code.

If you keep it, rewrite it as ONE short, self-contained question (under 30 words) in plain
English, the way a developer would type it into a docs search box. Remove names, stack
traces and long code. Keep any FastAPI-specific terms (Depends, APIRouter, etc.).

Reply with JSON only, no other text:
{{"keep": true or false, "question": "...", "reason": "a few words"}}

Title: {title}

Body:
{body}"""


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def next_id_number(existing: list[dict]) -> int:
    nums = [int(m.group(1)) for q in existing if (m := re.fullmatch(r"q(\d+)", q.get("id", "")))]
    return max(nums, default=0) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=80, help="how many candidates to process")
    args = ap.parse_args()

    existing = [json.loads(l) for l in QUESTIONS.read_text().splitlines() if l.strip()] \
        if QUESTIONS.exists() else []
    used_sources = {q.get("source") for q in existing}
    candidates = [json.loads(l) for l in CANDIDATES.read_text().splitlines() if l.strip()]
    candidates = [c for c in candidates if c["url"] not in used_sources][: args.limit]

    llm = make_llm(LLM_PROVIDER, LLM_MODEL)
    n, kept, dropped = next_id_number(existing), [], 0
    for i, c in enumerate(candidates, 1):
        resp = llm.complete("", [{"role": "user", "content": PROMPT.format(
            title=c["title"], body=c.get("body_excerpt", "")[:1500])}], max_tokens=300)
        data = parse_json(resp.text)
        if not data or not data.get("keep") or not data.get("question"):
            dropped += 1
            print(f"[{i}/{len(candidates)}] drop  {c['title'][:70]}")
            continue
        kept.append({
            "id": f"q{n:03d}",
            "question": data["question"].strip(),
            "expected_pages": c["answer_pages"] or c["comment_pages"],
            "source": c["url"],
            "tags": [],
            "notes": f"draft ({c['strength']} link): {data.get('reason', '')} | original: {c['title']}",
        })
        print(f"[{i}/{len(candidates)}] keep  q{n:03d} {data['question'][:70]}")
        n += 1

    OUT.write_text("".join(json.dumps(q) + "\n" for q in kept))
    print(f"\nkept {len(kept)}, dropped {dropped} -> {OUT}  (review before appending!)")


if __name__ == "__main__":
    main()
