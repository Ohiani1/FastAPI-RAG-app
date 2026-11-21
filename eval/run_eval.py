"""
Run the eval questions through the pipeline.

    python -m eval.run_eval                 # retrieval only: hit@k (cheap, no LLM calls)
    python -m eval.run_eval --answer        # also generate answers with citations

Retrieval hit@k = share of answerable questions where an expected page appears in the
top k retrieved chunks. Questions with "expected_pages": [] are treated as
unanswerable and should produce "I couldn't find this in the docs."
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from api.rag import answer_question, to_dict
from api.retrieval import retrieve

QUESTIONS = Path("eval/questions.jsonl")
RESULTS_DIR = Path("eval/results")


def first_hit_rank(retrieved_pages: list[str], expected: list[str]) -> int | None:
    for rank, page in enumerate(retrieved_pages, 1):
        if page in expected:
            return rank
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answer", action="store_true", help="also generate answers (calls the LLM)")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    questions = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    rows, ks = [], (1, 3, 5, args.k)

    for q in questions:
        expected = q.get("expected_pages", [])
        chunks, _ = retrieve(q["question"], args.k)
        rank = first_hit_rank([c.page_path for c in chunks], expected) if expected else None
        row = {"id": q["id"], "question": q["question"], "expected_pages": expected,
               "hit_rank": rank, "top_pages": [c.page_path for c in chunks[:5]]}
        if args.answer:
            row["result"] = to_dict(answer_question(q["question"]))
        rows.append(row)

        mark = "-" if not expected else (f"hit@{rank}" if rank else "MISS")
        line = f"{q['id']:>6}  {mark:<7} {q['question'][:70]}"
        if args.answer:
            r = row["result"]
            line += f"\n        found={r['found']} cited={[c['page_path'] for c in r['citations']]}"
        print(line)

    answerable = [r for r in rows if r["expected_pages"]]
    print(f"\n{len(answerable)} answerable, {len(rows) - len(answerable)} unanswerable")
    for k in ks:
        hits = sum(1 for r in answerable if r["hit_rank"] and r["hit_rank"] <= k)
        print(f"  hit@{k:<3} {hits}/{len(answerable)} = {hits / max(len(answerable), 1):.0%}")

    if args.answer:
        unans = [r for r in rows if not r["expected_pages"]]
        correct_nf = sum(1 for r in unans if not r["result"]["found"])
        print(f"  correctly said not-found: {correct_nf}/{len(unans)}")
        cited_right = sum(1 for r in answerable
                          if any(c["page_path"] in r["expected_pages"] for c in r["result"]["citations"]))
        print(f"  answer cites an expected page: {cited_right}/{len(answerable)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}{'-answers' if args.answer else ''}.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
