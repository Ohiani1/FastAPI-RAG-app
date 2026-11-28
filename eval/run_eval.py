"""
Evaluate the RAG pipeline on the eval questions.

    python -m eval.run_eval                                  # retrieval only (cheap)
    python -m eval.run_eval --answer                         # + generate answers
    python -m eval.run_eval --judge --label "baseline"       # + LLM judge, record the run
    python -m eval.run_eval --report                         # print the README results table

Questions come from eval/questions.jsonl and eval/extra_questions.jsonl.
"expected_pages": [] marks a question the docs don't answer.

Metrics
  Retrieval (answerable only)
    hit@k   share of questions with an expected page in the top k chunks
    MRR     mean of 1/rank of the first expected page (0 if missing from the top k)
  Answers (with --judge)
    pass            relevance >= 4 and groundedness >= 4 (see eval/judge.py)
    groundedness    average 1-5; low means the model is answering from memory
    false not-found answerable questions where the model said it couldn't find it
    correct decline unanswerable questions where it declined without inventing an answer
  Cost and latency (with --answer): cost per query, p50/p95 of the pipeline (judge excluded)
"""

import argparse
import csv
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from api.rag import generate, to_dict
from api.retrieval import retrieve
from common.db import close_pool
from common.config import (EMBED_PRICE_PER_M, EMBEDDING_MODEL, LLM_MODEL, LLM_PRICE_IN_PER_M,
                           LLM_PRICE_OUT_PER_M, LLM_PROVIDER, TOP_K)
from ingest.chunker import DOCS_ROOT

QUESTION_FILES = [Path("eval/questions.jsonl"), Path("eval/extra_questions.jsonl")]
RESULTS_DIR = Path("eval/results")
RUNS_CSV = RESULTS_DIR / "runs.csv"
KS = (1, 3, 5, 10)


# ---------- loading and checks ----------

def load_questions() -> list[dict]:
    questions = []
    for path in QUESTION_FILES:
        if not path.exists():
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip():
                try:
                    questions.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise SystemExit(f"{path}:{n} is not valid JSON ({e})")
    problems = []
    ids = [q["id"] for q in questions]
    problems += [f"duplicate id {i}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    for q in questions:
        for page in q.get("expected_pages", []):
            if not (DOCS_ROOT / page).exists():
                problems.append(f"{q['id']}: expected page not found in docs: {page}")
    if problems:
        raise SystemExit("Fix these in your question files first:\n  " + "\n  ".join(problems))
    return questions


# ---------- per-question work ----------

def run_one(q: dict, args) -> dict:
    expected = q.get("expected_pages", [])
    k = max(args.k, TOP_K)
    t0 = time.perf_counter()
    chunks, embed_tokens = retrieve(q["question"], k)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    pages = [c.page_path for c in chunks]
    rank = next((i for i, p in enumerate(pages, 1) if p in expected), None) if expected else None
    row = {"id": q["id"], "question": q["question"], "expected_pages": expected,
           "tags": q.get("tags", []), "hit_rank": rank, "top_pages": pages[:10]}

    if args.answer or args.judge:
        # The LLM sees the same TOP_K chunks a real user request would get.
        answer = generate(q["question"], chunks[:TOP_K], None, embed_tokens, retrieval_ms)
        row["result"] = to_dict(answer)
        if args.judge:
            from eval.judge import judge
            row["judge"] = judge(q["question"], answer.answer, chunks[:TOP_K], bool(expected))
    return row


# ---------- metrics ----------

def pct(values, p):
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["expected_pages"]]
    unans = [r for r in rows if not r["expected_pages"]]
    m = {"n_answerable": len(ans), "n_unanswerable": len(unans)}
    for k in KS:
        m[f"hit@{k}"] = sum(1 for r in ans if r["hit_rank"] and r["hit_rank"] <= k) / max(len(ans), 1)
    m["mrr"] = sum(1 / r["hit_rank"] for r in ans if r["hit_rank"]) / max(len(ans), 1)

    recent = [r for r in ans if "recent" in r["tags"]]
    if recent:
        m["recent_hit@5"] = sum(1 for r in recent if r["hit_rank"] and r["hit_rank"] <= 5) / len(recent)

    if rows and "result" in rows[0]:
        res = [r["result"] for r in rows]
        m["false_not_found"] = sum(1 for r in ans if not r["result"]["found"]) / max(len(ans), 1)
        lat = [x["latency_ms"]["total"] for x in res]
        m["p50_ms"], m["p95_ms"] = pct(lat, 50), pct(lat, 95)
        tok_in = statistics.mean(x["usage"]["input_tokens"] for x in res)
        tok_out = statistics.mean(x["usage"]["output_tokens"] for x in res)
        tok_emb = statistics.mean(x["usage"]["embedding_tokens"] for x in res)
        m["avg_input_tokens"], m["avg_output_tokens"] = round(tok_in), round(tok_out)
        if LLM_PRICE_IN_PER_M or LLM_PRICE_OUT_PER_M:
            m["cost_per_query"] = (tok_in * LLM_PRICE_IN_PER_M + tok_out * LLM_PRICE_OUT_PER_M
                                   + tok_emb * EMBED_PRICE_PER_M) / 1_000_000

    if rows and "judge" in rows[0]:
        ja = [r["judge"] for r in ans]
        m["answer_pass"] = sum(1 for j in ja if j.get("pass")) / max(len(ja), 1)
        for key in ("relevance", "groundedness", "citations"):
            vals = [j[key] for j in ja if key in j]
            m[key] = statistics.mean(vals) if vals else None
        m["correct_decline"] = sum(1 for r in unans if r["judge"].get("pass")) / max(len(unans), 1)
        if recent:
            m["recent_pass"] = sum(1 for r in recent if r["judge"].get("pass")) / len(recent)
    return m


def fmt(v, kind="pct"):
    if v is None:
        return "-"
    return {"pct": f"{v:.0%}", "num": f"{v:.2f}", "ms": f"{v:,.0f} ms",
            "usd": f"${v:.4f}"}[kind]


def print_summary(m: dict) -> None:
    print(f"\n{m['n_answerable']} answerable, {m['n_unanswerable']} unanswerable")
    print("Retrieval:  " + "  ".join(f"hit@{k} {fmt(m[f'hit@{k}'])}" for k in KS)
          + f"  MRR {fmt(m['mrr'], 'num')}")
    if "recent_hit@5" in m:
        print(f"  recent-page questions: hit@5 {fmt(m['recent_hit@5'])}")
    if "answer_pass" in m:
        print(f"Answers:    pass {fmt(m['answer_pass'])}  relevance {fmt(m['relevance'], 'num')}  "
              f"groundedness {fmt(m['groundedness'], 'num')}  citations {fmt(m['citations'], 'num')}")
        print(f"  correct declines (unanswerable): {fmt(m['correct_decline'])}   "
              f"false not-found (answerable): {fmt(m['false_not_found'])}")
        if "recent_pass" in m:
            print(f"  recent-page questions: pass {fmt(m['recent_pass'])}")
    elif "false_not_found" in m:
        print(f"Answers:    false not-found {fmt(m['false_not_found'])}  (add --judge for quality scores)")
    if "p50_ms" in m:
        cost = fmt(m.get("cost_per_query"), "usd") if "cost_per_query" in m else \
            f"n/a (set LLM_PRICE_IN_PER_M / LLM_PRICE_OUT_PER_M; avg {m['avg_input_tokens']} in / " \
            f"{m['avg_output_tokens']} out tokens)"
        print(f"Cost/query: {cost}   latency p50 {fmt(m['p50_ms'], 'ms')}  p95 {fmt(m['p95_ms'], 'ms')}")


# ---------- run log and README table ----------

CSV_FIELDS = ["timestamp", "label", "llm", "embedding", "top_k", "n_answerable", "n_unanswerable",
              "hit@1", "hit@3", "hit@5", "hit@10", "mrr", "recent_hit@5", "answer_pass", "relevance",
              "groundedness", "citations", "correct_decline", "false_not_found", "recent_pass",
              "cost_per_query", "p50_ms", "p95_ms", "results_file"]


def record_run(label: str, m: dict, results_file: Path) -> None:
    new = not RUNS_CSV.exists()
    with RUNS_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({**m, "timestamp": f"{datetime.now():%Y-%m-%d %H:%M}", "label": label,
                    "llm": f"{LLM_PROVIDER}:{LLM_MODEL or 'default'}", "embedding": EMBEDDING_MODEL,
                    "top_k": TOP_K, "results_file": results_file.name})


def print_report() -> None:
    if not RUNS_CSV.exists():
        raise SystemExit("No recorded runs yet. Run with --label first.")
    rows = list(csv.DictReader(RUNS_CSV.open()))

    def cell(r, key, kind="pct"):
        return fmt(float(r[key]), kind) if r.get(key) else "-"

    print("| Run | Questions | hit@5 | hit@10 | MRR | Answer pass | Groundedness | "
          "Correct declines | Cost/query | p50 | p95 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['label']} | {r['n_answerable']}+{r['n_unanswerable']} | {cell(r, 'hit@5')} | "
              f"{cell(r, 'hit@10')} | {cell(r, 'mrr', 'num')} | {cell(r, 'answer_pass')} | "
              f"{cell(r, 'groundedness', 'num')} | {cell(r, 'correct_decline')} | "
              f"{cell(r, 'cost_per_query', 'usd')} | {cell(r, 'p50_ms', 'ms')} | {cell(r, 'p95_ms', 'ms')} |")


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answer", action="store_true", help="generate answers (calls the LLM)")
    ap.add_argument("--judge", action="store_true", help="generate and grade answers")
    ap.add_argument("--label", help="record this run in eval/results/runs.csv under this name")
    ap.add_argument("--report", action="store_true", help="print the README results table")
    ap.add_argument("--k", type=int, default=10, help="retrieval depth for hit@k")
    ap.add_argument("--workers", type=int, default=4, help="questions processed in parallel")
    args = ap.parse_args()

    if args.report:
        print_report()
        return

    questions = load_questions()
    print(f"running {len(questions)} questions...")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(lambda q: run_one(q, args), questions))
    finally:
        close_pool()

    for r in rows:
        mark = "-" if not r["expected_pages"] else (f"hit@{r['hit_rank']}" if r["hit_rank"] else "MISS")
        line = f"{r['id']:>6}  {mark:<7} {r['question'][:68]}"
        if "judge" in r:
            j = r["judge"]
            line += "  PASS" if j.get("pass") else "  FAIL"
            if r["expected_pages"] and "relevance" in j:
                line += f" (rel {j['relevance']}, grnd {j['groundedness']}, cite {j['citations']})"
        elif "result" in r:
            line += f"  found={r['result']['found']}"
        print(line)

    m = summarize(rows)
    print_summary(m)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mode = "judged" if args.judge else "answers" if args.answer else "retrieval"
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{mode}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"\nsaved {out}")
    if args.label:
        record_run(args.label, m, out)
        print(f"recorded as '{args.label}' in {RUNS_CSV}  (see it with --report)")


if __name__ == "__main__":
    main()
