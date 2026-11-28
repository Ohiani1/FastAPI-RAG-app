# Ask the FastAPI Docs

A support assistant that answers FastAPI questions using only the official documentation,
cites the exact doc section for every claim, and says "I couldn't find this in the docs"
instead of guessing. Built for developers who'd otherwise dig through docs pages and GitHub Discussions.

> **Status:** ingestion, retrieval, answering API and evaluation harness are working.
> Hybrid search, reranking, frontend and deployment are in progress.

**Live demo:** coming soon

## How it works

```
FastAPI repo (docs/en + docs_src)
  -> ingest: resolve code includes, clean MkDocs markup, chunk at H2/H3 headings
  -> embed (text-embedding-3-small) -> Postgres + pgvector (Supabase)
  -> /ask: embed question -> top-k vector search -> LLM with numbered excerpts -> answer + citations
```
- 144 docs pages -> 1,056 heading-based chunks (~248k tokens; full embedding cost about $0.005)
- Each chunk stores page, heading path, URL with anchor, and the docs commit SHA
- Re-ingestion is incremental: only chunks whose content hash changed get re-embedded
- The LLM provider is swappable (Anthropic or OpenAI) through one wrapper

## Evaluation

[N] questions: [X] answerable questions taken from real GitHub Discussions where the accepted
answer cited a docs page, 9 about docs pages added in 2026 (to catch answers from model
memory), and 9 the docs don't answer (to test refusals).

Measured: retrieval hit@k and MRR against expected pages; answer quality scored by an LLM
judge (relevance, groundedness, citations), with hand spot-checks
(agreed with the judge on [9/10] checked verdicts).

| Run | Questions | hit@5 | hit@10 | MRR | Answer pass | Groundedness | Correct declines | Cost/query | p50 | p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline: vector-only, heading chunks | 49+17 | 80% | 84% | 0.70 | 63% | 4.71 | 94% | $0.0005 | 2,914 ms | 6,297 ms |
| baseline: vector-only, heading chunks | 40+9 | 75% | 80% | 0.65 | 60% | 4.54 | 100% | $0.0005 | 3,101 ms | 5,740 ms |

## What broke and how I fixed it

1. **Code examples were missing from chunks.** FastAPI docs pull code in with
   `{* ../../docs_src/... *}` includes, so naive chunking dropped every example.
   The chunker now resolves includes, including `ln[...]` line ranges.
2. **Retrieval found the right page, but the model still refused.** In the first 10-question
   run, 3 questions retrieved the expected page yet got "couldn't find this." A strict
   "answer only if fully covered" prompt made the model give up on partial answers.
   I changed the prompt to allow partial answers with stated gaps. This effect avoided the outright rejection of the LLM stating it doesn't know even though it retrieved the right docs. and stating what the doc has in relation to the question. thus avoid All or nothing situations.
3. [one from your baseline misses]

## Cost and latency
[from the baseline: cost per query, p50/p95]. An admin page is planned.

## Tradeoffs
- **pgvector instead of a dedicated vector database:** vectors, metadata, full-text search
  and app data share one Postgres, so there's nothing to keep in sync. At a few thousand
  chunks, a separate vector database adds cost without benefit.
- **Heading-based chunks instead of fixed size:** each chunk maps to one doc section,
  which gives clean citation anchors. Long sections are split at paragraph boundaries,
  never inside code.
- **No LangChain or LlamaIndex:** the pipeline is a few hundred lines I can trace and
  debug step by step.
- **At 100x scale:** [fill in later]

## Run it locally
```bash
git clone <your repo> && cd FastAPI-RAG-app
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add DATABASE_URL and API keys
git clone --filter=blob:none --sparse https://github.com/fastapi/fastapi.git data/fastapi
(cd data/fastapi && git sparse-checkout set docs/en docs_src)
python -m ingest.load           # chunk, embed, load
python -m uvicorn api.main:app --reload
python -m eval.run_eval         # retrieval eval
```