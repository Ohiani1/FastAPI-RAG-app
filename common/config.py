"""Settings shared by ingestion, the API and the eval scripts. All come from .env."""

import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL", "")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536  # must match vector(1536) in db/schema.sql

# Which LLM answers questions: "anthropic" or "openai"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")
LLM_MODEL = os.environ.get("LLM_MODEL", "")  # empty = provider default (see api/llm.py)

TOP_K = int(os.environ.get("TOP_K", "6"))
# If the best chunk's cosine similarity is below this, answer "not found" without calling
# the LLM. 0 disables it; tune it on your eval set later.
MIN_SIMILARITY = float(os.environ.get("MIN_SIMILARITY", "0"))
