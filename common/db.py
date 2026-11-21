"""Postgres connections with pgvector support."""

from __future__ import annotations

from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from common.config import DATABASE_URL

SCHEMA = Path(__file__).resolve().parent.parent / "db/schema.sql"
_pool: ConnectionPool | None = None


def connect() -> psycopg.Connection:
    """A single connection (used by scripts)."""
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn


def apply_schema() -> None:
    # Create the extension before register_vector, which needs the vector type to exist.
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute(SCHEMA.read_text())


def get_pool() -> ConnectionPool:
    """A small connection pool (used by the API, so each request doesn't reconnect)."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(DATABASE_URL, min_size=1, max_size=5,
                               configure=register_vector, open=True)
    return _pool
