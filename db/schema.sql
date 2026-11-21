-- Safe to run repeatedly. ingest/load.py applies it automatically.
create extension if not exists vector;

create table if not exists chunks (
    id            text primary key,          -- page_path#anchor (+ ~n if a section was split)
    page_path     text not null,             -- tutorial/query-params.md
    title         text not null,             -- page H1
    heading_path  text not null,             -- "Query Parameters > Optional parameters"
    anchor        text not null default '',
    url           text not null,             -- full docs URL with #anchor
    content       text not null,             -- text that was embedded
    content_hash  text not null,             -- sha256 of content, for incremental re-embedding
    version       text not null,             -- FastAPI repo commit SHA
    embedding     vector(1536),              -- text-embedding-3-small
    -- Full-text search column, ready for hybrid (BM25-style) search later
    fts           tsvector generated always as
                  (to_tsvector('english', heading_path || ' ' || content)) stored,
    updated_at    timestamptz not null default now()
);

create index if not exists chunks_embedding_idx on chunks using hnsw (embedding vector_cosine_ops);
create index if not exists chunks_fts_idx on chunks using gin (fts);
create index if not exists chunks_page_idx on chunks (page_path);
