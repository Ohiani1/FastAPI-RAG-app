"""
Chunk the docs, embed new or changed chunks, and sync them into Postgres.

Incremental by design: a chunk is only re-embedded when its content hash changes,
and chunks that no longer exist in the docs are deleted. After a `git pull` in
data/fastapi, re-running this only pays for what changed.

    python -m ingest.load              # full sync
    python -m ingest.load --dry-run    # show what would change, no API calls
"""

import argparse
import time

from common.db import apply_schema, connect
from common.embeddings import embed_texts
from ingest.chunker import build_chunks

EMBED_PRICE_PER_M = 0.02  # USD per 1M tokens for text-embedding-3-small


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    chunks, warnings = build_chunks()
    for w in warnings:
        print("warning:", w)
    version = chunks[0].version
    print(f"chunked {len(chunks)} chunks from {len({c.page_path for c in chunks})} pages "
          f"(docs version {version[:10]})")

    apply_schema()
    with connect() as conn:
        existing = dict(conn.execute("select id, content_hash from chunks").fetchall())
        current_ids = {c.id for c in chunks}
        to_embed = [c for c in chunks if existing.get(c.id) != c.content_hash]
        to_delete = [cid for cid in existing if cid not in current_ids]
        print(f"new or changed: {len(to_embed)}  unchanged: {len(chunks) - len(to_embed)}  "
              f"removed: {len(to_delete)}")
        if args.dry_run:
            return

        tokens = 0
        if to_embed:
            vectors, tokens = embed_texts([c.content for c in to_embed])
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    insert into chunks (id, page_path, title, heading_path, anchor, url,
                                        content, content_hash, version, embedding)
                    values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    on conflict (id) do update set
                        page_path = excluded.page_path, title = excluded.title,
                        heading_path = excluded.heading_path, anchor = excluded.anchor,
                        url = excluded.url, content = excluded.content,
                        content_hash = excluded.content_hash, version = excluded.version,
                        embedding = excluded.embedding, updated_at = now()
                    """,
                    [(c.id, c.page_path, c.title, c.heading_path, c.anchor, c.url,
                      c.content, c.content_hash, c.version, v) for c, v in zip(to_embed, vectors)],
                )
        if to_delete:
            conn.execute("delete from chunks where id = any(%s)", (to_delete,))
        # Unchanged chunks still belong to the new docs version.
        conn.execute("update chunks set version = %s where version <> %s", (version, version))
        conn.commit()

    cost = tokens / 1_000_000 * EMBED_PRICE_PER_M
    print(f"embedded {tokens:,} tokens (~${cost:.4f}) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
