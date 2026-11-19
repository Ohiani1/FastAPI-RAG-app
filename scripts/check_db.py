import os, psycopg
from dotenv import load_dotenv

load_dotenv()
with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    print(conn.execute("select version()").fetchone()[0])
    print(conn.execute(
        "select extversion from pg_extension where extname = 'vector'"
    ).fetchone())