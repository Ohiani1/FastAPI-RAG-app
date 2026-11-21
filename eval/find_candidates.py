"""
Find candidate eval questions from FastAPI GitHub Discussions (Questions category).

Keeps only answered discussions where the accepted answer (or another comment)
links to a FastAPI docs page, maps each link to a file in docs/en/docs/, and
writes the shortlist to eval/candidates.jsonl for you to review by hand.

Usage:
    # needs a GitHub token (no scopes needed for public repos) in .env as GITHUB_TOKEN
    python eval/find_candidates.py --max 1500
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

OWNER, REPO = "fastapi", "fastapi"
DOCS_ROOT = Path("data/fastapi/docs/en/docs")
API = "https://api.github.com/graphql"

# Links like https://fastapi.tiangolo.com/tutorial/query-params/#query-parameter-type-conversion
SITE_LINK = re.compile(r"https?://fastapi\.tiangolo\.com/([^\s)\]\"'<>#?]*)")
# Links to the source files, e.g. github.com/fastapi/fastapi/blob/master/docs/en/docs/tutorial/body.md
REPO_LINK = re.compile(r"docs/en/docs/([\w\-/]+\.md)")

BUG_WORDS = ("bug", "regression", "crash", "broken after", "breaking change")

CATEGORIES_QUERY = """
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    discussionCategories(first: 25) { nodes { id name slug } }
  }
}
"""

DISCUSSIONS_QUERY = """
query($owner: String!, $name: String!, $cat: ID!, $after: String) {
  repository(owner: $owner, name: $name) {
    discussions(first: 50, after: $after, categoryId: $cat, answered: true,
                orderBy: {field: CREATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title url body createdAt
        answer { body url }
        comments(first: 15) { nodes { body replies(first: 5) { nodes { body } } } }
      }
    }
  }
}
"""


def gql(token: str, query: str, variables: dict) -> dict:
    req = urllib.request.Request(
        API,
        data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.load(resp)
    if "errors" in data:
        sys.exit(f"GitHub API error: {data['errors']}")
    return data["data"]


def to_doc_file(url_path: str) -> str | None:
    """Map a docs-site path like 'tutorial/query-params/' to 'tutorial/query-params.md'."""
    parts = [p for p in url_path.strip("/").split("/") if p]
    if not parts:
        return None  # homepage link, not useful as an expected page
    attempts = [parts]
    if len(parts) > 1:
        attempts.append(parts[1:])  # drop a language prefix like 'es/' or 'zh-hant/'
    for p in attempts:
        stem = "/".join(p)
        for candidate in (f"{stem}.md", f"{stem}/index.md"):
            if (DOCS_ROOT / candidate).exists():
                return candidate
    return None


def doc_pages(text: str) -> set[str]:
    pages = set()
    for m in SITE_LINK.finditer(text or ""):
        if f := to_doc_file(m.group(1)):
            pages.add(f)
    for m in REPO_LINK.finditer(text or ""):
        if (DOCS_ROOT / m.group(1)).exists():
            pages.add(m.group(1))
    return pages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=1500, help="max discussions to scan")
    ap.add_argument("--out", default="eval/candidates.jsonl")
    args = ap.parse_args()

    load_dotenv()
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("Set GITHUB_TOKEN in .env (a classic token with no scopes is enough).")
    if not DOCS_ROOT.exists():
        sys.exit(f"{DOCS_ROOT} not found. Run this from the repo root after cloning the docs.")

    cats = gql(token, CATEGORIES_QUERY, {"owner": OWNER, "name": REPO})
    cats = cats["repository"]["discussionCategories"]["nodes"]
    questions = next((c for c in cats if c["slug"] == "questions"), None)
    if not questions:
        sys.exit(f"No 'questions' category found. Categories: {[c['slug'] for c in cats]}")

    scanned, after, results = 0, None, []
    while scanned < args.max:
        data = gql(token, DISCUSSIONS_QUERY,
                   {"owner": OWNER, "name": REPO, "cat": questions["id"], "after": after})
        conn = data["repository"]["discussions"]
        for d in conn["nodes"]:
            scanned += 1
            if any(w in d["title"].lower() for w in BUG_WORDS):
                continue
            answer_pages = doc_pages((d.get("answer") or {}).get("body", ""))
            comment_text = []
            for c in d["comments"]["nodes"]:
                comment_text.append(c["body"])
                comment_text += [r["body"] for r in c["replies"]["nodes"]]
            comment_pages = doc_pages("\n".join(comment_text)) - answer_pages
            if not (answer_pages or comment_pages):
                continue
            results.append({
                "title": d["title"],
                "url": d["url"],
                "created": d["createdAt"][:10],
                "answer_pages": sorted(answer_pages),
                "comment_pages": sorted(comment_pages),
                "strength": "answer" if answer_pages else "comment",
                "body_excerpt": (d["body"] or "")[:600],
            })
        print(f"scanned {scanned}, candidates so far {len(results)}", file=sys.stderr)
        if not conn["pageInfo"]["hasNextPage"]:
            break
        after = conn["pageInfo"]["endCursor"]

    # Strongest first: the accepted answer itself links a docs page.
    results.sort(key=lambda r: (r["strength"] != "answer", r["created"]), reverse=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    sections = Counter(
        (r["answer_pages"] or r["comment_pages"])[0].split("/")[0] for r in results
    )
    print(f"\nScanned {scanned} answered discussions, kept {len(results)} -> {args.out}")
    print(f"  linked in accepted answer: {sum(r['strength'] == 'answer' for r in results)}")
    print("  by docs section:", dict(sections.most_common()))


if __name__ == "__main__":
    main()