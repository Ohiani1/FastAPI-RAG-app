"""
Turn the FastAPI Markdown docs into heading-based chunks.

Steps per page:
  1. Resolve code includes ({* ../../docs_src/x.py hl[3] *} and {!> path !}) into real code blocks.
  2. Clean MkDocs-specific markup (admonitions, tabs, HTML tags, mkdocstrings lines).
  3. Split at H2/H3 headings (never inside code fences) and attach metadata.

Run directly to inspect the output:
    python -m ingest.chunker --stats
    python -m ingest.chunker --page tutorial/query-params.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_DIR = Path("data/fastapi")
DOCS_EN = REPO_DIR / "docs/en"          # include paths are relative to this folder
DOCS_ROOT = DOCS_EN / "docs"
SITE = "https://fastapi.tiangolo.com"

# Pages that are not useful for answering user questions (or are huge changelogs).
SKIP_PAGES = {
    "release-notes.md", "fastapi-people.md", "management.md", "management-tasks.md",
    "newsletter.md", "contributing.md", "translations.md", "external-links.md",
    "agent-banner.md", "translation-banner.md", "_llm-test.md", "help-fastapi.md",
}
SKIP_DIRS = {"img", "css", "js"}

MAX_CHARS = 6000   # ~1500 tokens; longer sections are split at paragraph boundaries
MIN_CHARS = 80     # drop chunks with almost no body text

INCLUDE_STAR = re.compile(r"^\{\*\s*(\S+)(.*?)\*\}\s*$")          # {* path hl[..] ln[..] *}
INCLUDE_BANG = re.compile(r"^\{!>?\s*(\S+?)\s*!\}\s*$")            # {!> path !} / {!path!}
LN_RANGE = re.compile(r"ln\[([^\]]+)\]")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*(?:\{\s*#([\w-]+)\s*\})?\s*$")
ADMONITION = re.compile(r"^/{3,4}\s*(\w+)?\s*(?:\|\s*(.*))?$")    # /// tip | Title, //// tab | X
HTML_TAG = re.compile(
    r"</?(?:div|span|font|img|a|abbr|dfn|br|p|details|summary|em|strong|b|i|u|sup|sub|"
    r"svg|g|path|rect|text|switch|use|stop|defs|foreignObject|iframe|video|source)\b[^>]*>",
    re.IGNORECASE,
)
LANG_BY_EXT = {".py": "python", ".html": "html", ".js": "javascript", ".toml": "toml",
               ".json": "json", ".sh": "bash", ".txt": "text", ".md": "markdown",
               ".yaml": "yaml", ".yml": "yaml", ".ini": "ini", ".css": "css"}


@dataclass
class Chunk:
    id: str             # unique: page path + anchor (+ part number if split)
    page_path: str      # e.g. tutorial/query-params.md (matches eval expected_pages)
    title: str          # page H1
    heading_path: str   # "Query Parameters > Optional parameters"
    anchor: str
    url: str            # https://fastapi.tiangolo.com/tutorial/query-params/#optional-parameters
    content: str        # text that gets embedded (includes the heading path)
    content_hash: str
    version: str        # git commit SHA of the docs


# ---------- step 1: includes ----------

def _select_lines(lines: list[str], spec: str) -> list[str]:
    """ln[1:2,12:16,29] -> those 1-based line numbers (inclusive ranges)."""
    keep = []
    for part in spec.split(","):
        part = part.strip()
        if ":" in part:
            a, b = part.split(":")
            keep.extend(range(int(a), int(b) + 1))
        elif part:
            keep.append(int(part))
    return [lines[i - 1] for i in keep if 0 < i <= len(lines)]


def resolve_includes(text: str, warnings: list[str]) -> str:
    out = []
    for line in text.splitlines():
        m = INCLUDE_STAR.match(line.strip()) or INCLUDE_BANG.match(line.strip())
        if not m:
            out.append(line)
            continue
        rel = m.group(1)
        src = (DOCS_EN / rel).resolve()
        if not src.exists():
            warnings.append(f"missing include: {rel}")
            continue
        code = src.read_text(encoding="utf-8").rstrip("\n").splitlines()
        opts = m.group(2) if m.re is INCLUDE_STAR else ""
        if ln := LN_RANGE.search(opts or ""):
            code = _select_lines(code, ln.group(1))
        lang = LANG_BY_EXT.get(src.suffix, "")
        out.extend([f"```{lang}", *code, "```"])
    return "\n".join(out)


# ---------- step 2: cleaning ----------

def clean(text: str) -> str:
    out, in_code = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            out.append(HTML_TAG.sub("", line) if "<font" in line or "<span" in line else line)
            continue
        if stripped.startswith(":::"):           # mkdocstrings directive (API reference)
            continue
        if m := ADMONITION.match(stripped):
            kind, title = m.group(1), m.group(2)
            if kind == "tab":
                out.append(f"**{title}:**")
            elif kind:
                label = kind.capitalize() + (f": {title}" if title else "")
                out.append(f"**{label}**")
            continue                              # bare /// or //// closer
        out.append(HTML_TAG.sub("", line))
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------- step 3: chunking ----------

def page_url(page_path: str) -> str:
    p = page_path[:-3]                            # drop .md
    if p == "index":
        return f"{SITE}/"
    if p.endswith("/index"):
        p = p[: -len("/index")]
    return f"{SITE}/{p}/"


def _split_long(body: str) -> list[str]:
    """Split an oversized section at blank lines, never inside a code fence."""
    if len(body) <= MAX_CHARS:
        return [body]
    blocks, cur, in_code = [], [], False
    for line in body.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        cur.append(line)
        if not in_code and line.strip() == "":
            blocks.append("\n".join(cur))
            cur = []
    if cur:
        blocks.append("\n".join(cur))
    parts, buf = [], ""
    for b in blocks:
        if buf and len(buf) + len(b) > MAX_CHARS:
            parts.append(buf.strip())
            buf = ""
        buf += b + "\n"
    if buf.strip():
        parts.append(buf.strip())
    return parts


def chunk_page(page_path: str, text: str, version: str) -> list[Chunk]:
    title, sections = page_path, []
    h2 = h3 = None
    cur = {"anchor": "", "heads": [], "lines": []}
    in_code = False

    def flush():
        if cur["lines"]:
            sections.append(dict(cur))

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADING.match(line)
        if m and len(m.group(1)) == 1:
            title = m.group(2)
            cur["anchor"] = m.group(3) or ""
            continue
        if m and len(m.group(1)) in (2, 3):
            flush()
            level, name, anchor = len(m.group(1)), m.group(2), m.group(3) or ""
            if level == 2:
                h2, h3 = name, None
            else:
                h3 = name
            cur = {"anchor": anchor, "heads": [h for h in (h2, h3) if h], "lines": []}
            continue
        cur["lines"].append(line)
    flush()

    chunks, base_url = [], page_url(page_path)
    for s in sections:
        body = "\n".join(s["lines"]).strip()
        heading_path = " > ".join([title, *s["heads"]])
        parts = _split_long(body)
        for i, part in enumerate(parts):
            if len(part) < MIN_CHARS:
                continue
            content = f"{heading_path}\n\n{part}"
            anchor = s["anchor"]
            cid = f"{page_path}#{anchor}" + (f"~{i + 1}" if len(parts) > 1 else "")
            chunks.append(Chunk(
                id=cid, page_path=page_path, title=title, heading_path=heading_path,
                anchor=anchor, url=base_url + (f"#{anchor}" if anchor else ""),
                content=content, content_hash=hashlib.sha256(content.encode()).hexdigest(),
                version=version,
            ))
    # Guard against duplicate ids (e.g. two headings without anchors)
    seen: dict[str, int] = {}
    for c in chunks:
        if c.id in seen:
            seen[c.id] += 1
            c.id = f"{c.id}~dup{seen[c.id]}"
        else:
            seen[c.id] = 0
    return chunks


def docs_version() -> str:
    return subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def iter_pages():
    for path in sorted(DOCS_ROOT.rglob("*.md")):
        rel = path.relative_to(DOCS_ROOT).as_posix()
        if rel in SKIP_PAGES or rel.split("/")[0] in SKIP_DIRS:
            continue
        yield rel, path


def build_chunks(pages: list[str] | None = None) -> tuple[list[Chunk], list[str]]:
    version, chunks, warnings = docs_version(), [], []
    for rel, path in iter_pages():
        if pages and rel not in pages:
            continue
        text = clean(resolve_includes(path.read_text(encoding="utf-8"), warnings))
        chunks.extend(chunk_page(rel, text, version))
    return chunks, warnings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", help="print chunks for one page, e.g. tutorial/query-params.md")
    ap.add_argument("--stats", action="store_true", help="print summary statistics")
    ap.add_argument("--out", help="write all chunks to a JSONL file")
    args = ap.parse_args()

    chunks, warnings = build_chunks([args.page] if args.page else None)
    if args.page:
        for c in chunks:
            print(f"=== {c.id}  ({len(c.content)} chars)\n{c.url}\n\n{c.content}\n")
    if args.out:
        with open(args.out, "w") as f:
            for c in chunks:
                f.write(json.dumps(asdict(c)) + "\n")
    if args.stats or not (args.page or args.out):
        sizes = sorted(len(c.content) for c in chunks)
        pages = {c.page_path for c in chunks}
        print(f"pages: {len(pages)}  chunks: {len(chunks)}")
        print(f"chars per chunk: min {sizes[0]}, median {sizes[len(sizes)//2]}, max {sizes[-1]}")
        print(f"approx tokens total: {sum(sizes)//4:,}")
    for w in warnings:
        print("warning:", w)


if __name__ == "__main__":
    main()
