"""Parse doocs/leetcode README_EN.md files into raw (language, code, complexity) records.

Walks solution/ lcof/ lcof2/ lcci/ directories. For each README_EN.md:
  1. Split into solution blocks by `<!-- solution:start --> ... <!-- solution:end -->`
     (fallback: treat whole file as one block).
  2. For each block, grab the FIRST complexity sentence (`time complexity is O(...)`)
     and EVERY language fence whose info-string maps to one of our 11 + python.
  3. Emit one RawRecord per (problem, solution_idx, language) — i.e. fan out
     across languages so a single block with python+java+cpp+go fences
     produces 4 records.

Output:
  data/interim/parsed/leetcode.jsonl
  data/audit/leetcode_parse_failures.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

from markdown_it import MarkdownIt

# Fence detection
SOLUTION_FENCE = re.compile(r"<!--\s*solution:(start|end)\s*-->", re.IGNORECASE)

# Primary: "time complexity is $O(...)$"
COMPLEXITY_RE = re.compile(
    r"time\s+complexity\s+is\s+\$?O\(([^$)]+)\)\$?",
    re.IGNORECASE,
)
# Fallback: any "$O(...)$" preceded by "time" within ~80 chars, same paragraph-ish.
COMPLEXITY_RE_LOOSE = re.compile(
    r"time[^.\n]{0,80}?\$O\(([^$)]+)\)\$",
    re.IGNORECASE,
)


# Map markdown fence info-string -> our canonical language identifier.
# Skip `python` since `python3` is more common and has the same handling.
FENCE_TO_LANG: dict[str, str] = {
    "python": "python", "python3": "python", "py": "python",
    "java": "java",
    "cpp": "cpp", "c++": "cpp", "cxx": "cpp",
    "c": "c",
    "cs": "csharp", "csharp": "csharp", "c#": "csharp",
    "go": "go", "golang": "go",
    "javascript": "javascript", "js": "javascript",
    "typescript": "typescript", "ts": "typescript",
    "php": "php",
    "ruby": "ruby", "rb": "ruby",
    "rust": "rust", "rs": "rust",
    "swift": "swift",
}


def split_solution_blocks(md: str) -> list[tuple[int, str]]:
    positions = list(SOLUTION_FENCE.finditer(md))
    if not positions:
        return [(0, md)]
    blocks: list[tuple[int, str]] = []
    idx = 0
    start_end = None
    for m in positions:
        tag = m.group(1).lower()
        if tag == "start":
            start_end = m.end()
        elif tag == "end" and start_end is not None:
            blocks.append((idx, md[start_end:m.start()]))
            idx += 1
            start_end = None
    return blocks if blocks else [(0, md)]


def extract_fences_and_complexity(
    block: str, mdit: MarkdownIt
) -> tuple[list[tuple[str, str]], str | None]:
    """Return ([(language, code), ...], complexity-string-or-None) for a block.

    The list contains every fence whose info-string maps to a language we
    track. The block's complexity sentence (parsed once) applies to all of
    them — that's the labeling mechanism.
    """
    tokens = mdit.parse(block)
    fences: list[tuple[str, str]] = []
    seen_languages_in_block: set[str] = set()
    for tok in tokens:
        if tok.type != "fence":
            continue
        info = (tok.info or "").strip().lower().split()[0] if tok.info else ""
        lang = FENCE_TO_LANG.get(info)
        if lang is None:
            continue
        # First-fence-per-language wins inside one block (the README convention
        # is "Solution 1 in Python ... Solution 1 in Java ..." — both share the
        # block's complexity, so we want one (lang, code) pair per language).
        if lang in seen_languages_in_block:
            continue
        seen_languages_in_block.add(lang)
        fences.append((lang, tok.content))

    m = COMPLEXITY_RE.search(block) or COMPLEXITY_RE_LOOSE.search(block)
    complexity = m.group(1).strip() if m else None
    return fences, complexity


def derive_problem_id(path: Path) -> str | None:
    # solution/0000-0099/0001.Two Sum/README_EN.md -> 0001
    for p in path.parts:
        m = re.match(r"^(\d{4})\.\s*(.*)$", p)
        if m:
            return m.group(1)
    return None


def iter_readmes(root: Path) -> Iterable[Path]:
    for pat in ("solution/**/README_EN.md", "lcof/**/README_EN.md",
                "lcof2/**/README_EN.md", "lcci/**/README_EN.md"):
        yield from root.glob(pat)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw_dir", default="data/raw/leetcode")
    ap.add_argument("--out", default="data/interim/parsed/leetcode.jsonl")
    ap.add_argument("--fail_log", default="data/audit/leetcode_parse_failures.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.raw_dir)
    if not root.exists():
        print(f"[02] ERROR: {root} not found - run 01_fetch_sources first", file=sys.stderr)
        return 1

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fail = Path(args.fail_log); fail.parent.mkdir(parents=True, exist_ok=True)

    mdit = MarkdownIt("commonmark")

    n_files = n_blocks = n_records = n_failed = 0
    per_lang: dict[str, int] = {}
    with out.open("w", encoding="utf-8") as fout, fail.open("w", encoding="utf-8") as ffail:
        for path in iter_readmes(root):
            n_files += 1
            if args.limit and n_files > args.limit:
                break
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                n_failed += 1
                ffail.write(json.dumps({"path": str(path), "err": str(e)}) + "\n")
                continue
            problem_id = derive_problem_id(path)
            for idx, block in split_solution_blocks(text):
                n_blocks += 1
                fences, comp = extract_fences_and_complexity(block, mdit)
                if not fences or not comp:
                    ffail.write(json.dumps({
                        "path": str(path), "solution_idx": idx,
                        "n_fences": len(fences), "has_complexity": bool(comp),
                    }) + "\n")
                    n_failed += 1
                    continue
                for lang, code in fences:
                    fout.write(json.dumps({
                        "source": "leetcode",
                        "language": lang,
                        "problem_id": problem_id,
                        "solution_idx": idx,
                        "code": code,
                        "raw_complexity": comp,
                        "path": str(path),
                    }, ensure_ascii=False) + "\n")
                    n_records += 1
                    per_lang[lang] = per_lang.get(lang, 0) + 1

    rate = 100 * n_records / n_blocks if n_blocks else 0
    print(f"[02] files={n_files} blocks={n_blocks} records={n_records} "
          f"failed={n_failed} fan_out_rate={rate:.1f}%", flush=True)
    for lang in sorted(per_lang):
        print(f"[02]   {lang:<12s} {per_lang[lang]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
