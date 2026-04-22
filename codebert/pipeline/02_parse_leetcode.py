"""Parse doocs/leetcode solutions by reading Solution*.py files directly.

Each problem directory contains:
    README_EN.md (or README.md)     <-- complexity annotation lives here
    Solution.py, Solution2.py, ...  <-- actual Python solutions

We scan the README for a time-complexity statement, then pair it with every
Solution*.py file in the same directory — one record per solution, each
sharing the directory's complexity label.

Why this beats markdown-code-fence extraction (previous approach):
  - Solution.py files are the canonical source, not copy-pasted into markdown.
  - Many problems have 2-4 Solution files per directory → 2-4x yield.
  - The complexity regex can scan the entire README (not just a fenced block).
  - Handles Chinese-only READMEs via the 时间复杂度 pattern.

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

# Match problem dirs like "0001.Two Sum"
_PROBLEM_DIR_RE = re.compile(r"^(\d{4})\.")
# Match Solution.py, Solution2.py, Solution_DP.py, etc.
_PYTHON_SOLUTION_RE = re.compile(r"^Solution[\w\d_]*\.py$")

# Complexity regex (adapted from tree/ project):
# - prefix form:  "Time complexity: O(...)" / "Time: O(...)" / "T(n) = O(...)"
# - Chinese:      "时间复杂度为 O(...)" / "时间复杂度: O(...)"
# - suffix form:  "O(...) time" / "O(...) time complexity"
# `O((m+n) log(m+n))` requires nested-paren support, listed FIRST in the
# alternation so the nested form wins over the flat one.
_O_EXPR = r"""
    O\s*\(
      (?:[^()\n]|\([^()\n]*\)){1,160}
    \)
    |
    O\s*\([^)\n]{1,160}\)
"""

_PREFIX_COMPLEXITY_RE = re.compile(
    rf"""
    \btime\s*(?:complexity)?
    \s*
    (?:[:\-=]|\bis\b|\bof\b|\:)
    \s*
    (?P<expr>{_O_EXPR})
    """,
    re.IGNORECASE | re.VERBOSE,
)

_CH_COMPLEXITY_RE = re.compile(
    rf"""
    时间复杂度
    [^O\n]{{0,20}}
    (?P<expr>{_O_EXPR})
    """,
    re.VERBOSE,
)

_SUFFIX_COMPLEXITY_RE = re.compile(
    rf"""
    (?P<expr>{_O_EXPR})
    \s+time\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_LATEX_MATH_DELIM_RE = re.compile(r"\$+")


def _strip_markup(text: str) -> str:
    # Inline superscripts first: O(n<sup>2</sup>) -> O(n^2)
    text = re.sub(r"<sup>\s*(\d+)\s*</sup>", r"^\1", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = text.replace("&nbsp;", " ")
    text = text.replace("&times;", "*")
    text = _LATEX_MATH_DELIM_RE.sub("", text)
    return text


def find_complexity_in_text(text: str, max_chars: int = 20000) -> str | None:
    head = _strip_markup(text[:max_chars])
    m = (_PREFIX_COMPLEXITY_RE.search(head)
         or _CH_COMPLEXITY_RE.search(head)
         or _SUFFIX_COMPLEXITY_RE.search(head))
    return m.group("expr").strip() if m else None


def _extract_problem_id(path: Path) -> str | None:
    for part in path.parts:
        m = _PROBLEM_DIR_RE.match(part)
        if m:
            return m.group(1)
    return None


def _read_readme(problem_dir: Path) -> str | None:
    """Prefer English README; fall back to Chinese (many problems lack EN)."""
    for name in ("README_EN.md", "README.md"):
        p = problem_dir / name
        if p.exists():
            try:
                return p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
    return None


def _iter_problem_dirs(repo_dir: Path) -> Iterable[Path]:
    """Yield each problem directory exactly once. EN-preferred, then CN fallback."""
    seen: set[Path] = set()
    roots = ("solution", "lcof", "lcof2", "lcci", "lcp")
    # First pass: EN READMEs
    for root in roots:
        base = repo_dir / root
        if not base.exists():
            continue
        for readme in base.rglob("README_EN.md"):
            if readme.parent not in seen:
                seen.add(readme.parent)
                yield readme.parent
    # Second pass: CN-only (dirs without EN README)
    for root in roots:
        base = repo_dir / root
        if not base.exists():
            continue
        for readme in base.rglob("README.md"):
            if readme.parent in seen:
                continue
            seen.add(readme.parent)
            yield readme.parent


def _python_solutions(problem_dir: Path) -> Iterable[Path]:
    for p in sorted(problem_dir.iterdir()):
        if p.is_file() and _PYTHON_SOLUTION_RE.match(p.name):
            yield p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw_dir", default="data/raw/leetcode")
    ap.add_argument("--out", default="data/interim/parsed/leetcode.jsonl")
    ap.add_argument("--fail_log", default="data/audit/leetcode_parse_failures.jsonl")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap on problem directories scanned (0 = all)")
    args = ap.parse_args()

    root = Path(args.raw_dir)
    if not root.exists():
        print(f"[02] ERROR: {root} not found — run 01_fetch_sources first", file=sys.stderr)
        return 1

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fail = Path(args.fail_log); fail.parent.mkdir(parents=True, exist_ok=True)

    n_problems = n_with_complexity = n_with_python = n_records = 0
    reject_counts: dict[str, int] = {}

    with out.open("w", encoding="utf-8") as fout, fail.open("w", encoding="utf-8") as ffail:
        for problem_dir in _iter_problem_dirs(root):
            n_problems += 1
            if args.limit and n_problems > args.limit:
                break

            readme = _read_readme(problem_dir)
            if readme is None:
                reject_counts["no_readme"] = reject_counts.get("no_readme", 0) + 1
                ffail.write(json.dumps({"dir": str(problem_dir), "reason": "no_readme"}) + "\n")
                continue

            raw_complexity = find_complexity_in_text(readme)
            if raw_complexity is None:
                reject_counts["no_complexity"] = reject_counts.get("no_complexity", 0) + 1
                ffail.write(json.dumps({"dir": str(problem_dir),
                                        "reason": "no_complexity_in_readme"}) + "\n")
                continue
            n_with_complexity += 1

            pid = _extract_problem_id(problem_dir) or problem_dir.name
            py_files = list(_python_solutions(problem_dir))
            if not py_files:
                reject_counts["no_python"] = reject_counts.get("no_python", 0) + 1
                ffail.write(json.dumps({"dir": str(problem_dir),
                                        "reason": "no_python_solution"}) + "\n")
                continue
            n_with_python += 1

            for idx, py_file in enumerate(py_files):
                try:
                    code = py_file.read_text(encoding="utf-8", errors="replace")
                except OSError as e:
                    reject_counts["read_error"] = reject_counts.get("read_error", 0) + 1
                    ffail.write(json.dumps({"file": str(py_file), "reason": "read_error",
                                            "error": str(e)}) + "\n")
                    continue
                if not code.strip():
                    reject_counts["empty"] = reject_counts.get("empty", 0) + 1
                    ffail.write(json.dumps({"file": str(py_file), "reason": "empty"}) + "\n")
                    continue
                fout.write(json.dumps({
                    "source": "leetcode",
                    "problem_id": pid,
                    "solution_idx": idx,
                    "code": code,
                    "raw_complexity": raw_complexity,
                    "path": str(py_file),
                }, ensure_ascii=False) + "\n")
                n_records += 1

    print(f"[02] scanned {n_problems} problem dirs", flush=True)
    print(f"[02]   with complexity in README:   {n_with_complexity}", flush=True)
    print(f"[02]   with Python solution files:  {n_with_python}", flush=True)
    print(f"[02]   total solution records:      {n_records}", flush=True)
    if reject_counts:
        print("[02] reject reasons:", flush=True)
        for reason, n in sorted(reject_counts.items(), key=lambda kv: -kv[1]):
            print(f"[02]   {n:>5} x {reason}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
