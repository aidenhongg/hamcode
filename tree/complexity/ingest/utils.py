"""Shared helpers for ingestion scripts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from ..schemas import POINT_SCHEMA, PointRecord


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()


def looks_like_python(code: str) -> bool:
    """Heuristic: does this source code look like Python (not Java/C++)?

    Cheap checks because we run this over thousands of samples.
    """
    if not code or len(code) < 5:
        return False
    # Java hard signals
    java_signals = ("public class", "public static void main", "System.out",
                    "import java.", "System.in", "private static ")
    if any(sig in code for sig in java_signals):
        return False
    # C++ hard signals
    cpp_signals = ("#include", "using namespace", "std::", "cout <<", "cin >>")
    if any(sig in code for sig in cpp_signals):
        return False
    # Python positive signals
    py_signals_count = sum(1 for sig in ("def ", "import ", "print(", "if __name__",
                                          "class ", "self.", "elif ", "return ")
                           if sig in code)
    return py_signals_count >= 1


def write_points(records: Iterable[PointRecord], path: Path) -> int:
    """Write PointRecord iterable to parquet. Always writes a file (possibly
    empty) so downstream orchestrators can distinguish "source ran but
    produced nothing" from "source never ran".
    """
    rows = [r.to_dict() for r in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Write an empty parquet with the correct schema
        empty = pa.Table.from_pylist([], schema=POINT_SCHEMA)
        pq.write_table(empty, path, compression="snappy")
        return 0
    table = pa.Table.from_pylist(rows, schema=POINT_SCHEMA)
    pq.write_table(table, path, compression="snappy")
    return len(rows)


def write_rejects(rejects: list[dict], path: Path) -> int:
    """Write reject log entries as JSONL for audit. OVERWRITES previous runs'
    rejects so diagnostic tools see only the current run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rejects:
            f.write(json.dumps(r) + "\n")
    return len(rejects)


# Patterns for mining complexity comments from source code. Matched against the
# first ~2KB of a file (header area). Two families:
#   1. Prefix form: "Time complexity: O(...)" / "Time: O(...)" / "T(n) = O(...)"
#   2. Suffix form: "O(...) time" / "O(...) time complexity"
# Nested-paren variant is listed FIRST in the alternation so `O((m+n) log(m+n))`
# wins over the simpler one-paren form.
_O_EXPR = r"""
    O\s*\(                                   # O(
      (?:[^()\n]|\([^()\n]*\)){1,160}        # body allowing one level of nested parens
    \)
    |
    O\s*\([^)\n]{1,160}\)                    # flat O(...)
"""

_PREFIX_COMPLEXITY_RE = re.compile(
    rf"""
    \btime\s*(?:complexity)?                 # 'time' / 'time complexity'
    \s*
    (?:[:\-=]|\bis\b|\bof\b|\:)              # separator: ':', '-', '=', 'is', 'of'
    \s*
    (?P<expr>{_O_EXPR})
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Chinese time-complexity prefix. doocs/leetcode's Chinese READMEs use phrases
# like "时间复杂度为 O(n)" or "时间复杂度：O(n)". 237 problems in doocs only have
# Chinese READMEs, so this pattern unlocks that data.
_CH_COMPLEXITY_RE = re.compile(
    rf"""
    时间复杂度                               # Chinese: 'time complexity'
    [^O\n]{{0,20}}                           # up to 20 chars: '为', ':', '：', spaces
    (?P<expr>{_O_EXPR})
    """,
    re.VERBOSE,
)

_SUFFIX_COMPLEXITY_RE = re.compile(
    rf"""
    (?P<expr>{_O_EXPR})
    \s+time\b                                # "O(...) time"
    """,
    re.IGNORECASE | re.VERBOSE,
)


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_LATEX_MATH_DELIM_RE = re.compile(r"\$+")


def strip_markup(text: str) -> str:
    """Strip HTML tags and LaTeX math delimiters so the complexity regex can
    match annotations like `$O(n^2)$` or `<code>O(n)</code>` uniformly.

    Also rewrites common HTML-encoded complexity forms:
        O(n<sup>2</sup>) -> O(n^2)
        O(n&nbsp;log&nbsp;n) -> O(n log n)
    """
    # Inline superscripts BEFORE stripping other tags
    text = re.sub(r"<sup>\s*(\d+)\s*</sup>", r"^\1", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = text.replace("&nbsp;", " ")
    text = text.replace("&times;", "*")
    text = _LATEX_MATH_DELIM_RE.sub("", text)
    return text


def find_complexity_in_text(text: str, max_chars: int = 4096) -> str | None:
    """Search for the first complexity annotation in the first `max_chars` of text.

    Strips HTML and LaTeX math delimiters first, then tries:
      1. English prefix form ("Time complexity: O(...)")
      2. Chinese prefix form ("时间复杂度: O(...)")
      3. English suffix form ("O(...) time")
    Returns the raw O(...) string or None.
    """
    head = strip_markup(text[:max_chars])
    m = _PREFIX_COMPLEXITY_RE.search(head)
    if m:
        return m.group("expr").strip()
    m = _CH_COMPLEXITY_RE.search(head)
    if m:
        return m.group("expr").strip()
    m = _SUFFIX_COMPLEXITY_RE.search(head)
    if m:
        return m.group("expr").strip()
    return None
