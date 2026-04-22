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
    """Write PointRecord iterable to parquet. Returns count written."""
    rows = []
    for r in records:
        rows.append(r.to_dict())
    if not rows:
        return 0
    table = pa.Table.from_pylist(rows, schema=POINT_SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="snappy")
    return len(rows)


def write_rejects(rejects: list[dict], path: Path) -> int:
    """Append reject log entries as JSONL for audit."""
    if not rejects:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
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
    (?:time\s*(?:complexity)?\s*[:\-=])      # 'Time:' / 'Time complexity:' / 'Time -'
    \s*
    (?P<expr>{_O_EXPR})
    """,
    re.IGNORECASE | re.VERBOSE,
)

_SUFFIX_COMPLEXITY_RE = re.compile(
    rf"""
    (?P<expr>{_O_EXPR})
    \s+time\b                                # "O(...) time"
    """,
    re.IGNORECASE | re.VERBOSE,
)


def find_complexity_in_text(text: str, max_chars: int = 2048) -> str | None:
    """Search for the first complexity annotation in the first `max_chars` of text.

    Tries the prefix form first ("Time complexity: O(...)"), then the suffix
    form ("O(...) time"). Returns the raw O(...) string or None.
    """
    head = text[:max_chars]
    m = _PREFIX_COMPLEXITY_RE.search(head)
    if m:
        return m.group("expr").strip()
    m = _SUFFIX_COMPLEXITY_RE.search(head)
    if m:
        return m.group("expr").strip()
    return None
