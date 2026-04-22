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
# first ~2KB of a file (header area). Captures the inner O(...) expression.
# We accept: "Time complexity:", "Time:", "T(n):", "Time Complexity :", etc.
_COMPLEXITY_RE = re.compile(
    r"""
    (?:time\s*(?:complexity)?\s*[:\-=])   # 'Time:' / 'Time complexity:' / 'Time -'
    \s*
    (?P<expr>
        O\s*\([^)\n]{1,120}\)             # O(...) without nested parens on one line
        |
        O\s*\((?:[^()\n]|\([^()\n]*\)){1,120}\)   # allow one level of nested parens
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def find_complexity_in_text(text: str, max_chars: int = 2048) -> str | None:
    """Search for the first complexity annotation in the first `max_chars` of text.

    Returns the raw O(...) string or None.
    """
    head = text[:max_chars]
    m = _COMPLEXITY_RE.search(head)
    if m:
        return m.group("expr").strip()
    return None
