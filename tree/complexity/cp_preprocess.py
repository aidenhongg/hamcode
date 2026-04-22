"""Strip competitive-programming boilerplate from a Python snippet so the
feature extractor sees only the algorithmic code.

The noise patterns we target (observed in CodeComplex / Codeforces data):

    1. Bulk imports:          `import sys, os, io`
    2. Fast I/O helper defs:  `def ri(): return int(sys.stdin.readline())`
    3. Library aliases:       `enum = enumerate`, `D = defaultdict`, `C = Counter`
    4. Author / pylint comments at the top of the file
    5. T-harness wrappers:    `for _ in range(int(input())):` where the loop's
                              only purpose is to iterate over test cases

Each of those inflates loop/method/statement counts and makes the feature
distribution diverge from LeetCode-style code. We strip them pre-feature-
extraction, not pre-training — we keep the original code in `PointRecord.code`
for auditability.
"""

from __future__ import annotations

import re

from tree_sitter_languages import get_parser


_parser = None


def _get_parser():
    global _parser
    if _parser is None:
        _parser = get_parser("python")
    return _parser


# --- Line-level heuristics (cheap, handle ~90% of the noise) -----------------

_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+\S+")
_COMMENT_RE = re.compile(r"^\s*#")
_BLANK_RE = re.compile(r"^\s*$")

# One-liner I/O helper def. Matches both `def x(): return sys.stdin.readline()`
# and `def x(y): sys.stdout.write(str(y))` styles.
_IO_HELPER_DEF_RE = re.compile(
    r"""
    ^\s*def\s+\w+\s*\([^)]*\)\s*:   # `def foo(...):`
    \s*(?:return\s+)?                # optional `return`
    [^\n]*                           # body on the same line
    (?:
        sys\.stdin | sys\.stdout |
        stdin\. | stdout\. |
        \binput\s*\( | \bprint\s*\(
    )
    """,
    re.VERBOSE,
)

# `NAME = OtherName` (simple aliases like `enum = enumerate` or `D = defaultdict`).
# Right side must be a single identifier or dotted name, no call/literal.
_ALIAS_RE = re.compile(r"^\s*(\w+)\s*=\s*([A-Za-z_][\w.]*)\s*(?:#.*)?$")


def _strip_lines(code: str) -> str:
    out: list[str] = []
    for line in code.split("\n"):
        if _IMPORT_RE.match(line):
            continue
        if _COMMENT_RE.match(line):
            continue
        if _IO_HELPER_DEF_RE.match(line):
            continue
        m = _ALIAS_RE.match(line)
        if m:
            lhs, rhs = m.group(1), m.group(2)
            # If RHS is lowercase and doesn't look like a user variable,
            # treat it as a library alias. Accepting `D = defaultdict`,
            # `enum = enumerate`, `Q = deque`, etc.
            if rhs not in lhs and "." not in lhs:
                continue
        out.append(line)
    result = "\n".join(out)
    return result if result.strip() else code


# --- AST-level stripping (handles T-harness + multi-line helpers) ------------

def _strip_t_harness(code: str) -> str:
    """Unwrap `for _ in range(int(input())):` wrappers whose body is the whole
    algorithm. We detect these at the AST level because the body's indentation
    varies.

    Heuristic: a for-loop at module level, with target == '_' (or a single-use
    name), and iterable == `range(int(input()))` or `range(t)` where `t` was
    just assigned from `int(input())`. The loop body becomes top-level.
    """
    try:
        tree = _get_parser().parse(code.encode("utf-8", errors="replace"))
    except Exception:
        return code

    root = tree.root_node
    src_bytes = code.encode("utf-8", errors="replace")

    def node_text(node) -> str:
        return src_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    # Find module-level statements
    rewrites: list[tuple[int, int, str]] = []  # (start, end, replacement)

    # Track recent `t = int(input())` assignments to find their referenced for-loops
    recent_t_names: set[str] = set()

    for stmt in root.children:
        t = stmt.type
        # Track `name = int(input())` assignments
        if t == "expression_statement" and len(stmt.children) >= 1:
            inner = stmt.children[0]
            if inner.type == "assignment":
                lhs = inner.child_by_field_name("left")
                rhs = inner.child_by_field_name("right")
                if lhs is not None and rhs is not None and lhs.type == "identifier":
                    if "int(input())" in node_text(rhs).replace(" ", ""):
                        recent_t_names.add(node_text(lhs))

        if t != "for_statement":
            continue

        # Need target = `_` or in recent_t_names, and iterable = range(int(input())) or range(t)
        target = stmt.child_by_field_name("left")
        iterable = stmt.child_by_field_name("right")
        body = stmt.child_by_field_name("body")
        if target is None or iterable is None or body is None:
            continue

        tgt_text = node_text(target).strip()
        iter_text = node_text(iterable).replace(" ", "")
        if tgt_text != "_":
            continue

        is_t_harness = (
            "range(int(input()))" in iter_text
            or any(f"range({name})" == iter_text for name in recent_t_names)
        )
        if not is_t_harness:
            continue

        # Replace the whole for_statement with its body (un-indented).
        body_text = node_text(body)
        # Body is usually a `block` node, indented one level. Unindent:
        unindented = _unindent_block(body_text)
        rewrites.append((stmt.start_byte, stmt.end_byte, unindented))

    if not rewrites:
        return code

    # Apply rewrites back-to-front so byte offsets stay valid
    out = src_bytes
    for start, end, repl in sorted(rewrites, key=lambda r: -r[0]):
        out = out[:start] + repl.encode("utf-8", errors="replace") + out[end:]
    return out.decode("utf-8", errors="replace")


def _unindent_block(block_text: str) -> str:
    lines = block_text.split("\n")
    # Find the minimum leading whitespace among non-blank lines
    indents = [
        len(line) - len(line.lstrip())
        for line in lines
        if line.strip()
    ]
    if not indents:
        return block_text
    base = min(indents)
    return "\n".join(
        (line[base:] if len(line) >= base else line)
        for line in lines
    )


# --- Template-heavy detector (row-level filter) ------------------------------

# Markers that indicate a competitive-programmer's pasted-in template library.
# If any of these appear, the record's feature counts will be dominated by
# library code rather than the algorithm — safer to drop the record entirely.
_TEMPLATE_MARKERS: tuple[str, ...] = (
    "sys.setrecursionlimit",
    "threading.stack_size",
    "class SortedList",
    "class UnionFind",
    "class DSU",
    "class SegmentTree",
    "class Fenwick",
    "class BIT",
    "class LazySegTree",
    "from io import BytesIO",
    "BytesIO, IOBase",
    "# pylint:",
    "# @ Author",
)


def is_template_heavy(code: str) -> bool:
    """Return True if the code contains competitive-programming template
    boilerplate significant enough that its features won't reflect the
    algorithm. Consumers use this to filter at ingest/feature-extract time.
    """
    if not code:
        return False
    # Fast path: explicit markers
    for m in _TEMPLATE_MARKERS:
        if m in code:
            return True
    # Size + def count proxy: many module-level defs suggest a template library
    if len(code) > 4000 and code.count("\ndef ") >= 6:
        return True
    return False


# --- Public entry point ------------------------------------------------------

def preprocess(code: str) -> str:
    """Strip CP boilerplate. Order matters:
      1. Unwrap T-harness (for _ in range(int(input()))) so its body is top-level
      2. Strip line-level boilerplate (imports, aliases, one-liner I/O helpers)

    Does NOT attempt to strip multi-line helper functions — that's too error-
    prone. For records where templates dominate, use `is_template_heavy()` at
    ingest time to drop the row entirely instead.
    """
    if not code:
        return code
    code = _strip_t_harness(code)
    code = _strip_lines(code)
    return code
