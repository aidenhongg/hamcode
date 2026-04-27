"""Strip LeetCode wrappers and time/space complexity comments before training.

This stage runs between 05_normalize_labels.py and 06_dedupe_filter.py. It:

  1. Removes top-level `class Solution(...)`-style wrappers in Python and
     Ruby — methods get lifted to module scope, dedented, and `self` is
     dropped from Python parameter lists. Other languages are passed
     through unchanged (their wrappers can't be lifted without breaking
     syntax).

  2. Removes time/space complexity comments across all 12 languages. The
     predicate matches: `Time:` / `Space:` headers (kamyu-style), `O(...)`
     anywhere in a comment, or the literal word `complexity`. Tree-sitter
     comment nodes are used so `O(n)` inside a string literal is left alone.
     Whole-line comments are deleted with their newline; end-of-line
     comments are stripped along with leading whitespace before them.

After both transforms, the resulting code is re-validated with
tree-sitter. Records whose stripped output no longer parses are dropped
into `data/audit/strip_failures.jsonl` with the original row + the
broken output for inspection.

Output:
    data/interim/stripped/combined.jsonl

Audit:
    data/audit/strip_log.jsonl
    data/audit/strip_failures.jsonl

Console summary: per-language wrap/comment counts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from common.parsers import parse, syntax_ok, walk as _walk_nodes


# ---------------------------------------------------------------------------
# Wrapper stripping (Python + Ruby only — see Q1 in plan)
# ---------------------------------------------------------------------------

_SOLUTION_NAME = re.compile(r"^Solution\d*$")
# Drop `self` (with optional trailing comma) from a def header. \s in default
# mode matches newlines, so multi-line `def foo(self,\n  x):` is also handled.
_SELF_PARAM_RE = re.compile(r"(def\s+\w+\s*\()self\s*(?:,\s*)?")


def _strip_indent(line: str, indent_cols: int) -> str:
    """Remove up to `indent_cols` columns of leading whitespace from one line."""
    stripped = 0
    j = 0
    while j < len(line) and stripped < indent_cols:
        ch = line[j]
        if ch == " ":
            stripped += 1
            j += 1
        elif ch == "\t":
            stripped += 8  # python convention; tabs are rare in LeetCode
            j += 1
        else:
            break
    return line[j:]


def _dedent_node(node, code_b: bytes, indent_cols: int) -> str:
    """Slice the node's source bytes and dedent every line *after* the first.

    Tree-sitter's start_byte points at the first non-whitespace token (e.g.
    `def`), so the first line of the slice already has no leading indent;
    only follow-on lines need stripping.
    """
    src = code_b[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
    lines = src.split("\n")
    return "\n".join(
        line if i == 0 else _strip_indent(line, indent_cols)
        for i, line in enumerate(lines)
    )


def _drop_self_from_def(src: str) -> str:
    """`def foo(self, x): ...` -> `def foo(x): ...`. No-op when `self` isn't
    the first positional parameter."""
    return _SELF_PARAM_RE.sub(r"\1", src, count=1)


def strip_solution_wrapper_python(code: str) -> tuple[str, bool]:
    """Lift methods of every top-level `class Solution[\\d]*(...)` to module
    scope. Returns (new_code, was_modified).

    Class-level statements that aren't methods (nested classes, attrs,
    docstrings) are dropped — LeetCode wrappers rarely have any, and
    keeping them would require recursive renaming. The post-strip syntax
    check catches anything we got wrong.
    """
    try:
        tree = parse("python", code)
    except Exception:
        return code, False
    code_b = code.encode("utf-8")
    root = tree.root_node

    targets = []
    for child in root.children:
        if child.type != "class_definition":
            continue
        name_node = child.child_by_field_name("name")
        if name_node is None:
            continue
        name = code_b[name_node.start_byte:name_node.end_byte].decode("utf-8")
        if _SOLUTION_NAME.match(name):
            targets.append(child)

    if not targets:
        return code, False

    out_chunks: list[str] = []
    last_end = 0
    for cls_node in targets:
        out_chunks.append(
            code_b[last_end:cls_node.start_byte].decode("utf-8", errors="replace")
        )
        body = cls_node.child_by_field_name("body")
        method_pieces: list[str] = []
        if body is not None:
            for stmt in body.children:
                if stmt.type in ("function_definition", "decorated_definition"):
                    indent = stmt.start_point[1]
                    raw = _dedent_node(stmt, code_b, indent)
                    method_pieces.append(_drop_self_from_def(raw))
        out_chunks.append("\n\n".join(method_pieces))
        last_end = cls_node.end_byte
    out_chunks.append(code_b[last_end:].decode("utf-8", errors="replace"))

    new_code = "".join(out_chunks).lstrip("\n")
    return new_code, True


def strip_solution_wrapper_ruby(code: str) -> tuple[str, bool]:
    """Same as Python variant but for Ruby. No `self` to drop in signatures."""
    try:
        tree = parse("ruby", code)
    except Exception:
        return code, False
    code_b = code.encode("utf-8")
    root = tree.root_node

    targets = []
    for child in root.children:
        if child.type != "class":
            continue
        name_node = child.child_by_field_name("name")
        if name_node is None:
            continue
        name = code_b[name_node.start_byte:name_node.end_byte].decode("utf-8")
        if _SOLUTION_NAME.match(name):
            targets.append(child)

    if not targets:
        return code, False

    out_chunks: list[str] = []
    last_end = 0
    for cls_node in targets:
        out_chunks.append(
            code_b[last_end:cls_node.start_byte].decode("utf-8", errors="replace")
        )
        # Tree-sitter Ruby wraps class methods in a `body_statement` child of
        # the class node (not directly under it like Python's class_definition).
        method_pieces: list[str] = []
        for child in cls_node.children:
            stmts = (
                child.children if child.type == "body_statement" else [child]
            )
            for n in stmts:
                if n.type == "method":
                    indent = n.start_point[1]
                    method_pieces.append(_dedent_node(n, code_b, indent))
        out_chunks.append("\n\n".join(method_pieces))
        last_end = cls_node.end_byte
    out_chunks.append(code_b[last_end:].decode("utf-8", errors="replace"))

    return "".join(out_chunks).lstrip("\n"), True


def strip_solution_wrapper(code: str, language: str) -> tuple[str, bool]:
    """Dispatch to language-specific wrapper stripping. Pass-through for
    languages where the wrapper can't be safely removed."""
    if language == "python":
        return strip_solution_wrapper_python(code)
    if language == "ruby":
        return strip_solution_wrapper_ruby(code)
    return code, False


# ---------------------------------------------------------------------------
# Complexity-comment stripping (all 12 languages)
# ---------------------------------------------------------------------------

# Predicate (Q2 option c): `O(...)` notation, `Time:` / `Space:` header,
# or the literal word `complexity` (case-insensitive).
_COMPLEXITY_PRED = re.compile(
    r"(?i)(?:O\([^)]+\)|\bTime\s*:|\bSpace\s*:|\bcomplexity\b)"
)


def strip_complexity_comments(code: str, language: str) -> tuple[str, int]:
    """Delete comment nodes whose text matches the complexity predicate.

    Whole-line comments are removed along with their trailing newline;
    end-of-line and inline comments are stripped along with the leading
    whitespace immediately preceding them so the surviving code line
    doesn't sprout trailing whitespace.

    Tree-sitter ensures `O(n)` inside a string literal is left alone — only
    nodes whose grammar type contains `comment` qualify.
    """
    try:
        tree = parse(language, code)
    except Exception:
        return code, 0
    code_b = code.encode("utf-8")
    spans: list[tuple[int, int]] = []
    for n in _walk_nodes(tree.root_node):
        if "comment" not in n.type:
            continue
        text = code_b[n.start_byte:n.end_byte].decode("utf-8", errors="replace")
        if _COMPLEXITY_PRED.search(text):
            spans.append((n.start_byte, n.end_byte))

    if not spans:
        return code, 0

    n_stripped = len(spans)
    out = bytearray(code_b)
    # Reverse order so earlier offsets stay valid as we splice
    for start, end in sorted(spans, reverse=True):
        line_end = out.find(b"\n", end)
        if line_end == -1:
            line_end = len(out)
        line_start = out.rfind(b"\n", 0, start) + 1
        head = bytes(out[line_start:start])
        tail = bytes(out[end:line_end])
        if head.strip() == b"" and tail.strip() == b"":
            # Whole line — drop including the trailing newline
            new_end = (
                line_end + 1
                if line_end < len(out) and out[line_end:line_end + 1] == b"\n"
                else line_end
            )
            del out[line_start:new_end]
        else:
            # End-of-line / inline comment. Eat leading whitespace before it.
            ws_start = start
            while ws_start > line_start and out[ws_start - 1:ws_start] in (b" ", b"\t"):
                ws_start -= 1
            del out[ws_start:end]

    return bytes(out).decode("utf-8", errors="replace"), n_stripped


# ---------------------------------------------------------------------------
# Top-level transform
# ---------------------------------------------------------------------------

def strip_record(code: str, language: str) -> tuple[str, dict]:
    """Apply both transforms; return (new_code, audit_dict)."""
    new_code, was_unwrapped = strip_solution_wrapper(code, language)
    new_code, n_stripped = strip_complexity_comments(new_code, language)
    return new_code, {
        "was_unwrapped": was_unwrapped,
        "n_comments_stripped": n_stripped,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_path", default="data/interim/normalized/combined.jsonl")
    ap.add_argument("--out", default="data/interim/stripped/combined.jsonl")
    ap.add_argument("--audit_log", default="data/audit/strip_log.jsonl")
    ap.add_argument("--fail_log", default="data/audit/strip_failures.jsonl")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path = Path(args.audit_log)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    fail_path = Path(args.fail_log)
    fail_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"[05b] {in_path} missing - did 05 run?", file=sys.stderr)
        return 1

    n_in = n_out = n_fail = 0
    n_unwrapped = 0
    n_with_comments = 0
    per_lang_unwrap: dict[str, int] = defaultdict(int)
    per_lang_comment: dict[str, int] = defaultdict(int)
    per_lang_fail: dict[str, int] = defaultdict(int)

    with in_path.open("r", encoding="utf-8") as fin, \
            out_path.open("w", encoding="utf-8") as fout, \
            audit_path.open("w", encoding="utf-8") as faud, \
            fail_path.open("w", encoding="utf-8") as ffail:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            rec = json.loads(line)
            language = rec.get("language") or "python"
            code = rec.get("code") or ""

            new_code, audit = strip_record(code, language)

            if not syntax_ok(language, new_code):
                n_fail += 1
                per_lang_fail[language] += 1
                ffail.write(json.dumps({
                    "row_idx": n_in,
                    "language": language,
                    "source": rec.get("source"),
                    "original": code,
                    "stripped": new_code,
                    "audit": audit,
                }, ensure_ascii=False) + "\n")
                continue

            rec["code"] = new_code
            rec["was_stripped"] = bool(
                audit["was_unwrapped"] or audit["n_comments_stripped"]
            )
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1

            if audit["was_unwrapped"]:
                n_unwrapped += 1
                per_lang_unwrap[language] += 1
            if audit["n_comments_stripped"]:
                n_with_comments += 1
                per_lang_comment[language] += 1

            faud.write(json.dumps({
                "language": language,
                "source": rec.get("source"),
                **audit,
            }, ensure_ascii=False) + "\n")

    print(f"[05b] in={n_in} out={n_out} fail={n_fail}", flush=True)
    print(f"[05b] unwrapped={n_unwrapped} comment_stripped={n_with_comments}",
          flush=True)
    print("[05b] per-language counts:", flush=True)
    langs_seen = (
        set(per_lang_unwrap) | set(per_lang_comment) | set(per_lang_fail)
    )
    for lang in sorted(langs_seen):
        u = per_lang_unwrap.get(lang, 0)
        c = per_lang_comment.get(lang, 0)
        f = per_lang_fail.get(lang, 0)
        print(f"[05b]   {lang:<12s} unwrap={u:<6d} comment_strip={c:<6d} fail={f}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
