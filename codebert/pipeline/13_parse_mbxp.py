"""MBXP / Multilingual MBPP label-transfer parser.

The trick: MBXP is the same MBPP problem set translated into 10+ languages.
Same problem ID across languages = (almost) same algorithm = same complexity.

So if we ALREADY have a labeled Python solution for MBPP problem 23, and
MBXP has a solution for the same problem 23 in Java, the Java sample inherits
the Python label. Repeat for every (problem, language) pair where we have
a Python anchor.

Anchor source: by default we read the labeled python rows from the upstream
parsed directory (combined Leetcode + CodeComplex) because those have already
been label-mapped through the normalizer. Specifically we look for rows where
the problem maps via problem_id. MBXP's problem_id format is
`MBPP/<n>` for python and `MBJP/<n>` for Java; we extract the integer.

Structural-similarity gate: simple rule — both code samples must share at
least one of {has-loop, has-recursion}. This catches the obvious case where
a language ports the algorithm cleanly but uses different control flow at the
boundary; it rejects pairs where the target language uses a stdlib call that
hides the algorithm (e.g. Python `sorted()` vs C custom quicksort).

Output:
    data/interim/parsed/mbxp.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from common.parsers import parse, walk
from common.parsers import LOOP_NODE_KINDS, FUNCTION_NODE_KINDS


# (mxeval split name -> our canonical language).
# Note: MBXP uses MBPP for python; MBJP for java; MBJSP for js; MBTSP for ts;
# MBCPP for cpp; MBCSP for csharp; MBGP for go; MBPHP for php; MBRBP for ruby;
# MBSWP for swift. Rust + C are not covered.
_MBXP_NAME_TO_LANG: dict[str, str] = {
    "mbpp": "python",
    "mbjp": "java",
    "mbjsp": "javascript",
    "mbtsp": "typescript",
    "mbcpp": "cpp",
    "mbcsp": "csharp",
    "mbgp": "go",
    "mbphp": "php",
    "mbrbp": "ruby",
    "mbswp": "swift",
}


def _has_loop(language: str, code: str) -> bool:
    try:
        tree = parse(language, code)
    except Exception:
        return False
    kinds = LOOP_NODE_KINDS.get(language, frozenset())
    for n in walk(tree.root_node):
        if n.type in kinds:
            return True
    return False


def _has_recursion(language: str, code: str) -> bool:
    """Cheap check: a function references its own name within its body."""
    try:
        tree = parse(language, code)
    except Exception:
        return False
    fn_kinds = FUNCTION_NODE_KINDS.get(language, frozenset())
    src = code.encode("utf-8")
    for n in walk(tree.root_node):
        if n.type not in fn_kinds:
            continue
        # Find the identifier name, then count occurrences within the function body.
        fname = None
        for child in n.children:
            if child.type in ("identifier", "name", "field_identifier"):
                fname = src[child.start_byte:child.end_byte].decode("utf-8", "replace")
                break
        if not fname:
            continue
        body = src[n.start_byte:n.end_byte].decode("utf-8", "replace")
        # Count occurrences of fname in body, minus the declaration itself.
        if body.count(fname) > 1:
            return True
    return False


def _structural_match(lang_a: str, code_a: str, lang_b: str, code_b: str) -> bool:
    a_loop = _has_loop(lang_a, code_a)
    a_rec = _has_recursion(lang_a, code_a)
    b_loop = _has_loop(lang_b, code_b)
    b_rec = _has_recursion(lang_b, code_b)
    return (a_loop == b_loop) and (a_rec == b_rec)


def _normalize_pid(s: str) -> str | None:
    """Pull integer task id out of 'MBPP/12', 'MBJP/12', etc. -> '12'."""
    if not s:
        return None
    head = s.split("/")[-1]
    digits = "".join(ch for ch in head if ch.isdigit())
    return digits or None


def _load_anchor_labels(anchor_path: Path) -> dict[str, str]:
    """Build {numeric_pid -> label} from a labeled Python parquet/jsonl file.

    Accepts jsonl rows of the form {problem_id, label, language='python'}.
    """
    anchor: dict[str, str] = {}
    if not anchor_path.exists():
        return anchor
    with anchor_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("language") != "python":
                continue
            pid = _normalize_pid(str(obj.get("problem_id") or ""))
            label = obj.get("label") or obj.get("pre_label")
            if pid and label and pid not in anchor:
                anchor[pid] = label
    return anchor


def _load_mbxp(raw_root: Path) -> list[dict]:
    """Read mxeval-flavored jsonl files under raw_root.

    Expected layout:
        data/raw/mbxp/
            mbpp.jsonl     (python)
            mbjp.jsonl     (java)
            mbjsp.jsonl    (js)
            ...
    """
    rows: list[dict] = []
    if not raw_root.exists():
        return rows
    for p in raw_root.glob("*.jsonl"):
        name = p.stem.lower()
        lang = _MBXP_NAME_TO_LANG.get(name)
        if lang is None:
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                code = obj.get("canonical_solution") or obj.get("solution") or obj.get("code")
                pid = _normalize_pid(str(obj.get("task_id") or obj.get("problem_id") or ""))
                if not code or not pid:
                    continue
                rows.append({"language": lang, "pid": pid, "code": code})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw_dir", default="data/raw/mbxp",
                    help="Directory containing mbpp.jsonl, mbjp.jsonl, ... (mxeval format)")
    ap.add_argument("--anchor", default="data/interim/parsed/leetcode.jsonl",
                    help="Path to a labeled-Python jsonl whose problem_id we use to "
                         "look up labels for transfer. We accept any of our parsers' "
                         "outputs that has {language: 'python', problem_id, label}.")
    ap.add_argument("--out", default="data/interim/parsed/mbxp.jsonl")
    args = ap.parse_args()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    raw_root = Path(args.raw_dir)
    anchor_path = Path(args.anchor)

    anchor = _load_anchor_labels(anchor_path)
    if not anchor:
        # Common case: MBXP doesn't share problem_ids with leetcode. Build the
        # anchor from MBXP's own python rows by hand-labeling them — for now
        # we just emit ZERO records when no anchor is supplied. The caller is
        # expected to point --anchor at a labeled jsonl with python rows whose
        # problem_id matches MBXP's task_id format.
        print(f"[13] no anchor labels found at {anchor_path}; nothing to transfer.",
              flush=True)
        out.write_text("", encoding="utf-8")
        return 0

    mbxp_rows = _load_mbxp(raw_root)
    if not mbxp_rows:
        print(f"[13] no mbxp rows found under {raw_root}; skipping.", flush=True)
        out.write_text("", encoding="utf-8")
        return 0

    # Build per-language python anchor code map for the structural gate.
    py_codes_by_pid: dict[str, str] = {}
    for r in mbxp_rows:
        if r["language"] == "python":
            py_codes_by_pid.setdefault(r["pid"], r["code"])

    n_emit = n_no_anchor = n_struct_mismatch = 0
    per_lang: dict[str, int] = {}

    with out.open("w", encoding="utf-8") as fout:
        for r in mbxp_rows:
            if r["language"] == "python":
                continue  # we don't need MBPP python rows; they're already labeled
            label = anchor.get(r["pid"])
            if not label:
                n_no_anchor += 1
                continue
            py_code = py_codes_by_pid.get(r["pid"])
            if py_code is None:
                # No paired Python anchor in MBXP — accept the transfer without
                # the structural gate (we trust the upstream label).
                pass
            else:
                if not _structural_match("python", py_code, r["language"], r["code"]):
                    n_struct_mismatch += 1
                    continue
            uid = "mbxp-" + hashlib.sha1(
                (r["language"] + "|" + r["pid"]).encode("utf-8")
            ).hexdigest()[:12]
            fout.write(json.dumps({
                "source": "mbxp",
                "language": r["language"],
                "problem_id": uid,
                "solution_idx": 0,
                "code": r["code"],
                "raw_complexity": label,
                "pre_label": label,
            }, ensure_ascii=False) + "\n")
            n_emit += 1
            per_lang[r["language"]] = per_lang.get(r["language"], 0) + 1

    print(f"[13] emitted={n_emit} no_anchor={n_no_anchor} "
          f"struct_mismatch={n_struct_mismatch}", flush=True)
    for lang in sorted(per_lang):
        print(f"[13]   {lang:<12s} {per_lang[lang]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
