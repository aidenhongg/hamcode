"""Supplementary ingest: mine Codeforces editorials + Python submissions.

Two HuggingFace datasets:
  - open-r1/codeforces             : problems with editorial/notes/description
  - open-r1/codeforces-submissions : user submissions, many languages

Flow:
  1. Scan problem editorials for a complexity statement (same regex family
     as 02_parse_leetcode).
  2. Stream the submissions dataset; keep Python submissions whose problem_id
     has a mineable complexity. Cap per problem so one popular problem doesn't
     dominate.

Yield is modest — many editorials don't state complexity in grep-friendly form.
Typical contribution: 500-2500 Python records supplementing doocs/leetcode.

Safe to skip entirely on flaky networks; the main pipeline uses other sources
as well. Output follows the same interim jsonl schema as 02_parse_leetcode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Reuse the battle-tested complexity extractor from 02.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

# Only import the regex helpers from 02; avoid its CLI.
_mod_path = _THIS.parent / "02_parse_leetcode.py"
import importlib.util
_spec = importlib.util.spec_from_file_location("parse_leetcode_mod", _mod_path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
find_complexity_in_text = _module.find_complexity_in_text


def _looks_like_python(code: str) -> bool:
    if not code or len(code) < 5:
        return False
    java = ("public class", "public static void main", "System.out",
            "import java.", "private static ")
    if any(s in code for s in java):
        return False
    cpp = ("#include", "using namespace", "std::", "cout <<", "cin >>")
    if any(s in code for s in cpp):
        return False
    hits = sum(1 for s in ("def ", "import ", "print(", "if __name__",
                           "class ", "self.", "elif ", "return ")
               if s in code)
    return hits >= 1


def _ingest_problems(hf_problems: str, limit: int | None):
    from datasets import load_dataset
    ds = load_dataset(hf_problems, split="train")
    by_problem: dict[str, tuple[str]] = {}   # problem_id -> raw_complexity
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        pid = row.get("id") or row.get("problem_id") or row.get("contest_id")
        if pid is None:
            continue
        for field in ("editorial", "editorial_text", "note", "description"):
            text = row.get(field) or ""
            if not text:
                continue
            raw = find_complexity_in_text(text, max_chars=8000)
            if raw is not None:
                by_problem[str(pid)] = raw
                break
    return by_problem


def _ingest_submissions(hf_submissions: str, by_problem: dict[str, str],
                         submissions_limit: int, per_problem_cap: int):
    from datasets import load_dataset
    ds = load_dataset(hf_submissions, split="train", streaming=True)
    counts: dict[str, int] = {}
    records: list[dict] = []
    for i, row in enumerate(ds):
        if i >= submissions_limit:
            break
        lang = (row.get("programmingLanguage") or row.get("language") or "").lower()
        if "python" not in lang:
            continue
        pid = (row.get("problem_id") or row.get("problemId")
               or f"{row.get('contestId')}-{row.get('index')}")
        pid = str(pid)
        if pid not in by_problem:
            continue
        code = row.get("source") or row.get("code") or ""
        if not _looks_like_python(code):
            continue
        if counts.get(pid, 0) >= per_problem_cap:
            continue
        counts[pid] = counts.get(pid, 0) + 1
        raw = by_problem[pid]
        records.append({
            "source": "codeforces",
            "problem_id": f"cf-{pid}",
            "solution_idx": counts[pid] - 1,
            "code": code,
            "raw_complexity": raw,
        })
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf_problems", default="open-r1/codeforces")
    ap.add_argument("--hf_submissions", default="open-r1/codeforces-submissions")
    ap.add_argument("--out", default="data/interim/parsed/codeforces.jsonl")
    ap.add_argument("--problems_limit", type=int, default=0,
                    help="0 = all problems")
    ap.add_argument("--submissions_limit", type=int, default=200_000)
    ap.add_argument("--per_problem_cap", type=int, default=5,
                    help="max Python solutions kept per problem")
    args = ap.parse_args()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[02b] loading problems from {args.hf_problems}...", flush=True)
        by_problem = _ingest_problems(
            args.hf_problems,
            args.problems_limit or None,
        )
    except Exception as e:
        print(f"[02b] ERROR loading problems: {e}", file=sys.stderr)
        out.write_text("", encoding="utf-8")
        print(f"[02b] wrote empty {out} — pipeline continues without Codeforces.", flush=True)
        return 0

    print(f"[02b] {len(by_problem)} problems have a mineable complexity", flush=True)
    if not by_problem:
        out.write_text("", encoding="utf-8")
        return 0

    try:
        print(f"[02b] streaming submissions from {args.hf_submissions} "
              f"(cap={args.submissions_limit})...", flush=True)
        records = _ingest_submissions(
            args.hf_submissions, by_problem,
            args.submissions_limit, args.per_problem_cap,
        )
    except Exception as e:
        print(f"[02b] ERROR loading submissions: {e}", file=sys.stderr)
        records = []

    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[02b] wrote {len(records)} records to {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
