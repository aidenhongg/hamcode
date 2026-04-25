"""Parse kamyu104/LeetCode-Solutions per-language directories.

Every solution file in this repo starts with header comments stating the time
and space complexity:

    // Time:  O(n)
    // Space: O(n)
    class Solution { ... }

(or `# Time: O(n)` for Python/Ruby). We extract the Time complexity and the
file body, tag with the language inferred from the per-language subdirectory.

The repo lays out:

    LeetCode-Solutions/
        C++/        *.cpp                          # C++
        C#/         *.cs                           # C#
        Python/     *.py                           # Python
        Java/       *.java                         # Java
        Go/         *.go                           # Go
        Kotlin/     *.kt                           # not in our 11 — skipped
        MySQL/      *.sql                          # not in our 11 — skipped
        PHP/        *.php                          # PHP
        Ruby/       *.rb                           # Ruby
        Rust/       *.rs                           # Rust
        Shell/      *.sh                           # not in our 11 - skipped
        Swift/      *.swift                        # Swift
        TypeScript/ *.ts                           # TypeScript

Output:
    data/interim/parsed/kamyu.jsonl
    data/audit/kamyu_parse_failures.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable

# (subdir-name-as-listed-in-repo, language-id, comment-prefix-regex)
_DIR_TO_LANG: list[tuple[str, str, str]] = [
    ("C++",        "cpp",        r"//\s*Time:\s*O\((.+?)\)"),
    ("C#",         "csharp",     r"//\s*Time:\s*O\((.+?)\)"),
    ("Python",     "python",     r"#\s*Time:\s*O\((.+?)\)"),
    ("Java",       "java",       r"//\s*Time:\s*O\((.+?)\)"),
    ("Go",         "go",         r"//\s*Time:\s*O\((.+?)\)"),
    ("PHP",        "php",        r"(?://|#)\s*Time:\s*O\((.+?)\)"),
    ("Ruby",       "ruby",       r"#\s*Time:\s*O\((.+?)\)"),
    ("Rust",       "rust",       r"//\s*Time:\s*O\((.+?)\)"),
    ("Swift",      "swift",      r"//\s*Time:\s*O\((.+?)\)"),
    ("TypeScript", "typescript", r"//\s*Time:\s*O\((.+?)\)"),
]

_EXT_FOR_LANG: dict[str, tuple[str, ...]] = {
    "cpp":        (".cpp", ".cc", ".cxx"),
    "csharp":     (".cs",),
    "python":     (".py",),
    "java":       (".java",),
    "go":         (".go",),
    "php":        (".php",),
    "ruby":       (".rb",),
    "rust":       (".rs",),
    "swift":      (".swift",),
    "typescript": (".ts",),
}


def _iter_files(root: Path, sub: str, exts: tuple[str, ...]) -> Iterable[Path]:
    base = root / sub
    if not base.exists():
        return
    for p in base.rglob("*"):
        if p.is_file() and p.suffix in exts:
            yield p


def parse_file(path: Path, time_re: re.Pattern[str]) -> tuple[str, str] | None:
    """Return (raw_complexity, code) or None if no Time: header."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    # Look at the first ~10 lines for the Time: header (header comments lead).
    head = "\n".join(text.splitlines()[:10])
    m = time_re.search(head)
    if not m:
        return None
    return m.group(1).strip(), text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw_dir", default="data/raw/kamyu",
                    help="Path to a clone of github.com/kamyu104/LeetCode-Solutions")
    ap.add_argument("--out", default="data/interim/parsed/kamyu.jsonl")
    ap.add_argument("--fail_log", default="data/audit/kamyu_parse_failures.jsonl")
    ap.add_argument("--limit", type=int, default=0,
                    help="Optional cap on total files visited (0 = no cap)")
    args = ap.parse_args()

    root = Path(args.raw_dir)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fail = Path(args.fail_log); fail.parent.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        print(f"[12] {root} not found - skipping (expected on a fresh checkout)",
              flush=True)
        out.write_text("", encoding="utf-8")
        return 0

    n_files = n_emit = n_no_header = 0
    per_lang: dict[str, int] = {}

    with out.open("w", encoding="utf-8") as fout, fail.open("w", encoding="utf-8") as ffail:
        for sub, lang, time_re_str in _DIR_TO_LANG:
            time_re = re.compile(time_re_str, re.IGNORECASE)
            exts = _EXT_FOR_LANG[lang]
            for path in _iter_files(root, sub, exts):
                n_files += 1
                if args.limit and n_files > args.limit:
                    break
                parsed = parse_file(path, time_re)
                if parsed is None:
                    n_no_header += 1
                    ffail.write(json.dumps({"path": str(path), "language": lang,
                                            "reason": "no_time_header"}) + "\n")
                    continue
                raw_comp, code = parsed
                pid = "km-" + hashlib.sha1(
                    (lang + "|" + path.stem).encode("utf-8")
                ).hexdigest()[:12]
                fout.write(json.dumps({
                    "source": "kamyu",
                    "language": lang,
                    "problem_id": pid,
                    "solution_idx": 0,
                    "code": code,
                    "raw_complexity": raw_comp,
                    "path": str(path),
                }, ensure_ascii=False) + "\n")
                n_emit += 1
                per_lang[lang] = per_lang.get(lang, 0) + 1

    print(f"[12] files={n_files} emitted={n_emit} no_header={n_no_header}", flush=True)
    for lang in sorted(per_lang):
        print(f"[12]   {lang:<12s} {per_lang[lang]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
