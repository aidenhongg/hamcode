"""Balance per-(language, class) sample counts.

Undersample over-represented (language, class) cells (cap) and oversample rare
ones via variable-renaming augmentation. Records added this way get
`augmented_from` pointing at the original `code_sha256`.

Cap: per-(language, class) target (default 620 = 500 train + 60 val + 60 test).
Aug cap: 4x to limit near-duplicate bias.

Per-language rename schemes are intentionally minimal and word-boundary-only;
we re-run a tree-sitter syntax check on every renamed snippet and discard any
that no longer parses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=SyntaxWarning)

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from common.labels import POINT_LABELS
from common.parsers import syntax_ok
from common.schemas import LANGUAGES


# -----------------------------------------------------------------------------
# Per-language augmentation schemes
# -----------------------------------------------------------------------------

# Identifier renames that work across most C-family languages without colliding
# with reserved words. Per-language overrides drop entries that ARE reserved
# in that language (e.g. `helper -> go` collides with the Go keyword).
_BASE_AUG_SCHEMES: tuple[dict[str, str], ...] = (
    {"i": "p", "j": "q"},
    {"x": "u", "y": "v"},
    {"res": "output", "out": "output"},
    {"tmp": "buf", "temp": "buf"},
    {"arr": "xs"},
    {"dp": "table"},
    {"dist": "distances"},
    {"ans": "final"},
    {"cnt": "count"},
    {"curr": "current"},
)

# Tokens we must NOT rename TO in a given language (would create syntax errors).
_PER_LANGUAGE_RESERVED: dict[str, frozenset[str]] = {
    "go":     frozenset({"go", "func", "type", "chan", "map", "range", "go_to"}),
    "rust":   frozenset({"fn", "let", "mut", "ref", "type", "mod", "use", "match",
                         "loop", "impl", "struct", "enum", "trait", "where",
                         "Vec", "Option", "Result", "Box"}),
    "swift":  frozenset({"let", "var", "func", "init", "deinit", "class",
                         "struct", "enum", "protocol", "where", "switch",
                         "guard", "Self"}),
    "c":      frozenset({"int", "char", "float", "double", "long", "short",
                         "void", "struct", "union", "enum", "typedef",
                         "static", "extern", "const", "register", "volatile",
                         "auto", "signed", "unsigned"}),
    "cpp":    frozenset({"int", "char", "void", "struct", "union", "enum",
                         "class", "typename", "namespace", "template",
                         "public", "private", "protected", "virtual",
                         "Vec", "auto"}),
    "csharp": frozenset({"int", "string", "void", "var", "dynamic",
                         "class", "struct", "interface", "namespace",
                         "abstract", "sealed", "readonly", "ref", "out"}),
    "java":   frozenset({"int", "char", "void", "boolean", "byte", "short",
                         "long", "float", "double", "class", "interface",
                         "enum", "record", "abstract", "final", "static",
                         "package", "import", "public", "private", "protected"}),
    "javascript": frozenset({"let", "var", "const", "function", "class",
                             "import", "export", "extends", "static"}),
    "typescript": frozenset({"let", "var", "const", "function", "class",
                             "interface", "type", "import", "export",
                             "namespace", "module", "any", "unknown",
                             "never", "void"}),
    "php":    frozenset({"function", "class", "interface", "trait",
                         "namespace", "use", "static", "public", "private",
                         "protected", "abstract", "final"}),
    "ruby":   frozenset({"def", "end", "class", "module", "if", "unless",
                         "while", "until", "do", "begin", "rescue"}),
    "python": frozenset({"def", "class", "lambda", "import", "from", "as",
                         "with", "yield", "async", "await", "match", "case"}),
}


def _scheme_safe_for(lang: str, scheme: dict[str, str]) -> dict[str, str]:
    """Drop scheme entries whose target would collide with `lang`'s reserved
    words. Returns a (possibly empty) sub-dict."""
    reserved = _PER_LANGUAGE_RESERVED.get(lang, frozenset())
    return {k: v for k, v in scheme.items() if v not in reserved}


def _rename(src: str, scheme: dict[str, str]) -> str:
    out = src
    for old, new in scheme.items():
        out = re.sub(rf"\b{re.escape(old)}\b", new, out)
    return out


def augment_record(rec: dict, scheme: dict[str, str]) -> dict | None:
    safe = _scheme_safe_for(rec["language"], scheme)
    if not safe:
        return None
    new_code = _rename(rec["code"], safe)
    if new_code == rec["code"]:
        return None
    if not syntax_ok(rec["language"], new_code):
        return None
    sha = hashlib.sha256(new_code.encode("utf-8")).hexdigest()
    clone = dict(rec)
    clone["code"] = new_code
    clone["code_sha256"] = sha
    clone["augmented_from"] = rec["code_sha256"]
    clone["source"] = f"{rec['source']}-aug"
    return clone


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_path", default="data/interim/filtered.jsonl")
    ap.add_argument("--out", default="data/interim/balanced.jsonl")
    ap.add_argument("--cap_per_class", type=int, default=620,
                    help="cap per (language, label) cell")
    ap.add_argument("--max_aug_ratio", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    in_path = Path(args.in_path)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    # bucket records by (language, label)
    buckets: dict[tuple[str, str], list[dict]] = {}
    for lang in LANGUAGES:
        for lab in POINT_LABELS:
            buckets[(lang, lab)] = []
    with in_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            rec = json.loads(line)
            key = (rec.get("language", "python"), rec.get("label"))
            if key in buckets:
                buckets[key].append(rec)

    # under- and over-sample per cell
    finals: list[dict] = []
    for lang in LANGUAGES:
        for lab in POINT_LABELS:
            recs = buckets[(lang, lab)]
            if not recs:
                continue
            rng.shuffle(recs)
            if len(recs) >= args.cap_per_class:
                finals.extend(recs[: args.cap_per_class])
                continue
            kept = list(recs)
            max_total = min(args.cap_per_class,
                            int(len(recs) * (1 + args.max_aug_ratio)))
            scheme_idx = 0
            tries = 0
            max_tries = len(_BASE_AUG_SCHEMES) * max(1, len(recs)) * 3
            while len(kept) < max_total and tries < max_tries:
                base = recs[scheme_idx % max(1, len(recs))]
                scheme = _BASE_AUG_SCHEMES[scheme_idx % len(_BASE_AUG_SCHEMES)]
                aug = augment_record(base, scheme)
                scheme_idx += 1
                tries += 1
                if aug is None:
                    continue
                if any(k["code_sha256"] == aug["code_sha256"] for k in kept):
                    continue
                kept.append(aug)
            finals.extend(kept)

    rng.shuffle(finals)
    per_cell: dict[tuple[str, str], int] = {}
    with out.open("w", encoding="utf-8") as fout:
        for rec in finals:
            key = (rec["language"], rec["label"])
            per_cell[key] = per_cell.get(key, 0) + 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[07] balanced total={len(finals)}", flush=True)
    for lang in LANGUAGES:
        any_for_lang = any(per_cell.get((lang, lab), 0) > 0 for lab in POINT_LABELS)
        if not any_for_lang:
            continue
        line = " | ".join(
            f"{lab}={per_cell.get((lang, lab), 0)}"
            for lab in POINT_LABELS
            if per_cell.get((lang, lab), 0) > 0
        )
        print(f"[07]   {lang:<12s} {line}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
