"""Dedupe + filter (multi-language).

1) Per-language tree-sitter syntax check (replaces ast.parse() — supports all 11 + Python).
2) Token-length gate via the LongCoder tokenizer (mandatory; the char heuristic
   was Python-tuned and underestimates token count by ~2x for the new languages).
3) MinHash LSH across *all* sources to drop near-duplicate code (threshold 0.85).
   Dedup buckets per language so a Python sort and a Rust sort don't collide.

Output:
    data/interim/filtered.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=SyntaxWarning)

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from datasketch import MinHash, MinHashLSH  # type: ignore

from common.parsers import parse, syntax_ok, walk
from common.schemas import LANG_SET


def _shingles(code: str, k: int = 5) -> set[str]:
    # word-ish shingles — robust to whitespace / minor cosmetic edits
    toks = code.split()
    return {" ".join(toks[i:i + k]) for i in range(max(0, len(toks) - k + 1))}


def _minhash(tokens: set[str], num_perm: int = 64) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for t in tokens:
        m.update(t.encode("utf-8"))
    return m


def _load_bpe_tokenizer():
    from transformers import AutoTokenizer  # type: ignore
    import transformers.utils.logging as hf_logging  # type: ignore
    hf_logging.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained("microsoft/longcoder-base")
    tok.model_max_length = 1_000_000
    return tok


def token_len(code: str, tokenizer) -> int:
    return len(tokenizer.encode(code, add_special_tokens=False))


def _node_count(language: str, code: str) -> int:
    try:
        tree = parse(language, code)
    except Exception:
        return 0
    return sum(1 for _ in walk(tree.root_node))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_path", default="data/interim/stripped/combined.jsonl",
                    help="Defaults to the strip-leakage stage's output. "
                         "Pass data/interim/normalized/combined.jsonl to bypass "
                         "the strip stage (e.g. for parity tests).")
    ap.add_argument("--out", default="data/interim/filtered.jsonl")
    ap.add_argument("--max_tokens", type=int, default=3900,
                    help="Token-count cap (LongCoder seq_len 4096 minus headroom for "
                         "CLS, SEP, bridge, and memory tokens).")
    ap.add_argument("--min_tokens", type=int, default=6)
    ap.add_argument("--threshold", type=float, default=0.85)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    if not in_path.exists():
        print(f"[06] {in_path} missing - did 05 run?", file=sys.stderr)
        return 1

    tokenizer = _load_bpe_tokenizer()

    # One LSH instance per language so cross-language coincidences don't dedup.
    per_lang_lsh: dict[str, MinHashLSH] = {}

    n_in = n_kept = n_no_lang = n_syntax_fail = n_dup = n_too_long = n_too_short = 0
    per_lang_kept: dict[str, int] = {}
    with in_path.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            n_in += 1
            rec = json.loads(line)
            code = rec.get("code", "")
            language = rec.get("language") or "python"
            if language not in LANG_SET:
                n_no_lang += 1
                continue

            # 1) syntax check
            if not syntax_ok(language, code):
                n_syntax_fail += 1
                continue
            rec["ast_nodes"] = _node_count(language, code)

            # 2) token length
            tl = token_len(code, tokenizer)
            rec["tokens_bpe"] = tl
            if tl > args.max_tokens:
                n_too_long += 1
                continue
            if tl < args.min_tokens:
                n_too_short += 1
                continue

            # 3) per-language near-dup via MinHash
            shingles = _shingles(code)
            if not shingles:
                n_too_short += 1
                continue
            m = _minhash(shingles)
            lsh = per_lang_lsh.get(language)
            if lsh is None:
                lsh = MinHashLSH(threshold=args.threshold, num_perm=64)
                per_lang_lsh[language] = lsh
            matches = lsh.query(m)
            if matches:
                n_dup += 1
                continue
            key = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
            lsh.insert(key, m)

            rec["language"] = language
            rec["code_sha256"] = hashlib.sha256(code.encode("utf-8")).hexdigest()
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_kept += 1
            per_lang_kept[language] = per_lang_kept.get(language, 0) + 1

    print(f"[06] in={n_in} kept={n_kept} syntax_fail={n_syntax_fail} dup={n_dup} "
          f"long={n_too_long} short={n_too_short} no_lang={n_no_lang}", flush=True)
    for lang in sorted(per_lang_kept):
        print(f"[06]   {lang:<12s} kept={per_lang_kept[lang]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
