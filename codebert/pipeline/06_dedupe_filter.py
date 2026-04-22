"""Dedupe + filter.

1) MinHash LSH across *all* sources to drop near-duplicate code (threshold 0.85).
2) ast.parse() sanity check.
3) Token-length gate via GraphCodeBERT tokenizer if available,
   else a char-length heuristic (~4.2 chars/token for Python).

Output:
    data/interim/filtered.jsonl
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import warnings
from pathlib import Path

# Silence Py3.12's noisy "invalid escape sequence" SyntaxWarnings emitted by
# ast.parse() on LeetCode solutions that embed regex patterns in strings.
warnings.filterwarnings("ignore", category=SyntaxWarning)

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from datasketch import MinHash, MinHashLSH  # type: ignore

_CHAR_PER_TOKEN = 4.2  # Python-ish average for RoBERTa BPE


def _shingles(code: str, k: int = 5) -> set[str]:
    # word-ish shingles — robust to whitespace / minor cosmetic edits
    toks = code.split()
    return {" ".join(toks[i:i + k]) for i in range(max(0, len(toks) - k + 1))}


def _minhash(tokens: set[str], num_perm: int = 64) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for t in tokens:
        m.update(t.encode("utf-8"))
    return m


def _load_gcb_tokenizer():
    try:
        from transformers import AutoTokenizer  # type: ignore
        import transformers.utils.logging as hf_logging  # type: ignore
        hf_logging.set_verbosity_error()   # silence per-sample "too long" spam
        tok = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
        # Bump effective max so the tokenizer doesn't warn on samples we'll drop anyway.
        tok.model_max_length = 1_000_000
        return tok
    except Exception as e:
        print(f"[06] tokenizer unavailable ({e}); falling back to char heuristic", flush=True)
        return None


def token_len(code: str, tokenizer) -> int:
    if tokenizer is None:
        return int(len(code) / _CHAR_PER_TOKEN)
    return len(tokenizer.encode(code, add_special_tokens=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_path", default="data/interim/normalized/combined.jsonl")
    ap.add_argument("--out", default="data/interim/filtered.jsonl")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--min_tokens", type=int, default=6)
    # 0.92 keeps distinct-algorithm solutions that share boilerplate while still
    # catching copy-paste duplicates. 0.85 was too aggressive — collapsed different
    # LeetCode solutions that happened to share a 5-word shingle.
    ap.add_argument("--threshold", type=float, default=0.92)
    ap.add_argument("--skip_tokenizer", action="store_true",
                    help="skip HF tokenizer; use char-length heuristic")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    if not in_path.exists():
        print(f"[06] {in_path} missing — did 05 run?", file=sys.stderr)
        return 1

    tokenizer = None if args.skip_tokenizer else _load_gcb_tokenizer()
    lsh = MinHashLSH(threshold=args.threshold, num_perm=64)

    n_in = n_kept = n_ast_fail = n_dup = n_too_long = n_too_short = 0
    with in_path.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            n_in += 1
            rec = json.loads(line)
            code = rec.get("code", "")

            # 1) ast.parse
            try:
                tree = ast.parse(code)
            except SyntaxError:
                n_ast_fail += 1
                continue
            rec["ast_nodes"] = sum(1 for _ in ast.walk(tree))

            # 2) token length
            tl = token_len(code, tokenizer)
            rec["tokens_graphcodebert"] = tl
            if tl > args.max_tokens:
                n_too_long += 1
                continue
            if tl < args.min_tokens:
                n_too_short += 1
                continue

            # 3) near-dup via MinHash
            shingles = _shingles(code)
            if not shingles:
                n_too_short += 1
                continue
            m = _minhash(shingles)
            matches = lsh.query(m)
            if matches:
                n_dup += 1
                continue
            key = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
            lsh.insert(key, m)

            rec["code_sha256"] = hashlib.sha256(code.encode("utf-8")).hexdigest()
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_kept += 1

    print(f"[06] in={n_in} kept={n_kept} ast_fail={n_ast_fail} dup={n_dup} "
          f"long={n_too_long} short={n_too_short}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
