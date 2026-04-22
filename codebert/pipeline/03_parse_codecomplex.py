"""Parse CodeComplex Python jsonl into the unified interim schema.

CodeComplex has 7 classes (time only, Python subset). We map their labels to
our 11-class scheme — NP-hard collapses to `exponential`.

Expected input format (per the paper — we accept variants):
    {"id": ..., "src"|"code"|"source": "...", "complexity"|"label": "O(n)", "problem": "..."}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Map CodeComplex labels (many surface forms) to our canonical labels.
_CC_LABEL_MAP: dict[str, str] = {
    "o(1)": "O(1)", "1": "O(1)", "constant": "O(1)",
    "o(logn)": "O(log n)", "o(log n)": "O(log n)", "logn": "O(log n)",
    "log(n)": "O(log n)", "logarithmic": "O(log n)",
    "o(n)": "O(n)", "n": "O(n)", "linear": "O(n)",
    "o(nlogn)": "O(n log n)", "o(n log n)": "O(n log n)",
    "o(n*logn)": "O(n log n)", "nlogn": "O(n log n)", "linearithmic": "O(n log n)",
    "o(n^2)": "O(n^2)", "o(n_2)": "O(n^2)", "o(n**2)": "O(n^2)", "quadratic": "O(n^2)",
    "o(n^3)": "O(n^3)", "o(n_3)": "O(n^3)", "o(n**3)": "O(n^3)", "cubic": "O(n^3)",
    "np": "exponential", "np_hard": "exponential", "np-hard": "exponential",
    "np hard": "exponential", "nphard": "exponential",
    "exponential": "exponential", "factorial": "exponential",
    "o(2^n)": "exponential", "o(2**n)": "exponential", "o(n!)": "exponential",
}


def map_cc_label(raw: str) -> str | None:
    if not raw:
        return None
    key = raw.strip().lower().replace(" ", "")
    if key in _CC_LABEL_MAP:
        return _CC_LABEL_MAP[key]
    # Fuzzy: try with minor whitespace/punctuation differences
    for src, dst in _CC_LABEL_MAP.items():
        if src.replace(" ", "") == key:
            return dst
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw_path", default="data/raw/codecomplex/python.jsonl")
    ap.add_argument("--out", default="data/interim/parsed/codecomplex.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    in_path = Path(args.raw_path)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"[03] {in_path} not found — writing empty {out} and continuing.", flush=True)
        out.write_text("", encoding="utf-8")
        return 0

    n_total = n_emit = n_rej = 0
    rej_reasons: dict[str, int] = {}
    with in_path.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            n_total += 1
            if args.limit and n_total > args.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                n_rej += 1; rej_reasons["json_decode"] = rej_reasons.get("json_decode", 0) + 1
                continue
            code = obj.get("src") or obj.get("code") or obj.get("source")
            raw_comp = obj.get("complexity") or obj.get("label") or obj.get("class")
            if not code or not raw_comp:
                n_rej += 1; rej_reasons["missing_fields"] = rej_reasons.get("missing_fields", 0) + 1
                continue
            label = map_cc_label(str(raw_comp))
            if not label:
                n_rej += 1
                rej_reasons[f"unmapped:{raw_comp}"] = rej_reasons.get(f"unmapped:{raw_comp}", 0) + 1
                continue
            pid = "cc-" + hashlib.sha1(code.encode("utf-8")).hexdigest()[:12]
            fout.write(json.dumps({
                "source": "codecomplex",
                "problem_id": pid,
                "solution_idx": 0,
                "code": code,
                "raw_complexity": str(raw_comp),
                "pre_label": label,
            }, ensure_ascii=False) + "\n")
            n_emit += 1

    print(f"[03] total={n_total} emit={n_emit} reject={n_rej}", flush=True)
    if rej_reasons:
        top = sorted(rej_reasons.items(), key=lambda kv: -kv[1])[:10]
        print(f"[03] top reject reasons: {top}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
