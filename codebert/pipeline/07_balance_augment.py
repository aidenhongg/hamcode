"""Balance per-class sample counts.

Undersample over-represented classes (cap) and oversample rare classes via
variable-renaming augmentation. Records added this way get `augmented_from`
pointing at the original `code_sha256`.

Cap: per-class target (default 620 = 500 train + 60 val + 60 test).
Aug cap: per-source 4x to limit near-duplicate bias.
"""

from __future__ import annotations

import argparse
import ast
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

# Benign identifier substitutions applied as a sweep. We validate with ast.parse.
_AUG_SCHEMES: tuple[dict[str, str], ...] = (
    {"i": "p", "j": "q"},
    {"x": "u", "y": "v"},
    {"res": "output", "out": "output"},
    {"tmp": "buf", "temp": "buf"},
    {"arr": "xs"},
    {"dp": "table"},
    {"dist": "distances"},
    {"helper": "go"},
    {"nums": "values"},
    {"ans": "final"},
    {"cnt": "count"},
    {"curr": "current"},
)


def _rename(src: str, scheme: dict[str, str]) -> str:
    out = src
    for old, new in scheme.items():
        out = re.sub(rf"\b{re.escape(old)}\b", new, out)
    return out


def _valid(code: str) -> bool:
    try:
        ast.parse(code); return True
    except SyntaxError:
        return False


def augment_record(rec: dict, scheme: dict[str, str]) -> dict | None:
    new_code = _rename(rec["code"], scheme)
    if new_code == rec["code"]:
        return None
    if not _valid(new_code):
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
    ap.add_argument("--cap_per_class", type=int, default=620)
    ap.add_argument("--max_aug_ratio", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    in_path = Path(args.in_path)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    # bucket records by label
    buckets: dict[str, list[dict]] = {lab: [] for lab in POINT_LABELS}
    with in_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            rec = json.loads(line)
            lab = rec.get("label")
            if lab in buckets:
                buckets[lab].append(rec)

    # under- and over-sample
    finals: list[dict] = []
    for lab in POINT_LABELS:
        recs = buckets[lab]
        rng.shuffle(recs)
        if len(recs) >= args.cap_per_class:
            finals.extend(recs[:args.cap_per_class])
            continue
        # oversample via augmentation
        kept = list(recs)
        max_total = min(args.cap_per_class, int(len(recs) * (1 + args.max_aug_ratio)))
        scheme_idx = 0
        while len(kept) < max_total and scheme_idx < len(_AUG_SCHEMES) * max(1, len(recs)):
            base = recs[scheme_idx % max(1, len(recs))]
            scheme = _AUG_SCHEMES[scheme_idx % len(_AUG_SCHEMES)]
            aug = augment_record(base, scheme)
            scheme_idx += 1
            if aug is None:
                continue
            if any(k["code_sha256"] == aug["code_sha256"] for k in kept):
                continue
            kept.append(aug)
        finals.extend(kept)

    # write
    rng.shuffle(finals)
    per_class: dict[str, int] = {}
    with out.open("w", encoding="utf-8") as fout:
        for rec in finals:
            per_class[rec["label"]] = per_class.get(rec["label"], 0) + 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[07] balanced total={len(finals)}", flush=True)
    for cls in POINT_LABELS:
        n = per_class.get(cls, 0)
        mark = " [THIN]" if n < 50 else ""
        print(f"[07]   {cls:<24} {n}{mark}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
