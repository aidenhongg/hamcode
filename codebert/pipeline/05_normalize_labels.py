"""Apply label normalizer to every parsed record, emit a combined jsonl.

Records with an already-canonical `pre_label` field (from CodeComplex, synthetic)
bypass the normalizer — but we validate pre_label is in POINT_LABELS.

Rejects (unnormalizable raw_complexity) go to audit/normalize_rejects.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Resolve sibling imports when invoked as `python pipeline/05_*.py`
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from common.labels import POINT_LABELS
from common.normalizer import normalize


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_dir", default="data/interim/parsed")
    ap.add_argument("--out", default="data/interim/normalized/combined.jsonl")
    ap.add_argument("--reject_log", default="data/audit/normalize_rejects.jsonl")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    rej = Path(args.reject_log); rej.parent.mkdir(parents=True, exist_ok=True)

    n_total = n_emit = n_rej = n_pre = 0
    per_class: dict[str, int] = {}
    reject_samples: dict[str, int] = {}

    with out.open("w", encoding="utf-8") as fout, rej.open("w", encoding="utf-8") as frej:
        for path in sorted(in_dir.glob("*.jsonl")):
            with path.open("r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    n_total += 1
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    pre = rec.get("pre_label")
                    if pre:
                        if pre not in POINT_LABELS:
                            # fix up common unseen forms
                            normed = normalize(pre)
                            if normed is None:
                                n_rej += 1
                                frej.write(json.dumps({
                                    **rec, "reject_reason": f"pre_label_unknown:{pre}"
                                }, ensure_ascii=False) + "\n")
                                continue
                            rec["label"] = normed
                        else:
                            rec["label"] = pre
                        n_pre += 1
                    else:
                        normed = normalize(rec.get("raw_complexity", ""))
                        if normed is None:
                            n_rej += 1
                            key = (rec.get("raw_complexity") or "")[:60]
                            reject_samples[key] = reject_samples.get(key, 0) + 1
                            frej.write(json.dumps({
                                **rec, "reject_reason": "normalizer_none"
                            }, ensure_ascii=False) + "\n")
                            continue
                        rec["label"] = normed
                    per_class[rec["label"]] = per_class.get(rec["label"], 0) + 1
                    # keep the record lean
                    rec.pop("pre_label", None)
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_emit += 1

    print(f"[05] total={n_total} emit={n_emit} reject={n_rej} pre_labeled={n_pre}", flush=True)
    for cls in POINT_LABELS:
        print(f"[05]   {cls:<24} {per_class.get(cls, 0)}", flush=True)
    top_rej = sorted(reject_samples.items(), key=lambda kv: -kv[1])[:15]
    if top_rej:
        print("[05] top 15 reject patterns:", flush=True)
        for k, v in top_rej:
            print(f"[05]   {v:>4}x  {k!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
