"""Audit train/val/test isolation. Prints overlap counts — all should be 0.

Usage:  python check_splits.py
"""
from __future__ import annotations

import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.labels import POINT_LABELS


def _load(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist() if path.exists() else []


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _normalized_sha(code: str) -> str:
    # Collapse whitespace so trivial formatting differences don't mask duplication.
    stripped = " ".join(code.split())
    return _sha(stripped)


def _diagnose(rows_a: list[dict], rows_b: list[dict], name_a: str, name_b: str,
              key_a: str = "code", key_b: str = "code") -> None:
    set_a_exact = {r[key_a] for r in rows_a}
    set_b_exact = {r[key_b] for r in rows_b}
    exact = set_a_exact & set_b_exact

    set_a_sha = {r.get("code_sha256") or _sha(r[key_a]) for r in rows_a}
    set_b_sha = {r.get("code_sha256") or _sha(r[key_b]) for r in rows_b}
    sha = set_a_sha & set_b_sha

    set_a_norm = {_normalized_sha(r[key_a]) for r in rows_a}
    set_b_norm = {_normalized_sha(r[key_b]) for r in rows_b}
    norm = set_a_norm & set_b_norm

    pid_a = {r["problem_id"] for r in rows_a if r.get("problem_id")}
    pid_b = {r["problem_id"] for r in rows_b if r.get("problem_id")}
    pid = pid_a & pid_b

    print(f"\n--- {name_a} vs {name_b} ---")
    print(f"  {name_a}: {len(rows_a)} rows   {name_b}: {len(rows_b)} rows")
    print(f"  exact code match:              {len(exact)}")
    print(f"  code_sha256 match:             {len(sha)}")
    print(f"  normalized-whitespace match:   {len(norm)}")
    print(f"  shared problem_id:             {len(pid)}   (should be 0)")

    if pid:
        print(f"    first few shared pids: {sorted(pid)[:5]}")
    if norm and not pid:
        print(f"    ⚠ same code, different problem_id — likely cross-source leak")
        # show one example
        for r in rows_a[:2000]:
            n = _normalized_sha(r["code"])
            if n in norm:
                print(f"      [{name_a}] pid={r.get('problem_id')} source={r.get('source')}")
                for r2 in rows_b:
                    if _normalized_sha(r2[key_b]) == n:
                        print(f"      [{name_b}] pid={r2.get('problem_id')} source={r2.get('source')}")
                        print(f"      snippet: {r['code'][:120]!r}")
                        break
                break


def _pairwise_audit(pair_rows: list[dict], pt_rows_by_split: dict[str, list[dict]]) -> None:
    """Check pairwise pairs respect split boundary (sanity)."""
    print("\n--- pairwise split integrity ---")
    by_split: dict[str, list[dict]] = {}
    for r in pair_rows:
        by_split.setdefault(r["split"], []).append(r)
    for sp, pairs in by_split.items():
        pt_codes = {r["code"] for r in pt_rows_by_split.get(sp, [])}
        crosses = 0
        for pr in pairs[:5000]:  # sample
            if pr["code_a"] not in pt_codes and pr["code_b"] not in pt_codes:
                crosses += 1
        print(f"  {sp}: {len(pairs)} pairs   sampled {min(5000, len(pairs))}   "
              f"pairs whose A and B aren't both in {sp} pointwise: {crosses}")


def main() -> int:
    root = Path("data/processed")
    train = _load(root / "train.parquet")
    val = _load(root / "val.parquet")
    test = _load(root / "test.parquet")

    print(f"pointwise totals: train={len(train)}  val={len(val)}  test={len(test)}")

    # Problem-id uniqueness: every split should have a disjoint problem_id set
    _diagnose(train, val, "train", "val")
    _diagnose(train, test, "train", "test")
    _diagnose(val, test, "val", "test")

    # Source distribution per split
    def _source_dist(rows: list[dict]) -> dict[str, int]:
        d: dict[str, int] = {}
        for r in rows:
            d[r["source"]] = d.get(r["source"], 0) + 1
        return d

    print("\n--- source distribution per split ---")
    for name, rows in (("train", train), ("val", val), ("test", test)):
        print(f"  {name}: {_source_dist(rows)}")

    # Per-class breakdown: records AND unique problems per split.
    def _class_breakdown(rows: list[dict]) -> tuple[dict[str, int], dict[str, int]]:
        recs: dict[str, int] = defaultdict(int)
        pids: dict[str, set[str]] = defaultdict(set)
        for r in rows:
            lab = r["label"]
            recs[lab] += 1
            pid = r.get("problem_id")
            if pid:
                pids[lab].add(pid)
        return dict(recs), {k: len(v) for k, v in pids.items()}

    train_rec, train_pid = _class_breakdown(train)
    val_rec, val_pid = _class_breakdown(val)
    test_rec, test_pid = _class_breakdown(test)

    print("\n--- records / unique problems per class per split ---")
    print(f"  {'class':<22} {'train':>16}   {'val':>14}   {'test':>14}")
    print(f"  {'':<22} {'recs / probs':>16}   {'recs / probs':>14}   {'recs / probs':>14}")
    print(f"  {'-'*22} {'-'*16}   {'-'*14}   {'-'*14}")
    for lab in POINT_LABELS:
        tr = train_rec.get(lab, 0); trp = train_pid.get(lab, 0)
        vr = val_rec.get(lab, 0);   vrp = val_pid.get(lab, 0)
        te = test_rec.get(lab, 0);  tep = test_pid.get(lab, 0)
        thin = " [THIN]" if tr < 100 else ""
        print(f"  {lab:<22} {tr:>7} / {trp:<6}   {vr:>5} / {vrp:<5}   "
              f"{te:>5} / {tep:<5}{thin}")

    # Flag classes where train has very few unique problems — model may overfit to them.
    narrow = []
    for lab in POINT_LABELS:
        probs = train_pid.get(lab, 0)
        recs = train_rec.get(lab, 0)
        # narrow = many records but few unique problems (e.g., heavy augmentation over 1-2 templates)
        if recs >= 50 and probs > 0 and recs / probs >= 5:
            narrow.append((lab, recs, probs, recs / probs))
    if narrow:
        print("\n  [WARN] train classes dominated by few unique problems "
              "(records/problem ratio >= 5):")
        for lab, r, p, ratio in narrow:
            print(f"    {lab:<22} {r} records across {p} problems  (ratio={ratio:.1f})")

    # Pairwise audit
    pair_train = _load(root / "pair_train.parquet")
    pair_val = _load(root / "pair_val.parquet")
    pair_test = _load(root / "pair_test.parquet")
    if pair_train or pair_val or pair_test:
        pt_by_split = {"train": train, "val": val, "test": test}
        _pairwise_audit(pair_train + pair_val + pair_test, pt_by_split)

    return 0


if __name__ == "__main__":
    sys.exit(main())
