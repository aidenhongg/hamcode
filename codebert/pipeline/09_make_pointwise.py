"""Emit the final pointwise parquet using the canonical schema.

Also writes convenience train/val/test parquet files to the same directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common.schemas import POINT_SCHEMA


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_path", default="data/interim/split.jsonl")
    ap.add_argument("--out", default="data/processed/pointwise.parquet")
    args = ap.parse_args()

    rows: list[dict] = []
    with open(args.in_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            rows.append({
                "id": (r.get("code_sha256") or str(i))[:16] + f"-{i:06d}",
                "source": r.get("source") or "unknown",
                "problem_id": r.get("problem_id"),
                "solution_idx": int(r.get("solution_idx") or 0),
                "code": r.get("code") or "",
                "code_sha256": r.get("code_sha256") or "",
                "label": r["label"],
                "raw_complexity": r.get("raw_complexity") or "",
                "tokens_graphcodebert": int(r.get("tokens_graphcodebert") or 0),
                "ast_nodes": int(r.get("ast_nodes") or 0),
                "augmented_from": r.get("augmented_from"),
                "split": r.get("split") or "train",
            })

    table = pa.Table.from_pylist(rows, schema=POINT_SCHEMA)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="zstd")
    for sp in ("train", "val", "test"):
        mask = pc.equal(table["split"], sp)
        pq.write_table(table.filter(mask), out.parent / f"{sp}.parquet", compression="zstd")
    print(f"[09] wrote {out} rows={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
