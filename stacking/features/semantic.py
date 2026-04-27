"""Compute pair-level semantic similarity features from pointwise CLS vectors.

Consumes the `point_cls_{split}.parquet` files written by bert_logits.py
(which contain a 768-d [CLS] vector per snippet, keyed by pointwise `id`).

For each pair row, look up CLS_A and CLS_B by joining pair rows against
the pointwise parquet on `code_sha256` (code of each side), then compute
cosine similarity, L2 distance, and a small stats bundle.

Writes `pair_sim_{split}.parquet` with columns:
    pair_id, cls_cosine, cls_l2, cls_mean_abs_diff, cls_max_abs_diff

CLI:
    python -m stacking.features.semantic \
        --in_splits data/processed \
        --extraction_dir runs/heads/extraction
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# -----------------------------------------------------------------------------
# Load CLS vectors from the wide-format parquet produced by bert_logits.py
# -----------------------------------------------------------------------------

def _load_cls_table(cls_path: Path) -> tuple[list[str], np.ndarray]:
    """Return (ids, cls_matrix) from point_cls_{split}.parquet."""
    tbl = pq.read_table(cls_path)
    ids = tbl.column("id").to_pylist()
    cls_cols = [c for c in tbl.schema.names if c.startswith("cls_")]
    # Sort by numeric suffix so dim order matches write order
    cls_cols.sort(key=lambda s: int(s.split("_", 1)[1]))
    mat = np.stack(
        [np.asarray(tbl.column(c).to_pylist(), dtype=np.float32) for c in cls_cols],
        axis=1,
    )
    return ids, mat


# -----------------------------------------------------------------------------
# Pair side lookup: map the pointwise parquet's `id` to its `code_sha256`
# so pair rows can find their CLS by A/B code contents.
# -----------------------------------------------------------------------------

def _sha_to_id_map(point_parquet: Path) -> dict[str, str]:
    tbl = pq.read_table(point_parquet)
    shas = tbl.column("code_sha256").to_pylist()
    ids = tbl.column("id").to_pylist()
    # Multiple ids can share a sha256 if dataset has duplicates — last one wins,
    # but features are identical so this is safe.
    return {s: i for s, i in zip(shas, ids)}


def _pair_shas(pair_parquet: Path) -> tuple[list[str], list[str], list[str]]:
    """Return (pair_ids, code_a_shas, code_b_shas)."""
    import hashlib
    tbl = pq.read_table(pair_parquet)
    pair_ids = tbl.column("pair_id").to_pylist()
    code_a = tbl.column("code_a").to_pylist()
    code_b = tbl.column("code_b").to_pylist()
    sha_a = [hashlib.sha256(c.encode("utf-8")).hexdigest() for c in code_a]
    sha_b = [hashlib.sha256(c.encode("utf-8")).hexdigest() for c in code_b]
    return pair_ids, sha_a, sha_b


# -----------------------------------------------------------------------------
# Similarity metrics
# -----------------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine between two (N, D) matrices."""
    na = np.linalg.norm(a, axis=1) + 1e-12
    nb = np.linalg.norm(b, axis=1) + 1e-12
    return (a * b).sum(axis=1) / (na * nb)


def _l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=1)


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(a - b), axis=1)


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.max(np.abs(a - b), axis=1)


# -----------------------------------------------------------------------------
# Per-split processor
# -----------------------------------------------------------------------------

def process_split(
    split: str,
    in_splits: Path,
    extraction_dir: Path,
    out_dir: Path,
) -> Path | None:
    pair_pq = in_splits / f"pair_{split}.parquet"
    point_pq = in_splits / f"{split}.parquet"
    cls_pq = extraction_dir / f"point_cls_{split}.parquet"

    if not pair_pq.exists():
        print(f"[sim] skip missing {pair_pq}", flush=True)
        return None
    if not point_pq.exists():
        print(f"[sim] skip missing {point_pq}", flush=True)
        return None
    if not cls_pq.exists():
        print(f"[sim] skip missing {cls_pq} — run bert_logits --point first", flush=True)
        return None

    # Build sha -> cls_row lookup
    sha_to_id = _sha_to_id_map(point_pq)
    cls_ids, cls_mat = _load_cls_table(cls_pq)
    id_to_row = {id_: row for row, id_ in enumerate(cls_ids)}

    pair_ids, sha_a, sha_b = _pair_shas(pair_pq)

    n = len(pair_ids)
    hidden = cls_mat.shape[1]
    a_rows = np.empty((n, hidden), dtype=np.float32)
    b_rows = np.empty((n, hidden), dtype=np.float32)
    missing = 0
    for i, (s_a, s_b) in enumerate(zip(sha_a, sha_b)):
        ia = sha_to_id.get(s_a)
        ib = sha_to_id.get(s_b)
        if ia is None or ib is None or ia not in id_to_row or ib not in id_to_row:
            # Missing — zero-fill (cosine will be undefined; use 0 with a flag if needed)
            a_rows[i] = 0.0
            b_rows[i] = 0.0
            missing += 1
            continue
        a_rows[i] = cls_mat[id_to_row[ia]]
        b_rows[i] = cls_mat[id_to_row[ib]]

    if missing:
        print(f"[sim] {split}: {missing}/{n} pairs had missing CLS rows — zero-filled", flush=True)

    cos = _cosine(a_rows, b_rows).astype(np.float32)
    l2 = _l2(a_rows, b_rows).astype(np.float32)
    mad = _mean_abs_diff(a_rows, b_rows).astype(np.float32)
    mxd = _max_abs_diff(a_rows, b_rows).astype(np.float32)

    out = pa.table({
        "pair_id": pair_ids,
        "cls_cosine": cos,
        "cls_l2": l2,
        "cls_mean_abs_diff": mad,
        "cls_max_abs_diff": mxd,
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"pair_sim_{split}.parquet"
    pq.write_table(out, dst, compression="zstd")
    print(f"[sim] wrote {dst}: rows={n}", flush=True)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_splits", default="data/processed")
    ap.add_argument("--extraction_dir", default="runs/heads/extraction")
    ap.add_argument("--out_dir", default=None,
                    help="where to write pair_sim_{split}.parquet — defaults to extraction_dir")
    args = ap.parse_args()

    in_splits = Path(args.in_splits)
    ext_dir = Path(args.extraction_dir)
    out_dir = Path(args.out_dir) if args.out_dir else ext_dir

    for sp in ("train", "val", "test"):
        process_split(sp, in_splits, ext_dir, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
