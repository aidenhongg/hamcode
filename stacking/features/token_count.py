"""Extract per-snippet BPE token counts.

For each row in {train,val,test}.parquet, runs the LongCoder tokenizer
without truncation and stores the resulting token count. Output:

    <out_dir>/point_token_count_<split>.parquet
        id            : row id (matches point_logits_<split>.parquet)
        code_sha256   : sha256 of the code string
        token_count   : int — len(tokenizer(code, add_special_tokens=False).input_ids)

Token count is the *true* pre-truncation BPE length, so a snippet that
exceeds max_seq_len produces a count > max_seq_len. Downstream features
(stacking/dataset.py) divide by max_seq_len to get a normalized signal.

CLI:
    python -m stacking.features.token_count \\
        --in_splits data/processed \\
        --out_dir runs/heads/extraction
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _code_sha(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def extract_token_counts(
    in_splits: Path,
    out_dir: Path,
    model_name: str = "microsoft/longcoder-base",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for sp in ("train", "val", "test"):
        src = in_splits / f"{sp}.parquet"
        if not src.exists():
            print(f"[token_count] skip missing {src}", flush=True)
            continue

        tbl = pq.read_table(src)
        ids = tbl.column("id").to_pylist()
        codes = tbl.column("code").to_pylist()

        counts = np.zeros(len(codes), dtype=np.int64)
        shas: list[str] = []
        for i, code in enumerate(tqdm(codes, desc=f"token_count:{sp}")):
            enc = tokenizer(code, add_special_tokens=False, truncation=False)
            counts[i] = len(enc["input_ids"])
            shas.append(_code_sha(code))

        out_path = out_dir / f"point_token_count_{sp}.parquet"
        pq.write_table(
            pa.table({
                "id": ids,
                "code_sha256": shas,
                "token_count": counts,
            }),
            out_path,
            compression="zstd",
        )
        print(f"[token_count] wrote {out_path} (n={len(ids)})", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_splits", default="data/processed")
    ap.add_argument("--out_dir", default="runs/heads/extraction")
    ap.add_argument("--model_name", default="microsoft/longcoder-base")
    args = ap.parse_args()

    extract_token_counts(
        in_splits=Path(args.in_splits),
        out_dir=Path(args.out_dir),
        model_name=args.model_name,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
