"""Out-of-fold (OOF) pointwise BERT training + extraction.

Fixes the leakage problem: the default pipeline trains a single pointwise
BERT on train.parquet, so its logits on any code in train.parquet (and
therefore on any code inside pair_train.parquet) are over-confident. Head
training on those logits learns a sharp distribution that doesn't match
val/test, leading to an optimistic train score and poor generalization.

This driver:
  1. Splits train.parquet into K folds by problem_id (splits that share
     a problem leak label information). If problem_id is missing/empty,
     falls back to code_sha256 for disjointness.
  2. For each fold i in [0..K-1]:
       a) Train pointwise BERT on the union of the other K-1 folds.
          (train.py is invoked via subprocess so we don't double-import.)
       b) Predict per-code logits + CLS vectors on held-out fold i.
       c) Append those rows to the running train extraction table.
     After K iterations every train code has a logit vector from a model
     that NEVER saw its problem.
  3. Train ONE final pointwise model on the union of all K-1 folds? No:
     train on ALL of train, use it to predict val/test. (val/test are
     already held out, so the full-train model is the correct source.)

Output layout (matches bert_logits.py):
    <out_dir>/point_logits_train.parquet  (OOF predictions, one row per train code)
    <out_dir>/point_cls_train.parquet     (OOF CLS vectors)
    <out_dir>/point_logits_val.parquet    (full-train-model predictions)
    <out_dir>/point_cls_val.parquet
    <out_dir>/point_logits_test.parquet
    <out_dir>/point_cls_test.parquet
    <out_dir>/point_meta.json             (incl. oof=True flag, n_folds, fold_assignments)
    <out_dir>/oof/fold_<k>/                (per-fold run directory with train.log + best/)

Design choices:
  - We call the existing `train.py` via subprocess rather than duplicating
    its training loop. Each fold run produces its own best/ checkpoint.
  - Fold splitting is by problem_id — critical to avoid cross-fold leakage
    since the upstream split generator already uses problem_id for
    train/val/test isolation.
  - The final full-train model is trained as fold K (index n_folds), used
    only for val/test logits.

CLI:
    python -m stacking.features.oof_point \
        --data_dir data/processed \
        --out_dir runs/heads/extraction \
        --n_folds 5 \
        --epochs 8 --batch_size 16 --grad_accum 2 --bf16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model import GraphCodeBERTClassifier
from stacking.features.bert_logits import (
    _IncrementalParquetWriter,
    _encode_point_batch,
    _forward_with_cls,
)


# -----------------------------------------------------------------------------
# Fold split: problem_id-disjoint, reproducible
# -----------------------------------------------------------------------------

def _fold_key(row: dict) -> str:
    """Return a stable group key for fold assignment.

    Problem_id wins when present. Otherwise fall back to code_sha256 so at
    least same code across fold boundaries is disallowed.
    """
    pid = row.get("problem_id")
    if pid:
        return str(pid)
    return row.get("code_sha256") or ""


def assign_folds(train_parquet: Path, n_folds: int, seed: int) -> dict[str, int]:
    """Return {fold_key: fold_idx}. Stable under same (train_parquet, n_folds, seed)."""
    tbl = pq.read_table(train_parquet)
    rows = tbl.to_pylist()
    # Get unique fold keys in a deterministic order
    keys = sorted({_fold_key(r) for r in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    assignment: dict[str, int] = {}
    for i, k in enumerate(keys):
        assignment[k] = i % n_folds
    return assignment


def split_train_parquet(
    train_parquet: Path,
    fold_assignment: dict[str, int],
    out_dir: Path,
    n_folds: int,
) -> list[tuple[Path, Path]]:
    """Write per-fold parquets: fold_<k>_train.parquet (the K-1 unions) and
    fold_<k>_heldout.parquet (the one fold). Returns list of (train_path, heldout_path).
    """
    tbl = pq.read_table(train_parquet)
    rows = tbl.to_pylist()

    # Group rows by fold
    by_fold: dict[int, list[dict]] = {k: [] for k in range(n_folds)}
    for r in rows:
        by_fold[fold_assignment[_fold_key(r)]].append(r)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[tuple[Path, Path]] = []
    for k in range(n_folds):
        held = by_fold[k]
        other = [r for f in range(n_folds) if f != k for r in by_fold[f]]
        held_p = out_dir / f"fold_{k:02d}_heldout.parquet"
        other_p = out_dir / f"fold_{k:02d}_train.parquet"
        pq.write_table(pa.Table.from_pylist(held, schema=tbl.schema),
                        held_p, compression="zstd")
        pq.write_table(pa.Table.from_pylist(other, schema=tbl.schema),
                        other_p, compression="zstd")
        paths.append((other_p, held_p))
    return paths


# -----------------------------------------------------------------------------
# Training a single fold via subprocess
# -----------------------------------------------------------------------------

def train_one_fold(
    fold_idx: int,
    train_path: Path,
    val_path: Path,
    test_path: Path,
    run_dir: Path,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    bf16: bool,
    num_workers: int,
    max_seq_len: int,
    eval_every_steps: int,
    patience: int,
    seed: int,
) -> None:
    """Invoke train.py as a subprocess with fold-specific parquet paths."""
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "train.py", "--point",
        "--train_parquet", str(train_path),
        "--val_parquet", str(val_path),
        "--test_parquet", str(test_path),
        "--output_dir", str(run_dir),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--grad_accum", str(grad_accum),
        "--lr", f"{lr}",
        "--num_workers", str(num_workers),
        "--max_seq_len", str(max_seq_len),
        "--eval_every_steps", str(eval_every_steps),
        "--patience", str(patience),
        "--seed", str(seed),
    ]
    if bf16:
        cmd.append("--bf16")
    else:
        cmd.append("--no_bf16")

    print(f"[oof] fold {fold_idx}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"fold {fold_idx} training failed (exit={proc.returncode}); "
                           f"see {run_dir / 'train.log'}")
    # Verify best/ exists
    best_dir = run_dir / "best"
    if not (best_dir / "pytorch_model.bin").exists():
        raise RuntimeError(f"fold {fold_idx}: no best/ checkpoint produced at {best_dir}")


# -----------------------------------------------------------------------------
# Inference helper (shared with bert_logits.py)
# -----------------------------------------------------------------------------

def extract_logits_and_cls(
    model: GraphCodeBERTClassifier,
    tokenizer,
    codes: list[str],
    ids: list[str],
    out_dir: Path,
    split_name: str,
    batch_size: int,
    max_seq_len: int,
    device: torch.device,
    fp16: bool,
) -> None:
    """Run the frozen model over (ids, codes) and write logits + CLS parquets
    (chunked, resumable). Identical format to bert_logits.extract_point."""
    model.eval()
    for pm in model.parameters():
        pm.requires_grad_(False)
    n_labels = int(model.num_labels)
    hidden = int(model.encoder.config.hidden_size)

    logit_writer = _IncrementalParquetWriter(out_dir, f"point_logits_{split_name}", "id")
    cls_writer = _IncrementalParquetWriter(out_dir, f"point_cls_{split_name}", "id")

    todo = [(i, id_, c) for i, (id_, c) in enumerate(zip(ids, codes)) if id_ not in logit_writer]
    if not todo:
        print(f"[oof-extract] {split_name}: all {len(ids)} already extracted", flush=True)
    else:
        print(f"[oof-extract] {split_name}: {len(todo)}/{len(ids)} rows to extract", flush=True)

    bs = batch_size
    for start in tqdm(range(0, len(todo), bs), desc=f"oof-extract:{split_name}"):
        chunk = todo[start:start + bs]
        _, batch_ids, batch_codes = zip(*chunk)
        enc = _encode_point_batch(list(batch_codes), tokenizer, max_seq_len)
        try:
            logits, cls = _forward_with_cls(model, enc, device, use_fp16=fp16)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs > 1:
                bs = max(1, bs // 2)
                print(f"[oof-extract] OOM; reducing batch to {bs}", flush=True)
                continue
            raise
        logits_np = logits.cpu().numpy().astype(np.float32)
        cls_np = cls.cpu().numpy().astype(np.float32)
        cols = {"id": list(batch_ids)}
        for k in range(n_labels):
            cols[f"point_logit_{k}"] = logits_np[:, k]
        logit_writer.add_batch(pa.table(cols))
        cls_cols = {"id": list(batch_ids)}
        for d in range(hidden):
            cls_cols[f"cls_{d}"] = cls_np[:, d]
        cls_writer.add_batch(pa.table(cls_cols))

    logit_writer.merge(out_dir / f"point_logits_{split_name}.parquet")
    cls_writer.merge(out_dir / f"point_cls_{split_name}.parquet")


def extract_from_checkpoint(
    ckpt_dir: Path,
    codes: list[str],
    ids: list[str],
    out_dir: Path,
    split_name: str,
    batch_size: int,
    max_seq_len: int,
    fp16: bool,
    device_str: str,
) -> None:
    device = torch.device(device_str)
    model = GraphCodeBERTClassifier.load_checkpoint(ckpt_dir, task="point")
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model.model_name)
    extract_logits_and_cls(
        model=model, tokenizer=tokenizer,
        codes=codes, ids=ids,
        out_dir=out_dir, split_name=split_name,
        batch_size=batch_size, max_seq_len=max_seq_len,
        device=device, fp16=fp16,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default="data/processed",
                    help="Dir with train.parquet + val.parquet + test.parquet.")
    ap.add_argument("--out_dir", default="runs/heads/extraction",
                    help="Where to write point_logits_* and point_cls_* parquets.")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    # Training hyperparams (passed to train.py). Generous upper bound; patience
    # is the real stopper. Earlier 8-epoch cap clearly under-trained the pair
    # model (see SUMMARY.md: v2 underperformed v1), so we loosen both.
    ap.add_argument("--epochs", type=int, default=40,
                    help="Hard upper bound per fold. Patience typically stops "
                         "training at ~15-20 epochs on the full train set.")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--no_bf16", dest="bf16", action="store_false")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--eval_every_steps", type=int, default=100)
    ap.add_argument("--patience", type=int, default=3,
                    help="Stop after this many consecutive evals with no improvement.")

    # Extraction hyperparams
    ap.add_argument("--extract_batch", type=int, default=32,
                    help="Batch size during per-fold inference extraction.")
    ap.add_argument("--extract_fp16", action="store_true", default=True,
                    help="fp16 autocast during extraction. bf16 not needed for inference.")
    ap.add_argument("--no_extract_fp16", dest="extract_fp16", action="store_false")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Safety: skip fold training if best/ already exists (resume)
    ap.add_argument("--resume", action="store_true",
                    help="If a fold's best/ already exists, skip retraining it.")
    ap.add_argument("--skip_final", action="store_true",
                    help="Skip the final full-train model + val/test extraction.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    oof_root = out_dir / "oof"
    oof_root.mkdir(exist_ok=True)

    train_pq = data_dir / "train.parquet"
    val_pq = data_dir / "val.parquet"
    test_pq = data_dir / "test.parquet"
    for p in (train_pq, val_pq, test_pq):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")

    print(f"[oof] assigning {args.n_folds} folds (seed={args.seed})", flush=True)
    fold_assignment = assign_folds(train_pq, args.n_folds, args.seed)
    (out_dir / "oof_fold_assignment.json").write_text(
        json.dumps({"n_folds": args.n_folds, "seed": args.seed,
                    "assignment": fold_assignment}, indent=2),
        encoding="utf-8",
    )

    # 1) Build per-fold parquet pairs
    fold_paths = split_train_parquet(train_pq, fold_assignment, oof_root / "splits",
                                       args.n_folds)

    # 2) For each fold: train + extract heldout predictions into running train tables
    logits_writer = _IncrementalParquetWriter(out_dir, "point_logits_train", "id")
    cls_writer = _IncrementalParquetWriter(out_dir, "point_cls_train", "id")

    for fold_idx, (tp, held_p) in enumerate(fold_paths):
        run_dir = oof_root / f"fold_{fold_idx:02d}"
        best_dir = run_dir / "best"
        if args.resume and (best_dir / "pytorch_model.bin").exists():
            print(f"[oof] fold {fold_idx}: using existing checkpoint at {best_dir}", flush=True)
        else:
            # Wipe stale run dir so we never mix epochs across invocations
            if run_dir.exists():
                shutil.rmtree(run_dir)
            train_one_fold(
                fold_idx=fold_idx,
                train_path=tp, val_path=val_pq, test_path=test_pq,
                run_dir=run_dir,
                epochs=args.epochs, batch_size=args.batch_size,
                grad_accum=args.grad_accum, lr=args.lr, bf16=args.bf16,
                num_workers=args.num_workers, max_seq_len=args.max_seq_len,
                eval_every_steps=args.eval_every_steps, patience=args.patience,
                seed=args.seed + fold_idx,
            )

        # Inference on held-out fold using this fold's best checkpoint.
        held_tbl = pq.read_table(held_p)
        held_ids = held_tbl.column("id").to_pylist()
        held_codes = held_tbl.column("code").to_pylist()
        # Reuse the same writers so all folds land in one merged parquet.
        device = torch.device(args.device)
        model = GraphCodeBERTClassifier.load_checkpoint(best_dir, task="point")
        model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model.model_name)
        model.eval()
        for pm in model.parameters():
            pm.requires_grad_(False)
        n_labels = int(model.num_labels)
        hidden = int(model.encoder.config.hidden_size)

        # Manual loop so we can write into the shared per-split writers.
        todo = [(id_, c) for id_, c in zip(held_ids, held_codes)
                 if id_ not in logits_writer]
        bs = args.extract_batch
        for start in tqdm(range(0, len(todo), bs),
                           desc=f"oof-infer:fold_{fold_idx}"):
            chunk = todo[start:start + bs]
            batch_ids = [c[0] for c in chunk]
            batch_codes = [c[1] for c in chunk]
            enc = _encode_point_batch(batch_codes, tokenizer, args.max_seq_len)
            try:
                logits, cls = _forward_with_cls(
                    model, enc, device, use_fp16=args.extract_fp16,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                print(f"[oof-infer] OOM; reducing batch to {bs}", flush=True)
                continue
            logits_np = logits.cpu().numpy().astype(np.float32)
            cls_np = cls.cpu().numpy().astype(np.float32)
            cols = {"id": batch_ids}
            for k in range(n_labels):
                cols[f"point_logit_{k}"] = logits_np[:, k]
            logits_writer.add_batch(pa.table(cols))
            cls_cols = {"id": batch_ids}
            for d in range(hidden):
                cls_cols[f"cls_{d}"] = cls_np[:, d]
            cls_writer.add_batch(pa.table(cls_cols))

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logits_writer.merge(out_dir / "point_logits_train.parquet")
    cls_writer.merge(out_dir / "point_cls_train.parquet")
    print(f"[oof] wrote OOF train predictions: {len(fold_paths)} folds combined", flush=True)

    # 3) Train one final model on ALL train data, extract val/test predictions.
    if args.skip_final:
        print("[oof] --skip_final set; leaving val/test extraction to caller", flush=True)
    else:
        final_dir = oof_root / "full"
        best_dir = final_dir / "best"
        if args.resume and (best_dir / "pytorch_model.bin").exists():
            print(f"[oof] final: using existing checkpoint at {best_dir}", flush=True)
        else:
            if final_dir.exists():
                shutil.rmtree(final_dir)
            train_one_fold(
                fold_idx=args.n_folds,   # informational index only
                train_path=train_pq,     # full train
                val_path=val_pq, test_path=test_pq,
                run_dir=final_dir,
                epochs=args.epochs, batch_size=args.batch_size,
                grad_accum=args.grad_accum, lr=args.lr, bf16=args.bf16,
                num_workers=args.num_workers, max_seq_len=args.max_seq_len,
                eval_every_steps=args.eval_every_steps, patience=args.patience,
                seed=args.seed,
            )

        for split, pq_path in (("val", val_pq), ("test", test_pq)):
            tbl = pq.read_table(pq_path)
            ids = tbl.column("id").to_pylist()
            codes = tbl.column("code").to_pylist()
            extract_from_checkpoint(
                ckpt_dir=best_dir,
                codes=codes, ids=ids,
                out_dir=out_dir, split_name=split,
                batch_size=args.extract_batch,
                max_seq_len=args.max_seq_len,
                fp16=args.extract_fp16,
                device_str=args.device,
            )

    # Metadata
    (out_dir / "point_meta.json").write_text(json.dumps({
        "oof": True,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "splits_dir": str(oof_root / "splits"),
        "fold_run_dirs": [str(oof_root / f"fold_{k:02d}") for k in range(args.n_folds)],
        "full_run_dir": str(oof_root / "full"),
        "note": "Train logits are out-of-fold (unbiased); val/test logits are from the full-train model.",
    }, indent=2), encoding="utf-8")

    print("[oof] done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
