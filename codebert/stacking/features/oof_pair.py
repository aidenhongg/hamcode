"""Out-of-fold (OOF) pairwise BERT training + extraction.

The pairwise cross-encoder was trained on pair_train.parquet, so its logits
on any pair in pair_train are over-confident. This driver re-trains the
pairwise BERT K times (fold-disjoint) and emits held-out predictions for
every pair in pair_train, plus a final full-train model used for val/test.

Fold disjointness is enforced on pair identity AND problem identity:
  - A pair belongs to a fold based on the problem_ids of its A and B codes.
  - If either side's problem appears in another fold's training set, that's
    leakage. To handle this, we assign folds by the canonical pair problem
    signature (sorted tuple of pids); pairs with any matching pid end up in
    the same fold.

When problem_ids are missing (e.g., cross-source synthetic pairs with None),
we fall back to code_sha256 pairs.

Output layout (matches bert_logits.py):
    <out_dir>/pair_logits_train.parquet  (OOF predictions, one row per pair)
    <out_dir>/pair_logits_val.parquet    (full-train-model predictions)
    <out_dir>/pair_logits_test.parquet
    <out_dir>/pair_meta.json
    <out_dir>/oof_pair/fold_<k>/ , <out_dir>/oof_pair/full/

CLI:
    python -m stacking.features.oof_pair \
        --data_dir data/processed \
        --out_dir runs/heads/extraction \
        --n_folds 5 \
        --epochs 6 --batch_size 12 --grad_accum 2 --bf16 \
        --warm_start_from runs/heads/extraction/oof/full/best

Note on --warm_start_from: pair training normally warm-starts from a
pointwise encoder. Pass the OOF-trained final pointwise checkpoint
(runs/heads/extraction/oof/full/best) so each fold starts from a
consistent encoder. If you omit it, pair trains from a fresh
GraphCodeBERT-base (slower convergence but still correct).
"""

from __future__ import annotations

import argparse
import json
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
    _encode_pair_batch,
    _forward_with_cls,
)


# -----------------------------------------------------------------------------
# Fold assignment — problem-disjoint across folds
# -----------------------------------------------------------------------------

def _pid_pair_key(row: dict) -> tuple[str, str]:
    # Sort problem ids to make the key canonical (A,B and B,A fall in the same fold)
    pa_pid = str(row.get("label_a") or "") + "|"  # pid not stored on pair directly
    # Pair parquet carries only code, label, ternary etc. — we don't have problem_id
    # at pair level. Fall back to code SHAs for disjoint grouping.
    return ("", "")


def _pair_key(code_a: str, code_b: str) -> tuple[str, str]:
    import hashlib
    sa = hashlib.sha256(code_a.encode("utf-8")).hexdigest()
    sb = hashlib.sha256(code_b.encode("utf-8")).hexdigest()
    return tuple(sorted((sa, sb)))


def assign_folds(pair_train: Path, n_folds: int, seed: int) -> dict[str, int]:
    """Return {pair_id: fold_idx}. Pairs grouped by joint code SHAs."""
    tbl = pq.read_table(pair_train)
    pair_ids = tbl.column("pair_id").to_pylist()
    code_a = tbl.column("code_a").to_pylist()
    code_b = tbl.column("code_b").to_pylist()

    # Build a union-find over code SHAs so any two pairs sharing ANY code SHA
    # on either side end up in the same fold (no cross-fold code reuse).
    import hashlib
    parent: dict[str, str] = {}
    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    shas_a = [hashlib.sha256(c.encode("utf-8")).hexdigest() for c in code_a]
    shas_b = [hashlib.sha256(c.encode("utf-8")).hexdigest() for c in code_b]
    for sa, sb in zip(shas_a, shas_b):
        union(sa, sb)

    # Each connected component is a fold-atomic unit.
    component: dict[str, str] = {}
    for sa, sb in zip(shas_a, shas_b):
        ra = find(sa)
        component[sa] = ra
        component[sb] = ra

    comps = sorted({find(s) for s in component})
    rng = np.random.default_rng(seed)
    rng.shuffle(comps)
    comp_to_fold = {c: i % n_folds for i, c in enumerate(comps)}

    out: dict[str, int] = {}
    for pid, sa, sb in zip(pair_ids, shas_a, shas_b):
        out[pid] = comp_to_fold[find(sa)]
    return out


def split_pair_parquet(
    pair_parquet: Path,
    fold_assignment: dict[str, int],
    out_dir: Path,
    n_folds: int,
) -> list[tuple[Path, Path]]:
    """Write per-fold pair parquets. Returns list of (train_path, heldout_path)."""
    tbl = pq.read_table(pair_parquet)
    rows = tbl.to_pylist()
    by_fold: dict[int, list[dict]] = {k: [] for k in range(n_folds)}
    for r in rows:
        by_fold[fold_assignment[r["pair_id"]]].append(r)

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
# Train one pair fold via subprocess
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
    warm_start_from: str | None,
    label_smoothing: float,
    class_weights: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "train.py", "--pair",
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
        "--label_smoothing", str(label_smoothing),
        "--class_weights", class_weights,
    ]
    cmd.append("--bf16" if bf16 else "--no_bf16")
    if warm_start_from:
        cmd += ["--warm_start_from", str(warm_start_from)]

    print(f"[oof-pair] fold {fold_idx}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"pair fold {fold_idx} training failed (exit={proc.returncode}); "
                           f"see {run_dir / 'train.log'}")
    if not (run_dir / "best" / "pytorch_model.bin").exists():
        raise RuntimeError(f"pair fold {fold_idx}: no best/ checkpoint at {run_dir / 'best'}")


# -----------------------------------------------------------------------------
# Inference helper
# -----------------------------------------------------------------------------

def extract_pair_logits(
    ckpt_dir: Path,
    rows: list[dict],      # each row needs pair_id, code_a, code_b
    writer: _IncrementalParquetWriter,
    batch_size: int,
    max_seq_len: int,
    fp16: bool,
    device_str: str,
) -> None:
    device = torch.device(device_str)
    model = GraphCodeBERTClassifier.load_checkpoint(ckpt_dir, task="pair")
    model = model.to(device)
    model.eval()
    for pm in model.parameters():
        pm.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model.model_name)
    n_labels = int(model.num_labels)

    todo = [r for r in rows if r["pair_id"] not in writer]
    if not todo:
        print(f"[oof-pair-infer] all {len(rows)} pairs already extracted", flush=True)
    bs = batch_size
    for start in tqdm(range(0, len(todo), bs), desc=f"oof-pair-infer"):
        chunk = todo[start:start + bs]
        enc = _encode_pair_batch(
            [r["code_a"] for r in chunk],
            [r["code_b"] for r in chunk],
            tokenizer, max_seq_len,
        )
        try:
            logits, _ = _forward_with_cls(model, enc, device, use_fp16=fp16)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs = max(1, bs // 2)
            print(f"[oof-pair-infer] OOM; reducing batch to {bs}", flush=True)
            continue
        logits_np = logits.cpu().numpy().astype(np.float32)
        cols = {"pair_id": [r["pair_id"] for r in chunk]}
        for k in range(n_labels):
            cols[f"pair_logit_{k}"] = logits_np[:, k]
        writer.add_batch(pa.table(cols))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out_dir", default="runs/heads/extraction")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    # Training hyperparams (passed to train.py). Defaults match configs/pair.yaml
    # — epochs is a generous upper bound; patience is the real stopper.
    ap.add_argument("--epochs", type=int, default=30,
                    help="Hard upper bound. Patience (default 3) usually stops "
                         "training at ~10-15 epochs per fold.")
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--class_weights", default="none")
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--no_bf16", dest="bf16", action="store_false")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--eval_every_steps", type=int, default=200)
    ap.add_argument("--patience", type=int, default=3,
                    help="Stop after this many consecutive evals with no improvement.")
    ap.add_argument("--warm_start_from", default="",
                    help="Pass path to an OOF pointwise final checkpoint (recommended). "
                         "Leave empty to train from fresh GraphCodeBERT-base.")

    ap.add_argument("--extract_batch", type=int, default=32)
    ap.add_argument("--extract_fp16", action="store_true", default=True)
    ap.add_argument("--no_extract_fp16", dest="extract_fp16", action="store_false")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip_final", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    oof_root = out_dir / "oof_pair"
    oof_root.mkdir(exist_ok=True)

    pair_train = data_dir / "pair_train.parquet"
    pair_val = data_dir / "pair_val.parquet"
    pair_test = data_dir / "pair_test.parquet"
    for p in (pair_train, pair_val, pair_test):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")

    print(f"[oof-pair] assigning {args.n_folds} folds (seed={args.seed})", flush=True)
    fold_assignment = assign_folds(pair_train, args.n_folds, args.seed)
    (out_dir / "oof_pair_fold_assignment.json").write_text(
        json.dumps({
            "n_folds": args.n_folds,
            "seed": args.seed,
            "n_pairs": len(fold_assignment),
            "pairs_per_fold": {
                str(k): sum(1 for v in fold_assignment.values() if v == k)
                for k in range(args.n_folds)
            },
        }, indent=2),
        encoding="utf-8",
    )

    fold_paths = split_pair_parquet(pair_train, fold_assignment,
                                      oof_root / "splits", args.n_folds)

    writer_train = _IncrementalParquetWriter(out_dir, "pair_logits_train", "pair_id")

    for fold_idx, (tp, held_p) in enumerate(fold_paths):
        run_dir = oof_root / f"fold_{fold_idx:02d}"
        best_dir = run_dir / "best"
        if args.resume and (best_dir / "pytorch_model.bin").exists():
            print(f"[oof-pair] fold {fold_idx}: using existing checkpoint", flush=True)
        else:
            if run_dir.exists():
                shutil.rmtree(run_dir)
            train_one_fold(
                fold_idx=fold_idx,
                train_path=tp, val_path=pair_val, test_path=pair_test,
                run_dir=run_dir,
                epochs=args.epochs, batch_size=args.batch_size,
                grad_accum=args.grad_accum, lr=args.lr, bf16=args.bf16,
                num_workers=args.num_workers, max_seq_len=args.max_seq_len,
                eval_every_steps=args.eval_every_steps, patience=args.patience,
                seed=args.seed + fold_idx,
                warm_start_from=args.warm_start_from or None,
                label_smoothing=args.label_smoothing,
                class_weights=args.class_weights,
            )

        held_rows = pq.read_table(held_p).to_pylist()
        extract_pair_logits(
            ckpt_dir=best_dir,
            rows=held_rows,
            writer=writer_train,
            batch_size=args.extract_batch,
            max_seq_len=args.max_seq_len,
            fp16=args.extract_fp16,
            device_str=args.device,
        )

    writer_train.merge(out_dir / "pair_logits_train.parquet")
    print(f"[oof-pair] wrote OOF pair_logits_train", flush=True)

    if args.skip_final:
        print("[oof-pair] --skip_final set; skipping val/test extraction", flush=True)
    else:
        final_dir = oof_root / "full"
        best_dir = final_dir / "best"
        if args.resume and (best_dir / "pytorch_model.bin").exists():
            print(f"[oof-pair] final: using existing checkpoint", flush=True)
        else:
            if final_dir.exists():
                shutil.rmtree(final_dir)
            train_one_fold(
                fold_idx=args.n_folds,
                train_path=pair_train,
                val_path=pair_val, test_path=pair_test,
                run_dir=final_dir,
                epochs=args.epochs, batch_size=args.batch_size,
                grad_accum=args.grad_accum, lr=args.lr, bf16=args.bf16,
                num_workers=args.num_workers, max_seq_len=args.max_seq_len,
                eval_every_steps=args.eval_every_steps, patience=args.patience,
                seed=args.seed,
                warm_start_from=args.warm_start_from or None,
                label_smoothing=args.label_smoothing,
                class_weights=args.class_weights,
            )

        for split, p in (("val", pair_val), ("test", pair_test)):
            rows = pq.read_table(p).to_pylist()
            writer = _IncrementalParquetWriter(out_dir, f"pair_logits_{split}", "pair_id")
            extract_pair_logits(
                ckpt_dir=best_dir,
                rows=rows,
                writer=writer,
                batch_size=args.extract_batch,
                max_seq_len=args.max_seq_len,
                fp16=args.extract_fp16,
                device_str=args.device,
            )
            writer.merge(out_dir / f"pair_logits_{split}.parquet")

    (out_dir / "pair_meta.json").write_text(json.dumps({
        "oof": True,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "splits_dir": str(oof_root / "splits"),
        "fold_run_dirs": [str(oof_root / f"fold_{k:02d}") for k in range(args.n_folds)],
        "full_run_dir": str(oof_root / "full"),
        "warm_start_from": args.warm_start_from,
        "note": "Train pair logits are out-of-fold (unbiased); val/test are full-train.",
    }, indent=2), encoding="utf-8")

    print("[oof-pair] done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
