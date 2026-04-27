"""K-fold out-of-fold (OOF) per-language LoRA logit extraction.

The Phase-B LoRA bundle (`runs/lora-{ts}/`) trains LoRA[L] on language-L's full
train.parquet. Re-running that LoRA on the same train rows then produces over-
confident logits because the model has effectively memorized chunks of them.
The binary stacking head trained on those logits learns to over-weight the
sharpness of the encoder output, hurts on val/test where logits are honest.

This driver fixes the leak. For each language L:

  1. Assign train rows to K folds, problem_id-disjoint within L.
  2. For each fold k in [0..K-1]:
     a. Train LoRA[L, fold k] on the K-1-fold union (subprocess call to
        lora_train.py).
     b. Extract fold-k held-out logits + pooled vectors with that LoRA.
  3. Concatenate held-out predictions across folds -> point_logits_train.parquet.
     Every train row's logits come from a LoRA that NEVER saw that row's problem.
  4. For val/test: load the regular Phase-B LoRA from `--full_lora_root` and
     extract logits. Those LoRAs were trained on full L-train but never saw
     val/test (problem-id stratified split -> no leakage).

Output layout (matches stacking/features/extract_lora_features.py):

    <out_dir>/point_logits_train.parquet          (OOF, one row per train code)
    <out_dir>/point_cls_train.parquet
    <out_dir>/point_logits_val.parquet
    <out_dir>/point_cls_val.parquet
    <out_dir>/point_logits_test.parquet
    <out_dir>/point_cls_test.parquet
    <out_dir>/point_meta.json                     (oof=True + fold assignments)
    <out_dir>/oof_lora/{lang}/fold_{k}/
        data/{train,val,test}.parquet             (per-fold input to lora_train.py)
        run/{lang}/                                (saved LoRA bundle for the fold)
        heldout.parquet                            (the fold's left-out rows)

Cost on a 4090: ~30 min per LoRA fold * K folds * up-to-12 languages.
At K=3 / 12 langs: ~18 hours (default). K=5 ~= 30 hours; use --n_folds 5
if you want tighter OOF estimates and have the budget. --languages can
restrict to a subset for faster iteration.

CLI:
    python -m stacking.features.oof_lora \\
        --base_run runs/multi-fullft-{ts}/best \\
        --full_lora_root runs/lora-{ts}/ \\
        --in_splits data/processed \\
        --out_dir runs/heads/extraction \\
        --n_folds 3
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.schemas import LANGUAGES, LANG_SET
from stacking.features.bert_logits import _IncrementalParquetWriter


# Lazy imports of the LoRA runner helpers -- they pull in transformers/torch
# which is fine but we delay until main() so --help is fast.
def _import_extract_helpers():
    from stacking.features.extract_lora_features import (
        _load_backbone_with_lora, _forward,
    )
    return _load_backbone_with_lora, _forward


# ---------------------------------------------------------------------------
# Fold assignment + per-fold parquet construction
# ---------------------------------------------------------------------------

def _fold_key(row: dict) -> str:
    """Stable group key for fold assignment.

    Problem_id wins when present; otherwise we fall back to code_sha256 so at
    least same code can't span folds.
    """
    pid = row.get("problem_id")
    if pid:
        return str(pid)
    return row.get("code_sha256") or row.get("id") or ""


def assign_folds_per_language(
    rows: list[dict], n_folds: int, seed: int,
) -> dict[str, int]:
    """Return {fold_key: fold_idx}. Stable under same (rows, n_folds, seed).

    Deterministically shuffles the *unique* fold keys so two rows that share a
    problem_id always land in the same fold.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    keys = sorted({_fold_key(r) for r in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    return {k: i % n_folds for i, k in enumerate(keys)}


def write_per_fold_parquets(
    lang_train_rows: list[dict],
    lang_val_table: pa.Table,
    lang_test_table: pa.Table,
    fold_assignment: dict[str, int],
    n_folds: int,
    schema: pa.Schema,
    lang_root: Path,
) -> list[tuple[Path, Path, Path]]:
    """Build the on-disk per-fold layout.

    Returns: [(fold_dir, fold_data_dir, heldout_pq), ...]
    fold_data_dir contains train.parquet (K-1 folds), val.parquet (full lang
    val), test.parquet (full lang test) -- that's what lora_train.py expects
    under --data_dir.
    """
    by_fold: dict[int, list[dict]] = {k: [] for k in range(n_folds)}
    for r in lang_train_rows:
        fk = fold_assignment.get(_fold_key(r))
        if fk is None:
            continue
        by_fold[fk].append(r)

    paths: list[tuple[Path, Path, Path]] = []
    for k in range(n_folds):
        held = by_fold[k]
        other = [r for f in range(n_folds) if f != k for r in by_fold[f]]
        fold_dir = lang_root / f"fold_{k:02d}"
        data_dir = fold_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        pq.write_table(
            pa.Table.from_pylist(other, schema=schema),
            data_dir / "train.parquet", compression="zstd",
        )
        pq.write_table(lang_val_table, data_dir / "val.parquet", compression="zstd")
        pq.write_table(lang_test_table, data_dir / "test.parquet", compression="zstd")

        held_pq = fold_dir / "heldout.parquet"
        pq.write_table(
            pa.Table.from_pylist(held, schema=schema),
            held_pq, compression="zstd",
        )
        paths.append((fold_dir, data_dir, held_pq))
    return paths


def filter_table_by_lang(table: pa.Table, language: str) -> pa.Table:
    if "language" not in table.column_names:
        raise RuntimeError(
            f"table has no `language` column; re-run pipeline 09. cols={table.column_names}"
        )
    mask = pc.equal(table["language"], language)
    return table.filter(mask)


# ---------------------------------------------------------------------------
# Per-fold LoRA training (subprocess to lora_train.py)
# ---------------------------------------------------------------------------

def _adapter_present(run_dir: Path, language: str) -> bool:
    return (run_dir / language / "adapter_model.safetensors").exists()


def train_one_fold_lora(
    base_run: Path, language: str,
    fold_data_dir: Path, fold_run_dir: Path,
    epochs: int, batch_size: int, grad_accum: int, lr: float,
    num_workers: int, seed: int, bf16: bool,
) -> Path:
    """Invoke lora_train.py via subprocess. Returns path to the saved LoRA dir."""
    cmd = [
        sys.executable, "lora_train.py",
        "--base_run", str(base_run),
        "--language", language,
        "--data_dir", str(fold_data_dir),
        "--output_root", str(fold_run_dir),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--grad_accum", str(grad_accum),
        "--lr", f"{lr}",
        "--num_workers", str(num_workers),
        "--seed", str(seed),
    ]
    if bf16:
        cmd.append("--bf16")
    else:
        cmd.append("--no_bf16")
    print(f"[oof-lora] $ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"LoRA fold training failed (exit={proc.returncode}); see "
            f"{fold_run_dir}/{language}/train.log"
        )
    bundle = fold_run_dir / language
    if not (bundle / "adapter_model.safetensors").exists():
        raise RuntimeError(f"no adapter_model.safetensors at {bundle} after training")
    return bundle


# ---------------------------------------------------------------------------
# Logit + pooled-vector extraction for an arbitrary row subset
# ---------------------------------------------------------------------------

def extract_logits_for_rows(
    base_run: Path, lora_dir: Path, language: str,
    ids: list[str], codes: list[str],
    logits_writer: _IncrementalParquetWriter,
    cls_writer: _IncrementalParquetWriter,
    max_seq_len: int, bridge_stride: int, batch: int,
    device: torch.device, use_amp: bool,
) -> None:
    """Run the LoRA-adapted backbone over (ids, codes) in `language`, writing
    logits + pooled vectors via the resumable parquet writers.

    Skips ids that the writer already has (resume). Halves batch on OOM.
    """
    if not ids:
        return
    todo = [(i, c) for i, c in zip(ids, codes) if i not in logits_writer]
    if not todo:
        print(f"[oof-lora]   ({language}) all {len(ids)} ids already extracted",
              flush=True)
        return
    print(f"[oof-lora]   ({language}) {len(todo)}/{len(ids)} rows to extract",
          flush=True)

    _load_backbone_with_lora, _forward = _import_extract_helpers()
    from transformers import AutoTokenizer

    pmodel = _load_backbone_with_lora(base_run, lora_dir, device)
    tokenizer = AutoTokenizer.from_pretrained(pmodel.base_model.model.model_name)
    bs = batch

    try:
        for start in tqdm(range(0, len(todo), bs),
                           desc=f"oof-lora extract {language}",
                           dynamic_ncols=True, mininterval=1.0):
            chunk = todo[start:start + bs]
            b_ids = [c[0] for c in chunk]
            b_codes = [c[1] for c in chunk]
            try:
                logits, pooled = _forward(
                    pmodel, tokenizer, b_codes, language,
                    max_seq_len, bridge_stride, device, use_amp,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs <= 1:
                    raise
                bs = max(1, bs // 2)
                print(f"[oof-lora] OOM; reducing batch to {bs}", flush=True)
                continue
            l_arr = logits.cpu().numpy().astype(np.float32)
            p_arr = pooled.cpu().numpy().astype(np.float32)
            cols: dict[str, list | np.ndarray] = {"id": b_ids}
            for k in range(l_arr.shape[1]):
                cols[f"point_logit_{k}"] = l_arr[:, k]
            logits_writer.add_batch(pa.table(cols))
            ccols: dict[str, list | np.ndarray] = {"id": b_ids}
            for d in range(p_arr.shape[1]):
                ccols[f"cls_{d}"] = p_arr[:, d]
            cls_writer.add_batch(pa.table(ccols))
    finally:
        del pmodel
        if device.type == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base_run", required=True,
                    help="Phase-A best/ checkpoint directory")
    ap.add_argument("--full_lora_root", required=True,
                    help="Phase-B bundle root (used for val/test extraction). "
                         "Must contain {lang}/adapter_model.safetensors per language.")
    ap.add_argument("--in_splits", default="data/processed",
                    help="Dir with train/val/test.parquet from pipeline/09")
    ap.add_argument("--out_dir", default="runs/heads/extraction",
                    help="Destination for point_{logits,cls}_{split}.parquet")
    ap.add_argument("--n_folds", type=int, default=3,
                    help="K for K-fold OOF (default 3 for ~18h on 4090; "
                         "use 5 for tighter estimates at ~30h)")
    ap.add_argument("--seed", type=int, default=42)

    # Per-fold lora_train.py hyperparams
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--no_bf16", dest="bf16", action="store_false")

    # Extraction hyperparams
    ap.add_argument("--batch", type=int, default=8, help="extraction batch size")
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--bridge_stride", type=int, default=128)
    ap.add_argument("--use_amp", action="store_true", default=True)
    ap.add_argument("--no_amp", dest="use_amp", action="store_false")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Scoping + resume
    ap.add_argument("--languages", nargs="+", default=None,
                    help="Restrict OOF to a subset (default: all 12)")
    ap.add_argument("--min_problems_for_oof", type=int, default=20,
                    help="If a language has fewer unique problem_ids than this "
                         "(per fold scaled), skip OOF for that language and use "
                         "the full LoRA from --full_lora_root for its train rows.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip per-fold training when an adapter is already saved")
    args = ap.parse_args()

    base_run = Path(args.base_run)
    full_lora_root = Path(args.full_lora_root)
    in_splits = Path(args.in_splits)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    oof_root = out_dir / "oof_lora"; oof_root.mkdir(exist_ok=True)

    if not (base_run / "codebert_meta.json").exists():
        raise FileNotFoundError(f"missing Phase-A best/ at {base_run}")
    if not full_lora_root.exists():
        raise FileNotFoundError(f"missing full_lora_root at {full_lora_root}")

    train_pq = in_splits / "train.parquet"
    val_pq = in_splits / "val.parquet"
    test_pq = in_splits / "test.parquet"
    for p in (train_pq, val_pq, test_pq):
        if not p.exists():
            raise FileNotFoundError(p)

    languages: list[str] = args.languages if args.languages else list(LANGUAGES)
    for lang in languages:
        if lang not in LANG_SET:
            raise ValueError(f"unknown language: {lang}")

    device = torch.device(args.device)
    print(f"[oof-lora] device={device} n_folds={args.n_folds} "
          f"languages={languages}", flush=True)

    # Read full splits once
    train_table_full = pq.read_table(train_pq)
    val_table_full = pq.read_table(val_pq)
    test_table_full = pq.read_table(test_pq)
    schema = train_table_full.schema

    # ---------- Train OOF ----------
    train_logits_writer = _IncrementalParquetWriter(out_dir, "point_logits_train", "id")
    train_cls_writer = _IncrementalParquetWriter(out_dir, "point_cls_train", "id")

    fold_assignments_log: dict[str, dict] = {}
    languages_skipped_oof: list[str] = []

    for lang in languages:
        l_train_table = filter_table_by_lang(train_table_full, lang)
        if l_train_table.num_rows == 0:
            print(f"[oof-lora] {lang}: 0 train rows, skipping", flush=True)
            continue
        l_val_table = filter_table_by_lang(val_table_full, lang)
        l_test_table = filter_table_by_lang(test_table_full, lang)
        l_train_rows = l_train_table.to_pylist()
        unique_pids = {_fold_key(r) for r in l_train_rows}

        # Sanity: skip OOF if we can't make K balanced folds.
        if len(unique_pids) < args.n_folds * args.min_problems_for_oof / args.n_folds:
            print(f"[oof-lora] {lang}: only {len(unique_pids)} unique problem_ids "
                  f"(< {args.min_problems_for_oof}); skipping OOF -- using "
                  f"full LoRA from {full_lora_root}/{lang} for its train rows",
                  flush=True)
            languages_skipped_oof.append(lang)
            full_lora_dir = full_lora_root / lang
            if not (full_lora_dir / "adapter_model.safetensors").exists():
                print(f"[oof-lora] WARN: full LoRA missing for {lang}; "
                      f"train rows for this lang will be omitted from output", flush=True)
                continue
            ids = l_train_table.column("id").to_pylist()
            codes = l_train_table.column("code").to_pylist()
            extract_logits_for_rows(
                base_run, full_lora_dir, lang, ids, codes,
                train_logits_writer, train_cls_writer,
                args.max_seq_len, args.bridge_stride, args.batch,
                device, args.use_amp,
            )
            continue

        l_assign = assign_folds_per_language(l_train_rows, args.n_folds, args.seed)
        fold_assignments_log[lang] = {
            "n_unique_pids": len(unique_pids),
            "n_folds": args.n_folds,
            "fold_sizes": [
                sum(1 for _, fk in l_assign.items() if fk == k)
                for k in range(args.n_folds)
            ],
        }

        lang_root = oof_root / lang
        lang_root.mkdir(exist_ok=True)
        (lang_root / "fold_assignment.json").write_text(
            json.dumps({"fold_assignment": l_assign,
                        "n_folds": args.n_folds, "seed": args.seed},
                       indent=2),
            encoding="utf-8",
        )

        fold_paths = write_per_fold_parquets(
            l_train_rows, l_val_table, l_test_table,
            l_assign, args.n_folds, schema, lang_root,
        )

        for k, (fold_dir, fold_data_dir, heldout_pq) in enumerate(fold_paths):
            run_dir = fold_dir / "run"
            already_trained = args.resume and _adapter_present(run_dir, lang)
            if already_trained:
                print(f"[oof-lora] {lang}/fold_{k:02d}: adapter already present, "
                      f"skipping training", flush=True)
            else:
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                train_one_fold_lora(
                    base_run=base_run, language=lang,
                    fold_data_dir=fold_data_dir, fold_run_dir=run_dir,
                    epochs=args.epochs, batch_size=args.batch_size,
                    grad_accum=args.grad_accum, lr=args.lr,
                    num_workers=args.num_workers,
                    seed=args.seed + k, bf16=args.bf16,
                )

            # Extract held-out logits for this fold using its LoRA.
            held_table = pq.read_table(heldout_pq)
            ids = held_table.column("id").to_pylist()
            codes = held_table.column("code").to_pylist()
            extract_logits_for_rows(
                base_run, run_dir / lang, lang, ids, codes,
                train_logits_writer, train_cls_writer,
                args.max_seq_len, args.bridge_stride, args.batch,
                device, args.use_amp,
            )

    train_logits_writer.merge(out_dir / "point_logits_train.parquet")
    train_cls_writer.merge(out_dir / "point_cls_train.parquet")
    print(f"[oof-lora] train OOF complete; wrote {out_dir}/point_{{logits,cls}}_train.parquet",
          flush=True)

    # ---------- Val/test extraction with full Phase-B LoRAs ----------
    for split_name, split_table in (("val", val_table_full), ("test", test_table_full)):
        logits_w = _IncrementalParquetWriter(out_dir, f"point_logits_{split_name}", "id")
        cls_w = _IncrementalParquetWriter(out_dir, f"point_cls_{split_name}", "id")
        for lang in languages:
            l_table = filter_table_by_lang(split_table, lang)
            if l_table.num_rows == 0:
                continue
            full_lora_dir = full_lora_root / lang
            if not (full_lora_dir / "adapter_model.safetensors").exists():
                print(f"[oof-lora] skip {split_name}/{lang}: missing "
                      f"{full_lora_dir}/adapter_model.safetensors", flush=True)
                continue
            ids = l_table.column("id").to_pylist()
            codes = l_table.column("code").to_pylist()
            extract_logits_for_rows(
                base_run, full_lora_dir, lang, ids, codes,
                logits_w, cls_w,
                args.max_seq_len, args.bridge_stride, args.batch,
                device, args.use_amp,
            )
        logits_w.merge(out_dir / f"point_logits_{split_name}.parquet")
        cls_w.merge(out_dir / f"point_cls_{split_name}.parquet")
        print(f"[oof-lora] {split_name} extraction complete", flush=True)

    # ---------- Meta ----------
    (out_dir / "point_meta.json").write_text(json.dumps({
        "oof": True,
        "source": "lora_per_language_kfold",
        "n_folds": args.n_folds,
        "seed": args.seed,
        "base_run": str(base_run),
        "full_lora_root": str(full_lora_root),
        "languages": languages,
        "languages_skipped_oof": languages_skipped_oof,
        "fold_assignments": fold_assignments_log,
        "max_seq_len": args.max_seq_len,
        "bridge_stride": args.bridge_stride,
        "note": ("Train logits are out-of-fold per language: each train row's "
                 "logits come from a LoRA[L, fold k] that never saw that row's "
                 "problem_id. Val/test logits come from full-train LoRAs at "
                 "full_lora_root. languages_skipped_oof have too few problem_ids "
                 "for K-fold and reuse the full-train LoRA for their train rows."),
    }, indent=2), encoding="utf-8")
    print("[oof-lora] done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
