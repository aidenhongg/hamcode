"""Deterministic train/val/test splitting that preserves problem_id integrity.

Same problem must never appear in multiple splits — otherwise the model can
memorize per-problem idioms during training and cheat at test time. For rows
without a problem_id, we fall back to splitting by code_sha256.
"""

from __future__ import annotations

import hashlib

import pandas as pd


def _bucket(key: str, mod: int = 10_000) -> int:
    return int(hashlib.sha256(key.encode("utf-8", errors="replace")).hexdigest(), 16) % mod


def assign_splits(df: pd.DataFrame, train_ratio: float = 0.8,
                  val_ratio: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """Return df with a new/overwritten 'split' column of train|val|test.

    Grouping key = problem_id if present, else code_sha256. A deterministic hash
    of (seed, key) decides the bucket so runs are reproducible.
    """
    assert 0 < train_ratio < 1 and 0 <= val_ratio < 1 and train_ratio + val_ratio < 1
    mod = 10_000
    train_cut = int(train_ratio * mod)
    val_cut = train_cut + int(val_ratio * mod)

    def pick(row: pd.Series) -> str:
        key = row.get("problem_id") or row.get("code_sha256") or row.get("id") or ""
        bucket = _bucket(f"{seed}::{key}", mod)
        if bucket < train_cut:
            return "train"
        if bucket < val_cut:
            return "val"
        return "test"

    out = df.copy()
    out["split"] = out.apply(pick, axis=1)
    return out


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Rows = labels, columns = splits, values = counts."""
    if "split" not in df.columns or "label" not in df.columns:
        raise ValueError("df needs 'label' and 'split' columns")
    return (
        df.groupby(["label", "split"]).size().unstack(fill_value=0)
    )
