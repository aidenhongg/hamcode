"""Join AST + pointwise BERT logit + semantic-similarity features into a
head-ready matrix.

Filters to the B>=A subset (drops ternary=='B_faster'), derives the binary
label y (1 for A_faster, 0 for same), builds the feature matrix from
pointwise BERT logits + AST diffs + CLS similarity, and fits a z-score
scaler on the TRAIN split only.

The scaler discipline is enforced: the fitted scaler object is saved to
disk and reused on val/test. Boolean AST features (recursion_present etc.)
bypass the scaler.

Variant: only `v1` is supported. After dropping pairwise BERT fine-tuning
the v2/v3 inputs (pair logits) no longer exist; v1 = "A pt + B pt": pointwise
logits for A and B concatenated with diff/|diff|.
"""

from __future__ import annotations

import json
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc
from sklearn.preprocessing import StandardScaler

from .features.ast_features import FEATURE_KIND, FEATURE_NAMES, diff_columns


Variant = Literal["v1"]
VARIANT_DESCRIPTIONS = {
    "v1": "A + B pointwise logits (+ diff, |diff|)",
}


# -----------------------------------------------------------------------------
# Filter + label construction
# -----------------------------------------------------------------------------

def filter_b_ge_a(pair_tbl) -> "pa.Table":
    """Drop B_faster rows. Returns filtered pyarrow Table."""
    kept = pair_tbl.filter(pc.not_equal(pair_tbl["ternary"], "B_faster"))
    remaining = set(kept.column("ternary").to_pylist())
    assert "B_faster" not in remaining, "filter left B_faster rows"
    return kept


def binary_labels(pair_tbl) -> np.ndarray:
    """0 = same, 1 = A_faster. Assumes filter_b_ge_a already applied."""
    t = pair_tbl.column("ternary").to_pylist()
    y = np.asarray([1 if v == "A_faster" else 0 for v in t], dtype=np.int64)
    return y


def log_imbalance(y: np.ndarray, split: str) -> None:
    counts = Counter(y.tolist())
    n0 = counts.get(0, 0); n1 = counts.get(1, 0)
    ratio = n1 / max(1, n0)
    print(f"[dataset] {split}: n0(same)={n0} n1(A_faster)={n1} ratio={ratio:.2f}", flush=True)
    if ratio > 2.5 or (n0 > 0 and ratio < 0.4):
        warnings.warn(f"imbalance ratio {ratio:.2f} in {split} — class weights recommended")


# -----------------------------------------------------------------------------
# Column selection per variant
# -----------------------------------------------------------------------------

def _point_logit_columns(n_labels: int = 11) -> list[str]:
    return [f"point_logit_{k}" for k in range(n_labels)]


# -----------------------------------------------------------------------------
# Joined feature matrix
# -----------------------------------------------------------------------------

@dataclass
class FeatureMatrix:
    pair_ids: list[str]
    X: np.ndarray               # (N, D) float32
    y: np.ndarray               # (N,) int64
    columns: list[str]          # names, length D
    scaled_mask: np.ndarray     # (D,) bool — True for columns that went through scaler
    variant: str
    languages: list[str] = field(default_factory=list)  # (N,) per-row language

    def num_features(self) -> int:
        return self.X.shape[1]


def _build_ast_pair_block(tbl) -> tuple[np.ndarray, list[str]]:
    """Load pair-level differenced AST features from ast_pair_{split}.parquet."""
    cols = diff_columns()
    mat = np.stack(
        [np.asarray(tbl.column(c).to_pylist(), dtype=np.float32) for c in cols],
        axis=1,
    )
    return mat, cols


def _load_or_empty(path: Path, name: str) -> "pq.Table | None":
    if not path.exists():
        print(f"[dataset] {name} missing: {path}", flush=True)
        return None
    return pq.read_table(path)


def _join_point_logits(
    pair_tbl,
    point_pq: Path,
    point_logits_tbl,
) -> tuple[np.ndarray, list[str]]:
    """For each pair row, look up pointwise logits of A and B by code_sha256.

    Returns a (N, 4*n_labels) matrix: [logits_A; logits_B; diff; |diff|]
    """
    import hashlib

    # Build sha -> id via pointwise parquet, then id -> row in logits
    ppt = pq.read_table(point_pq)
    shas = ppt.column("code_sha256").to_pylist()
    ids = ppt.column("id").to_pylist()
    sha_to_id = {s: i for s, i in zip(shas, ids)}

    logit_ids = point_logits_tbl.column("id").to_pylist()
    id_to_row = {id_: i for i, id_ in enumerate(logit_ids)}

    logit_cols = _point_logit_columns()
    n_labels = len(logit_cols)
    logit_raw = np.stack(
        [np.asarray(point_logits_tbl.column(c).to_pylist(), dtype=np.float32)
         for c in logit_cols],
        axis=1,
    )

    code_a = pair_tbl.column("code_a").to_pylist()
    code_b = pair_tbl.column("code_b").to_pylist()
    n = len(code_a)

    a_mat = np.zeros((n, n_labels), dtype=np.float32)
    b_mat = np.zeros((n, n_labels), dtype=np.float32)
    missing = 0
    for i, (ca, cb) in enumerate(zip(code_a, code_b)):
        sa = hashlib.sha256(ca.encode("utf-8")).hexdigest()
        sb = hashlib.sha256(cb.encode("utf-8")).hexdigest()
        ia = sha_to_id.get(sa)
        ib = sha_to_id.get(sb)
        if ia is None or ib is None or ia not in id_to_row or ib not in id_to_row:
            missing += 1
            continue
        a_mat[i] = logit_raw[id_to_row[ia]]
        b_mat[i] = logit_raw[id_to_row[ib]]

    if missing:
        print(f"[dataset] point logits: {missing}/{n} pair sides unmapped, zero-filled", flush=True)

    diff = b_mat - a_mat
    absd = np.abs(diff)
    X = np.concatenate([a_mat, b_mat, diff, absd], axis=1)
    cols = (
        [f"point_A_logit_{k}" for k in range(n_labels)]
        + [f"point_B_logit_{k}" for k in range(n_labels)]
        + [f"point_diff_logit_{k}" for k in range(n_labels)]
        + [f"point_abs_diff_logit_{k}" for k in range(n_labels)]
    )
    return X, cols


def _join_similarity(pair_ids: list[str], sim_tbl) -> tuple[np.ndarray, list[str]]:
    cols = ["cls_cosine", "cls_l2", "cls_mean_abs_diff", "cls_max_abs_diff"]
    available = [c for c in cols if c in sim_tbl.schema.names]
    if not available:
        return np.zeros((len(pair_ids), 0), dtype=np.float32), []
    id_to_row = {id_: i for i, id_ in enumerate(sim_tbl.column("pair_id").to_pylist())}
    mat = np.zeros((len(pair_ids), len(available)), dtype=np.float32)
    raw = {c: np.asarray(sim_tbl.column(c).to_pylist(), dtype=np.float32) for c in available}
    missing = 0
    for i, pid in enumerate(pair_ids):
        r = id_to_row.get(pid)
        if r is None:
            missing += 1
            continue
        for j, c in enumerate(available):
            mat[i, j] = raw[c][r]
    if missing:
        print(f"[dataset] sim: {missing}/{len(pair_ids)} missing, zero-filled", flush=True)
    return mat, available


# -----------------------------------------------------------------------------
# Public build function
# -----------------------------------------------------------------------------

def build_feature_matrix(
    split: str,
    variant: Variant,
    in_splits: Path,
    extraction_dir: Path,
    include_ast: bool = True,
    include_sim: bool = True,
) -> FeatureMatrix:
    """Assemble features for one split + variant. Filters B>=A and builds y."""
    pair_pq = in_splits / f"pair_{split}.parquet"
    point_pq = in_splits / f"{split}.parquet"

    pair_tbl = pq.read_table(pair_pq)
    pair_tbl = filter_b_ge_a(pair_tbl)
    pair_ids = pair_tbl.column("pair_id").to_pylist()
    if "language" in pair_tbl.schema.names:
        languages = pair_tbl.column("language").to_pylist()
    else:
        languages = ["unknown"] * len(pair_ids)
    y = binary_labels(pair_tbl)
    log_imbalance(y, split)

    blocks: list[np.ndarray] = []
    col_names: list[str] = []
    scaled_flags: list[bool] = []

    # AST
    if include_ast:
        ast_pq = extraction_dir / f"ast_pair_{split}.parquet"
        ast_tbl = _load_or_empty(ast_pq, "ast_pair")
        if ast_tbl is not None:
            # Align by pair_id
            id_to_row = {id_: i for i, id_ in enumerate(ast_tbl.column("pair_id").to_pylist())}
            cols = diff_columns()
            raw = {c: np.asarray(ast_tbl.column(c).to_pylist(), dtype=np.float32) for c in cols}
            mat = np.zeros((len(pair_ids), len(cols)), dtype=np.float32)
            for i, pid in enumerate(pair_ids):
                r = id_to_row.get(pid)
                if r is None:
                    continue
                for j, c in enumerate(cols):
                    mat[i, j] = raw[c][r]
            blocks.append(mat)
            col_names.extend(cols)
            # bool features (from FEATURE_KIND) are NOT scaled — identify them by pattern
            for c in cols:
                # Column name is ast_{A|B|diff|abs_diff}__<base_feature>
                base = c.split("__", 1)[-1]
                kind = FEATURE_KIND.get(base, "cont")
                # Scaling: booleans skip; counts/cont get scaled (log1p'd below)
                scaled_flags.append(kind != "bool")

    # Pointwise BERT logits (v1 only — pair logits are gone)
    logit_pq = extraction_dir / f"point_logits_{split}.parquet"
    logit_tbl = _load_or_empty(logit_pq, "point_logits")
    if logit_tbl is not None:
        mat, cols = _join_point_logits(pair_tbl, point_pq, logit_tbl)
        blocks.append(mat)
        col_names.extend(cols)
        scaled_flags.extend([True] * len(cols))

    # Similarity
    if include_sim:
        sim_pq = extraction_dir / f"pair_sim_{split}.parquet"
        sim_tbl = _load_or_empty(sim_pq, "pair_sim")
        if sim_tbl is not None:
            mat, cols = _join_similarity(pair_ids, sim_tbl)
            if cols:
                blocks.append(mat)
                col_names.extend(cols)
                scaled_flags.extend([True] * len(cols))

    if not blocks:
        raise RuntimeError(
            f"No feature blocks built for {split}/{variant}. Check extraction_dir "
            f"has ast_pair_{split}.parquet, point_logits_{split}.parquet, "
            f"pair_sim_{split}.parquet."
        )

    X = np.concatenate(blocks, axis=1)
    scaled_mask = np.asarray(scaled_flags, dtype=bool)

    # Log-transform counts (positive integers) — detect by AST feature name prefix.
    # Apply to ast_a__ / ast_b__ / ast_diff__ / ast_abs_diff__ columns that are 'count'.
    for j, name in enumerate(col_names):
        if name.startswith(("ast_a__", "ast_b__", "ast_diff__", "ast_abs_diff__")):
            base = name.split("__", 1)[-1]
            if FEATURE_KIND.get(base) == "count":
                # log1p for raw A/B/|diff|; signed diff uses signed log1p
                if name.startswith("ast_diff__"):
                    X[:, j] = np.sign(X[:, j]) * np.log1p(np.abs(X[:, j]))
                else:
                    X[:, j] = np.log1p(np.clip(X[:, j], 0, None))

    assert len(col_names) == X.shape[1] == len(scaled_mask)
    return FeatureMatrix(
        pair_ids=pair_ids, X=X.astype(np.float32), y=y,
        columns=col_names, scaled_mask=scaled_mask, variant=variant,
        languages=languages,
    )


# -----------------------------------------------------------------------------
# Scaler fit/apply
# -----------------------------------------------------------------------------

def fit_scaler(train: FeatureMatrix) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train.X[:, train.scaled_mask])
    return scaler


def apply_scaler(fm: FeatureMatrix, scaler: StandardScaler) -> FeatureMatrix:
    X2 = fm.X.copy()
    X2[:, fm.scaled_mask] = scaler.transform(X2[:, fm.scaled_mask]).astype(np.float32)
    return FeatureMatrix(
        pair_ids=fm.pair_ids, X=X2, y=fm.y,
        columns=fm.columns, scaled_mask=fm.scaled_mask, variant=fm.variant,
        languages=fm.languages,
    )


def save_scaler(scaler: StandardScaler, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: Path) -> StandardScaler:
    return joblib.load(path)


# -----------------------------------------------------------------------------
# Convenience: build all three splits with shared scaler
# -----------------------------------------------------------------------------

def build_all_splits(
    variant: Variant,
    in_splits: Path,
    extraction_dir: Path,
    out_dir: Path | None = None,
) -> tuple[FeatureMatrix, FeatureMatrix, FeatureMatrix]:
    """Build train/val/test, fit scaler on train, apply to val+test.

    If out_dir is given, save scaler + schema to disk.
    """
    train = build_feature_matrix("train", variant, in_splits, extraction_dir)
    val = build_feature_matrix("val", variant, in_splits, extraction_dir)
    test = build_feature_matrix("test", variant, in_splits, extraction_dir)

    # All three must have identical column schema
    assert train.columns == val.columns == test.columns, (
        "column schema drift across splits — check extraction cache "
        f"train={len(train.columns)} val={len(val.columns)} test={len(test.columns)}"
    )

    scaler = fit_scaler(train)
    train_s = apply_scaler(train, scaler)
    val_s = apply_scaler(val, scaler)
    test_s = apply_scaler(test, scaler)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_scaler(scaler, out_dir / "scaler.joblib")
        (out_dir / "schema.json").write_text(json.dumps({
            "variant": variant,
            "variant_desc": VARIANT_DESCRIPTIONS[variant],
            "columns": train.columns,
            "n_features": train.num_features(),
            "scaled_mask": train.scaled_mask.tolist(),
            "n_train": int(train.X.shape[0]),
            "n_val": int(val.X.shape[0]),
            "n_test": int(test.X.shape[0]),
        }, indent=2), encoding="utf-8")

    return train_s, val_s, test_s
