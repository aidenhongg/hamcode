"""End-to-end prediction CLI for the stacking head.

Loads a trained head directory (contains scaler + head artifacts), re-extracts
AST / pointwise BERT logits / CLS-cosine for a fresh (code_a, code_b) pair,
and emits JSON.

IMPORTANT: The caller is expected to submit pairs in canonical order (B
same-or-slower than A). The head's output is:
    same    = same complexity tier
    A_faster = B is strictly slower than A

Usage:
    python -m stacking.predict_head \
        --head_dir runs/heads/xgb-v1-s42 \
        --point_ckpt runs/point-20260422-065105/point-20260422-065105/best \
        --pair examples/linear.py examples/quadratic.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stacking import dataset as ds
from stacking.features.ast_features import (
    diff_columns, extract_differenced, extract_features,
)
from stacking.features.bert_logits import (
    _encode_point_batch, _forward_with_cls, load_frozen_model,
)
from stacking.heads.base import HeadRegistry


def _head_from_dir(head_dir: Path):
    """Instantiate and load a head from a train_head.run() output directory."""
    cfg = json.loads((head_dir / "config.json").read_text(encoding="utf-8"))
    head_name = cfg["head"]
    head_cls = HeadRegistry.get(head_name)
    return head_cls.load(head_dir / "head"), cfg


def _pointwise_features(model, tokenizer, code: str, device: torch.device,
                          fp16: bool, max_seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (logits_11d, cls_768d)."""
    enc = _encode_point_batch([code], tokenizer, max_seq_len)
    logits, cls = _forward_with_cls(model, enc, device, use_fp16=fp16)
    return logits[0].cpu().numpy(), cls[0].cpu().numpy()


def predict(
    head_dir: Path,
    point_ckpt: Path,
    code_a: str,
    code_b: str,
    fp16: bool = True,
    max_seq_len: int = 512,
) -> dict:
    head, cfg = _head_from_dir(head_dir)
    scaler = ds.load_scaler(head_dir / "scaler.joblib")
    schema = json.loads((head_dir / "schema.json").read_text(encoding="utf-8"))
    expected_cols = schema["columns"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # AST differenced (always)
    ast_row = extract_differenced(code_a, code_b).astype(np.float32)

    # Pointwise BERT block (only variant supported is v1)
    pmodel, ptok = load_frozen_model(point_ckpt)
    pmodel = pmodel.to(device)
    la, cls_a = _pointwise_features(pmodel, ptok, code_a, device, fp16, max_seq_len)
    lb, cls_b = _pointwise_features(pmodel, ptok, code_b, device, fp16, max_seq_len)
    diff = lb - la
    absd = np.abs(diff)
    point_block = np.concatenate([la, lb, diff, absd]).astype(np.float32)
    # cls cosine
    na = np.linalg.norm(cls_a) + 1e-12
    nb = np.linalg.norm(cls_b) + 1e-12
    cos = float((cls_a @ cls_b) / (na * nb))
    l2 = float(np.linalg.norm(cls_a - cls_b))
    mad = float(np.mean(np.abs(cls_a - cls_b)))
    mxd = float(np.max(np.abs(cls_a - cls_b)))
    sim_block = np.array([cos, l2, mad, mxd], dtype=np.float32)

    # Assemble in the same order as training, then scale
    ast_cols = diff_columns()  # 84
    feature_dict: dict[str, float] = {}
    for i, c in enumerate(ast_cols):
        feature_dict[c] = float(ast_row[i])
    for c in expected_cols:
        if c.startswith("point_A_logit_"):
            k = int(c.rsplit("_", 1)[-1])
            feature_dict[c] = float(point_block[k])
        elif c.startswith("point_B_logit_"):
            k = int(c.rsplit("_", 1)[-1])
            feature_dict[c] = float(point_block[11 + k])
        elif c.startswith("point_diff_logit_"):
            k = int(c.rsplit("_", 1)[-1])
            feature_dict[c] = float(point_block[22 + k])
        elif c.startswith("point_abs_diff_logit_"):
            k = int(c.rsplit("_", 1)[-1])
            feature_dict[c] = float(point_block[33 + k])
    sim_names = ["cls_cosine", "cls_l2", "cls_mean_abs_diff", "cls_max_abs_diff"]
    for i, c in enumerate(sim_names):
        feature_dict[c] = float(sim_block[i])

    # Apply log1p to AST count columns to match training transform
    from stacking.features.ast_features import FEATURE_KIND
    for c, v in list(feature_dict.items()):
        if c.startswith(("ast_a__", "ast_b__", "ast_diff__", "ast_abs_diff__")):
            base = c.split("__", 1)[-1]
            if FEATURE_KIND.get(base) == "count":
                if c.startswith("ast_diff__"):
                    feature_dict[c] = float(np.sign(v) * np.log1p(abs(v)))
                else:
                    feature_dict[c] = float(np.log1p(max(0.0, v)))

    # Build X in schema order
    X = np.array([[feature_dict.get(c, 0.0) for c in expected_cols]], dtype=np.float32)
    mask = np.asarray(schema["scaled_mask"], dtype=bool)
    X[:, mask] = scaler.transform(X[:, mask]).astype(np.float32)

    proba = head.predict_proba(X)[0]
    pred = int(np.argmax(proba))
    return {
        "head": cfg["head"],
        "seed": cfg["seed"],
        "label": "A_faster" if pred == 1 else "same",
        "label_idx": pred,
        "prob_same": float(proba[0]),
        "prob_A_faster": float(proba[1]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--head_dir", required=True)
    ap.add_argument("--point_ckpt", required=True)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--pair", nargs=2, metavar=("A", "B"),
                        help="Python file paths for code_a and code_b")
    group.add_argument("--code_strings", nargs=2, metavar=("A", "B"),
                        help="raw code strings (escape newlines)")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no_fp16", dest="fp16", action="store_false")
    ap.add_argument("--max_seq_len", type=int, default=512)
    args = ap.parse_args()

    if args.pair:
        code_a = Path(args.pair[0]).read_text(encoding="utf-8")
        code_b = Path(args.pair[1]).read_text(encoding="utf-8")
    else:
        code_a, code_b = args.code_strings

    result = predict(
        head_dir=Path(args.head_dir),
        point_ckpt=Path(args.point_ckpt),
        code_a=code_a, code_b=code_b,
        fp16=args.fp16, max_seq_len=args.max_seq_len,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
