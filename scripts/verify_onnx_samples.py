"""End-to-end ONNX verification on real pair_test samples.

For each language, picks 10 random pairs from pair_test.parquet and runs them
through both:
  - the per-language ONNX module (via onnxruntime CPU)
  - the patched PyTorch reference pipeline (same code as export_onnx.py)

Reports max-abs-diff in probabilities, plus prediction-vs-truth agreement so
you can sanity-check accuracy on the sample.

Usage:
    python scripts/verify_onnx_samples.py \
        --backbone_run runs/multi-fullft-20260426-002447/best \
        --lora_root runs/lora-20260426-073548 \
        --heads_root runs/heads/per_lang \
        --scaler runs/heads/scaler.joblib \
        --schema runs/heads/schema.json \
        --onnx_dir runs/onnx-export \
        --pair_parquet data/processed/pair_test.parquet \
        --n_pairs 10 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import build_point_inputs
from model import _last_token_pool
from stacking.features.ast_features import extract_differenced

from export_onnx import (  # type: ignore
    HEAD_SELECTION,
    HIDDEN,
    MAX_SEQ_LEN,
    N_LABELS,
    PairwiseFusedFeatures,
    PairwiseFusedMLP,
    _build_lang_features_module,
    load_mlp_head,
)

logger = logging.getLogger("verify_onnx_samples")


@torch.no_grad()
def _compute_a_features(features: PairwiseFusedFeatures, code_a: str, lang: str,
                         tokenizer, max_seq_len: int = MAX_SEQ_LEN
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Run snippet A through the same LoRA-adapted encoder + Phase B 11-d
    classifier baked into <lang>.onnx and return (logits_a[11], pooled_a[768]).

    This is the verification-time stand-in for whatever external precompute
    the caller uses in production. Using `features.peft_model` directly
    guarantees bit-exact parity with what the ONNX graph computes for B
    (and thus what it would have computed for A in the old batched build)."""
    bundle = build_point_inputs(code_a, tokenizer, max_seq_len, 128, language=lang)
    inner = features.peft_model.base_model.model
    input_ids = torch.from_numpy(bundle.input_ids).unsqueeze(0)
    attn = torch.from_numpy(bundle.attention_mask).unsqueeze(0)
    g_attn = torch.from_numpy(bundle.global_attention_mask).unsqueeze(0)
    types = torch.from_numpy(bundle.token_type_ids).unsqueeze(0)
    enc_out = inner.encoder(
        input_ids=input_ids, attention_mask=attn,
        global_attention_mask=g_attn, token_type_ids=types,
        return_dict=True,
    )
    pooled = _last_token_pool(enc_out.last_hidden_state, attn).squeeze(0)  # [H]
    logits = inner.classifier(pooled)                                       # [11]
    return logits.numpy().astype(np.float32), pooled.numpy().astype(np.float32)


def _build_pair_inputs(features: PairwiseFusedFeatures,
                        code_a: str, code_b: str, lang: str, tokenizer,
                        max_seq_len: int = MAX_SEQ_LEN) -> dict:
    """Tokenize B and build the per-language ONNX/PyTorch input dict.

    A's logits and pooled vector are precomputed via `_compute_a_features`,
    mirroring the production contract where the caller supplies them."""
    bundle_b = build_point_inputs(code_b, tokenizer, max_seq_len, 128, language=lang)
    ast84 = extract_differenced(code_a, code_b, language=lang).astype(np.float32)
    logits_a, pooled_a = _compute_a_features(features, code_a, lang, tokenizer, max_seq_len)
    return {
        "input_ids":             bundle_b.input_ids[None, :].astype(np.int64),
        "attention_mask":        bundle_b.attention_mask[None, :].astype(np.int64),
        "global_attention_mask": bundle_b.global_attention_mask[None, :].astype(np.int64),
        "token_type_ids":        bundle_b.token_type_ids[None, :].astype(np.int64),
        "ast_diff_features":     ast84,
        "logits_a":              logits_a,
        "pooled_a":              pooled_a,
    }


def _sample_pairs(pair_parquet: Path, lang: str, n: int, seed: int) -> list[dict]:
    table = pq.read_table(pair_parquet)
    import pyarrow.compute as pc
    table = table.filter(pc.equal(table["language"], lang))
    n_total = table.num_rows
    if n_total == 0:
        return []
    take = min(n, n_total)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=take, replace=False)
    rows: list[dict] = []
    for i in sorted(int(x) for x in idx):
        rows.append({
            "pair_id":  table["pair_id"][i].as_py(),
            "code_a":   table["code_a"][i].as_py(),
            "code_b":   table["code_b"][i].as_py(),
            "label_a":  table["label_a"][i].as_py(),
            "label_b":  table["label_b"][i].as_py(),
            "ternary":  table["ternary"][i].as_py(),
        })
    return rows


_ORT_INPUT_KEYS = (
    "input_ids", "attention_mask", "global_attention_mask",
    "token_type_ids", "ast_diff_features", "logits_a", "pooled_a",
)


def _ort_feeds(inputs: dict) -> dict:
    return {k: inputs[k] for k in _ORT_INPUT_KEYS}


def _torch_kwargs(inputs: dict) -> dict:
    return {k: torch.from_numpy(inputs[k]) for k in _ORT_INPUT_KEYS}


def _run_one_pair_mlp(fused: PairwiseFusedMLP, ort_sess, inputs: dict
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Returns (ort_probs[2], pt_probs[2]) for an MLP-head module."""
    ort_probs = ort_sess.run(["probabilities"], _ort_feeds(inputs))[0]
    with torch.no_grad():
        pt_probs = fused(**_torch_kwargs(inputs)).numpy()
    return np.asarray(ort_probs).reshape(-1), np.asarray(pt_probs).reshape(-1)


def _run_one_pair_lgbm(features: PairwiseFusedFeatures, lgbm, ort_sess, inputs: dict
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Returns (ort_probs[2], pt_probs[2]) for an LGBM-head module.

    PyTorch reference: feature pipeline -> [1, 132] -> sklearn lgbm.predict_proba.
    ORT: full merged graph -> probabilities[2].
    """
    ort_probs = ort_sess.run(["probabilities"], _ort_feeds(inputs))[0]
    with torch.no_grad():
        feat = features(**_torch_kwargs(inputs)).numpy()  # [1, 132]
    pt_probs = lgbm.predict_proba(feat)[0]  # [2]
    return np.asarray(ort_probs).reshape(-1), np.asarray(pt_probs).reshape(-1)


def verify_language(
    lang: str, head_kind: str, seed: int,
    backbone_dir: Path, lora_root: Path, heads_root: Path,
    scaler, scaled_mask: list[bool],
    onnx_dir: Path, pair_parquet: Path,
    n_pairs: int, sample_seed: int, max_seq_len: int,
) -> dict:
    pairs = _sample_pairs(pair_parquet, lang, n_pairs, sample_seed)
    if not pairs:
        return {"language": lang, "n_pairs": 0, "skipped": True}

    onnx_path = onnx_dir / f"{lang}.onnx"
    ort_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    lora_dir = lora_root / lang
    head_dir = heads_root / lang / f"{head_kind}-v1-s{seed}" / "head"
    features = _build_lang_features_module(backbone_dir, lora_dir, scaler, scaled_mask)

    if head_kind == "mlp":
        mlp = load_mlp_head(head_dir)
        fused = PairwiseFusedMLP(features=features, mlp=mlp).eval()
        lgbm = None
    elif head_kind == "lgbm":
        lgbm = joblib.load(head_dir / "lgbm.pkl")
        fused = None
    else:
        raise ValueError(head_kind)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/longcoder-base", use_fast=True)

    diffs: list[float] = []
    rows: list[dict] = []
    correct = 0
    t0 = time.time()
    for k, pair in enumerate(pairs):
        inputs = _build_pair_inputs(features, pair["code_a"], pair["code_b"],
                                     lang, tokenizer, max_seq_len)
        if head_kind == "mlp":
            ort_probs, pt_probs = _run_one_pair_mlp(fused, ort_sess, inputs)
        else:
            ort_probs, pt_probs = _run_one_pair_lgbm(features, lgbm, ort_sess, inputs)
        diff = float(np.max(np.abs(ort_probs - pt_probs)))
        diffs.append(diff)
        ort_pred = "A_faster" if float(ort_probs[1]) > float(ort_probs[0]) else "same"
        pt_pred  = "A_faster" if float(pt_probs[1])  > float(pt_probs[0])  else "same"
        truth = pair["ternary"]
        is_correct = ort_pred == truth
        correct += 1 if is_correct else 0
        rows.append({
            "pair_id":          pair["pair_id"],
            "label_a":          pair["label_a"],
            "label_b":          pair["label_b"],
            "truth":            truth,
            "ort_pred":         ort_pred,
            "pt_pred":          pt_pred,
            "ort_prob_same":    float(ort_probs[0]),
            "ort_prob_Afaster": float(ort_probs[1]),
            "pt_prob_same":     float(pt_probs[0]),
            "pt_prob_Afaster":  float(pt_probs[1]),
            "diff":             diff,
            "ort_correct":      is_correct,
        })
        logger.info("  [%s/%d] %s truth=%s ort=%s diff=%.2e",
                    k + 1, len(pairs), pair["pair_id"], truth, ort_pred, diff)

    elapsed = time.time() - t0
    return {
        "language": lang,
        "head": head_kind,
        "seed": seed,
        "n_pairs": len(pairs),
        "max_diff": max(diffs) if diffs else 0.0,
        "mean_diff": sum(diffs) / max(1, len(diffs)),
        "ort_accuracy": correct / max(1, len(pairs)),
        "elapsed_s": round(elapsed, 1),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone_run", required=True)
    ap.add_argument("--lora_root", required=True)
    ap.add_argument("--heads_root", required=True)
    ap.add_argument("--scaler", required=True)
    ap.add_argument("--schema", required=True)
    ap.add_argument("--onnx_dir", required=True)
    ap.add_argument("--pair_parquet", required=True)
    ap.add_argument("--n_pairs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seq_len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--report", default=None,
                    help="optional path to write JSON report; default prints summary only")
    ap.add_argument("--only_lang", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    scaled_mask = list(schema["scaled_mask"])
    scaler = joblib.load(args.scaler)
    backbone_dir = Path(args.backbone_run)
    lora_root = Path(args.lora_root)
    heads_root = Path(args.heads_root)
    onnx_dir = Path(args.onnx_dir)
    pair_parquet = Path(args.pair_parquet)

    all_results = []
    for lang, head_kind, seed in HEAD_SELECTION:
        if args.only_lang and args.only_lang != lang:
            continue
        logger.info("=== %s (%s s%d) ===", lang, head_kind, seed)
        rec = verify_language(
            lang=lang, head_kind=head_kind, seed=seed,
            backbone_dir=backbone_dir, lora_root=lora_root, heads_root=heads_root,
            scaler=scaler, scaled_mask=scaled_mask,
            onnx_dir=onnx_dir, pair_parquet=pair_parquet,
            n_pairs=args.n_pairs, sample_seed=args.seed,
            max_seq_len=args.seq_len,
        )
        all_results.append(rec)
        if rec.get("skipped"):
            logger.info("  skipped (no rows)")
        else:
            logger.info("  max diff=%.3e mean diff=%.3e ort_acc=%.0f%% (%ds)",
                        rec["max_diff"], rec["mean_diff"],
                        100 * rec["ort_accuracy"], rec["elapsed_s"])

    # Summary table
    print()
    print(f"{'lang':<12} {'head':<6} {'seed':>4} {'n':>3} {'max_diff':>10} "
          f"{'mean_diff':>10} {'ort_acc':>8}  {'time':>6}")
    print("-" * 70)
    for rec in all_results:
        if rec.get("skipped"):
            print(f"{rec['language']:<12} {'—':<6}  —     0  (skipped)")
            continue
        print(f"{rec['language']:<12} {rec['head']:<6} "
              f"{rec['seed']:>4} {rec['n_pairs']:>3} "
              f"{rec['max_diff']:>10.3e} {rec['mean_diff']:>10.3e} "
              f"{rec['ort_accuracy']*100:>7.0f}%  {rec['elapsed_s']:>5.0f}s")
    overall_max = max((r["max_diff"] for r in all_results if not r.get("skipped")), default=0.0)
    overall_acc = (sum(r["ort_accuracy"] * r["n_pairs"] for r in all_results if not r.get("skipped"))
                   / max(1, sum(r["n_pairs"] for r in all_results if not r.get("skipped"))))
    print("-" * 70)
    print(f"{'overall':<12} {'':<6}  —    — {overall_max:>10.3e} "
          f"{'':>10} {overall_acc*100:>7.0f}%")

    if args.report:
        Path(args.report).write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        logger.info("report written to %s", args.report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
