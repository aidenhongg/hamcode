"""A/B test: does the model cheat off the kamyu `# Time: O(...)` headers?

Eval kamyu rows of val.parquet twice: with the original code, and with the
leaky header lines stripped. Report accuracy delta per label.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.labels import IDX_TO_LABEL, LABEL_TO_IDX
from data import build_point_inputs
from model import LongCoderClassifier


LEAKY = re.compile(
    r"^[ \t]*(?://|#)[ \t]*(?:Time|Space)[ \t]*:.*\n?",
    re.MULTILINE,
)


def strip_leak(code: str) -> str:
    return LEAKY.sub("", code)


def predict_batch(model, tokenizer, codes, langs, device, max_seq_len, bridge_stride):
    preds = []
    for code, lang in zip(codes, langs):
        b = build_point_inputs(code, tokenizer, max_seq_len, bridge_stride, language=lang)
        input_ids = torch.from_numpy(b.input_ids).unsqueeze(0).to(device)
        attn = torch.from_numpy(b.attention_mask).unsqueeze(0).to(device)
        g_attn = torch.from_numpy(b.global_attention_mask).unsqueeze(0).to(device)
        types = torch.from_numpy(b.token_type_ids).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attn,
                global_attention_mask=g_attn,
                token_type_ids=types,
            )
        preds.append(int(torch.argmax(logits[0]).cpu().item()))
    return preds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--val", default="data/processed/val.parquet")
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--bridge_stride", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="0 = all kamyu rows")
    args = ap.parse_args()

    t = pq.read_table(args.val).to_pandas()
    kamyu = t[t["source"] == "kamyu"].reset_index(drop=True)
    if args.limit:
        kamyu = kamyu.iloc[: args.limit].reset_index(drop=True)
    print(f"kamyu val rows: {len(kamyu)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    model = LongCoderClassifier.load_checkpoint(args.model_dir)
    model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model.model_name)

    codes_orig = list(kamyu["code"])
    codes_clean = [strip_leak(c) for c in codes_orig]
    langs = list(kamyu["language"])
    labels = [LABEL_TO_IDX[l] for l in kamyu["label"]]

    n_changed = sum(1 for a, b in zip(codes_orig, codes_clean) if a != b)
    print(f"rows with header stripped: {n_changed}/{len(kamyu)}", flush=True)

    t0 = time.time()
    print("predicting on ORIGINAL code ...", flush=True)
    preds_orig = predict_batch(model, tokenizer, codes_orig, langs, device,
                               args.max_seq_len, args.bridge_stride)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    print("predicting on STRIPPED code ...", flush=True)
    preds_clean = predict_batch(model, tokenizer, codes_clean, langs, device,
                                args.max_seq_len, args.bridge_stride)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    correct_orig = sum(int(p == y) for p, y in zip(preds_orig, labels))
    correct_clean = sum(int(p == y) for p, y in zip(preds_clean, labels))
    n = len(labels)
    acc_orig = correct_orig / n
    acc_clean = correct_clean / n

    print()
    print(f"{'metric':<24s} {'orig':>10s} {'stripped':>10s} {'delta':>10s}")
    print(f"{'accuracy':<24s} {acc_orig:>10.4f} {acc_clean:>10.4f} {acc_clean-acc_orig:>+10.4f}")

    print()
    print("per-label accuracy:")
    print(f"  {'label':<18s} {'n':>4s} {'orig':>8s} {'stripped':>10s} {'delta':>8s}")
    by_label: dict[int, list[int]] = {}
    for y in labels:
        by_label.setdefault(y, []).append(0)
    label_idxs = sorted(set(labels))
    for li in label_idxs:
        idxs = [i for i, y in enumerate(labels) if y == li]
        n_l = len(idxs)
        o = sum(int(preds_orig[i] == labels[i]) for i in idxs) / n_l
        c = sum(int(preds_clean[i] == labels[i]) for i in idxs) / n_l
        print(f"  {IDX_TO_LABEL[li]:<18s} {n_l:>4d} {o:>8.3f} {c:>10.3f} {c-o:>+8.3f}")

    flips = sum(1 for a, b in zip(preds_orig, preds_clean) if a != b)
    print()
    print(f"prediction flips when header stripped: {flips}/{n} ({100*flips/n:.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
