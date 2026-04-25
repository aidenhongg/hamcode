"""Inference CLI for the pointwise LongCoder complexity classifier.

Usage:
    python predict.py --model_dir runs/point-v1/best --input examples/two_sum.py
    python predict.py --model_dir runs/point-v1/best --stdin < code.py

Outputs a JSON object with top prediction + full probability vector.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.labels import IDX_TO_LABEL
from data import build_point_inputs
from model import LongCoderClassifier


def _predict_point(
    model: LongCoderClassifier,
    tokenizer,
    code: str,
    device: torch.device,
    max_seq_len: int,
    bridge_stride: int,
) -> dict:
    b = build_point_inputs(code, tokenizer, max_seq_len, bridge_stride)
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
    probs = torch.softmax(logits[0], dim=-1).cpu().numpy().tolist()
    idx = int(np.argmax(probs))
    return {
        "label": IDX_TO_LABEL[idx], "label_idx": idx,
        "confidence": float(probs[idx]),
        "probs": {IDX_TO_LABEL[i]: float(p) for i, p in enumerate(probs)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--bridge_stride", type=int, default=128)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Python file path")
    group.add_argument("--stdin", action="store_true", help="Read snippet from stdin")
    args = ap.parse_args()

    model = LongCoderClassifier.load_checkpoint(args.model_dir, merge_lora=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model.model_name)

    code = sys.stdin.read() if args.stdin else Path(args.input).read_text(encoding="utf-8")
    result = _predict_point(model, tokenizer, code, device,
                            args.max_seq_len, args.bridge_stride)
    out = {"task": "point", **({"file": args.input} if args.input else {}), **result}

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
