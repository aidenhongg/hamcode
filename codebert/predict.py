"""Inference CLI for GraphCodeBERT complexity classifier.

Usage:
    python predict.py --model_dir runs/point-v1/best --input examples/two_sum.py
    python predict.py --model_dir runs/point-v1/best --stdin < code.py
    python predict.py --model_dir runs/pair-v1/best  --pair a.py b.py

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

from common.labels import IDX_TO_LABEL, PAIR_LABELS
from data import build_pair_inputs, build_point_inputs
from model import GraphCodeBERTClassifier


def _predict_point(model, tokenizer, code: str, device, max_seq_len: int,
                    max_dfg: int, use_dfg: bool) -> dict:
    if use_dfg:
        b = build_point_inputs(code, tokenizer, max_seq_len, max_dfg)
        input_ids = torch.from_numpy(b.input_ids).unsqueeze(0).to(device)
        attn = torch.from_numpy(b.attention_mask).unsqueeze(0).to(device)
        pos = torch.from_numpy(b.position_ids).unsqueeze(0).to(device)
    else:
        enc = tokenizer(code, truncation=True, padding="max_length",
                        max_length=max_seq_len, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        pos = torch.arange(max_seq_len, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn, position_ids=pos)
    probs = torch.softmax(logits[0], dim=-1).cpu().numpy().tolist()
    idx = int(np.argmax(probs))
    return {
        "label": IDX_TO_LABEL[idx], "label_idx": idx,
        "confidence": float(probs[idx]),
        "probs": {IDX_TO_LABEL[i]: float(p) for i, p in enumerate(probs)},
    }


def _predict_pair(model, tokenizer, code_a: str, code_b: str, device,
                  max_seq_len: int, max_dfg: int, use_dfg: bool) -> dict:
    if use_dfg:
        b = build_pair_inputs(code_a, code_b, tokenizer, max_seq_len, max_dfg)
        input_ids = torch.from_numpy(b.input_ids).unsqueeze(0).to(device)
        attn = torch.from_numpy(b.attention_mask).unsqueeze(0).to(device)
        pos = torch.from_numpy(b.position_ids).unsqueeze(0).to(device)
    else:
        enc = tokenizer(code_a, code_b, truncation="longest_first",
                        padding="max_length", max_length=max_seq_len,
                        return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        pos = torch.arange(max_seq_len, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn, position_ids=pos)
    probs = torch.softmax(logits[0], dim=-1).cpu().numpy().tolist()
    idx = int(np.argmax(probs))
    return {
        "label": PAIR_LABELS[idx], "label_idx": idx,
        "confidence": float(probs[idx]),
        "probs": {PAIR_LABELS[i]: float(p) for i, p in enumerate(probs)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--max_dfg_nodes", type=int, default=64)
    ap.add_argument("--use_dfg", action="store_true", default=False,
                    help="use tree-sitter DFG (only if the model was trained with --use_dfg)")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Python file path (single-snippet mode)")
    group.add_argument("--stdin", action="store_true", help="Read single snippet from stdin")
    group.add_argument("--pair", nargs=2, metavar=("A", "B"),
                       help="Two python files for pairwise comparison")
    args = ap.parse_args()

    model = GraphCodeBERTClassifier.load_checkpoint(args.model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model.model_name)

    if args.pair:
        if model.task != "pair":
            print(f"[predict] WARNING: checkpoint task is '{model.task}' but --pair passed", file=sys.stderr)
        code_a = Path(args.pair[0]).read_text(encoding="utf-8")
        code_b = Path(args.pair[1]).read_text(encoding="utf-8")
        result = _predict_pair(model, tokenizer, code_a, code_b, device,
                               args.max_seq_len, args.max_dfg_nodes, args.use_dfg)
        out = {"task": "pair", "file_a": args.pair[0], "file_b": args.pair[1], **result}
    else:
        code = sys.stdin.read() if args.stdin else Path(args.input).read_text(encoding="utf-8")
        result = _predict_point(model, tokenizer, code, device,
                                args.max_seq_len, args.max_dfg_nodes, args.use_dfg)
        out = {"task": "point", **({"file": args.input} if args.input else {}), **result}

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
