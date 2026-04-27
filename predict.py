"""Inference CLI for the Hamcode pointwise complexity classifier.

Two deployment modes:

  1. Phase-A only (universal full-FT):
        python predict.py \\
            --model_dir runs/multi-fullft-20260425/best \\
            --input examples/two_sum.py

  2. Phase-A + Phase-B (per-language LoRA bundle, recommended):
        python predict.py \\
            --bundle runs/lora-20260425/ \\
            --base_run runs/multi-fullft-20260425/best \\
            --input examples/two_sum.cpp

In bundle mode, the language is auto-detected from the input file extension
(use --language to override). For --stdin input, --language is required.

Outputs JSON with the top label, confidence, and full probability vector.
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

from common.labels import IDX_TO_LABEL, POINT_LABELS
from common.schemas import LANG_SET
from data import build_point_inputs
from model import LongCoderClassifier


# Filename extension -> canonical language id.
EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".java": "java",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp",
    ".c": "c", ".h": "c",
    ".cs": "csharp",
    ".go": "go",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".php": "php",
    ".rb": "ruby",
    ".rs": "rust",
    ".swift": "swift",
}


def detect_language(path: str | None, override: str | None) -> str:
    if override:
        if override not in LANG_SET:
            raise SystemExit(f"--language must be one of {sorted(LANG_SET)}; got {override!r}")
        return override
    if path is None:
        raise SystemExit("--language is required for --stdin input")
    ext = Path(path).suffix.lower()
    lang = EXT_TO_LANG.get(ext)
    if lang is None:
        raise SystemExit(
            f"could not infer language from extension {ext!r}; "
            f"pass --language explicitly (one of {sorted(LANG_SET)})"
        )
    return lang


def _load_phase_a(model_dir: str, device: torch.device):
    model = LongCoderClassifier.load_checkpoint(model_dir)
    model.eval()
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model.model_name)
    return model, tokenizer


def _load_bundle(base_run: str, bundle_dir: str, language: str,
                 device: torch.device):
    """Load Phase-A backbone, then attach the per-language LoRA adapter and
    swap in the per-language classifier head from `bundle_dir/{language}/`.
    """
    from peft import PeftModel
    from safetensors.torch import load_file as load_safetensors

    backbone = LongCoderClassifier.load_checkpoint(base_run)
    backbone.eval()
    lang_dir = Path(bundle_dir) / language
    if not lang_dir.exists():
        raise SystemExit(f"bundle missing adapter for {language!r}: {lang_dir}")
    pmodel = PeftModel.from_pretrained(backbone, str(lang_dir))
    # Swap in the per-language head.
    head_path = lang_dir / "head.safetensors"
    if head_path.exists():
        sd = load_safetensors(str(head_path))
        with torch.no_grad():
            inner = pmodel.base_model.model
            inner.classifier.weight.copy_(sd["weight"])
            inner.classifier.bias.copy_(sd["bias"])
    pmodel.eval()
    pmodel = pmodel.to(device)

    # Restore the kwargs absorber for safety (peft-internal calls).
    inner = pmodel.base_model.model
    if not getattr(inner.forward, "_kwargs_absorbed", False):
        orig_fwd = inner.forward
        def _absorbing_fwd(input_ids, attention_mask, global_attention_mask,
                            token_type_ids, **_):
            return orig_fwd(input_ids=input_ids, attention_mask=attention_mask,
                            global_attention_mask=global_attention_mask,
                            token_type_ids=token_type_ids)
        _absorbing_fwd._kwargs_absorbed = True
        inner.forward = _absorbing_fwd

    tokenizer = AutoTokenizer.from_pretrained(backbone.model_name)
    return pmodel, tokenizer


def _predict(model, tokenizer, code: str, language: str, device: torch.device,
             max_seq_len: int, bridge_stride: int) -> dict:
    b = build_point_inputs(code, tokenizer, max_seq_len, bridge_stride,
                           language=language)
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
    g_model = ap.add_mutually_exclusive_group(required=True)
    g_model.add_argument("--model_dir", help="Phase-A only: path to a best/ checkpoint")
    g_model.add_argument("--bundle", help="Phase-B bundle root (runs/lora-{ts}/)")
    ap.add_argument("--base_run", help="Required with --bundle: Phase-A best/ dir")
    ap.add_argument("--language", default=None,
                    help="Language id; required for --stdin, optional otherwise")
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--bridge_stride", type=int, default=128)
    g_in = ap.add_mutually_exclusive_group(required=True)
    g_in.add_argument("--input", help="Path to a code file")
    g_in.add_argument("--stdin", action="store_true", help="Read code from stdin")
    args = ap.parse_args()

    language = detect_language(args.input, args.language)
    code = sys.stdin.read() if args.stdin else Path(args.input).read_text(encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_dir:
        model, tokenizer = _load_phase_a(args.model_dir, device)
        result = _predict(model, tokenizer, code, language, device,
                          args.max_seq_len, args.bridge_stride)
        out = {"task": "point", "language": language,
               "model": "phase_a",
               **({"file": args.input} if args.input else {}),
               **result}
    else:
        if not args.base_run:
            raise SystemExit("--bundle requires --base_run pointing at the Phase-A best/ dir")
        model, tokenizer = _load_bundle(args.base_run, args.bundle, language, device)
        result = _predict(model, tokenizer, code, language, device,
                          args.max_seq_len, args.bridge_stride)
        out = {"task": "point", "language": language,
               "model": "phase_b_bundle",
               "bundle": args.bundle,
               **({"file": args.input} if args.input else {}),
               **result}
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
