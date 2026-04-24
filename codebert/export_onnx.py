"""Export pointwise GraphCodeBERT to ONNX (fp32) with (logits, cls) outputs.

Target: transformers.js / onnxruntime-web in a Chrome extension. The exported
graph returns two tensors so the v1 stacking head can consume both:
    logits: (B, 11)   -- pre-softmax classifier output
    cls:    (B, 768)  -- encoder last_hidden_state[:, 0, :]

Output layout (HF + transformers.js friendly):
    <out_dir>/
      config.json            -- RoBERTa encoder config
      tokenizer.json + friends
      label_map.json         -- id2label / label2id for the 11 pointwise classes
      onnx/model.onnx        -- the exported graph (dynamic batch + sequence)

Usage:
    pip install onnx onnxruntime   # needed for export + parity check
    python export_onnx.py \
        --ckpt runs/heads/extraction/oof/full/best \
        --out_dir exports/graphcodebert-point
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.labels import IDX_TO_LABEL, POINT_LABELS
from model import GraphCodeBERTClassifier


class PointExportWrapper(nn.Module):
    """Expose (logits, cls) as two ONNX outputs. Dropout is a no-op in eval."""

    def __init__(self, base: GraphCodeBERTClassifier) -> None:
        super().__init__()
        self.encoder = base.encoder
        self.classifier = base.classifier

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )
        cls = out.last_hidden_state[:, 0, :]
        logits = self.classifier(cls)
        return logits, cls


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="runs/heads/extraction/oof/full/best",
                    help="Checkpoint dir (pytorch_model.bin + codebert_meta.json).")
    ap.add_argument("--out_dir", default="exports/graphcodebert-point")
    ap.add_argument("--max_seq_len", type=int, default=512,
                    help="Trace length; dynamic_axes allows any length at inference.")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--atol", type=float, default=1e-4,
                    help="Max abs diff tolerated between PyTorch and onnxruntime.")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    if not (ckpt / "pytorch_model.bin").exists():
        raise FileNotFoundError(f"no pytorch_model.bin at {ckpt}")

    print(f"[export] loading {ckpt}", flush=True)
    model = GraphCodeBERTClassifier.load_checkpoint(ckpt, task="point")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    wrapper = PointExportWrapper(model).eval()

    out_dir = Path(args.out_dir)
    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "model.onnx"

    # The model was trained with explicit position_ids = arange(seq_len) (see
    # data.py simple path), so we feed arange here — keeping that consistent
    # downstream is the extension's responsibility.
    B, S = 1, args.max_seq_len
    input_ids = torch.zeros((B, S), dtype=torch.long)
    attention_mask = torch.ones((B, S), dtype=torch.long)
    position_ids = torch.arange(S, dtype=torch.long).unsqueeze(0).expand(B, S).contiguous()

    print(f"[export] tracing to {onnx_path} (opset={args.opset})", flush=True)
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask, position_ids),
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "position_ids"],
        output_names=["logits", "cls"],
        dynamic_axes={
            "input_ids":      {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "position_ids":   {0: "batch", 1: "sequence"},
            "logits":         {0: "batch"},
            "cls":            {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"[export] wrote {onnx_path} ({size_mb:.1f} MB)", flush=True)

    # PT <-> onnxruntime numeric parity on the dummy batch. Cheap sanity — the
    # F1 check on the full test set is what the user waived, not this.
    try:
        import onnxruntime as ort
    except ImportError:
        print("[export] onnxruntime not installed; skipping parity check "
              "(pip install onnxruntime to enable)", flush=True)
    else:
        print("[export] running PT<->ORT parity check ...", flush=True)
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_logits, ort_cls = sess.run(None, {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy(),
            "position_ids": position_ids.numpy(),
        })
        with torch.no_grad():
            pt_logits, pt_cls = wrapper(input_ids, attention_mask, position_ids)
        d_logits = float(np.max(np.abs(pt_logits.numpy() - ort_logits)))
        d_cls = float(np.max(np.abs(pt_cls.numpy() - ort_cls)))
        print(f"[export] max abs diff  logits={d_logits:.2e}  cls={d_cls:.2e}", flush=True)
        if d_logits > args.atol or d_cls > args.atol:
            raise RuntimeError(
                f"parity check failed (atol={args.atol}): "
                f"logits={d_logits:.2e} cls={d_cls:.2e}"
            )

    # HF/transformers.js-shaped sidecar files.
    model.encoder.config.save_pretrained(out_dir)
    tokenizer = AutoTokenizer.from_pretrained(model.model_name)
    tokenizer.save_pretrained(out_dir)

    label_map = {
        "id2label": {str(i): lab for i, lab in IDX_TO_LABEL.items()},
        "label2id": {lab: i for i, lab in enumerate(POINT_LABELS)},
        "task": "point",
    }
    (out_dir / "label_map.json").write_text(
        json.dumps(label_map, indent=2), encoding="utf-8",
    )

    print(f"[export] done. Artifacts in {out_dir}/", flush=True)
    print(f"[export]   onnx/model.onnx     -- outputs (logits:{len(POINT_LABELS)}, cls:768)")
    print(f"[export]   config.json         -- RoBERTa encoder config")
    print(f"[export]   tokenizer.json + friends")
    print(f"[export]   label_map.json      -- id/label mappings for the 11 classes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
