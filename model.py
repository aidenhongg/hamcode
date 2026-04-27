"""LongCoder pointwise classifier.

Encoder: microsoft/longcoder-base (Longformer, 4096 max positions). LongCoder
was pretrained autoregressively for code completion. We adapt it to
bidirectional classification by:
  - leaving HF's default bidirectional Longformer attention on (no causal mask),
  - pooling the last non-pad hidden state instead of [CLS]. The first position
    has effectively no learned representation under the pretraining objective,
    while the last position carries the prefix's compressed summary.

attention_window: bumped from the pretrained default 1024 to 2048 so it
matches our training/deployment seq_len. With seq_len <= attention_window
the sliding-window kernel degenerates to full attention, which keeps the
ONNX FullAttentionReplacement (scripts/longcoder_onnx_attention.py)
mathematically valid. Pretrained weights still load — the window is a
runtime config, not a weight shape — so this is a forward-pass behavior
change, not a weight-init change.

The classifier head is a single linear over `hidden_size`. Bridge + memory
token construction lives in data.py - the model itself just consumes the
standard Longformer input bundle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import LongformerModel

from common.labels import NUM_POINT_LABELS


DEFAULT_ATTENTION_WINDOW = 2048


class LongCoderClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/longcoder-base",
        dropout: float = 0.1,
        attention_window: int = DEFAULT_ATTENTION_WINDOW,
    ) -> None:
        super().__init__()
        self.task = "point"
        self.num_labels = NUM_POINT_LABELS
        self.model_name = model_name
        self.attention_window = attention_window
        # Override pretrained attention_window so seq <= window everywhere
        # we run (training at 2048; ONNX deploy at 2048).
        self.encoder = LongformerModel.from_pretrained(
            model_name, attention_window=attention_window,
        )
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, self.num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        global_attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        **_unused: object,
    ) -> torch.Tensor:
        # **_unused absorbs `inputs_embeds`, `output_attentions`, etc. that
        # peft / transformers wrappers inject by default. We only consume the
        # four bundle tensors built by data.build_point_inputs.
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        pooled = _last_token_pool(out.last_hidden_state, attention_mask)
        return self.classifier(self.dropout(pooled))

    # ---- save / load -----------------------------------------------------

    def save_checkpoint(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), out / "pytorch_model.bin")
        self.encoder.config.save_pretrained(out)
        (out / "codebert_meta.json").write_text(
            json.dumps({
                "task": "point",
                "num_labels": self.num_labels,
                "model_name": self.model_name,
                "backbone": "longformer",
                "attention_window": self.attention_window,
            }, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load_checkpoint(cls, path: str | Path, **_: Any) -> "LongCoderClassifier":
        p = Path(path)
        meta = json.loads((p / "codebert_meta.json").read_text(encoding="utf-8"))
        # Older meta files (pre-attention_window-bump) don't carry the field;
        # fall back to the new default. Such checkpoints predate the seq=2048
        # change and need to be retrained anyway.
        attention_window = int(meta.get("attention_window", DEFAULT_ATTENTION_WINDOW))
        model = cls(model_name=meta["model_name"], attention_window=attention_window)
        state = torch.load(p / "pytorch_model.bin", map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=True)
        if missing or unexpected:
            print(f"[model] load: missing={len(missing)} unexpected={len(unexpected)}")
        return model


def _last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_idx = attention_mask.sum(dim=1).long() - 1
    last_idx = last_idx.clamp_min(0)
    batch_idx = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch_idx, last_idx]


def build_model(model_name: str, **kwargs: Any) -> LongCoderClassifier:
    return LongCoderClassifier(model_name=model_name, **kwargs)
