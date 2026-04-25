"""LongCoder pointwise classifier.

Encoder: microsoft/longcoder-base (Longformer, 4096 max positions, sliding
window 1024). LongCoder was pretrained autoregressively for code completion.
We adapt it to bidirectional classification by:
  - leaving HF's default bidirectional Longformer attention on (no causal mask),
  - pooling the last non-pad hidden state instead of [CLS]. The first position
    has effectively no learned representation under the pretraining objective,
    while the last position carries the prefix's compressed summary.

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


class LongCoderClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/longcoder-base",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.task = "point"
        self.num_labels = NUM_POINT_LABELS
        self.model_name = model_name
        self.encoder = LongformerModel.from_pretrained(model_name)
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
            }, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load_checkpoint(cls, path: str | Path, **_: Any) -> "LongCoderClassifier":
        p = Path(path)
        meta = json.loads((p / "codebert_meta.json").read_text(encoding="utf-8"))
        model = cls(model_name=meta["model_name"])
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
