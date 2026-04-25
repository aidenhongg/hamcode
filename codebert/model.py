"""GraphCodeBERT pointwise classifier with DFG-aware attention.

Encoder: microsoft/graphcodebert-base. Classifier head: 11-way pointwise.

The model itself is a thin wrapper over HF RobertaModel. The DFG magic is in
data.py's collator, which builds:
  - input_ids      [B, seq]
  - attention_mask [B, seq, seq]   (3D: graph-guided attention)
  - position_ids   [B, seq]        (DFG nodes get position 0 / UNK_POS)

HF RobertaModel natively accepts a 3D attention_mask — it expands to
[B, 1, seq, seq] in `get_extended_attention_mask`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel

from common.labels import NUM_POINT_LABELS


class GraphCodeBERTClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/graphcodebert-base",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.task = "point"
        self.num_labels = NUM_POINT_LABELS
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, self.num_labels)
        self.model_name = model_name

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """attention_mask may be 2D [B, S] or 3D [B, S, S] — HF handles both."""
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))

    # ---- save / load -----------------------------------------------------

    def save_checkpoint(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), out / "pytorch_model.bin")
        self.encoder.config.save_pretrained(out)
        (out / "codebert_meta.json").write_text(
            f'{{"task":"point","num_labels":{self.num_labels},'
            f'"model_name":"{self.model_name}"}}',
            encoding="utf-8",
        )

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> "GraphCodeBERTClassifier":
        import json
        p = Path(path)
        meta = json.loads((p / "codebert_meta.json").read_text(encoding="utf-8"))
        model = cls(model_name=meta["model_name"])
        state = torch.load(p / "pytorch_model.bin", map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=True)
        if missing or unexpected:
            print(f"[model] load: missing={len(missing)} unexpected={len(unexpected)}")
        return model


def build_model(model_name: str, **kwargs: Any) -> GraphCodeBERTClassifier:
    return GraphCodeBERTClassifier(model_name=model_name, **kwargs)
