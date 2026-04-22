"""GraphCodeBERT classifier with DFG-aware attention.

Both --point and --pair share an encoder (microsoft/graphcodebert-base).
Difference is only the input construction (done in data.py) and the classifier
head size: 11 for --point, 3 for --pair.

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

from common.labels import NUM_PAIR_LABELS, NUM_POINT_LABELS


class GraphCodeBERTClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/graphcodebert-base",
        task: str = "point",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert task in ("point", "pair"), task
        self.task = task
        self.num_labels = NUM_POINT_LABELS if task == "point" else NUM_PAIR_LABELS
        # use_safetensors=False avoids HF's auto-discovery of safetensors
        # converted on PR branches (refs/pr/N) — that lookup hangs on flaky
        # LFS redirects for models like microsoft/graphcodebert-base whose
        # main branch only ships pytorch_model.bin.
        self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=False)
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
            f'{{"task":"{self.task}","num_labels":{self.num_labels},'
            f'"model_name":"{self.model_name}"}}',
            encoding="utf-8",
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        task: str | None = None,
    ) -> "GraphCodeBERTClassifier":
        import json
        p = Path(path)
        meta = json.loads((p / "codebert_meta.json").read_text(encoding="utf-8"))
        use_task = task or meta["task"]
        model = cls(model_name=meta["model_name"], task=use_task)
        state = torch.load(p / "pytorch_model.bin", map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=(use_task == meta["task"]))
        if missing or unexpected:
            print(f"[model] load: missing={len(missing)} unexpected={len(unexpected)}")
        return model

    def load_warm_start_encoder(self, warm_start_dir: str | Path) -> tuple[list[str], list[str]]:
        """Copy encoder weights from a pointwise checkpoint; reinit classifier."""
        p = Path(warm_start_dir)
        state = torch.load(p / "pytorch_model.bin", map_location="cpu")
        enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
        missing, unexpected = self.encoder.load_state_dict(enc_state, strict=False)
        return list(missing), list(unexpected)


def build_model(model_name: str, task: str, **kwargs: Any) -> GraphCodeBERTClassifier:
    return GraphCodeBERTClassifier(model_name=model_name, task=task, **kwargs)
