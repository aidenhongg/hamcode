"""LongCoder pointwise classifier with optional LoRA finetuning.

Encoder: microsoft/longcoder-base (Longformer, 4096 max positions, sliding
window 1024). LongCoder was pretrained autoregressively for code completion.
We adapt it to bidirectional classification by:
  - leaving HF's default bidirectional Longformer attention on (no causal mask),
  - pooling the last non-pad hidden state instead of [CLS]. The first position
    has effectively no learned representation under the pretraining objective,
    while the last position carries the prefix's compressed summary.

Two finetuning paths:
  * Full FT — every parameter trainable.
  * LoRA   — base weights frozen, low-rank adapters injected on configurable
             attention projections; can additionally restrict adapters to a
             top-K layer band so the bottom (12 - K) layers stay fully frozen.
             When the bottom band is fully frozen, the per-sample activation
             at the K/(12-K) boundary is reusable across epochs and can be
             cached on disk (see data._FrozenActivationCache and
             cache_activations.py). Forward then resumes from layer K.

Bridge + memory token construction lives in data.py — the model itself just
consumes the standard Longformer input bundle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import LongformerModel

from common.labels import NUM_POINT_LABELS

DEFAULT_LORA_TARGETS = ("query", "value", "query_global", "value_global")


@dataclass
class LoraSpec:
    enabled: bool = False
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = DEFAULT_LORA_TARGETS
    freeze_depth: int = 0  # 0 = LoRA on all layers; K = layers 0..K-1 fully frozen, LoRA on K..11

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled, "r": self.r, "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
            "freeze_depth": self.freeze_depth,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "LoraSpec":
        if not d or not d.get("enabled"):
            return cls(enabled=False)
        return cls(
            enabled=True,
            r=int(d.get("r", 16)),
            alpha=int(d.get("alpha", 32)),
            dropout=float(d.get("dropout", 0.05)),
            target_modules=tuple(d.get("target_modules", DEFAULT_LORA_TARGETS)),
            freeze_depth=int(d.get("freeze_depth", 0)),
        )


class LongCoderClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/longcoder-base",
        dropout: float = 0.1,
        lora: LoraSpec | None = None,
    ) -> None:
        super().__init__()
        self.task = "point"
        self.num_labels = NUM_POINT_LABELS
        self.model_name = model_name
        self.lora = lora or LoraSpec(enabled=False)

        base = LongformerModel.from_pretrained(model_name)
        if self.lora.enabled:
            from peft import LoraConfig, get_peft_model, TaskType
            n_layers = base.config.num_hidden_layers
            cfg_kwargs: dict = dict(
                r=self.lora.r,
                lora_alpha=self.lora.alpha,
                lora_dropout=self.lora.dropout,
                target_modules=list(self.lora.target_modules),
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            if self.lora.freeze_depth > 0:
                cfg_kwargs["layers_to_transform"] = list(
                    range(self.lora.freeze_depth, n_layers)
                )
                cfg_kwargs["layers_pattern"] = "layer"
            cfg = LoraConfig(**cfg_kwargs)
            base = get_peft_model(base, cfg)
            if self.lora.freeze_depth > 0:
                _freeze_below(base, self.lora.freeze_depth)
        self.encoder = base

        hidden = self._base_longformer.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, self.num_labels)

    # ---- helpers ---------------------------------------------------------

    @property
    def _base_longformer(self) -> LongformerModel:
        """The underlying LongformerModel, regardless of PEFT wrapping."""
        enc = self.encoder
        try:
            from peft import PeftModel
            if isinstance(enc, PeftModel):
                return enc.base_model.model
        except ImportError:
            pass
        return enc

    @property
    def freeze_depth(self) -> int:
        return self.lora.freeze_depth

    # ---- forward ---------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        global_attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        cached_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cached_hidden is None:
            return self._forward_full(
                input_ids, attention_mask, global_attention_mask, token_type_ids,
            )
        return self._forward_with_cached_prefix(
            cached_hidden, attention_mask, global_attention_mask,
        )

    def _forward_full(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        global_attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        pooled = _last_token_pool(out.last_hidden_state, attention_mask)
        return self.classifier(self.dropout(pooled))

    def _forward_with_cached_prefix(
        self,
        cached_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        global_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Resume Longformer from layer K with cached_hidden as that layer's input.

        Mirrors LongformerEncoder.forward starting at index K = self.freeze_depth.
        cached_hidden must be the (un-padded) post-layer-(K-1) activation produced
        by cache_activations.compute_prefix_activation on the same inputs.

        Pads cached_hidden + masks to a multiple of attention_window (HF's
        sliding-window kernel hard-asserts this), runs layers K..11, then
        unpads back to the original seq_len before pooling.
        """
        if self.freeze_depth <= 0:
            raise ValueError(
                "cached_hidden requires freeze_depth > 0; got freeze_depth=0."
            )
        lf = self._base_longformer
        # The cache may store activations as bf16 (default) but the encoder weights
        # are fp32. Under autocast the dtypes auto-align; without autocast (eval)
        # bf16 input + fp32 Linear raises. Always cast cached_hidden to the encoder's
        # parameter dtype so the layer call works in both contexts. Autocast will
        # downcast for the matmul if it's active.
        cached_hidden = cached_hidden.to(dtype=next(lf.parameters()).dtype)
        orig_len = cached_hidden.size(1)
        hidden, padded_attn, padded_global = _pad_for_longformer(
            lf, cached_hidden, attention_mask, global_attention_mask,
        )
        extended, is_index_masked, is_index_global_attn, is_global_attn = setup_longformer_masks(
            lf, padded_attn, padded_global,
        )
        for layer_module in lf.encoder.layer[self.freeze_depth:]:
            hidden = layer_module(
                hidden,
                attention_mask=extended,
                layer_head_mask=None,
                is_index_masked=is_index_masked,
                is_index_global_attn=is_index_global_attn,
                is_global_attn=is_global_attn,
                output_attentions=False,
            )[0]
        hidden = hidden[:, :orig_len]
        pooled = _last_token_pool(hidden, attention_mask)
        return self.classifier(self.dropout(pooled))

    # ---- save / load -----------------------------------------------------

    def save_checkpoint(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        meta = {
            "task": "point",
            "num_labels": self.num_labels,
            "model_name": self.model_name,
            "backbone": "longformer",
            "lora": self.lora.to_dict(),
        }
        if self.lora.enabled:
            self.encoder.save_pretrained(out / "adapter")
            torch.save(self.classifier.state_dict(), out / "classifier.pt")
            self._base_longformer.config.save_pretrained(out)
        else:
            torch.save(self.state_dict(), out / "pytorch_model.bin")
            self.encoder.config.save_pretrained(out)
        (out / "codebert_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8",
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        merge_lora: bool = False,
        **_: Any,
    ) -> "LongCoderClassifier":
        """Load a checkpoint produced by save_checkpoint.

        merge_lora=True merges the adapter into the base weights and detaches
        PEFT — useful for inference / extraction (skips per-forward LoRA
        matmul). Training callers should leave merge_lora=False.
        """
        p = Path(path)
        meta = json.loads((p / "codebert_meta.json").read_text(encoding="utf-8"))
        lora = LoraSpec.from_dict(meta.get("lora"))
        # Build with full weights only — we'll attach the adapter below if needed.
        model = cls(
            model_name=meta["model_name"],
            lora=LoraSpec(enabled=False),
        )
        if lora.enabled:
            from peft import PeftModel
            adapter_dir = p / "adapter"
            if not adapter_dir.exists():
                raise FileNotFoundError(
                    f"meta says lora=true but {adapter_dir} is missing"
                )
            peft_model = PeftModel.from_pretrained(model.encoder, adapter_dir)
            if merge_lora:
                model.encoder = peft_model.merge_and_unload()
                model.lora = LoraSpec(enabled=False)
            else:
                model.encoder = peft_model
                model.lora = lora
                if lora.freeze_depth > 0:
                    _freeze_below(model.encoder, lora.freeze_depth)
            cls_path = p / "classifier.pt"
            if cls_path.exists():
                model.classifier.load_state_dict(torch.load(cls_path, map_location="cpu"))
        else:
            state = torch.load(p / "pytorch_model.bin", map_location="cpu")
            missing, unexpected = model.load_state_dict(state, strict=True)
            if missing or unexpected:
                print(f"[model] load: missing={len(missing)} unexpected={len(unexpected)}")
        return model


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_idx = attention_mask.sum(dim=1).long() - 1
    last_idx = last_idx.clamp_min(0)
    batch_idx = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch_idx, last_idx]


def _freeze_below(model: nn.Module, depth: int) -> None:
    """Set requires_grad=False on every parameter in `encoder.layer.{i}` for i < depth.

    Belt-and-suspenders to PEFT's `layers_to_transform`: PEFT only injects
    adapters into the requested layers, but LayerNorm + residual weights in
    the bottom band would otherwise still be trainable (they are by default
    on `LongformerModel.from_pretrained`). We freeze them so the cached
    activation at the K-boundary stays valid across optimizer steps.
    """
    n_frozen = 0
    for name, param in model.named_parameters():
        for i in range(depth):
            if f"layer.{i}." in name:
                param.requires_grad = False
                n_frozen += 1
                break
    return n_frozen


def setup_longformer_masks(
    lf: LongformerModel,
    attention_mask: torch.Tensor,
    global_attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Reproduce LongformerModel.forward's mask preprocessing.

    Returns (extended_attn_mask, is_index_masked, is_index_global_attn, is_global_attn).
    Used by both the partial-forward path (model._forward_with_cached_prefix)
    and the cache-population helper (cache_activations.compute_prefix_activation).
    """
    if global_attention_mask is not None:
        attention_mask = lf._merge_to_attention_mask(attention_mask, global_attention_mask)
    extended = lf.get_extended_attention_mask(attention_mask, attention_mask.shape)[:, 0, 0, :]
    return extended, extended < 0, extended > 0, (extended > 0).flatten().any().item()


def _longformer_window_size(lf: LongformerModel) -> int:
    w = lf.config.attention_window
    return w[0] if isinstance(w, list) else int(w)


def _pad_for_longformer(
    lf: LongformerModel,
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    global_attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Pad hidden states + masks to a multiple of attention_window.

    HF's sliding-window kernel asserts seq_len % attention_window == 0.
    LongformerModel.forward calls _pad_to_window_size for raw inputs; we
    replicate the right-pad on already-embedded hidden states. Padded
    positions get attention_mask=0 (and global_attention_mask=0) so the
    layers ignore them.
    """
    window = _longformer_window_size(lf)
    seq_len = hidden.size(1)
    rem = seq_len % window
    if rem == 0:
        return hidden, attention_mask, global_attention_mask
    pad = window - rem
    hidden_p = torch.nn.functional.pad(hidden, (0, 0, 0, pad), value=0.0)
    attn_p = torch.nn.functional.pad(attention_mask, (0, pad), value=0)
    g_attn_p = (torch.nn.functional.pad(global_attention_mask, (0, pad), value=0)
                if global_attention_mask is not None else None)
    return hidden_p, attn_p, g_attn_p


def build_model(
    model_name: str,
    lora: LoraSpec | dict | None = None,
    **kwargs: Any,
) -> LongCoderClassifier:
    if isinstance(lora, dict):
        lora = LoraSpec.from_dict(lora)
    return LongCoderClassifier(model_name=model_name, lora=lora, **kwargs)
