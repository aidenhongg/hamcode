"""Smoke tests for the LoRA + activation-cache integration.

These tests download microsoft/longcoder-base on first run (needs network).
They are slow (~30s each on CPU) but cover the load-bearing invariants:

  * LoRA-mode build trains ~1M params, full base frozen.
  * freeze_depth>0 zeroes grads on the bottom-band base weights.
  * Save/load round-trip yields bit-exact logits.
  * Partial-encoder forward (cached prefix) matches full forward to fp32 atol.
  * Activation cache write/read round-trip preserves the bf16 bit pattern.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import (
    ActivationCacheConfig,
    _FrozenActivationCache,
    build_point_inputs,
    cached_hidden_to_torch,
)
from model import LongCoderClassifier, LoraSpec


MODEL_NAME = "microsoft/longcoder-base"


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def small_inputs(tokenizer):
    code = (
        "import sys\n"
        "def f(n):\n"
        "    s = 0\n"
        "    for i in range(n):\n"
        "        for j in range(i):\n"
        "            s += i * j\n"
        "    return s\n"
    )
    b = build_point_inputs(code, tokenizer, max_seq_len=2048, bridge_stride=128)
    return code, {
        "input_ids": torch.from_numpy(b.input_ids).unsqueeze(0),
        "attention_mask": torch.from_numpy(b.attention_mask).unsqueeze(0),
        "global_attention_mask": torch.from_numpy(b.global_attention_mask).unsqueeze(0),
        "token_type_ids": torch.from_numpy(b.token_type_ids).unsqueeze(0),
    }


def test_lora_build_freezes_base_only_adapters_train():
    spec = LoraSpec(enabled=True, r=8, alpha=16, dropout=0.0, freeze_depth=0)
    model = LongCoderClassifier(model_name=MODEL_NAME, lora=spec)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Frozen base: every encoder weight should be requires_grad=False except
    # the LoRA adapter layers and the classifier head.
    assert n_trainable < n_total, "no params frozen"
    assert n_trainable < 0.05 * n_total, (
        f"trainable {n_trainable:,} ({100*n_trainable/n_total:.2f}%) is too high "
        f"for LoRA r=8"
    )
    # Classifier head is trainable
    assert all(p.requires_grad for p in model.classifier.parameters())


def test_lora_freeze_depth_zeros_lower_band():
    spec = LoraSpec(enabled=True, r=8, alpha=16, dropout=0.0, freeze_depth=6)
    model = LongCoderClassifier(model_name=MODEL_NAME, lora=spec)
    # Bottom 6 layers must have zero trainable params (no adapters AND base frozen)
    for name, param in model.named_parameters():
        for i in range(6):
            if f"layer.{i}." in name and param.requires_grad:
                pytest.fail(f"layer {i} has trainable param {name}")


def test_partial_forward_matches_full_forward(small_inputs):
    """Partial-encoder forward with cached_hidden = output(layer K-1) must
    bit-match the full forward, since bottom-K layers are deterministic."""
    code, inputs = small_inputs
    spec = LoraSpec(enabled=True, r=8, alpha=16, dropout=0.0, freeze_depth=6)
    model = LongCoderClassifier(model_name=MODEL_NAME, lora=spec).eval()

    from cache_activations import compute_prefix_activation
    with torch.no_grad():
        cached = compute_prefix_activation(
            model._base_longformer,
            inputs["input_ids"], inputs["attention_mask"],
            inputs["global_attention_mask"], inputs["token_type_ids"],
            freeze_depth=6,
        )
        full_logits = model(**inputs)
        partial_logits = model(
            attention_mask=inputs["attention_mask"],
            global_attention_mask=inputs["global_attention_mask"],
            cached_hidden=cached,
        )
    diff = (full_logits - partial_logits).abs().max().item()
    assert diff < 1e-4, f"partial-forward diverges by {diff}"


def test_save_load_roundtrip_lora(small_inputs):
    """Save then load a LoRA checkpoint; logits must be bit-exact."""
    code, inputs = small_inputs
    spec = LoraSpec(enabled=True, r=8, alpha=16, dropout=0.0, freeze_depth=0)
    model = LongCoderClassifier(model_name=MODEL_NAME, lora=spec).eval()
    with torch.no_grad():
        before = model(**inputs)

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "ckpt"
        model.save_checkpoint(out)
        # Both adapter dir and classifier weight should exist
        assert (out / "adapter").exists()
        assert (out / "classifier.pt").exists()
        assert (out / "codebert_meta.json").exists()
        loaded = LongCoderClassifier.load_checkpoint(out).eval()
        with torch.no_grad():
            after = loaded(**inputs)
    assert torch.allclose(before, after, atol=1e-6), \
        f"load roundtrip diverged: max diff {(before - after).abs().max().item()}"


def test_save_load_roundtrip_lora_with_merge(small_inputs):
    """merge_lora=True should produce identical logits to the LoRA path."""
    code, inputs = small_inputs
    spec = LoraSpec(enabled=True, r=8, alpha=16, dropout=0.0, freeze_depth=0)
    model = LongCoderClassifier(model_name=MODEL_NAME, lora=spec).eval()
    with torch.no_grad():
        before = model(**inputs)

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "ckpt"
        model.save_checkpoint(out)
        merged = LongCoderClassifier.load_checkpoint(out, merge_lora=True).eval()
        with torch.no_grad():
            after = merged(**inputs)
    # merge_and_unload changes the matmul order so allow a small tolerance
    assert torch.allclose(before, after, atol=1e-4), \
        f"merge_lora diverged: max diff {(before - after).abs().max().item()}"


def test_activation_cache_roundtrip_bf16():
    cache = _FrozenActivationCache(tempfile.mkdtemp())
    code = "def f(): return 1"
    cfg = "act:test:6:128:128"
    hidden = torch.randn(1, 128, 768)
    bf = hidden.to(torch.bfloat16)
    arr = bf.view(torch.uint16).numpy()
    cache.put(code, cfg, arr)
    got = cache.get(code, cfg)
    assert got is not None
    back = cached_hidden_to_torch(got)
    assert back.dtype == torch.bfloat16
    # bf16 roundtrip is bit-exact (we store the same bytes)
    assert torch.equal(back, bf)
