"""Phase-3 ONNX parity smoke test at seq=2048.

Patches the LongformerSelfAttention layers with FullAttentionReplacement
and asserts logits parity between the original Longformer kernel and the
patched one at the new attention_window=2048. Slow + heavy: skipped
unless RUN_ONNX_SMOKE=1 is set, since it loads the LongCoder backbone
and runs two CPU forwards at seq=2048.

This test does NOT trace ONNX — it just checks the PyTorch parity that
backs the export. The actual ONNX runtime check lives in
scripts/verify_onnx_samples.py for end-to-end pipeline runs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _have_longcoder() -> bool:
    try:
        from transformers import AutoTokenizer, LongformerModel
        AutoTokenizer.from_pretrained("microsoft/longcoder-base")
        # LongformerModel.from_pretrained pulls the weights — skip if cache cold.
        # We don't actually instantiate here to keep the gate cheap; we rely on
        # the user setting RUN_ONNX_SMOKE=1 only on warm-cache machines.
        return True
    except Exception:
        return False


_GATE = os.environ.get("RUN_ONNX_SMOKE") == "1" and _have_longcoder()


@pytest.mark.skipif(
    not _GATE,
    reason="Set RUN_ONNX_SMOKE=1 (and warm the HF cache) to enable",
)
def test_full_attention_replacement_parity_at_2048():
    """Original LongformerSelfAttention vs FullAttentionReplacement at seq=2048.

    The replacement is mathematically equivalent only at seq <= window. With
    attention_window=2048 (set by LongCoderClassifier default) this holds.
    Max-abs-diff in encoder output should be < 1e-3 in fp32.
    """
    import torch

    sys.path.insert(0, str(REPO / "scripts"))
    from longcoder_onnx_attention import (  # type: ignore
        patch_longformer_attention,
        _make_synthetic_inputs,
    )
    from model import build_model

    inputs = _make_synthetic_inputs(seq_len=2048, n_globals=8)

    bb_orig = build_model("microsoft/longcoder-base").eval()
    for p in bb_orig.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        out_orig = bb_orig.encoder(**inputs).last_hidden_state

    bb_patched = build_model("microsoft/longcoder-base").eval()
    for p in bb_patched.parameters():
        p.requires_grad_(False)
    n_replaced = patch_longformer_attention(bb_patched)
    assert n_replaced == 12, f"expected 12 attention swaps, got {n_replaced}"
    with torch.no_grad():
        out_patched = bb_patched.encoder(**inputs).last_hidden_state

    diff = (out_orig - out_patched).abs()
    max_abs = float(diff.max().item())
    assert max_abs < 1e-3, (
        f"FullAttentionReplacement parity FAILED at seq=2048: "
        f"max_abs_diff={max_abs:.6f} (expected < 1e-3)"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
