"""Phase-3 seq=2048 + attention_window=2048 invariants.

Cheap config + tokenizer-only checks always run. Heavy checks (loading
LongCoder, forward passes) are skipped unless explicitly opted into via
env var RUN_LONGCODER_SMOKE=1 — the model load pulls ~570 MB and a CPU
forward at seq=2048 is slow.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# Config invariants — cheap, always run
# ---------------------------------------------------------------------------

def test_point_yaml_seq_2048_batch_4():
    cfg = yaml.safe_load((REPO / "configs" / "point.yaml").read_text(encoding="utf-8"))
    assert cfg["max_seq_len"] == 2048
    assert cfg["batch_size"] == 4
    assert cfg["grad_accum"] == 8
    # bridge_stride untouched
    assert cfg["bridge_stride"] == 128


def test_lora_yaml_seq_2048_batch_4():
    cfg = yaml.safe_load((REPO / "configs" / "lora.yaml").read_text(encoding="utf-8"))
    assert cfg["max_seq_len"] == 2048
    assert cfg["batch_size"] == 4
    assert cfg["grad_accum"] == 8


def test_train_config_default_seq_2048():
    from train import Config
    cfg = Config()
    assert cfg.max_seq_len == 2048
    assert cfg.batch_size == 4
    assert cfg.grad_accum == 8


def test_lora_config_default_seq_2048():
    from lora_train import LoraCfg
    cfg = LoraCfg()
    assert cfg.max_seq_len == 2048
    assert cfg.batch_size == 4
    assert cfg.grad_accum == 8


def test_predict_max_seq_len_default_2048():
    """The CLI default propagates to deployment."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("predict", REPO / "predict.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # parse_args is callable; we'd want to inspect the parser default
    # Easiest: read source and grep for the literal — already covered by ONNX
    # constants below. Instead verify the source contains the literal.
    src = (REPO / "predict.py").read_text(encoding="utf-8")
    assert 'default=2048' in src


def test_onnx_max_seq_len_2048():
    src = (REPO / "scripts" / "export_onnx.py").read_text(encoding="utf-8")
    assert "MAX_SEQ_LEN = 2048" in src
    # Sanity: the FullAttentionReplacement docstring is consistent.
    src2 = (REPO / "scripts" / "longcoder_onnx_attention.py").read_text(encoding="utf-8")
    assert "2048 <= 2048" in src2


def test_model_default_attention_window_2048():
    """LongCoderClassifier picks up the new default attention_window."""
    from model import DEFAULT_ATTENTION_WINDOW
    assert DEFAULT_ATTENTION_WINDOW == 2048


# ---------------------------------------------------------------------------
# Tokenizer / dataset shape — needs HF cache but no model weights
# ---------------------------------------------------------------------------

def _have_longcoder_tokenizer() -> bool:
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained("microsoft/longcoder-base")
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _have_longcoder_tokenizer(),
    reason="LongCoder tokenizer not in HF cache; needs network or pre-warm",
)
def test_build_point_inputs_shape_at_2048():
    from transformers import AutoTokenizer
    from data import build_point_inputs

    tok = AutoTokenizer.from_pretrained("microsoft/longcoder-base")
    # Code long enough to exceed bridge_stride=128 so we exercise the bridge
    # insertion path; ~600 BPE tokens with this loop body.
    code = (
        "def f(arr):\n"
        + "".join(
            f"    arr[{i}] = arr[{i}] * arr[{i+1}] + arr[{i+2}]\n"
            for i in range(60)
        )
        + "    return arr\n"
    )
    bundle = build_point_inputs(
        code, tok, max_seq_len=2048, bridge_stride=128, language="python",
    )
    assert bundle.input_ids.shape == (2048,)
    assert bundle.attention_mask.shape == (2048,)
    assert bundle.global_attention_mask.shape == (2048,)
    assert bundle.token_type_ids.shape == (2048,)
    # At ~600 code tokens with stride 128 we expect ~4 bridge tokens.
    n_bridge = int((bundle.token_type_ids == 1).sum())
    assert n_bridge >= 1, f"expected bridge tokens at stride=128 with ~600 code tokens, got {n_bridge}"


# ---------------------------------------------------------------------------
# Heavy: load LongCoder, assert attention_window, optionally forward
# ---------------------------------------------------------------------------

_LONGCODER_GATE = (
    os.environ.get("RUN_LONGCODER_SMOKE") == "1"
    and _have_longcoder_tokenizer()
)


@pytest.mark.skipif(
    not _LONGCODER_GATE,
    reason="Set RUN_LONGCODER_SMOKE=1 (and warm the HF cache) to enable",
)
def test_longcoder_attention_window_is_2048():
    from model import build_model
    m = build_model("microsoft/longcoder-base")
    aw = m.encoder.config.attention_window
    if isinstance(aw, list):
        assert all(w == 2048 for w in aw), aw
        assert len(aw) == 12
    else:
        assert aw == 2048


@pytest.mark.skipif(
    not _LONGCODER_GATE,
    reason="Set RUN_LONGCODER_SMOKE=1 (and warm the HF cache) to enable",
)
def test_longcoder_forward_at_seq_2048_no_oom():
    """Single-batch CPU forward at seq=2048. Slow (~30-60s on CPU) but proves
    the attention layers don't blow up at the new seq_len."""
    import torch
    from data import build_point_inputs
    from model import build_model
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("microsoft/longcoder-base")
    code = "def add(a, b):\n    return a + b\n"
    bundle = build_point_inputs(
        code, tok, max_seq_len=2048, bridge_stride=128, language="python",
    )
    m = build_model("microsoft/longcoder-base").eval()
    with torch.no_grad():
        logits = m(
            input_ids=torch.from_numpy(bundle.input_ids).unsqueeze(0),
            attention_mask=torch.from_numpy(bundle.attention_mask).unsqueeze(0),
            global_attention_mask=torch.from_numpy(bundle.global_attention_mask).unsqueeze(0),
            token_type_ids=torch.from_numpy(bundle.token_type_ids).unsqueeze(0),
        )
    assert logits.shape == (1, 11)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
