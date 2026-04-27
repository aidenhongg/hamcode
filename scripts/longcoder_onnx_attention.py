"""ONNX-clean drop-in replacement for ``LongformerSelfAttention``.

Background
----------
``transformers.models.longformer.modeling_longformer.LongformerSelfAttention``
uses advanced indexing patterns (``aten::index`` / ``index_put``) that fail to
ONNX-export correctly across all known exporter paths (legacy tracer, dynamo,
optimum). At ``seq_len <= attention_window`` (our deployment case: 2048 <= 2048)
sliding-window attention degenerates to full attention, so we can replace the
custom kernel with a vanilla MatMul-based attention layer that uses only
ONNX-friendly ops (MatMul / Add / Softmax / Where / Reshape / Transpose).

Mathematical equivalence
------------------------
Longformer fuses two parallel attention modes:

  - LOCAL queries attend to a CONCATENATED key set:
      [global_keys (n_global cols)] || [local_keys (window cols)]
    The "global keys" here are NOT from ``self.key_global``; they are
    ``self.key`` evaluated at global positions only. Same for V. So at
    seq_len <= window, the local-query branch is mathematically:

        scores = Q_local @ [K_local_at_globals || K_local]^T  / sqrt(Dh)
        # mask: first n_global cols are valid only at global positions,
        #       both halves are masked at padded positions.
        probs  = softmax(scores)
        out    = probs @ [V_local_at_globals || V_local]

    Operationally, since the "at globals" packing is just a column-mask
    against the full-seq tensor, we set K_ext = concat([K_local, K_local], -2)
    and V_ext = concat([V_local, V_local], -2), then mask the first half's
    columns to be valid only where is_index_global_attn AND NOT is_index_masked.
    This yields the same softmax weights as the original packed form.

  - GLOBAL queries do full attention with Q_global / K_global / V_global
    (all from the ``*_global`` linear projections).

The combined output is selected via ``torch.where(is_index_global_attn, ...)``,
mirroring how the original code overwrites local-query outputs at global
positions with global-query outputs.

Critical detail: pad rows must be zeroed AFTER softmax (matches original line
~590 of ``modeling_longformer.py``). The "corner mask" applied by
``_mask_invalid_locations`` in the sliding-window kernel is a NO-OP at
``seq_len == attention_window``, so we can safely skip it.

Usage
-----
    from longcoder_onnx_attention import patch_longformer_attention
    patch_longformer_attention(model)   # in-place; returns count of replacements

Run as ``__main__`` to execute the synthetic-input parity gate.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# Make the project root importable when run as __main__.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from transformers.models.longformer.modeling_longformer import LongformerSelfAttention

logger = logging.getLogger("longcoder_onnx_attention")


# ---------------------------------------------------------------------------
# FullAttentionReplacement
# ---------------------------------------------------------------------------

class FullAttentionReplacement(nn.Module):
    """Drop-in replacement for ``LongformerSelfAttention`` (full-attention path).

    Valid when ``seq_len <= attention_window`` (so sliding window collapses to
    full attention). All ops are ONNX-friendly; tested at opset 17.

    The 6 projection layers are captured BY REFERENCE from the original
    module — this preserves any peft-LoRA wrappers that were attached to those
    submodules. Do not deep-copy or rebuild the projections; that would break
    LoRA hooking.
    """

    def __init__(self, orig: LongformerSelfAttention) -> None:
        super().__init__()
        # Capture references to the (possibly peft-wrapped) projection layers.
        self.query = orig.query
        self.key = orig.key
        self.value = orig.value
        self.query_global = orig.query_global
        self.key_global = orig.key_global
        self.value_global = orig.value_global

        self.num_heads: int = int(orig.num_heads)
        self.head_dim: int = int(orig.head_dim)
        self.embed_dim: int = int(orig.embed_dim)
        # one_sided_attn_window_size: each local query attends to keys in
        # [i - W, i + W]. At seq_len <= 2W+1 the band may cover everything;
        # at seq_len just equal to attention_window=2W, edge queries still
        # have band-mask cutouts. We must replicate this banded structure to
        # match the original sliding-window kernel exactly.
        self.one_sided_attn_window_size: int = int(orig.one_sided_attn_window_size)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, D] -> [B, H, L, Dh]."""
        B, L, _ = x.shape
        # view -> [B, L, H, Dh] -> transpose -> [B, H, L, Dh]
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        layer_head_mask: torch.Tensor | None = None,
        is_index_masked: torch.Tensor | None = None,
        is_index_global_attn: torch.Tensor | None = None,
        is_global_attn: bool | torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Compute full attention with dual-key (global || local) structure.

        Inputs:
          hidden_states: [B, L, D] (caller convention; we keep batch-first)
          is_index_masked: [B, L] bool — True at padded query positions
          is_index_global_attn: [B, L] bool — True at global tokens

        ``attention_mask`` and ``is_global_attn`` are accepted for interface
        compatibility but unused: the masking information needed is fully
        captured by the two bool tensors above. ``layer_head_mask`` is None
        in our use case; if non-None, we multiply post-softmax probs as in the
        original. ``output_attentions`` is False in inference.
        """
        if is_index_masked is None or is_index_global_attn is None:
            raise RuntimeError(
                "FullAttentionReplacement.forward requires is_index_masked and "
                "is_index_global_attn (they are derived in LongformerEncoder). "
                "Got None — did you call this layer outside the standard encoder?"
            )

        B, L, D = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim
        scale = 1.0 / math.sqrt(Dh)

        # ---- Step 1: project. ------------------------------------------------
        Q_local = self._split_heads(self.query(hidden_states))   # [B, H, L, Dh]
        K_local = self._split_heads(self.key(hidden_states))
        V_local = self._split_heads(self.value(hidden_states))
        Q_global = self._split_heads(self.query_global(hidden_states))
        K_global = self._split_heads(self.key_global(hidden_states))
        V_global = self._split_heads(self.value_global(hidden_states))

        # ---- Step 2: LOCAL-query attention with concat-and-mask key set. ----
        # The "global key" half is K_local restricted to global positions; the
        # second half is full K_local. Same for V. Operationally this is just
        # K_local concatenated with itself, with the first half's columns
        # masked out at non-global positions (see column-validity mask below).
        # K_ext / V_ext: [B, H, 2L, Dh]
        K_ext = torch.cat([K_local, K_local], dim=-2)
        V_ext = torch.cat([V_local, V_local], dim=-2)

        # scores: [B, H, L, 2L]
        scores_local = torch.matmul(Q_local, K_ext.transpose(-1, -2)) * scale

        # Build a single [B, 1, L, 2L] validity mask over the 2L key axis.
        #   - first L cols  (global keys, no band limit):
        #         valid iff is_index_global_attn AND NOT is_index_masked
        #     (broadcasts over query rows — same column validity for every i)
        #   - second L cols (local keys, banded to |j-i| <= W):
        #         valid iff (NOT is_index_masked) AND (NOT is_index_global_attn)
        #         AND within band of query i.
        #     This second condition mirrors the original
        #     ``remove_from_windowed_attention_mask = (attention_mask != 0)``
        #     trick: in the windowed local attention path, the original masks
        #     out BOTH padded AND global positions on the key side, since
        #     global positions are accounted for separately in the global-key
        #     half (the first L cols).
        not_masked = ~is_index_masked                                 # [B, L]
        not_global = ~is_index_global_attn                            # [B, L]
        first_half_valid = is_index_global_attn & not_masked          # [B, L]
        # Broadcast first-half validity to query-row axis: [B, 1, L, L_key]
        global_valid_4d = first_half_valid[:, None, None, :].expand(B, 1, L, L)

        # Per-query band + column-validity for local-key half.
        device = scores_local.device
        idx = torch.arange(L, device=device)
        # band[i, j] = True iff |j - i| <= W. Shape [L, L].
        band = (idx[None, :] - idx[:, None]).abs() <= self.one_sided_attn_window_size
        # local_key_col_valid[B, L_key] = not_masked & not_global
        local_key_col_valid = not_masked & not_global                  # [B, L]
        # local_valid[B, L_query, L_key] = band & col_valid_at_key
        local_valid = band[None, :, :] & local_key_col_valid[:, None, :]   # [B, L, L]
        local_valid_4d = local_valid[:, None, :, :]                   # [B, 1, L, L]

        # Concat along the key axis: [B, 1, L, 2L]
        valid_4d = torch.cat([global_valid_4d, local_valid_4d], dim=-1)
        invalid_4d = ~valid_4d

        neg_inf = torch.finfo(scores_local.dtype).min
        scores_local = scores_local.masked_fill(invalid_4d, neg_inf)

        # Softmax in fp32 for stability (matches original), cast back to input dtype.
        probs_local = torch.softmax(scores_local, dim=-1, dtype=torch.float32).to(scores_local.dtype)

        if layer_head_mask is not None:
            probs_local = layer_head_mask.view(1, -1, 1, 1) * probs_local

        # Zero out probs at padded query rows (matches original line ~590).
        probs_local = probs_local.masked_fill(
            is_index_masked[:, None, :, None], 0.0
        )

        attn_local_h = torch.matmul(probs_local, V_ext)               # [B, H, L, Dh]
        attn_local = attn_local_h.transpose(1, 2).contiguous().view(B, L, D)

        # ---- Step 3: GLOBAL-query attention with K_global / V_global. -------
        scores_global = torch.matmul(Q_global, K_global.transpose(-1, -2)) * scale
        scores_global = scores_global.masked_fill(
            is_index_masked[:, None, None, :], neg_inf
        )
        probs_global = torch.softmax(scores_global, dim=-1, dtype=torch.float32).to(scores_global.dtype)

        if layer_head_mask is not None:
            probs_global = layer_head_mask.view(1, -1, 1, 1) * probs_global

        probs_global = probs_global.masked_fill(
            is_index_masked[:, None, :, None], 0.0
        )

        attn_global_h = torch.matmul(probs_global, V_global)          # [B, H, L, Dh]
        attn_global = attn_global_h.transpose(1, 2).contiguous().view(B, L, D)

        # ---- Step 4: Combine — global-output where global query, else local.
        # is_index_global_attn: [B, L] bool -> broadcast to [B, L, 1] over [B, L, D].
        gate = is_index_global_attn[:, :, None]
        output = torch.where(gate, attn_global, attn_local)

        # ---- Step 5: Return tuple matching LongformerSelfAttention contract. -
        # LongformerAttention only indexes [0] and forwards [1:] — so when
        # output_attentions=False we just return (output,). We provide None
        # placeholders if attn weights are requested (downstream uses are
        # tolerant of None for our deployment).
        if output_attentions:
            return (output, probs_local, probs_global)
        return (output,)


# ---------------------------------------------------------------------------
# Patch helper
# ---------------------------------------------------------------------------

def patch_longformer_attention(model: nn.Module) -> int:
    """Replace every ``LongformerSelfAttention`` in ``model`` with
    ``FullAttentionReplacement``. In-place. Returns the count of replacements.

    Walks all submodules. For any module that has an attribute named ``self``
    of type ``LongformerSelfAttention``, swap it. This naming is fixed by
    transformers' ``LongformerAttention`` — the self-attention sub-block is
    always called ``self``.
    """
    n_replaced = 0
    for module in model.modules():
        # Look for the LongformerAttention container, which always carries a
        # `self` attribute of LongformerSelfAttention type.
        attr = getattr(module, "self", None)
        if isinstance(attr, LongformerSelfAttention):
            replacement = FullAttentionReplacement(attr)
            module.self = replacement
            n_replaced += 1
    logger.info("patched %d LongformerSelfAttention layers", n_replaced)
    return n_replaced


# ---------------------------------------------------------------------------
# Synthetic-input parity test
# ---------------------------------------------------------------------------

def _make_synthetic_inputs(seq_len: int, n_globals: int) -> dict[str, torch.Tensor]:
    """Deterministic inputs mirroring make_dummy_inputs in scripts/export_onnx.py.

    Last 8 positions are padded; first / last-real / strided positions are global.
    """
    import numpy as np
    rng = np.random.default_rng(0)
    B = 1
    input_ids = torch.from_numpy(
        rng.integers(low=4, high=50000, size=(B, seq_len), dtype=np.int64)
    )
    real = seq_len - 8
    attention_mask = torch.zeros(B, seq_len, dtype=torch.int64)
    attention_mask[:, :real] = 1

    global_attention_mask = torch.zeros(B, seq_len, dtype=torch.int64)
    # Place ~n_globals globals: position 0, last-real, then strided.
    global_attention_mask[:, 0] = 1
    global_attention_mask[:, real - 1] = 1
    # Stride globals to cover roughly n_globals positions across the real range.
    if n_globals > 2:
        stride = max(1, real // (n_globals - 1))
        global_attention_mask[:, ::stride] = 1
        # Re-zero pad positions just in case the stride hit them.
        global_attention_mask[:, real:] = 0

    token_type_ids = torch.zeros(B, seq_len, dtype=torch.int64)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "global_attention_mask": global_attention_mask,
        "token_type_ids": token_type_ids,
    }


def parity_test(
    backbone_path: Path,
    seq_len: int = 2048,
    n_globals: int = 8,
    atol: float = 1e-4,
) -> dict[str, Any]:
    """Run original Longformer attention vs patched on synthetic input.

    Loads the backbone twice (the patch is in-place) and compares logits.
    Returns a dict with metrics; raises nothing — caller decides on threshold.
    """
    from model import LongCoderClassifier

    inputs = _make_synthetic_inputs(seq_len=seq_len, n_globals=n_globals)
    n_global_actual = int(inputs["global_attention_mask"].sum().item())
    logger.info(
        "parity_test: seq_len=%d, requested_globals=%d, actual=%d, padded=%d",
        seq_len, n_globals, n_global_actual, int((inputs["attention_mask"] == 0).sum().item()),
    )

    # ---- Original Longformer attention. ----
    bb_orig = LongCoderClassifier.load_checkpoint(backbone_path)
    bb_orig.eval()
    for p in bb_orig.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        logits_orig = bb_orig(**inputs)

    # ---- Patched attention (fresh load). ----
    bb_patched = LongCoderClassifier.load_checkpoint(backbone_path)
    bb_patched.eval()
    for p in bb_patched.parameters():
        p.requires_grad_(False)
    n_replaced = patch_longformer_attention(bb_patched)
    with torch.no_grad():
        logits_patched = bb_patched(**inputs)

    diff = (logits_orig - logits_patched).abs()
    max_abs_diff = float(diff.max().item())
    mean_abs_diff = float(diff.mean().item())

    # Sanity: encoder output (pre-classifier) parity too.
    with torch.no_grad():
        out_orig_enc = bb_orig.encoder(**inputs).last_hidden_state
        out_patched_enc = bb_patched.encoder(**inputs).last_hidden_state
    enc_diff = (out_orig_enc - out_patched_enc).abs()
    enc_max = float(enc_diff.max().item())
    enc_mean = float(enc_diff.mean().item())

    result = {
        "n_replaced": n_replaced,
        "logits_max_abs_diff": max_abs_diff,
        "logits_mean_abs_diff": mean_abs_diff,
        "encoder_max_abs_diff": enc_max,
        "encoder_mean_abs_diff": enc_mean,
        "atol": atol,
        "passed": max_abs_diff < atol,
        "logits_orig_first": logits_orig.flatten()[:6].tolist(),
        "logits_patched_first": logits_patched.flatten()[:6].tolist(),
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    backbone_path = _PROJECT_ROOT / "runs" / "multi-fullft-20260426-002447" / "best"
    if not backbone_path.exists():
        print(f"ERROR: backbone path not found: {backbone_path}", file=sys.stderr)
        return 1

    result = parity_test(backbone_path=backbone_path, seq_len=2048, n_globals=8, atol=1e-4)

    print("\n" + "=" * 60)
    print("Phase-1 parity test (original vs patched, PyTorch-vs-PyTorch)")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    if not result["passed"]:
        print(
            f"FAIL: logits max_abs_diff={result['logits_max_abs_diff']:.3e} "
            f">= atol={result['atol']:.3e}",
            file=sys.stderr,
        )
        return 1

    print(
        f"PASS: logits max_abs_diff={result['logits_max_abs_diff']:.3e} "
        f"< atol={result['atol']:.3e}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
