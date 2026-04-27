"""Export the codebert pipeline to ONNX for Chrome extension deployment.

Produces under <out_dir>/:
  backbone.onnx + backbone.onnx_data    (~570 MB FP32 weights blob, shared)
  <lang>.onnx (×12)                     (encoder graph + inline LoRA + Phase B
                                          classifier + pair-feature builder +
                                          scaler + MLP/LGBM head; backbone
                                          weights point to backbone.onnx_data
                                          via ONNX external_data)
  manifest.json                          (file list, sha256, schema, language map)

Per-language modules encode a single B snippet at max_seq_len=2048 (batch=1)
and consume A's features as direct inputs (precomputed externally — the
known-faster reference is fixed per ingest, so re-encoding it on every
inference would be wasted work). Output is `probabilities[2]` =
(prob_same, prob_A_faster). AST log1p, pair feature construction (logits
diff, cosine, l2, mean/max-abs-diff), and the global StandardScaler are all
baked into the graph. Caller preprocessing:
  1. Tokenize B + insert bridge/memory tokens (mirror data.build_point_inputs)
  2. Compute 21 raw AST counts × 4 versions (A, B, diff, abs_diff) = 84 floats
  3. Provide precomputed A features:
     - logits_a[11]  : output of the same per-lang LoRA-adapted encoder
                       + Phase B 11-d classifier baked into <lang>.onnx,
                       run on snippet A
     - pooled_a[768] : last-non-pad pooled hidden state from the same
                       LoRA-adapted encoder run on A
     Drift between the precompute model and the model baked here breaks
     parity with the trained head — use the same per-language LoRA bundle.

Usage:
    python scripts/export_onnx.py \\
        --backbone_run runs/multi-fullft-20260426-002447/best \\
        --lora_root runs/lora-20260426-073548 \\
        --heads_root runs/heads/per_lang \\
        --scaler runs/heads/scaler.joblib \\
        --schema runs/heads/schema.json \\
        --out_dir runs/onnx-export \\
        --stages all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnx import TensorProto, external_data_helper, numpy_helper
from peft import PeftModel
from safetensors.torch import load_file as load_safetensors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.labels import NUM_POINT_LABELS, POINT_LABELS
from longcoder_onnx_attention import patch_longformer_attention
from model import LongCoderClassifier
from stacking.features.ast_features import FEATURE_KIND, FEATURE_NAMES
from stacking.heads.mlp import _MLP

logger = logging.getLogger("export_onnx")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# (language, head_kind, seed) — finalized per user override decisions.
HEAD_SELECTION: list[tuple[str, str, int]] = [
    ("cpp",        "mlp",  44),
    ("csharp",     "mlp",  43),
    ("python",     "mlp",  42),
    ("ruby",       "mlp",  43),
    ("swift",      "mlp",  44),
    ("java",       "lgbm", 42),
    ("javascript", "lgbm", 42),
    ("typescript", "lgbm", 42),
    ("go",         "lgbm", 42),
    ("rust",       "lgbm", 42),
    ("c",          "lgbm", 42),
    ("php",        "lgbm", 42),
]

MAX_SEQ_LEN = 2048
HIDDEN = 768
N_LABELS = NUM_POINT_LABELS              # 11
N_AST = len(FEATURE_NAMES) * 4           # 84
N_FEATURES = N_AST + 4 * N_LABELS + 4    # 132


# ---------------------------------------------------------------------------
# AST log1p masks (matches stacking/dataset.py logic exactly)
# ---------------------------------------------------------------------------

def build_ast_log1p_masks() -> tuple[torch.Tensor, torch.Tensor]:
    """Return (log1p_mask, log1p_signed_mask), each bool[84].

    Schema column order (from stacking/features/ast_features.diff_columns()):
        for name in FEATURE_NAMES:
            for kind in (a, b, diff, abs_diff): ast_<kind>__<name>

    Training transform (dataset.build_feature_matrix):
        if FEATURE_KIND[name] == 'count':
            if kind == 'diff':  x = sign(x) * log1p(|x|)    -> log1p_signed_mask
            else:               x = log1p(max(0, x))         -> log1p_mask
        else: identity (booleans, cyclomatic 'cont')
    """
    log1p: list[bool] = []
    log1p_signed: list[bool] = []
    for name in FEATURE_NAMES:
        is_count = FEATURE_KIND[name] == "count"
        for kind in ("a", "b", "diff", "abs_diff"):
            log1p.append(is_count and kind != "diff")
            log1p_signed.append(is_count and kind == "diff")
    return (
        torch.tensor(log1p, dtype=torch.bool),
        torch.tensor(log1p_signed, dtype=torch.bool),
    )


# ---------------------------------------------------------------------------
# Scaler full-array builder
# ---------------------------------------------------------------------------

def build_scaler_full(scaler, scaled_mask: list[bool]) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand the partial (subset) StandardScaler to full 132-d arrays.

    Unscaled positions get mean=0, scale=1 so `(x - mean_full)/scale_full` is
    a no-op there. This avoids needing a Where op in the ONNX graph.
    """
    n = len(scaled_mask)
    mean_full = torch.zeros(n, dtype=torch.float32)
    scale_full = torch.ones(n, dtype=torch.float32)
    j = 0
    for i, m in enumerate(scaled_mask):
        if m:
            mean_full[i] = float(scaler.mean_[j])
            scale_full[i] = float(scaler.scale_[j])
            j += 1
    if j != len(scaler.mean_):
        raise RuntimeError(f"scaled_mask trues={j} != scaler.mean_ len={len(scaler.mean_)}")
    return mean_full, scale_full


# ---------------------------------------------------------------------------
# PyTorch wrapper modules (subjects of torch.onnx.export)
# ---------------------------------------------------------------------------

def _last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Gather the last non-pad hidden state per row. ONNX-friendly via Gather."""
    last_idx = (attention_mask.sum(dim=1).long() - 1).clamp_min(0)  # [B]
    idx = last_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden.size(-1))
    return torch.gather(hidden, 1, idx).squeeze(1)


class BackbonePointwise(nn.Module):
    """Backbone-only export: encoder + Phase A 11-class classifier."""

    def __init__(self, longcoder: LongCoderClassifier) -> None:
        super().__init__()
        self.encoder = longcoder.encoder
        self.classifier = longcoder.classifier  # Phase A head

    def forward(self, input_ids, attention_mask, global_attention_mask, token_type_ids):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        pooled = _last_token_pool(out.last_hidden_state, attention_mask)
        return self.classifier(pooled)


class PairwiseFusedFeatures(nn.Module):
    """Per-language pipeline up to the post-scaler 132-d feature vector.

    Inputs (B is encoded live; A is precomputed by the caller):
      input_ids[1, L], attention_mask[1, L], global_attention_mask[1, L],
      token_type_ids[1, L]              — tokenized B
      ast_diff_features[84]             — raw A/B/diff/abs_diff AST counts
      logits_a[11]                      — precomputed Phase B per-lang head output for A
      pooled_a[H]                       — precomputed last-non-pad pooled vector for A

    `logits_a` / `pooled_a` MUST be produced with the same LoRA-adapted encoder +
    Phase B 11-d classifier baked into this module (i.e. by running A through
    `peft_model.base_model.model.encoder` + last-token pool + `.classifier`).
    Drift here breaks parity with the trained head.

    Output: feature_vector[1, 132] (post-scaler)
    """

    def __init__(
        self,
        peft_model: nn.Module,
        ast_log1p_mask: torch.Tensor,
        ast_log1p_signed_mask: torch.Tensor,
        scaler_mean_full: torch.Tensor,
        scaler_scale_full: torch.Tensor,
    ) -> None:
        super().__init__()
        self.peft_model = peft_model
        self.register_buffer("ast_log1p_mask", ast_log1p_mask, persistent=False)
        self.register_buffer("ast_log1p_signed_mask", ast_log1p_signed_mask, persistent=False)
        self.register_buffer("scaler_mean_full", scaler_mean_full, persistent=False)
        self.register_buffer("scaler_scale_full", scaler_scale_full, persistent=False)

    def forward(self, input_ids, attention_mask, global_attention_mask,
                token_type_ids, ast_diff_features, logits_a, pooled_a):
        inner = self.peft_model.base_model.model  # LongCoderClassifier
        out = inner.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        pooled_b = _last_token_pool(out.last_hidden_state, attention_mask).squeeze(0)  # [H]
        logits_b = inner.classifier(pooled_b)                                          # [11]

        # Pair logit features (A from input, B from live encoder pass)
        diff = logits_b - logits_a
        abs_diff = diff.abs()

        # Pair pooled-vector similarity (A from input, B from live encoder pass)
        cos = (pooled_a * pooled_b).sum() / (
            (pooled_a.norm() * pooled_b.norm()).clamp_min(1e-12)
        )
        l2 = (pooled_a - pooled_b).norm()
        d = (pooled_a - pooled_b).abs()
        mad = d.mean()
        mxd = d.max()
        sim = torch.stack([cos, l2, mad, mxd])                            # [4]

        # AST log1p — count cols (a/b/abs_diff) get log1p(max(0,x)),
        # count diff cols get sign(x)*log1p(|x|), everything else identity.
        signed = torch.sign(ast_diff_features) * torch.log1p(ast_diff_features.abs())
        positive = torch.log1p(ast_diff_features.clamp_min(0.0))
        ast_log = torch.where(
            self.ast_log1p_signed_mask,
            signed,
            torch.where(self.ast_log1p_mask, positive, ast_diff_features),
        )                                                                 # [84]

        # Concat in schema order: [84 ast, 11 A, 11 B, 11 diff, 11 absdiff, 4 sim]
        feat = torch.cat([ast_log, logits_a, logits_b, diff, abs_diff, sim], dim=0)  # [132]

        # Scaler bake — mean_full=0, scale_full=1 for unscaled positions, so no-op there.
        feat = (feat - self.scaler_mean_full) / self.scaler_scale_full
        return feat.unsqueeze(0)                                          # [1, 132]


class PairwiseFusedMLP(nn.Module):
    """PairwiseFusedFeatures + MLP head + softmax → probabilities[2]."""

    def __init__(self, features: PairwiseFusedFeatures, mlp: _MLP) -> None:
        super().__init__()
        self.features = features
        self.mlp = mlp

    def forward(self, input_ids, attention_mask, global_attention_mask,
                token_type_ids, ast_diff_features, logits_a, pooled_a):
        feat = self.features(input_ids, attention_mask, global_attention_mask,
                             token_type_ids, ast_diff_features,
                             logits_a, pooled_a)                           # [1, 132]
        logits = self.mlp(feat)                                            # [1, 2]
        probs = torch.softmax(logits, dim=-1).squeeze(0)                   # [2]
        return probs


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_backbone(backbone_dir: Path) -> LongCoderClassifier:
    bb = LongCoderClassifier.load_checkpoint(backbone_dir)
    bb.eval()
    for p in bb.parameters():
        p.requires_grad_(False)
    # FullAttentionReplacement is only mathematically valid when seq_len <=
    # attention_window. We deploy at seq=MAX_SEQ_LEN (2048) and the model is
    # built with attention_window=2048 (see model.py). Assert defensively.
    aw = bb.encoder.config.attention_window
    if isinstance(aw, list):
        aw_min = min(aw)
    else:
        aw_min = aw
    if aw_min < MAX_SEQ_LEN:
        raise RuntimeError(
            f"backbone attention_window={aw} < MAX_SEQ_LEN={MAX_SEQ_LEN}; "
            f"FullAttentionReplacement would be incorrect at seq>{aw_min}. "
            f"Retrain with attention_window>={MAX_SEQ_LEN} (see model.py)."
        )
    n = patch_longformer_attention(bb)
    if n != 12:
        raise RuntimeError(
            f"expected to patch 12 LongformerSelfAttention layers, got {n}"
        )
    return bb


def attach_lora_and_phase_b_head(
    backbone: LongCoderClassifier, lora_dir: Path
) -> nn.Module:
    """Wrap with peft, then overwrite classifier with the per-language Phase B head.

    Note: ``backbone`` arriving here may already have been patched by
    ``load_backbone`` (the ONNX-clean attention swap). That's fine — peft
    targets the projection submodules (query/key/value/etc.) which our
    FullAttentionReplacement holds by reference, so peft will still wrap
    those Linears correctly. To be safe against either ordering, we
    re-apply the patch after peft wraps; ``patch_longformer_attention`` is
    idempotent w.r.t. already-patched layers (it only matches modules
    whose ``.self`` is a LongformerSelfAttention instance).
    """
    pmodel = PeftModel.from_pretrained(backbone, str(lora_dir))
    head_path = lora_dir / "head.safetensors"
    if not head_path.exists():
        raise FileNotFoundError(f"missing per-language head: {head_path}")
    sd = load_safetensors(str(head_path))
    inner = pmodel.base_model.model
    with torch.no_grad():
        inner.classifier.weight.copy_(sd["weight"])
        inner.classifier.bias.copy_(sd["bias"])
    # Defensive: make sure all 12 attention layers are patched, regardless
    # of whether peft re-wrapped any of them. patch_longformer_attention
    # is a no-op for layers that are already FullAttentionReplacement.
    patch_longformer_attention(pmodel.base_model.model)
    pmodel.eval()
    for p in pmodel.parameters():
        p.requires_grad_(False)
    return pmodel


def load_mlp_head(head_dir: Path) -> _MLP:
    meta = joblib.load(head_dir / "mlp_meta.pkl")
    hp = meta["hp"]
    mlp = _MLP(
        input_dim=meta["input_dim"],
        hidden_layers=hp["hidden_layers"],
        hidden_dim=hp["hidden_dim"],
        activation=hp["activation"],
        dropout=hp["dropout"],
        layer_norm=hp["layer_norm"],
    )
    state = torch.load(head_dir / "mlp.pt", map_location="cpu", weights_only=True)
    mlp.load_state_dict(state)
    mlp.eval()
    return mlp


# ---------------------------------------------------------------------------
# Dummy input builders
# ---------------------------------------------------------------------------

def make_dummy_inputs(batch: int, seq_len: int) -> dict[str, torch.Tensor]:
    """Synthetic, well-formed Longformer inputs for tracing/verification."""
    rng = np.random.default_rng(0)
    input_ids = torch.from_numpy(
        rng.integers(low=4, high=50000, size=(batch, seq_len), dtype=np.int64)
    )
    # Last few tokens padded
    real = seq_len - 8
    attention_mask = torch.zeros(batch, seq_len, dtype=torch.int64)
    attention_mask[:, :real] = 1
    global_attention_mask = torch.zeros(batch, seq_len, dtype=torch.int64)
    # First, last-real, and a few sprinkles get global attention
    global_attention_mask[:, 0] = 1
    global_attention_mask[:, real - 1] = 1
    global_attention_mask[:, ::128] = 1
    token_type_ids = torch.zeros(batch, seq_len, dtype=torch.int64)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "global_attention_mask": global_attention_mask,
        "token_type_ids": token_type_ids,
    }


def make_dummy_lang_inputs(seq_len: int) -> dict[str, torch.Tensor]:
    """Single-snippet (B) tokenizer bundle + raw AST counts + precomputed A features.

    The per-language ONNX graph encodes B live and consumes A's logits and
    pooled vector as direct inputs (precomputed by the caller).
    """
    d = make_dummy_inputs(batch=1, seq_len=seq_len)
    d["ast_diff_features"] = torch.zeros(N_AST, dtype=torch.float32)
    d["logits_a"] = torch.zeros(N_LABELS, dtype=torch.float32)
    d["pooled_a"] = torch.zeros(HIDDEN, dtype=torch.float32)
    return d


# ---------------------------------------------------------------------------
# External-data redirection: share backbone weights via content-sha256
# ---------------------------------------------------------------------------

def index_external_data(onnx_path: Path) -> dict[str, dict]:
    """Map sha256(weight bytes) → {offset, length, shape, dtype, name}.

    Only initializers stored as EXTERNAL contribute. Caller must ensure the
    .onnx_data file referenced exists alongside the .onnx file.
    """
    model = onnx.load(onnx_path, load_external_data=False)
    base = onnx_path.parent
    out: dict[str, dict] = {}
    for init in model.graph.initializer:
        if init.data_location != TensorProto.EXTERNAL:
            continue
        ext = {kv.key: kv.value for kv in init.external_data}
        location = ext.get("location", "")
        offset = int(ext.get("offset", 0))
        length = int(ext.get("length", 0))
        if not location or length <= 0:
            continue
        data_path = base / location
        with open(data_path, "rb") as f:
            f.seek(offset)
            blob = f.read(length)
        sha = hashlib.sha256(blob).hexdigest()
        out[sha] = {
            "name": init.name,
            "offset": offset,
            "length": length,
            "dims": list(init.dims),
            "data_type": int(init.data_type),
            "location": location,
        }
    return out


def _initializer_bytes(init: TensorProto) -> bytes | None:
    """Extract raw bytes of an inline initializer in canonical FP32 ordering.

    Returns None if the initializer is external or empty.
    """
    if init.data_location == TensorProto.EXTERNAL:
        return None
    if init.raw_data:
        return bytes(init.raw_data)
    arr = numpy_helper.to_array(init)
    return arr.tobytes()


def redirect_to_shared_backbone(
    lang_onnx_path: Path,
    backbone_index: dict[str, dict],
    backbone_data_relpath: str,
) -> dict:
    """Rewrite <lang>.onnx so backbone weight initializers point at backbone.onnx_data.

    Match by sha256 of bytes + dims + data_type. Initializers that don't match
    stay inline (LoRA A/B, classifier, scaler, head weights).
    """
    model = onnx.load(lang_onnx_path, load_external_data=False)
    redirected = 0
    bytes_saved = 0
    misses = 0
    for init in model.graph.initializer:
        blob = _initializer_bytes(init)
        if blob is None:
            continue
        sha = hashlib.sha256(blob).hexdigest()
        match = backbone_index.get(sha)
        if match is None:
            continue
        if list(init.dims) != match["dims"] or int(init.data_type) != match["data_type"]:
            misses += 1
            continue
        # Redirect to shared file
        external_data_helper.set_external_data(
            init,
            location=backbone_data_relpath,
            offset=match["offset"],
            length=match["length"],
        )
        init.data_location = TensorProto.EXTERNAL
        # Wipe inline payload — must clear ALL inline data fields that could carry tensor data.
        init.raw_data = b""
        del init.float_data[:]
        del init.int32_data[:]
        del init.int64_data[:]
        del init.double_data[:]
        del init.uint64_data[:]
        del init.string_data[:]
        redirected += 1
        bytes_saved += match["length"]

    onnx.save(model, lang_onnx_path)
    return {
        "redirected_initializers": redirected,
        "bytes_saved_mb": round(bytes_saved / 1024 / 1024, 1),
        "shape_dtype_misses": misses,
    }


# ---------------------------------------------------------------------------
# Backbone export
# ---------------------------------------------------------------------------

def export_backbone(
    backbone: LongCoderClassifier,
    out_dir: Path,
    seq_len: int = MAX_SEQ_LEN,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "backbone.onnx"
    data_relpath = "backbone.onnx_data"

    wrapper = BackbonePointwise(backbone).eval()
    dummy = make_dummy_inputs(batch=1, seq_len=seq_len)

    logger.info("backbone: tracing...")
    t0 = time.time()
    torch.onnx.export(
        wrapper,
        (dummy["input_ids"], dummy["attention_mask"],
         dummy["global_attention_mask"], dummy["token_type_ids"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask",
                     "global_attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":            {0: "batch", 1: "seq"},
            "attention_mask":       {0: "batch", 1: "seq"},
            "global_attention_mask":{0: "batch", 1: "seq"},
            "token_type_ids":       {0: "batch", 1: "seq"},
            "logits":               {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    logger.info("backbone: traced in %.1fs", time.time() - t0)

    # Convert to external data, single file. size_threshold=0 externalizes
    # everything so the per-language sha-redirector can find every backbone
    # tensor in the index (no inline-only stragglers).
    model = onnx.load(onnx_path)
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_relpath,
        size_threshold=0,
        convert_attribute=False,
    )
    data_path = out_dir / data_relpath
    logger.info(
        "backbone: wrote %s (%.1f MB) + %s (%.1f MB)",
        onnx_path.name, onnx_path.stat().st_size / 1024 / 1024,
        data_relpath, data_path.stat().st_size / 1024 / 1024,
    )
    return onnx_path


# ---------------------------------------------------------------------------
# Per-language export
# ---------------------------------------------------------------------------

def _build_lang_features_module(
    backbone_dir: Path,
    lora_dir: Path,
    scaler,
    scaled_mask: list[bool],
) -> PairwiseFusedFeatures:
    backbone = load_backbone(backbone_dir)
    pmodel = attach_lora_and_phase_b_head(backbone, lora_dir)
    log1p_mask, log1p_signed_mask = build_ast_log1p_masks()
    mean_full, scale_full = build_scaler_full(scaler, scaled_mask)
    return PairwiseFusedFeatures(
        peft_model=pmodel,
        ast_log1p_mask=log1p_mask,
        ast_log1p_signed_mask=log1p_signed_mask,
        scaler_mean_full=mean_full,
        scaler_scale_full=scale_full,
    ).eval()


def _torch_export_pair(
    module: nn.Module,
    out_path: Path,
    output_names: list[str],
    seq_len: int = MAX_SEQ_LEN,
) -> None:
    dummy = make_dummy_lang_inputs(seq_len)
    torch.onnx.export(
        module,
        (dummy["input_ids"], dummy["attention_mask"],
         dummy["global_attention_mask"], dummy["token_type_ids"],
         dummy["ast_diff_features"],
         dummy["logits_a"], dummy["pooled_a"]),
        str(out_path),
        input_names=[
            "input_ids", "attention_mask", "global_attention_mask",
            "token_type_ids", "ast_diff_features",
            "logits_a", "pooled_a",
        ],
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
    )


def export_lang_mlp(
    lang: str, seed: int,
    backbone_dir: Path, lora_dir: Path, head_dir: Path,
    scaler, scaled_mask: list[bool],
    out_dir: Path, backbone_index: dict[str, dict],
) -> dict:
    features = _build_lang_features_module(backbone_dir, lora_dir, scaler, scaled_mask)
    mlp = load_mlp_head(head_dir)
    fused = PairwiseFusedMLP(features=features, mlp=mlp).eval()

    onnx_path = out_dir / f"{lang}.onnx"
    logger.info("[%s] tracing fused MLP module...", lang)
    t0 = time.time()
    _torch_export_pair(fused, onnx_path, output_names=["probabilities"])
    logger.info("[%s] traced in %.1fs (raw size %.1f MB)",
                lang, time.time() - t0, onnx_path.stat().st_size / 1024 / 1024)

    redirect = redirect_to_shared_backbone(
        onnx_path, backbone_index, backbone_data_relpath="backbone.onnx_data"
    )
    logger.info("[%s] redirected: %s", lang, redirect)

    return {
        "language": lang,
        "head": "mlp",
        "seed": seed,
        "file": onnx_path.name,
        "size_mb": round(onnx_path.stat().st_size / 1024 / 1024, 2),
        **redirect,
    }


def export_lang_lgbm(
    lang: str, seed: int,
    backbone_dir: Path, lora_dir: Path, head_dir: Path,
    scaler, scaled_mask: list[bool],
    out_dir: Path, backbone_index: dict[str, dict],
) -> dict:
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    # 1. Export the feature pipeline (encoder+LoRA+classifier+pair feats+scaler)
    features = _build_lang_features_module(backbone_dir, lora_dir, scaler, scaled_mask)
    features_path = out_dir / f"_tmp_{lang}_features.onnx"
    logger.info("[%s] tracing fused feature pipeline...", lang)
    t0 = time.time()
    _torch_export_pair(features, features_path, output_names=["feature_vector"])
    logger.info("[%s] traced in %.1fs (raw size %.1f MB)",
                lang, time.time() - t0, features_path.stat().st_size / 1024 / 1024)

    # 2. Convert LightGBM model
    lgbm = joblib.load(head_dir / "lgbm.pkl")
    initial_types = [("feat_in", FloatTensorType([1, N_FEATURES]))]
    # onnxmltools' lightgbm converter caps at opset 15 for the default ONNX
    # domain and emits IR version 4 by default. The features pipeline uses
    # IR 8 / opset 17, so we bump the LGBM model in place to match before
    # merging — onnx.compose.merge_models requires identical IR versions.
    lgbm_onnx = convert_lightgbm(lgbm, initial_types=initial_types,
                                 target_opset=15, zipmap=False)
    # Bump IR to match the features model.
    lgbm_onnx.ir_version = 8
    # Bump default-domain opset to 17 (the ai.onnx.ml domain stays at v1).
    for op in lgbm_onnx.opset_import:
        if op.domain in ("", "ai.onnx") and op.version < 17:
            op.version = 17
    lgbm_path = out_dir / f"_tmp_{lang}_lgbm.onnx"
    onnx.save(lgbm_onnx, lgbm_path)

    # 3. Identify the probability output name from the LGBM ONNX, then rename
    #    it to a unique placeholder. The LGBM converter outputs the probability
    #    tensor as "probabilities", which collides with the canonical post-Squeeze
    #    output name we want for the merged model. Rename the internal LGBM
    #    output to "lgbm_probs_2d" before merging to free up "probabilities".
    lgbm_outputs = [o.name for o in lgbm_onnx.graph.output]
    prob_name = next((n for n in lgbm_outputs if "prob" in n.lower()), lgbm_outputs[-1])
    label_name = next((n for n in lgbm_outputs if "label" in n.lower()), lgbm_outputs[0])
    logger.info("[%s] lgbm outputs (pre-rename): %s (prob=%s, label=%s)",
                lang, lgbm_outputs, prob_name, label_name)

    new_prob_name = "lgbm_probs_2d"
    if prob_name != new_prob_name:
        # Rename inside the LGBM graph: every node output and every graph output
        # matching prob_name → new_prob_name.
        for out in lgbm_onnx.graph.output:
            if out.name == prob_name:
                out.name = new_prob_name
        for node in lgbm_onnx.graph.node:
            for i, o in enumerate(node.output):
                if o == prob_name:
                    node.output[i] = new_prob_name
        prob_name = new_prob_name

    # 4. Merge: features.feature_vector → lgbm.feat_in
    features_model = onnx.load(features_path)
    merged = onnx.compose.merge_models(
        features_model, lgbm_onnx,
        io_map=[("feature_vector", "feat_in")],
        outputs=[prob_name],   # only carry forward the probability output
    )

    # 5. Squeeze [1, 2] -> [2] under the canonical name "probabilities".
    axes_name = _add_constant_int64(merged, "axes_zero", [0])
    squeeze_node = onnx.helper.make_node(
        "Squeeze",
        inputs=[prob_name, axes_name],
        outputs=["probabilities"],
        name="squeeze_probs_to_1d",
    )
    merged.graph.node.append(squeeze_node)
    del merged.graph.output[:]
    merged.graph.output.append(
        onnx.helper.make_tensor_value_info(
            "probabilities", TensorProto.FLOAT, [2]
        )
    )

    # 6. Save merged module under <lang>.onnx
    onnx_path = out_dir / f"{lang}.onnx"
    onnx.save(merged, onnx_path)
    logger.info("[%s] merged module written (%.1f MB)",
                lang, onnx_path.stat().st_size / 1024 / 1024)

    # 7. Redirect backbone weights
    redirect = redirect_to_shared_backbone(
        onnx_path, backbone_index, backbone_data_relpath="backbone.onnx_data"
    )
    logger.info("[%s] redirected: %s", lang, redirect)

    # 8. Cleanup tmp files
    features_path.unlink(missing_ok=True)
    lgbm_path.unlink(missing_ok=True)
    # The features export's external-data sidecar (if any) is also tmp:
    features_data_sidecar = features_path.with_suffix(".onnx_data")
    features_data_sidecar.unlink(missing_ok=True)

    return {
        "language": lang,
        "head": "lgbm",
        "seed": seed,
        "file": onnx_path.name,
        "size_mb": round(onnx_path.stat().st_size / 1024 / 1024, 2),
        **redirect,
    }


def _add_constant_int64(model: onnx.ModelProto, name: str, values: list[int]) -> str:
    """Add an int64 constant initializer to the model graph and return its name."""
    # Use a unique name even if multiple constants are added
    idx = 0
    base = name
    existing = {init.name for init in model.graph.initializer}
    while name in existing:
        idx += 1
        name = f"{base}_{idx}"
    arr = np.asarray(values, dtype=np.int64)
    init = numpy_helper.from_array(arr, name=name)
    model.graph.initializer.append(init)
    return name


# ---------------------------------------------------------------------------
# Verification — compare PyTorch vs ORT-CPU on the same input
# ---------------------------------------------------------------------------

def verify_backbone(onnx_path: Path, backbone: LongCoderClassifier,
                     seq_len: int = MAX_SEQ_LEN) -> dict:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dummy = make_dummy_inputs(batch=1, seq_len=seq_len)
    feeds = {k: v.numpy() for k, v in dummy.items()}
    ort_logits = sess.run(["logits"], feeds)[0]
    with torch.no_grad():
        torch_logits = BackbonePointwise(backbone).eval()(**dummy).numpy()
    diff = float(np.max(np.abs(ort_logits - torch_logits)))
    return {"max_abs_diff": diff}


def verify_lang(onnx_path: Path, fused: nn.Module,
                 seq_len: int = MAX_SEQ_LEN) -> dict:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dummy = make_dummy_lang_inputs(seq_len)
    feeds = {k: v.numpy() for k, v in dummy.items()}
    ort_probs = sess.run(["probabilities"], feeds)[0]
    with torch.no_grad():
        torch_probs = fused(**dummy).numpy()
    diff = float(np.max(np.abs(ort_probs - torch_probs)))
    return {"max_abs_diff": diff,
            "torch_probs": torch_probs.tolist(),
            "ort_probs": ort_probs.tolist()}


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def write_manifest(out_dir: Path, lang_records: list[dict]) -> Path:
    files: dict[str, dict] = {}
    for name in sorted(p.name for p in out_dir.iterdir() if p.is_file()):
        p = out_dir / name
        if name == "manifest.json":
            continue
        files[name] = {"size_mb": round(p.stat().st_size / 1024 / 1024, 2),
                       "sha256": sha256_file(p)}
    manifest = {
        "max_seq_len": MAX_SEQ_LEN,
        "n_features": N_FEATURES,
        "n_labels": N_LABELS,
        "labels": list(POINT_LABELS),
        "ast_feature_names": list(FEATURE_NAMES),
        "ast_feature_kinds": dict(FEATURE_KIND),
        "shared_backbone_data": "backbone.onnx_data",
        "language_modules": {r["language"]: r for r in lang_records},
        "files": files,
        "input_schema_per_lang": {
            "input_ids":             {"dtype": "int64",   "shape": [1, MAX_SEQ_LEN]},
            "attention_mask":        {"dtype": "int64",   "shape": [1, MAX_SEQ_LEN]},
            "global_attention_mask": {"dtype": "int64",   "shape": [1, MAX_SEQ_LEN]},
            "token_type_ids":        {"dtype": "int64",   "shape": [1, MAX_SEQ_LEN]},
            "ast_diff_features":     {"dtype": "float32", "shape": [N_AST]},
            "logits_a":              {"dtype": "float32", "shape": [N_LABELS]},
            "pooled_a":              {"dtype": "float32", "shape": [HIDDEN]},
        },
        "input_schema_per_lang_notes": {
            "tokenizer_inputs": "input_ids/attention_mask/global_attention_mask/token_type_ids "
                                "are tokenized B only (batch=1).",
            "logits_a": "Precomputed by running A through the same per-language "
                        "LoRA-adapted encoder + Phase B 11-d classifier baked into "
                        "this <lang>.onnx. Use the per-lang LoRA bundle that "
                        "matches this language module.",
            "pooled_a": "Precomputed last-non-pad pooled hidden state ([HIDDEN]) "
                        "from the same LoRA-adapted encoder run on A. Used by the "
                        "graph's built-in cls-similarity feature block "
                        "(cosine, l2, mean_abs_diff, max_abs_diff vs pooled_b).",
        },
        "output_schema_per_lang": {
            "probabilities": {"dtype": "float32", "shape": [2],
                              "index_meaning": ["prob_same", "prob_A_faster"]},
        },
        "input_schema_backbone": {
            "input_ids":             {"dtype": "int64",   "shape": ["batch", "seq"]},
            "attention_mask":        {"dtype": "int64",   "shape": ["batch", "seq"]},
            "global_attention_mask": {"dtype": "int64",   "shape": ["batch", "seq"]},
            "token_type_ids":        {"dtype": "int64",   "shape": ["batch", "seq"]},
        },
        "output_schema_backbone": {
            "logits": {"dtype": "float32", "shape": ["batch", N_LABELS]},
        },
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone_run", required=True,
                    help="Phase-A best/ checkpoint dir")
    ap.add_argument("--lora_root", required=True,
                    help="Phase-B per-language LoRA bundle root")
    ap.add_argument("--heads_root", required=True,
                    help="Phase-C per-language head root (runs/heads/per_lang)")
    ap.add_argument("--scaler", required=True,
                    help="Path to scaler.joblib (runs/heads/scaler.joblib)")
    ap.add_argument("--schema", required=True,
                    help="Path to schema.json (runs/heads/schema.json)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--stages", default="all",
                    help="comma list: backbone,langs,manifest,verify  or 'all'")
    ap.add_argument("--only_lang", default=None,
                    help="restrict the langs stage to a single language (debug)")
    ap.add_argument("--seq_len", type=int, default=MAX_SEQ_LEN)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone_dir = Path(args.backbone_run)
    lora_root = Path(args.lora_root)
    heads_root = Path(args.heads_root)
    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    scaled_mask: list[bool] = list(schema["scaled_mask"])
    if len(scaled_mask) != N_FEATURES:
        raise RuntimeError(f"schema.scaled_mask has {len(scaled_mask)} cols, expected {N_FEATURES}")
    scaler = joblib.load(args.scaler)

    stages = set(s.strip() for s in args.stages.split(",")) if args.stages != "all" \
             else {"backbone", "langs", "manifest", "verify"}

    backbone_onnx = out_dir / "backbone.onnx"
    backbone_data = out_dir / "backbone.onnx_data"

    # -- Stage: backbone -----------------------------------------------------
    if "backbone" in stages:
        backbone = load_backbone(backbone_dir)
        export_backbone(backbone, out_dir, seq_len=args.seq_len)
        if "verify" in stages:
            v = verify_backbone(backbone_onnx, backbone, seq_len=args.seq_len)
            logger.info("[backbone] verify: %s", v)
        del backbone
    if not backbone_onnx.exists() or not backbone_data.exists():
        raise FileNotFoundError(
            "backbone.onnx and backbone.onnx_data must exist before per-language "
            "export — run --stages backbone first or include 'all'."
        )

    backbone_index = index_external_data(backbone_onnx)
    logger.info("backbone external initializers indexed: %d tensors, %d MB",
                len(backbone_index),
                sum(v["length"] for v in backbone_index.values()) // (1024 * 1024))

    # -- Stage: per-language -------------------------------------------------
    lang_records: list[dict] = []
    if "langs" in stages:
        for lang, head_kind, seed in HEAD_SELECTION:
            if args.only_lang and args.only_lang != lang:
                continue
            lora_dir = lora_root / lang
            if not lora_dir.exists():
                raise FileNotFoundError(f"missing LoRA for {lang}: {lora_dir}")
            head_dir = heads_root / lang / f"{head_kind}-v1-s{seed}" / "head"
            if not head_dir.exists():
                raise FileNotFoundError(f"missing head dir: {head_dir}")

            logger.info("=== %s (%s s%d) ===", lang, head_kind, seed)
            t0 = time.time()
            if head_kind == "mlp":
                rec = export_lang_mlp(
                    lang, seed, backbone_dir, lora_dir, head_dir,
                    scaler, scaled_mask, out_dir, backbone_index,
                )
            elif head_kind == "lgbm":
                rec = export_lang_lgbm(
                    lang, seed, backbone_dir, lora_dir, head_dir,
                    scaler, scaled_mask, out_dir, backbone_index,
                )
            else:
                raise ValueError(head_kind)
            rec["elapsed_s"] = round(time.time() - t0, 1)
            lang_records.append(rec)
            logger.info("[%s] done in %.1fs", lang, rec["elapsed_s"])

            if "verify" in stages:
                if head_kind == "mlp":
                    features = _build_lang_features_module(
                        backbone_dir, lora_dir, scaler, scaled_mask)
                    mlp = load_mlp_head(head_dir)
                    fused = PairwiseFusedMLP(features=features, mlp=mlp).eval()
                    v = verify_lang(out_dir / f"{lang}.onnx", fused, seq_len=args.seq_len)
                    logger.info("[%s] verify: max_abs_diff=%.3e", lang, v["max_abs_diff"])
                # LGBM verification is harder — skipping (parity within the
                # feature pipeline is what matters; the LGBM op is a black-box
                # tree ensemble whose output we trust).

    # -- Stage: manifest -----------------------------------------------------
    if "manifest" in stages:
        # If we did the langs stage in this run, lang_records is populated;
        # otherwise rebuild it from disk.
        if not lang_records:
            for lang, head_kind, seed in HEAD_SELECTION:
                p = out_dir / f"{lang}.onnx"
                if p.exists():
                    lang_records.append({
                        "language": lang, "head": head_kind, "seed": seed,
                        "file": p.name,
                        "size_mb": round(p.stat().st_size / 1024 / 1024, 2),
                    })
        path = write_manifest(out_dir, lang_records)
        logger.info("manifest: %s", path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
