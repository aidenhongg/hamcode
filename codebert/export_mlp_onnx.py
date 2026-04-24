"""Export the stacking MLP v1 head to a fused ONNX graph.

The exported graph bakes the full preprocessing chain (log1p on count AST
columns + StandardScaler on scaled_mask columns) together with the MLP and
softmax. The Chrome extension builds a raw 132-dim feature vector in the
schema-defined column order and reads P(A_faster) off the output.

Usage:
    pip install onnx onnxruntime
    python export_mlp_onnx.py \
        --head_dir runs/heads/mlp-v1-s44 \
        --out_dir exports/head-mlp-v1

Output layout:
    <out_dir>/
      onnx/model.onnx       # input: features (?, 132) f32 -> output: probs (?, 2) f32
      schema.json           # verbatim copy; JS must honor this column order
      head_config.json      # HPs + label map + preprocessing summary
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stacking.features.ast_features import FEATURE_KIND
from stacking.heads.mlp import MLPHead


class FusedMLPExport(nn.Module):
    """log1p + scaler + MLP + softmax, all in one traced graph."""

    def __init__(
        self,
        mlp_module: nn.Module,
        mean: np.ndarray,
        scale: np.ndarray,
        log1p_nondiff_mask: np.ndarray,
        log1p_diff_mask: np.ndarray,
    ) -> None:
        super().__init__()
        self.mlp = mlp_module
        # register_buffer keeps them in state_dict but non-trainable; torch.onnx
        # bakes them as Constant nodes in the graph.
        self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("scale", torch.from_numpy(scale.astype(np.float32)))
        self.register_buffer("m_nondiff", torch.from_numpy(log1p_nondiff_mask).bool())
        self.register_buffer("m_diff", torch.from_numpy(log1p_diff_mask).bool())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 1) per-column log1p. Nondiff path clips negatives to 0; diff path is
        #    sign-preserving on |x|. torch.where with a (F,) bool mask broadcasts
        #    over batch in both PyTorch and ONNX opset>=9.
        log_nondiff = torch.log1p(torch.clamp(features, min=0.0))
        log_diff = torch.sign(features) * torch.log1p(torch.abs(features))
        x = torch.where(self.m_nondiff, log_nondiff, features)
        x = torch.where(self.m_diff, log_diff, x)
        # 2) standardize. For unscaled cols mean=0, scale=1 → identity.
        x = (x - self.mean) / self.scale
        # 3) MLP (dropout no-op at eval) + softmax.
        logits = self.mlp(x)
        return torch.softmax(logits, dim=-1)


def _build_log1p_masks(columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (nondiff_mask, diff_mask) over columns for AST count features.

    Matches the logic in stacking/dataset.py:312-322 exactly:
      - ast_a__<count>, ast_b__<count>, ast_abs_diff__<count> → clipped log1p
      - ast_diff__<count>                                     → signed log1p
      - bool / cont AST and all non-AST columns                → passthrough
    """
    n = len(columns)
    nondiff = np.zeros(n, dtype=bool)
    diff = np.zeros(n, dtype=bool)
    prefixes = ("ast_a__", "ast_b__", "ast_diff__", "ast_abs_diff__")
    for i, name in enumerate(columns):
        if not name.startswith(prefixes):
            continue
        base = name.split("__", 1)[-1]
        if FEATURE_KIND.get(base) != "count":
            continue
        if name.startswith("ast_diff__"):
            diff[i] = True
        else:
            nondiff[i] = True
    return nondiff, diff


def _build_scaler_vectors(
    scaler, scaled_mask: np.ndarray, n_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Spread sklearn's compact mean_/scale_ vectors back over all columns.

    Unscaled columns get (mean=0, scale=1) so the uniform (x-mean)/scale op
    acts as identity on them.
    """
    mean = np.zeros(n_features, dtype=np.float32)
    scale = np.ones(n_features, dtype=np.float32)
    mean[scaled_mask] = np.asarray(scaler.mean_, dtype=np.float32)
    scale[scaled_mask] = np.asarray(scaler.scale_, dtype=np.float32)
    # Defensive against a zero-variance column (sklearn usually replaces with 1).
    scale[scale == 0.0] = 1.0
    return mean, scale


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--head_dir", default="runs/heads/mlp-v1-s44",
                    help="Dir with head/, scaler.joblib, schema.json, config.json.")
    ap.add_argument("--out_dir", default="exports/head-mlp-v1")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--atol", type=float, default=1e-5,
                    help="Parity tolerance between sklearn+torch reference and onnxruntime.")
    ap.add_argument("--parity_n", type=int, default=64,
                    help="Rows of synthetic input for the parity check.")
    ap.add_argument("--parity_seed", type=int, default=0)
    args = ap.parse_args()

    head_dir = Path(args.head_dir)
    schema_path = head_dir / "schema.json"
    scaler_path = head_dir / "scaler.joblib"
    head_artifact = head_dir / "head"
    config_path = head_dir / "config.json"
    for p in (schema_path, scaler_path, head_artifact, config_path):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    columns: list[str] = schema["columns"]
    n_features: int = schema["n_features"]
    scaled_mask = np.asarray(schema["scaled_mask"], dtype=bool)
    if len(columns) != n_features or scaled_mask.shape != (n_features,):
        raise ValueError(
            f"schema mismatch: columns={len(columns)} n_features={n_features} "
            f"scaled_mask={scaled_mask.shape}"
        )

    print(f"[mlp-export] loading scaler + head from {head_dir}", flush=True)
    scaler = joblib.load(scaler_path)
    mean, scale = _build_scaler_vectors(scaler, scaled_mask, n_features)

    head = MLPHead.load(head_artifact)
    mlp_module = head.model.cpu().eval()
    for p in mlp_module.parameters():
        p.requires_grad_(False)

    if head._input_dim != n_features:
        raise ValueError(
            f"head input_dim={head._input_dim} disagrees with schema n_features={n_features}"
        )

    m_nondiff, m_diff = _build_log1p_masks(columns)
    print(f"[mlp-export] log1p mask: nondiff={int(m_nondiff.sum())} "
          f"diff={int(m_diff.sum())} (of {n_features})", flush=True)
    print(f"[mlp-export] scaler cols: {int(scaled_mask.sum())} of {n_features} "
          f"({int((~scaled_mask).sum())} passthrough)", flush=True)

    wrapper = FusedMLPExport(mlp_module, mean, scale, m_nondiff, m_diff).eval()

    out_dir = Path(args.out_dir)
    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "model.onnx"

    dummy = torch.zeros((1, n_features), dtype=torch.float32)
    print(f"[mlp-export] tracing to {onnx_path} (opset={args.opset})", flush=True)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(onnx_path),
        input_names=["features"],
        output_names=["probs"],
        dynamic_axes={"features": {0: "batch"}, "probs": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    size_kb = onnx_path.stat().st_size / 1e3
    print(f"[mlp-export] wrote {onnx_path} ({size_kb:.1f} KB)", flush=True)

    # Parity: compare ONNX against a reference that uses the *actual* sklearn
    # scaler and the *actual* _MLP state dict. If this passes, it validates
    # both the buffer construction (mean/scale vs sklearn) and the ONNX trace.
    try:
        import onnxruntime as ort
    except ImportError:
        print("[mlp-export] onnxruntime not installed; skipping parity check "
              "(pip install onnxruntime to enable)", flush=True)
    else:
        rng = np.random.default_rng(args.parity_seed)
        x_raw = (rng.standard_normal((args.parity_n, n_features)) * 2.0).astype(np.float32)

        # Reference: log1p per mask -> real sklearn scaler -> real _MLP -> softmax.
        clipped = np.log1p(np.clip(x_raw, 0.0, None))
        signed = np.sign(x_raw) * np.log1p(np.abs(x_raw))
        x_ref = np.where(m_nondiff[None, :], clipped, x_raw)
        x_ref = np.where(m_diff[None, :], signed, x_ref)
        x_scaled = x_ref.copy()
        x_scaled[:, scaled_mask] = scaler.transform(x_ref[:, scaled_mask]).astype(np.float32)
        with torch.no_grad():
            logits = mlp_module(torch.from_numpy(x_scaled.astype(np.float32)))
            ref_probs = torch.softmax(logits, dim=-1).cpu().numpy()

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_probs = sess.run(None, {"features": x_raw})[0]
        max_abs = float(np.max(np.abs(ref_probs - ort_probs)))
        print(f"[mlp-export] parity max abs diff probs = {max_abs:.2e} "
              f"(atol={args.atol})", flush=True)
        if max_abs > args.atol:
            raise RuntimeError(f"parity check failed: {max_abs:.2e} > atol {args.atol}")

    # Ship the schema verbatim — JS needs exactly this column order.
    shutil.copy2(schema_path, out_dir / "schema.json")

    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    head_config = {
        "head": source_config.get("head", "mlp"),
        "variant": source_config.get("variant", "v1"),
        "seed": source_config.get("seed"),
        "hp": source_config.get("hp"),
        "labels": {"0": "same", "1": "A_faster"},
        "n_features": n_features,
        "input_name": "features",
        "output_name": "probs",
        "preprocessing": {
            "log1p_count_columns_nondiff": int(m_nondiff.sum()),
            "log1p_count_columns_diff": int(m_diff.sum()),
            "scaled_columns": int(scaled_mask.sum()),
            "passthrough_columns": int((~scaled_mask).sum()),
            "note": (
                "log1p and StandardScaler are baked into the ONNX graph. "
                "JS must feed raw features in schema['columns'] order."
            ),
        },
        "source_head_dir": str(head_dir),
    }
    (out_dir / "head_config.json").write_text(
        json.dumps(head_config, indent=2), encoding="utf-8",
    )

    print(f"[mlp-export] done. Artifacts in {out_dir}/", flush=True)
    print(f"[mlp-export]   onnx/model.onnx     -- features({n_features}) -> probs(2)")
    print(f"[mlp-export]   schema.json         -- column order (JS contract)")
    print(f"[mlp-export]   head_config.json    -- HPs + labels + preprocessing summary")
    return 0


if __name__ == "__main__":
    sys.exit(main())
