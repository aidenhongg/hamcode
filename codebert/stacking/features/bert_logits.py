"""Extract frozen pointwise LongCoder pre-softmax logits (+ pooled vectors) for
downstream stacking.

Loads a pointwise checkpoint and extracts 11-d logits + 768-d pooled
last-token representation per code. Uses bf16 autocast on Ampere/Ada/Hopper/
Blackwell, fp16 on Turing, fp32 on CPU. Writes incrementally to parquet so a
killed process can be resumed: already-written ids are skipped on restart.

Does NOT fine-tune. The model's state_dict is loaded read-only.

The "cls_*" parquet columns are kept for naming-stability with downstream
readers (semantic.py, dataset.py); the values are LongCoder's pooled
last-token hidden state, not a [CLS] vector.

CLI:
    python -m stacking.features.bert_logits \
        --ckpt runs/point-20260422-065105/point-20260422-065105/best \
        --in_splits data/processed \
        --out_dir runs/heads/extraction \
        --batch 4 --fp16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    import transformers.utils.logging as _hf_log  # type: ignore
    _hf_log.set_verbosity_error()
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data import build_point_inputs
from model import LongCoderClassifier


# -----------------------------------------------------------------------------
# SHA256 helper
# -----------------------------------------------------------------------------

def code_sha(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


# -----------------------------------------------------------------------------
# Model loading (frozen)
# -----------------------------------------------------------------------------

def load_frozen_model(ckpt_dir: str | Path) -> tuple[LongCoderClassifier, "AutoTokenizer"]:
    """Load a pointwise LongCoderClassifier checkpoint in eval + no-grad mode.

    For LoRA checkpoints we merge the adapter into the base weights so the
    extraction forward pays no per-step LoRA matmul cost.
    """
    p = Path(ckpt_dir)
    has_full = (p / "pytorch_model.bin").exists()
    has_adapter = (p / "adapter").exists()
    if not (has_full or has_adapter):
        raise FileNotFoundError(
            f"No checkpoint at {p}. Expected either pytorch_model.bin (full FT) "
            f"or adapter/ + classifier.pt (LoRA), plus codebert_meta.json."
        )
    model = LongCoderClassifier.load_checkpoint(p, merge_lora=True)
    model.eval()
    for pm in model.parameters():
        pm.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model.model_name)
    return model, tokenizer


# -----------------------------------------------------------------------------
# Inference — LongCoder bundles (bridge + memory tokens, global attention)
# -----------------------------------------------------------------------------

def _encode_point_batch(
    codes: list[str],
    tokenizer,
    max_seq_len: int,
    bridge_stride: int = 128,
) -> dict:
    bundles = [
        build_point_inputs(c, tokenizer, max_seq_len, bridge_stride)
        for c in codes
    ]
    return {
        "input_ids": torch.from_numpy(np.stack([b.input_ids for b in bundles])),
        "attention_mask": torch.from_numpy(np.stack([b.attention_mask for b in bundles])),
        "global_attention_mask": torch.from_numpy(
            np.stack([b.global_attention_mask for b in bundles])
        ),
        "token_type_ids": torch.from_numpy(np.stack([b.token_type_ids for b in bundles])),
    }


def _autocast_dtype(device: torch.device, use_amp: bool) -> "torch.dtype | None":
    """Return the best autocast dtype for this device, or None to disable.

    Preference order on CUDA: bf16 if supported (Ampere, Ada, Hopper, Blackwell),
    else fp16 (Turing, Volta). On CPU we return None — mixed precision on CPU
    provides little speedup and risks numerical drift in this workload.
    """
    if not use_amp or device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@torch.no_grad()
def _forward_with_cls(model, enc, device, use_fp16: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the encoder; return (logits, pooled_vec). pooled_vec shape (B, hidden).

    Always uses the full-forward path. Extraction-time LoRA checkpoints have
    already been merge_and_unload'd in load_frozen_model so this stays branch
    free.
    """
    input_ids = enc["input_ids"].to(device, non_blocking=True)
    attn = enc["attention_mask"].to(device, non_blocking=True)
    g_attn = enc["global_attention_mask"].to(device, non_blocking=True)
    types = enc["token_type_ids"].to(device, non_blocking=True)

    dtype = _autocast_dtype(device, use_amp=use_fp16)

    def _run() -> tuple[torch.Tensor, torch.Tensor]:
        from model import _last_token_pool
        enc_out = model.encoder(
            input_ids=input_ids,
            attention_mask=attn,
            global_attention_mask=g_attn,
            token_type_ids=types,
            return_dict=True,
        )
        pooled = _last_token_pool(enc_out.last_hidden_state, attn)
        return model.classifier(model.dropout(pooled)), pooled

    if dtype is not None:
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits, pooled = _run()
    else:
        logits, pooled = _run()
    return logits.float(), pooled.float()


# -----------------------------------------------------------------------------
# Resumable parquet writer
# -----------------------------------------------------------------------------

class _IncrementalParquetWriter:
    """Append-only parquet chunks under out_dir/<name>/chunk_NNNNN.parquet.

    `known_keys` file tracks which rows are already written (key column
    specified at construction). Call .add_batch(rows: pa.Table) to append.
    """

    def __init__(self, out_dir: Path, name: str, key_col: str) -> None:
        self.name = name
        self.root = out_dir / name
        self.root.mkdir(parents=True, exist_ok=True)
        self.key_col = key_col
        self.done_file = self.root / "_done_keys.txt"
        self.done: set[str] = set()
        if self.done_file.exists():
            for line in self.done_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    self.done.add(line)
        # next chunk idx
        existing = sorted(self.root.glob("chunk_*.parquet"))
        self._next_chunk = 0
        for p in existing:
            try:
                n = int(p.stem.split("_")[-1])
                self._next_chunk = max(self._next_chunk, n + 1)
            except ValueError:
                pass

    def add_batch(self, table: pa.Table) -> None:
        """Write a chunk + append its keys to the done file."""
        if table.num_rows == 0:
            return
        dst = self.root / f"chunk_{self._next_chunk:05d}.parquet"
        pq.write_table(table, dst, compression="zstd")
        keys = table.column(self.key_col).to_pylist()
        with self.done_file.open("a", encoding="utf-8") as f:
            for k in keys:
                f.write(f"{k}\n")
                self.done.add(k)
        self._next_chunk += 1

    def merge(self, final_path: Path) -> None:
        """Concat all chunks into a single parquet."""
        chunks = sorted(self.root.glob("chunk_*.parquet"))
        if not chunks:
            return
        tbls = [pq.read_table(p) for p in chunks]
        merged = pa.concat_tables(tbls)
        pq.write_table(merged, final_path, compression="zstd")

    def __contains__(self, key: str) -> bool:
        return key in self.done


# -----------------------------------------------------------------------------
# Pointwise extraction
# -----------------------------------------------------------------------------

def extract_point(
    ckpt_dir: Path,
    in_splits: Path,
    out_dir: Path,
    max_seq_len: int = 4096,
    bridge_stride: int = 128,
    batch_size: int = 4,
    fp16: bool = True,
    device: str | None = None,
    also_cls: bool = True,
) -> None:
    model, tokenizer = load_frozen_model(ckpt_dir)
    dev = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[point] device={dev} fp16={fp16 and dev.type=='cuda'} "
          f"batch={batch_size} seq={max_seq_len}", flush=True)
    model = model.to(dev)

    out_dir.mkdir(parents=True, exist_ok=True)
    hidden = int(model._base_longformer.config.hidden_size)
    n_labels = int(model.num_labels)

    for sp in ("train", "val", "test"):
        src = in_splits / f"{sp}.parquet"
        if not src.exists():
            print(f"[point] skip missing {src}", flush=True)
            continue
        tbl = pq.read_table(src)
        ids = tbl.column("id").to_pylist()
        codes = tbl.column("code").to_pylist()
        shas = [code_sha(c) for c in codes]

        logit_writer = _IncrementalParquetWriter(out_dir, f"point_logits_{sp}", "id")
        cls_writer = (_IncrementalParquetWriter(out_dir, f"point_cls_{sp}", "id")
                      if also_cls else None)

        # Filter out already-done ids
        todo = [
            (i, id_, code)
            for i, (id_, code) in enumerate(zip(ids, codes))
            if id_ not in logit_writer
        ]
        if not todo:
            print(f"[point] {sp}: all {len(ids)} rows already extracted", flush=True)
        else:
            print(f"[point] {sp}: {len(todo)}/{len(ids)} rows to extract", flush=True)

        # Batch loop
        for start in tqdm(range(0, len(todo), batch_size), desc=f"point:{sp}"):
            chunk = todo[start:start + batch_size]
            _, batch_ids, batch_codes = zip(*chunk)
            enc = _encode_point_batch(list(batch_codes), tokenizer, max_seq_len, bridge_stride)
            try:
                logits, cls_vec = _forward_with_cls(model, enc, dev, use_fp16=fp16)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch_size > 1:
                    print(f"[point] OOM at batch {batch_size} — falling back to batch//2", flush=True)
                    batch_size = max(1, batch_size // 2)
                    continue
                raise

            logits_np = logits.cpu().numpy().astype(np.float32)
            cls_np = cls_vec.cpu().numpy().astype(np.float32) if also_cls else None

            cols = {"id": list(batch_ids)}
            for k in range(n_labels):
                cols[f"point_logit_{k}"] = logits_np[:, k]
            logit_writer.add_batch(pa.table(cols))

            if also_cls and cls_np is not None:
                cls_cols = {"id": list(batch_ids)}
                # store CLS as one "vec" column with fixed-size list, easier to read back
                # Use pa.large_list for portability; values are float32.
                for d in range(hidden):
                    cls_cols[f"cls_{d}"] = cls_np[:, d]
                cls_writer.add_batch(pa.table(cls_cols))

        # Merge
        final = out_dir / f"point_logits_{sp}.parquet"
        logit_writer.merge(final)
        print(f"[point] merged {final}", flush=True)
        if also_cls and cls_writer is not None:
            final_cls = out_dir / f"point_cls_{sp}.parquet"
            cls_writer.merge(final_cls)
            print(f"[point] merged {final_cls}", flush=True)

    # Metadata
    (out_dir / "point_meta.json").write_text(json.dumps({
        "ckpt_dir": str(ckpt_dir),
        "model_name": model.model_name,
        "num_labels": n_labels,
        "hidden_size": hidden,
        "max_seq_len": max_seq_len,
        "bridge_stride": bridge_stride,
    }, indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True,
                    help="path to pointwise checkpoint directory with pytorch_model.bin")
    ap.add_argument("--in_splits", default="data/processed")
    ap.add_argument("--out_dir", default="runs/heads/extraction")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--bridge_stride", type=int, default=128)
    ap.add_argument("--batch", type=int, default=4,
                    help="Default 4 (tuned for RTX 5090 / 32GB VRAM at 4096 seq). "
                         "Halve for tighter VRAM budgets; quarter on CPU.")
    ap.add_argument("--fp16", action="store_true", default=True,
                    help="Enable CUDA autocast (bf16 if the GPU supports it, else fp16). "
                         "Ignored on CPU. Default on.")
    ap.add_argument("--no_fp16", dest="fp16", action="store_false")
    ap.add_argument("--device", default=None,
                    help="cuda | cpu | cuda:0. Defaults to cuda if available.")
    ap.add_argument("--no_cls", action="store_true",
                    help="skip CLS vector extraction")
    args = ap.parse_args()

    extract_point(
        ckpt_dir=Path(args.ckpt),
        in_splits=Path(args.in_splits),
        out_dir=Path(args.out_dir),
        max_seq_len=args.max_seq_len,
        bridge_stride=args.bridge_stride,
        batch_size=args.batch,
        fp16=args.fp16,
        device=args.device,
        also_cls=not args.no_cls,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
