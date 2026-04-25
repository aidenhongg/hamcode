"""Pre-compute the frozen-prefix activation for every snippet in a parquet split.

Used by recipe B (LoRA on top K layers, bottom 12-K layers fully frozen).
The activation right before layer K is fully determined by:
  - the input bundle (input_ids, attention_mask, global_attention_mask, token_type_ids)
  - the base LongCoder weights at the pinned revision
  - the freeze depth K

Caching it on disk turns every per-epoch forward through layers 0..K-1 into
a single np.load(). The cache is keyed (in data._FrozenActivationCache) on
(model_name, freeze_depth, max_seq_len, bridge_stride, code_sha256).

Usage:
    python cache_activations.py \
        --data_dir data/processed \
        --model_name microsoft/longcoder-base \
        --freeze_depth 6 \
        --max_seq_len 4096 --bridge_stride 128 \
        --batch_size 4

Resumable — already-cached samples are skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, LongformerModel

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import (
    ActivationCacheConfig,
    _FrozenActivationCache,
    build_point_inputs,
)
from model import _pad_for_longformer, setup_longformer_masks


def _autocast_dtype(device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@torch.no_grad()
def compute_prefix_activation(
    lf: LongformerModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    global_attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    freeze_depth: int,
) -> torch.Tensor:
    """Run LongformerModel through layers 0..freeze_depth-1, return the post-K-1 hidden state.

    Pads inputs to a multiple of attention_window (HF asserts on this) and
    returns the activation un-padded so the cache stores at the canonical
    seq_len. Output shape: [batch, orig_seq_len, hidden].
    """
    orig_len = attention_mask.size(1)
    embedding_output = lf.embeddings(
        input_ids=input_ids,
        position_ids=None,
        token_type_ids=token_type_ids,
        inputs_embeds=None,
    )
    hidden, padded_attn, padded_global = _pad_for_longformer(
        lf, embedding_output, attention_mask, global_attention_mask,
    )
    extended, is_index_masked, is_index_global_attn, is_global_attn = setup_longformer_masks(
        lf, padded_attn, padded_global,
    )
    for layer in lf.encoder.layer[:freeze_depth]:
        hidden = layer(
            hidden,
            attention_mask=extended,
            layer_head_mask=None,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=False,
        )[0]
    return hidden[:, :orig_len]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default="data/processed",
                    help="Dir containing train/val/test.parquet (and any other splits).")
    ap.add_argument("--splits", default="train,val,test",
                    help="csv list of split parquet stems to cache.")
    ap.add_argument("--model_name", default="microsoft/longcoder-base")
    ap.add_argument("--freeze_depth", type=int, required=True,
                    help="Number of bottom layers to run for the cached activation.")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--bridge_stride", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4,
                    help="Forward batch size on GPU.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--store_dtype", default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--force", action="store_true",
                    help="Re-compute even if a cache entry exists.")
    args = ap.parse_args()

    if args.freeze_depth <= 0:
        raise ValueError("--freeze_depth must be > 0 for activation caching.")

    cfg = ActivationCacheConfig(
        model_name=args.model_name,
        freeze_depth=args.freeze_depth,
        cache_dir=args.cache_dir,
    )
    cache = _FrozenActivationCache(cfg.cache_dir)
    cache_key = cfg.cache_key(args.max_seq_len, args.bridge_stride)
    print(f"[cache] key={cache_key}", flush=True)
    print(f"[cache] root={cache.root}", flush=True)

    device = torch.device(args.device)
    print(f"[cache] device={device} dtype={args.store_dtype}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(f"[cache] loading {args.model_name} ...", flush=True)
    lf = LongformerModel.from_pretrained(args.model_name).to(device)
    lf.eval()
    for p in lf.parameters():
        p.requires_grad_(False)

    store_dtype = {
        "float32": np.float32, "float16": np.float16, "bfloat16": "bf16-as-uint16",
    }[args.store_dtype]
    autocast_dt = _autocast_dtype(device)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    data_dir = Path(args.data_dir)

    for split in splits:
        pq_path = data_dir / f"{split}.parquet"
        if not pq_path.exists():
            print(f"[cache] skip missing {pq_path}", flush=True)
            continue
        tbl = pq.read_table(pq_path)
        codes = tbl.column("code").to_pylist()
        ids = tbl.column("id").to_pylist()

        todo = []
        for i, (id_, code) in enumerate(zip(ids, codes)):
            if not args.force and cache.has(code, cache_key):
                continue
            todo.append((i, id_, code))
        if not todo:
            print(f"[cache] {split}: all {len(ids)} already cached", flush=True)
            continue
        print(f"[cache] {split}: {len(todo)}/{len(ids)} to compute", flush=True)

        bs = args.batch_size
        for start in tqdm(range(0, len(todo), bs), desc=f"cache:{split}"):
            chunk = todo[start:start + bs]
            bundles = [
                build_point_inputs(c, tokenizer, args.max_seq_len, args.bridge_stride)
                for _, _, c in chunk
            ]
            input_ids = torch.from_numpy(np.stack([b.input_ids for b in bundles])).to(device)
            attn = torch.from_numpy(np.stack([b.attention_mask for b in bundles])).to(device)
            g_attn = torch.from_numpy(np.stack([b.global_attention_mask for b in bundles])).to(device)
            types = torch.from_numpy(np.stack([b.token_type_ids for b in bundles])).to(device)
            try:
                if autocast_dt is not None:
                    with torch.autocast(device_type="cuda", dtype=autocast_dt):
                        hidden = compute_prefix_activation(
                            lf, input_ids, attn, g_attn, types, args.freeze_depth,
                        )
                else:
                    hidden = compute_prefix_activation(
                        lf, input_ids, attn, g_attn, types, args.freeze_depth,
                    )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs > 1:
                    bs = max(1, bs // 2)
                    print(f"[cache] OOM; reducing batch to {bs}", flush=True)
                    continue
                raise

            arr_np = _to_numpy(hidden, args.store_dtype)
            for (_, _, code), row in zip(chunk, arr_np):
                cache.put(code, cache_key, row)

    print("[cache] done.", flush=True)
    return 0


def _to_numpy(hidden: torch.Tensor, dtype_name: str) -> np.ndarray:
    """Move + downcast to the requested storage dtype.

    bfloat16 isn't a numpy dtype; we view as uint16 (same byte width) so
    np.save preserves the bit pattern. Read-side does the inverse view.
    """
    h = hidden.detach()
    if dtype_name == "float32":
        return h.float().cpu().numpy()
    if dtype_name == "float16":
        return h.to(torch.float16).cpu().numpy()
    if dtype_name == "bfloat16":
        bf = h.to(torch.bfloat16).cpu()
        return bf.view(torch.uint16).numpy()
    raise ValueError(dtype_name)


if __name__ == "__main__":
    sys.exit(main())
