"""Per-language LoRA pointwise feature extractor (Phase B → stacker input).

Iterates over each split parquet, groups rows by language, loads the
matching `runs/lora-{ts}/{lang}/` adapter+head on top of the frozen Phase-A
backbone, and writes:

    <out_dir>/point_logits_{train,val,test}.parquet
    <out_dir>/point_cls_{train,val,test}.parquet
    <out_dir>/point_meta.json

Each language pass loads exactly one (backbone+adapter+head) configuration,
runs inference over that language's slice of the split, and unloads. The
shared Phase-A backbone is loaded once per process; per-language adapters
are swapped in/out via peft.PeftModel.

NOTE on leakage: This script does NOT do K-fold OOF extraction; train-set
logits come from the same LoRA[L] that was trained on (a superset of) those
rows, so they are over-confident. If you need OOF discipline (recommended
when training the stacker on tightly-fit logits), see follow-up TODO:
oof_lora.py would re-train K LoRAs[L, fold-k] per language and stitch held-out
predictions. Current default is single-pass.

CLI:
    python -m stacking.features.extract_lora_features \\
        --base_run runs/multi-fullft-20260425/best \\
        --lora_root runs/lora-20260425 \\
        --in_splits data/processed \\
        --out_dir runs/heads/extraction \\
        --batch 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.schemas import LANG_SET
from data import build_point_inputs
from model import LongCoderClassifier, _last_token_pool


def _autocast_dtype(device: torch.device, use_amp: bool):
    if not use_amp or device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _load_backbone_with_lora(base_run: Path, lora_dir: Path,
                             device: torch.device) -> LongCoderClassifier:
    """Load Phase-A backbone, attach LoRA adapter, swap in per-language head."""
    from peft import PeftModel
    from safetensors.torch import load_file as load_safetensors

    backbone = LongCoderClassifier.load_checkpoint(base_run)
    backbone.eval()
    pmodel = PeftModel.from_pretrained(backbone, str(lora_dir))
    head_path = lora_dir / "head.safetensors"
    if head_path.exists():
        sd = load_safetensors(str(head_path))
        with torch.no_grad():
            inner = pmodel.base_model.model
            inner.classifier.weight.copy_(sd["weight"])
            inner.classifier.bias.copy_(sd["bias"])
    pmodel.eval()
    pmodel = pmodel.to(device)

    # peft kwargs absorber for safety
    inner = pmodel.base_model.model
    if not getattr(inner.forward, "_kwargs_absorbed", False):
        orig_fwd = inner.forward
        def _fwd(input_ids, attention_mask, global_attention_mask,
                  token_type_ids, **_):
            return orig_fwd(input_ids=input_ids, attention_mask=attention_mask,
                            global_attention_mask=global_attention_mask,
                            token_type_ids=token_type_ids)
        _fwd._kwargs_absorbed = True
        inner.forward = _fwd

    for p in pmodel.parameters():
        p.requires_grad_(False)
    return pmodel


@torch.no_grad()
def _forward(pmodel, tokenizer, codes: list[str], language: str,
              max_seq_len: int, bridge_stride: int, device: torch.device,
              use_amp: bool):
    bundles = [build_point_inputs(c, tokenizer, max_seq_len, bridge_stride,
                                   language=language) for c in codes]
    input_ids = torch.from_numpy(np.stack([b.input_ids for b in bundles])).to(
        device, non_blocking=True)
    attn = torch.from_numpy(np.stack([b.attention_mask for b in bundles])).to(
        device, non_blocking=True)
    g_attn = torch.from_numpy(np.stack([b.global_attention_mask for b in bundles])).to(
        device, non_blocking=True)
    types = torch.from_numpy(np.stack([b.token_type_ids for b in bundles])).to(
        device, non_blocking=True)

    inner = pmodel.base_model.model

    def _run():
        enc_out = inner.encoder(
            input_ids=input_ids, attention_mask=attn,
            global_attention_mask=g_attn, token_type_ids=types,
            return_dict=True,
        )
        pooled = _last_token_pool(enc_out.last_hidden_state, attn)
        logits = inner.classifier(inner.dropout(pooled))
        return logits, pooled

    dtype = _autocast_dtype(device, use_amp)
    if dtype is not None:
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits, pooled = _run()
    else:
        logits, pooled = _run()
    return logits.float(), pooled.float()


def _extract_split(split_path: Path, base_run: Path, lora_root: Path,
                    out_dir: Path, split_name: str, batch: int,
                    max_seq_len: int, bridge_stride: int,
                    device: torch.device, use_amp: bool) -> None:
    """For one split parquet, iterate per-language and write logit + cls parquets."""
    tbl = pq.read_table(split_path)
    if "language" not in tbl.column_names:
        raise RuntimeError(f"{split_path} missing `language` column - re-run pipeline 09.")

    n_total = tbl.num_rows
    if n_total == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Write empty parquets with the right schema
        pq.write_table(pa.table({"id": pa.array([], type=pa.string())}),
                       out_dir / f"point_logits_{split_name}.parquet")
        pq.write_table(pa.table({"id": pa.array([], type=pa.string())}),
                       out_dir / f"point_cls_{split_name}.parquet")
        print(f"[lora-extract] {split_name}: empty split", flush=True)
        return

    ids_all = tbl.column("id").to_pylist()
    codes_all = tbl.column("code").to_pylist()
    langs_all = tbl.column("language").to_pylist()

    rows_logits: list[dict] = []
    rows_cls: list[dict] = []

    tokenizer = None
    n_labels = 11
    hidden = 768

    by_lang: dict[str, list[int]] = {}
    for i, lang in enumerate(langs_all):
        by_lang.setdefault(lang, []).append(i)

    for lang, indices in by_lang.items():
        if lang not in LANG_SET:
            print(f"[lora-extract] {split_name}: skipping unknown lang {lang!r} "
                  f"({len(indices)} rows)", flush=True)
            continue
        lora_dir = lora_root / lang
        if not lora_dir.exists():
            raise FileNotFoundError(
                f"missing LoRA dir for {lang}: {lora_dir}. "
                f"Did Phase B finish for this language?")
        print(f"[lora-extract] {split_name}/{lang}: {len(indices)} rows", flush=True)
        pmodel = _load_backbone_with_lora(base_run, lora_dir, device)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                pmodel.base_model.model.model_name
            )
        # Per-language inference
        for start in tqdm(range(0, len(indices), batch),
                           desc=f"lora:{split_name}:{lang}"):
            batch_idx = indices[start:start + batch]
            ids_b = [ids_all[i] for i in batch_idx]
            codes_b = [codes_all[i] for i in batch_idx]
            try:
                logits, pooled = _forward(pmodel, tokenizer, codes_b, lang,
                                           max_seq_len, bridge_stride, device, use_amp)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                # Fall back to batch=1 for this batch only
                for i in range(len(batch_idx)):
                    one = [codes_b[i]]
                    one_id = [ids_b[i]]
                    logits1, pooled1 = _forward(pmodel, tokenizer, one, lang,
                                                  max_seq_len, bridge_stride,
                                                  device, use_amp)
                    arr = logits1.cpu().numpy()
                    pl = pooled1.cpu().numpy()
                    row = {"id": one_id[0]}
                    for k in range(arr.shape[1]):
                        row[f"point_logit_{k}"] = float(arr[0, k])
                    rows_logits.append(row)
                    crow = {"id": one_id[0]}
                    for d in range(pl.shape[1]):
                        crow[f"cls_{d}"] = float(pl[0, d])
                    rows_cls.append(crow)
                continue
            arr = logits.cpu().numpy()
            pl = pooled.cpu().numpy()
            n_labels = arr.shape[1]
            hidden = pl.shape[1]
            for r in range(arr.shape[0]):
                row = {"id": ids_b[r]}
                for k in range(arr.shape[1]):
                    row[f"point_logit_{k}"] = float(arr[r, k])
                rows_logits.append(row)
                crow = {"id": ids_b[r]}
                for d in range(pl.shape[1]):
                    crow[f"cls_{d}"] = float(pl[r, d])
                rows_cls.append(crow)
        # release GPU memory between languages
        del pmodel
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_dir.mkdir(parents=True, exist_ok=True)
    # Write out — preserve original row order so downstream joins are stable.
    id_to_logit = {r["id"]: r for r in rows_logits}
    id_to_cls = {r["id"]: r for r in rows_cls}
    ordered_logit = [id_to_logit[i] for i in ids_all if i in id_to_logit]
    ordered_cls = [id_to_cls[i] for i in ids_all if i in id_to_cls]
    pq.write_table(pa.Table.from_pylist(ordered_logit),
                    out_dir / f"point_logits_{split_name}.parquet",
                    compression="zstd")
    pq.write_table(pa.Table.from_pylist(ordered_cls),
                    out_dir / f"point_cls_{split_name}.parquet",
                    compression="zstd")
    print(f"[lora-extract] {split_name}: wrote {len(ordered_logit)} logit rows, "
          f"{len(ordered_cls)} cls rows  (n_labels={n_labels} hidden={hidden})",
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base_run", required=True,
                    help="Phase-A best/ checkpoint dir")
    ap.add_argument("--lora_root", required=True,
                    help="Phase-B bundle root: contains <lang>/ subdirs with adapter + head")
    ap.add_argument("--in_splits", default="data/processed")
    ap.add_argument("--out_dir", default="runs/heads/extraction")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--bridge_stride", type=int, default=128)
    ap.add_argument("--use_amp", action="store_true", default=True)
    ap.add_argument("--no_amp", dest="use_amp", action="store_false")
    args = ap.parse_args()

    base_run = Path(args.base_run)
    lora_root = Path(args.lora_root)
    in_splits = Path(args.in_splits)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (base_run / "codebert_meta.json").exists():
        raise FileNotFoundError(f"missing Phase-A checkpoint at {base_run}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[lora-extract] device={device}", flush=True)

    for sp in ("train", "val", "test"):
        src = in_splits / f"{sp}.parquet"
        if not src.exists():
            print(f"[lora-extract] skip missing {src}", flush=True)
            continue
        _extract_split(src, base_run, lora_root, out_dir, sp,
                        args.batch, args.max_seq_len, args.bridge_stride,
                        device, args.use_amp)

    (out_dir / "point_meta.json").write_text(json.dumps({
        "oof": False,
        "source": "lora_per_language",
        "base_run": str(base_run),
        "lora_root": str(lora_root),
        "max_seq_len": args.max_seq_len,
        "bridge_stride": args.bridge_stride,
        "note": ("Train logits come from the same LoRA[L] those rows "
                 "trained against; expect over-confidence. Use OOF if you "
                 "need clean train logits."),
    }, indent=2), encoding="utf-8")
    print("[lora-extract] done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
