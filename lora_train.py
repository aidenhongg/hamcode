"""Per-language LoRA fine-tune on top of the Phase-A universal backbone.

Loads the frozen Phase-A backbone, applies a peft LoRA adapter to the encoder
attention modules, and trains it end-to-end with a fresh per-language 11-class
classifier head. Outputs (per language) under runs/lora-{ts}/{lang}/:

    adapter_model.safetensors    (~7 MB, peft format)
    adapter_config.json          (peft)
    head.safetensors             (~33 KB; the 11-class classifier head)
    meta.json                    (language, base_run_id, lora_cfg digest)
    metrics.jsonl                (per-eval metrics, like train.py)
    config.json                  (resolved config snapshot)
    test_metrics.json            (final eval on the language-filtered test slice)

Usage:
    python lora_train.py \\
        --base_run runs/multi-fullft-20260425/best \\
        --language cpp \\
        --data_dir data/processed \\
        --output_root runs/lora-20260425/

Defaults inherit from configs/lora.yaml.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import random
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from safetensors.torch import save_file as save_safetensors
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import AutoTokenizer

logger = logging.getLogger("lora_train")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import PointDataset, compute_class_weights, make_collator
from metrics import pointwise_metrics, pointwise_metrics_per_language, pretty_confusion
from model import LongCoderClassifier
from common.labels import LABEL_TO_IDX, POINT_LABELS
from common.schemas import LANG_SET


# ------------------------------------------------------------------ config

@dataclass
class LoraCfg:
    base_run: str = ""
    language: str = ""
    data_dir: str = "data/processed"
    output_root: str = ""
    train_parquet: str = ""
    val_parquet: str = ""
    test_parquet: str = ""

    # LoRA
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: tuple[str, ...] = (
        "query", "key", "value", "query_global", "key_global", "value_global",
    )

    # Training
    max_seq_len: int = 2048
    bridge_stride: int = 128
    batch_size: int = 4
    grad_accum: int = 8
    lr: float = 2e-4
    warmup_ratio: float = 0.10
    weight_decay: float = 0.0
    epochs: int = 20
    label_smoothing: float = 0.05
    bf16: bool = True
    class_weights: str = "auto"
    balanced_sampler: bool = True
    seed: int = 42
    eval_every_steps: int = 200
    patience: int = 4
    min_delta: float = 1e-4
    wandb_project: str = "codebert-complexity-lora"

    num_workers: int = 4
    prefetch_factor: int = 4
    cache_dir: str = ""


def load_cfg(cli: argparse.Namespace) -> LoraCfg:
    cfg_path = Path("configs") / "lora.yaml"
    base: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}
    for k, v in vars(cli).items():
        if v is None:
            continue
        base[k] = v
    if isinstance(base.get("target_modules"), list):
        base["target_modules"] = tuple(base["target_modules"])
    cfg = LoraCfg(**{k: v for k, v in base.items() if k in LoraCfg.__dataclass_fields__})
    if not cfg.output_root:
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        cfg.output_root = f"runs/lora-{ts}"
    return cfg


# ------------------------------------------------------------------ utils

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def linear_warmup_linear_decay(optimizer, num_warmup: int, num_total: int):
    def lr_lambda(step: int) -> float:
        if step < num_warmup:
            return step / max(1, num_warmup)
        return max(0.0, (num_total - step) / max(1, num_total - num_warmup))
    return LambdaLR(optimizer, lr_lambda)


def setup_logging(out_dir: Path, level: int = logging.INFO) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = []
    stream = logging.StreamHandler(sys.stderr)
    stream.setLevel(level)
    handlers.append(stream)
    fileh = logging.FileHandler(out_dir / "train.log", encoding="utf-8")
    fileh.setLevel(logging.DEBUG)
    handlers.append(fileh)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-5s %(name)s: %(message)s", "%H:%M:%S"
    )
    for h in handlers:
        h.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in handlers:
        root.addHandler(h)


# ------------------------------------------------------------------ model build

def build_lora_model(cfg: LoraCfg, device) -> tuple[nn.Module, nn.Module]:
    """Returns (peft_wrapped_model, classifier_head_module).

    The classifier head is fresh-initialized 11-class linear (per user spec:
    no warm-start from Phase A's head). It lives on the wrapped model as
    `model.classifier` (peft `modules_to_save` flag will save its state_dict
    alongside the adapter).
    """
    from peft import LoraConfig, TaskType, get_peft_model

    base_run = Path(cfg.base_run)
    if not (base_run / "codebert_meta.json").exists():
        raise FileNotFoundError(
            f"--base_run must point at a Phase-A best/ dir with codebert_meta.json; "
            f"got {base_run}")
    base = LongCoderClassifier.load_checkpoint(base_run)

    # Replace the classifier head with a fresh-init 11-class linear.
    # (Per spec: per-language head trained from scratch, no Phase-A warm-start.)
    hidden = base.encoder.config.hidden_size
    base.classifier = nn.Linear(hidden, len(POINT_LABELS))
    nn.init.xavier_uniform_(base.classifier.weight)
    nn.init.zeros_(base.classifier.bias)

    lora_cfg = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=list(cfg.target_modules),
        modules_to_save=["classifier"],   # save head state alongside the adapter
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    pmodel = get_peft_model(base, lora_cfg)

    # peft injects inputs_embeds=None / output_attentions etc. by default;
    # the model.py forward already absorbs **_unused so this is a no-op
    # safety net for older check-outs.
    inner = pmodel.base_model.model
    if not getattr(inner.forward, "_kwargs_absorbed", False):
        orig_fwd = inner.forward

        def _absorbing_fwd(input_ids, attention_mask, global_attention_mask,
                           token_type_ids, **_):
            return orig_fwd(input_ids=input_ids, attention_mask=attention_mask,
                            global_attention_mask=global_attention_mask,
                            token_type_ids=token_type_ids)
        _absorbing_fwd._kwargs_absorbed = True
        inner.forward = _absorbing_fwd

    pmodel = pmodel.to(device)
    return pmodel, pmodel.base_model.model.classifier


# ------------------------------------------------------------------ eval

def evaluate(model, loader, device, bf16: bool = True,
              language: str | None = None) -> dict:
    """If `language` is set, the returned metrics include a `per_language`
    field with a single key — keeps the artifact format identical to train.py
    so downstream reporters can read both Phase A and Phase B uniformly."""
    from contextlib import nullcontext
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    use_amp = bf16 and device.type == "cuda"
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if use_amp else nullcontext())
    with torch.no_grad(), autocast_ctx:
        for batch in tqdm(loader, desc="eval", leave=False, dynamic_ncols=True,
                          mininterval=1.0):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                global_attention_mask=batch["global_attention_mask"],
                token_type_ids=batch["token_type_ids"],
            )
            preds = logits.argmax(dim=-1).cpu().tolist()
            labels = batch["labels"].cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)
    if language is not None and all_preds:
        langs = [language] * len(all_preds)
        return pointwise_metrics_per_language(all_preds, all_labels, langs)
    return pointwise_metrics(all_preds, all_labels)


def _log_safe(met: dict) -> dict:
    flat = {}
    for k, v in met.items():
        if isinstance(v, (int, float)):
            flat[k] = v
    for cls, stats in met.get("per_class", {}).items():
        flat[f"f1[{cls}]"] = stats["f1"]
    for lang, lm in met.get("per_language", {}).items():
        flat[f"acc[{lang}]"] = lm.get("accuracy", 0.0)
        flat[f"f1[{lang}]"] = lm.get("macro_f1", 0.0)
        flat[f"w1[{lang}]"] = lm.get("within_1_tier_accuracy", 0.0)
        flat[f"n[{lang}]"] = lm.get("n", 0)
    return flat


# ------------------------------------------------------------------ save

def save_lora_artifact(pmodel, head: nn.Module, cfg: LoraCfg, out_dir: Path,
                        best_val_metrics: dict | None = None) -> None:
    """Save adapter + per-language head + meta to a compact directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pmodel.save_pretrained(str(out_dir))    # adapter_config.json + adapter_model.safetensors
    save_safetensors({"weight": head.weight.detach().cpu().contiguous(),
                      "bias":   head.bias.detach().cpu().contiguous()},
                     str(out_dir / "head.safetensors"))
    meta = {
        "language": cfg.language,
        "base_run": cfg.base_run,
        "lora_r": cfg.r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "target_modules": list(cfg.target_modules),
        "best_val_metrics": best_val_metrics,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base_run", required=True,
                    help="Path to Phase-A best/ checkpoint directory")
    ap.add_argument("--language", required=True,
                    help="Canonical language id (one of common.schemas.LANGUAGES)")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--output_root", default=None)
    ap.add_argument("--r", type=int, default=None)
    ap.add_argument("--lora_alpha", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--grad_accum", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--bf16", action="store_true", default=None)
    ap.add_argument("--no_bf16", dest="bf16", action="store_false")
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args)
    if cfg.language not in LANG_SET:
        print(f"[lora] unknown language: {cfg.language!r}", file=sys.stderr)
        return 2
    set_seed(cfg.seed)

    out_dir = Path(cfg.output_root) / cfg.language
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)
    (out_dir / "config.json").write_text(json.dumps({
        **{k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
        "target_modules": list(cfg.target_modules),
    }, indent=2), encoding="utf-8")
    logger.info("config: %s", cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)
    if device.type == "cuda":
        logger.info("gpu: %s (%.1f GB)", torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_run, use_fast=True) \
                if (Path(cfg.base_run) / "tokenizer.json").exists() \
                else AutoTokenizer.from_pretrained("microsoft/longcoder-base")
    data_root = Path(cfg.data_dir)

    train_path = Path(cfg.train_parquet) if cfg.train_parquet else data_root / "train.parquet"
    val_path = Path(cfg.val_parquet) if cfg.val_parquet else data_root / "val.parquet"
    test_path = Path(cfg.test_parquet) if cfg.test_parquet else data_root / "test.parquet"

    ds_kwargs = dict(
        tokenizer=tokenizer,
        max_seq_len=cfg.max_seq_len,
        bridge_stride=cfg.bridge_stride,
        cache_dir=cfg.cache_dir or None,
        language_filter=cfg.language,
    )
    train_ds = PointDataset(train_path, **ds_kwargs)
    val_ds = PointDataset(val_path, **ds_kwargs)
    test_ds = PointDataset(test_path, **ds_kwargs)
    logger.info("train=%d val=%d test=%d (filter=language==%s)",
                len(train_ds), len(val_ds), len(test_ds), cfg.language)

    if len(train_ds) == 0:
        logger.error("no training rows for language=%s; aborting", cfg.language)
        return 3

    collate = make_collator()
    dl_kwargs = dict(
        collate_fn=collate,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )

    # Per-language balanced sampler (just by label, since the slice is single-language).
    train_sampler = None
    train_shuffle = True
    if cfg.balanced_sampler:
        labels_raw = train_ds.table.column("label").to_pylist()
        cnt: dict[int, int] = {}
        for lab in labels_raw:
            i = LABEL_TO_IDX[lab]; cnt[i] = cnt.get(i, 0) + 1
        per_w = {i: 1.0 / max(1, n) for i, n in cnt.items()}
        sw = [per_w[LABEL_TO_IDX[l]] for l in labels_raw]
        train_sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)
        train_shuffle = False
        logger.info("balanced sampler counts: %s", dict(sorted(cnt.items())))

    dl_train = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=train_shuffle,
                          sampler=train_sampler, **dl_kwargs)
    dl_val = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **dl_kwargs) \
                if len(val_ds) > 0 else None
    dl_test = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, **dl_kwargs) \
                if len(test_ds) > 0 else None

    # Model
    pmodel, head = build_lora_model(cfg, device)
    n_total = sum(p.numel() for p in pmodel.parameters())
    n_train = sum(p.numel() for p in pmodel.parameters() if p.requires_grad)
    logger.info("params total=%s trainable=%s (%.2f%%)",
                f"{n_total:,}", f"{n_train:,}", 100.0 * n_train / max(1, n_total))

    # Class weights from the language-filtered slice
    weights = None
    if cfg.class_weights == "auto":
        from data import compute_class_weights as _ccw
        # compute_class_weights reads parquet directly; pass a proxy parquet by
        # writing the filtered table to a temp file.
        import pyarrow.parquet as _pq, tempfile, os as _os
        tmp = Path(out_dir) / "_train_filtered.parquet"
        _pq.write_table(train_ds.table, str(tmp))
        try:
            w = _ccw(tmp)
            weights = torch.tensor(w, dtype=torch.float).to(device)
        finally:
            try:
                _os.remove(tmp)
            except OSError:
                pass
        logger.info("class weights (auto, inv-sqrt-freq): %s",
                    [round(x, 3) for x in weights.tolist()])
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)

    no_decay = ("bias", "LayerNorm.weight")
    trainable = [(n, p) for n, p in pmodel.named_parameters() if p.requires_grad]
    grouped = [
        {"params": [p for n, p in trainable if not any(nd in n for nd in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in trainable if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(grouped, lr=cfg.lr)
    total_steps = max(1, (len(dl_train) // cfg.grad_accum) * cfg.epochs)
    warmup = int(total_steps * cfg.warmup_ratio)
    scheduler = linear_warmup_linear_decay(optimizer, warmup, total_steps)

    use_bf16 = cfg.bf16 and torch.cuda.is_available()

    wandb_run = None
    if cfg.wandb_project and os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                name=f"{cfg.language}-{Path(cfg.output_root).name}",
                config={k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
            )
        except Exception as e:
            logger.warning("wandb init failed: %s", e)

    metrics_log = (out_dir / "metrics.jsonl").open("a", encoding="utf-8")
    loss_log = (out_dir / "train_loss.jsonl").open("a", encoding="utf-8")
    best_f1 = -1.0
    best_metrics: dict | None = None
    patience = 0
    step = 0

    logger.info("start: total_steps=%d warmup=%d", total_steps, warmup)
    try:
        for epoch in range(cfg.epochs):
            pmodel.train()
            optimizer.zero_grad(set_to_none=True)
            pbar = tqdm(dl_train, desc=f"epoch {epoch+1}/{cfg.epochs}",
                        dynamic_ncols=True, leave=True, mininterval=1.0)
            for local_step, batch in enumerate(pbar):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                kwargs = dict(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    global_attention_mask=batch["global_attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                )
                if use_bf16:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = pmodel(**kwargs)
                        loss = loss_fn(logits, batch["labels"]) / cfg.grad_accum
                else:
                    logits = pmodel(**kwargs)
                    loss = loss_fn(logits, batch["labels"]) / cfg.grad_accum
                loss.backward()
                if (local_step + 1) % cfg.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in pmodel.parameters() if p.requires_grad], 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    loss_val = float(loss.item() * cfg.grad_accum)
                    loss_log.write(json.dumps({"step": step, "epoch": epoch,
                                               "loss": loss_val,
                                               "lr": scheduler.get_last_lr()[0]}) + "\n")
                    pbar.set_postfix(step=step, loss=f"{loss_val:.3f}",
                                     lr=f"{scheduler.get_last_lr()[0]:.2e}")
                    if dl_val is not None and step % cfg.eval_every_steps == 0:
                        met = evaluate(pmodel, dl_val, device, bf16=cfg.bf16,
                                       language=cfg.language)
                        rec = {"step": step, "epoch": epoch, "split": "val",
                               **_log_safe(met)}
                        metrics_log.write(json.dumps(rec) + "\n")
                        metrics_log.flush()
                        if wandb_run:
                            wandb_run.log({"val/" + k: v for k, v in _log_safe(met).items()},
                                           step=step)
                        f1 = met["macro_f1"]
                        delta = f1 - best_f1
                        logger.info("eval step=%d macro_f1=%.4f Δ=%+.4f",
                                    step, f1, delta)
                        if delta > cfg.min_delta:
                            best_f1 = f1
                            best_metrics = met
                            patience = 0
                            save_lora_artifact(pmodel, head, cfg, out_dir,
                                               best_val_metrics=_log_safe(met))
                            logger.info("saved best @ macro_f1=%.4f", f1)
                        else:
                            patience += 1
                            if patience >= cfg.patience:
                                logger.info("early stop at step=%d (best=%.4f)",
                                            step, best_f1)
                                metrics_log.close(); loss_log.close()
                                return _final_test(pmodel, dl_test, out_dir, cfg, best_metrics)
                        pmodel.train()
            pbar.close()
        # End of training without early-stop trigger; ensure we saved at least once.
        if best_metrics is None:
            save_lora_artifact(pmodel, head, cfg, out_dir, best_val_metrics=None)
            logger.info("no eval triggered (val empty?); saved final state")
    except Exception:
        logger.error("training failed:\n%s", traceback.format_exc())
        metrics_log.close(); loss_log.close()
        raise
    metrics_log.close(); loss_log.close()
    return _final_test(pmodel, dl_test, out_dir, cfg, best_metrics)


def _final_test(pmodel, dl_test, out_dir: Path, cfg: LoraCfg,
                 best_metrics: dict | None) -> int:
    if dl_test is None:
        logger.info("no test set")
        return 0
    device = next(pmodel.parameters()).device
    met = evaluate(pmodel, dl_test, device, bf16=cfg.bf16, language=cfg.language)
    met["seed"] = cfg.seed
    met["language"] = cfg.language
    met["best_val"] = _log_safe(best_metrics) if best_metrics else None
    (out_dir / "test_metrics.json").write_text(json.dumps(met, indent=2),
                                                encoding="utf-8")
    logger.info("=== TEST METRICS (%s) ===", cfg.language)
    logger.info("accuracy=%.4f macro_f1=%.4f within_1_tier=%.4f",
                met["accuracy"], met["macro_f1"], met["within_1_tier_accuracy"])
    logger.info("\n%s", pretty_confusion(met["confusion_matrix"], POINT_LABELS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
