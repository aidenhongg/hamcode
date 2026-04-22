"""Train GraphCodeBERT on pointwise or pairwise code-complexity tasks.

Usage:
    python train.py --point --data_dir data/processed --epochs 8
    python train.py --pair  --data_dir data/processed --warm_start_from runs/point-v1/best

Loads defaults from configs/point.yaml (or pair.yaml). CLI flags override.

Writes to --output_dir:
    best/                  HF-format checkpoint (model + tokenizer config)
    last/                  most recent epoch checkpoint
    metrics.jsonl          eval results, one line per eval
    config.json            resolved config
    (optional) wandb run if --wandb_project set and env has WANDB_API_KEY
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
import random
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

logger = logging.getLogger("train")


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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import (
    PairDataset,
    PointDataset,
    compute_class_weights,
    make_collator,
)
from metrics import pairwise_metrics, pointwise_metrics, pretty_confusion
from model import build_model
from common.labels import PAIR_LABELS, POINT_LABELS


# ------------------------------------------------------------------ config

@dataclass
class Config:
    task: str = "point"
    model_name: str = "microsoft/graphcodebert-base"
    data_dir: str = "data/processed"
    output_dir: str = ""
    max_seq_len: int = 512
    max_dfg_nodes: int = 64
    batch_size: int = 16
    grad_accum: int = 2
    lr: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    epochs: int = 8
    label_smoothing: float = 0.0
    bf16: bool = True
    class_weights: str = "auto"           # auto | none | PATH
    warm_start_from: str = ""
    seed: int = 42
    eval_every_steps: int = 200
    patience: int = 3
    wandb_project: str = "codebert-complexity"
    dry_run: bool = False
    resume_from: str = ""
    num_workers: int = 4
    prefetch_factor: int = 4
    cache_dir: str = ""
    use_dfg: bool = False   # DEFAULT: simple tokenization, no DFG. Set True for paper-faithful.


def load_config(cli: argparse.Namespace) -> Config:
    cfg_path = Path("configs") / f"{cli.task}.yaml"
    base: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}
    base["task"] = cli.task
    # Overlay CLI flags (non-None only)
    for k, v in vars(cli).items():
        if v is None or k in {"point", "pair"}:
            continue
        base[k] = v
    cfg = Config(**{k: v for k, v in base.items() if k in Config.__dataclass_fields__})
    if not cfg.output_dir:
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        cfg.output_dir = f"runs/{cfg.task}-{ts}"
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


def resolve_class_weights(cfg: Config) -> torch.Tensor | None:
    if cfg.task != "point" or cfg.class_weights == "none":
        return None
    if cfg.class_weights == "auto":
        train_pq = Path(cfg.data_dir) / "train.parquet"
        w = compute_class_weights(train_pq)
        print(f"[train] class weights (auto, inv-sqrt-freq): {[round(x, 3) for x in w]}")
        return torch.tensor(w, dtype=torch.float)
    # path to json
    w = json.loads(Path(cfg.class_weights).read_text(encoding="utf-8"))
    return torch.tensor(w, dtype=torch.float)


# ------------------------------------------------------------------ train loop

def evaluate(model, loader, device, task: str, max_batches: int | None = None) -> dict:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    iterable = tqdm(loader, desc="eval", leave=False, dynamic_ncols=True, mininterval=1.0)
    with torch.no_grad():
        for i, batch in enumerate(iterable):
            if max_batches is not None and i >= max_batches:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                position_ids=batch["position_ids"],
            )
            preds = logits.argmax(dim=-1).cpu().tolist()
            labels = batch["labels"].cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)
    iterable.close() if hasattr(iterable, "close") else None
    if task == "point":
        return pointwise_metrics(all_preds, all_labels)
    return pairwise_metrics(all_preds, all_labels)


def save_state(model, optimizer, scheduler, scaler, step, epoch, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(out)
    torch.save(optimizer.state_dict(), out / "optimizer.pt")
    torch.save(scheduler.state_dict(), out / "scheduler.pt")
    if scaler is not None:
        torch.save(scaler.state_dict(), out / "scaler.pt")
    rng = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "py": random.getstate(),
    }
    torch.save(rng, out / "rng_state.pt")
    (out / "step.json").write_text(
        json.dumps({"step": step, "epoch": epoch}),
        encoding="utf-8",
    )


def load_state(model, optimizer, scheduler, scaler, in_dir: Path):
    state = torch.load(in_dir / "pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state, strict=True)
    optimizer.load_state_dict(torch.load(in_dir / "optimizer.pt", map_location="cpu"))
    scheduler.load_state_dict(torch.load(in_dir / "scheduler.pt", map_location="cpu"))
    if scaler is not None and (in_dir / "scaler.pt").exists():
        scaler.load_state_dict(torch.load(in_dir / "scaler.pt", map_location="cpu"))
    rng = torch.load(in_dir / "rng_state.pt", map_location="cpu")
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    np.random.set_state(rng["numpy"])
    random.setstate(rng["py"])
    meta = json.loads((in_dir / "step.json").read_text(encoding="utf-8"))
    return meta["step"], meta["epoch"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--point", dest="task", action="store_const", const="point")
    mode.add_argument("--pair", dest="task", action="store_const", const="pair")
    # Overrides (None means "inherit from yaml")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--max_seq_len", type=int, default=None)
    ap.add_argument("--max_dfg_nodes", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--grad_accum", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--warmup_ratio", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--label_smoothing", type=float, default=None)
    ap.add_argument("--bf16", action="store_true", default=None)
    ap.add_argument("--no_bf16", dest="bf16", action="store_false")
    ap.add_argument("--class_weights", default=None)
    ap.add_argument("--warm_start_from", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--eval_every_steps", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--dry_run", action="store_true", default=None)
    ap.add_argument("--resume_from", default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--prefetch_factor", type=int, default=None)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--use_dfg", action="store_true", default=None,
                    help="Paper-faithful mode: tree-sitter DFG + 2D graph attention mask")
    ap.add_argument("--no_dfg", dest="use_dfg", action="store_false",
                    help="Simple mode (default): standard 1D tokenization, no DFG")
    args = ap.parse_args()

    cfg = load_config(args)
    set_seed(cfg.seed)
    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    setup_logging(out_root)
    (out_root / "config.json").write_text(json.dumps(cfg.__dict__, indent=2), encoding="utf-8")
    logger.info("config: %s", cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)
    if device.type == "cuda":
        logger.info("gpu: %s  (%.1fGB)", torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    # --- data -----
    logger.info("loading tokenizer (%s) ...", cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    data_root = Path(cfg.data_dir)
    logger.info("tokenizer ready (vocab=%d)", tokenizer.vocab_size)

    if cfg.task == "point":
        train_path = data_root / "train.parquet"
        val_path = data_root / "val.parquet"
        test_path = data_root / "test.parquet"
        DS = PointDataset
    else:
        train_path = data_root / "pair_train.parquet"
        val_path = data_root / "pair_val.parquet"
        test_path = data_root / "pair_test.parquet"
        DS = PairDataset

    logger.info("loading datasets from %s ...", data_root)
    ds_kwargs = dict(
        tokenizer=tokenizer,
        max_seq_len=cfg.max_seq_len,
        max_dfg_nodes=cfg.max_dfg_nodes,
        cache_dir=cfg.cache_dir or None,
        use_dfg=cfg.use_dfg,
    )
    train_ds = DS(train_path, **ds_kwargs)
    val_ds = DS(val_path, **ds_kwargs)
    test_ds = DS(test_path, **ds_kwargs)
    logger.info("train=%d  val=%d  test=%d  use_dfg=%s",
                len(train_ds), len(val_ds), len(test_ds), cfg.use_dfg)

    if cfg.dry_run:
        # Cheap smoke mode
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, list(range(min(100, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(20, len(val_ds)))))
        test_ds = Subset(test_ds, list(range(min(20, len(test_ds)))))
        cfg.epochs = 1

    collate = make_collator()
    dl_kwargs = dict(
        collate_fn=collate,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )
    dl_train = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **dl_kwargs)
    dl_val = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **dl_kwargs)
    dl_test = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, **dl_kwargs)
    logger.info("DataLoader: bs=%d num_workers=%d prefetch_factor=%d pin_memory=%s",
                cfg.batch_size, cfg.num_workers, cfg.prefetch_factor,
                device.type == "cuda")

    # --- model -----
    logger.info("building model (%s, task=%s) ...", cfg.model_name, cfg.task)
    model = build_model(cfg.model_name, cfg.task)
    if cfg.warm_start_from:
        miss, unex = model.load_warm_start_encoder(cfg.warm_start_from)
        logger.info("warm-start from %s (missing=%d unexpected=%d)",
                    cfg.warm_start_from, len(miss), len(unex))
    logger.info("moving model to %s ...", device)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("model params total=%s trainable=%s", f"{n_params:,}", f"{n_trainable:,}")

    # --- loss -----
    weights = resolve_class_weights(cfg)
    if weights is not None:
        weights = weights.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)

    # --- optim / sched -----
    no_decay = ("bias", "LayerNorm.weight")
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(grouped, lr=cfg.lr)
    total_steps = (len(dl_train) // cfg.grad_accum) * cfg.epochs
    warmup = int(total_steps * cfg.warmup_ratio)
    scheduler = linear_warmup_linear_decay(optimizer, warmup, total_steps)

    # --- amp -----
    use_bf16 = cfg.bf16 and torch.cuda.is_available()
    scaler = None  # bf16 doesn't need GradScaler

    # --- resume -----
    step = 0
    start_epoch = 0
    if cfg.resume_from:
        step, start_epoch = load_state(model, optimizer, scheduler, scaler, Path(cfg.resume_from))
        print(f"[train] resumed at epoch={start_epoch} step={step}", flush=True)

    # --- wandb (optional) -----
    wandb_run = None
    if cfg.wandb_project and os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            wandb_run = wandb.init(project=cfg.wandb_project, name=Path(cfg.output_dir).name,
                                    config=cfg.__dict__)
        except Exception as e:
            print(f"[train] wandb init failed: {e} — continuing without wandb", flush=True)

    metrics_log = (out_root / "metrics.jsonl").open("a", encoding="utf-8")
    loss_log = (out_root / "train_loss.jsonl").open("a", encoding="utf-8")
    best_f1 = -1.0
    best_epoch = -1
    patience = 0

    logger.info("starting training: %d epochs, %d total steps, warmup=%d",
                cfg.epochs, total_steps, warmup)
    logger.info("(first epoch's early steps are slow while tree-sitter builds the DFG cache)")

    # --- train loop -----
    try:
        for epoch in range(start_epoch, cfg.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            pbar = tqdm(
                dl_train,
                desc=f"epoch {epoch+1}/{cfg.epochs}",
                dynamic_ncols=True,
                leave=True,
                mininterval=1.0,
            )
            for local_step, batch in enumerate(pbar):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                if use_bf16:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            position_ids=batch["position_ids"],
                        )
                        loss = loss_fn(logits, batch["labels"]) / cfg.grad_accum
                else:
                    logits = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        position_ids=batch["position_ids"],
                    )
                    loss = loss_fn(logits, batch["labels"]) / cfg.grad_accum
                loss.backward()
                if (local_step + 1) % cfg.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    loss_val = float(loss.item() * cfg.grad_accum)
                    loss_log.write(json.dumps({"step": step, "epoch": epoch, "loss": loss_val,
                                               "lr": scheduler.get_last_lr()[0]}) + "\n")
                    pbar.set_postfix(step=step, loss=f"{loss_val:.3f}",
                                     lr=f"{scheduler.get_last_lr()[0]:.2e}")
                    # Verbose early so the user sees progress while DFG cache warms up;
                    # taper to 1/20 the eval cadence after the first 10 global steps.
                    log_interval = 1 if step <= 10 else max(1, cfg.eval_every_steps // 10)
                    if step % log_interval == 0:
                        logger.info("epoch=%d step=%d lr=%.2e loss=%.4f",
                                    epoch, step, scheduler.get_last_lr()[0], loss_val)
                    if step > 0 and step % cfg.eval_every_steps == 0:
                        met = evaluate(model, dl_val, device, cfg.task)
                        rec = {"step": step, "epoch": epoch, "split": "val", **_log_safe(met)}
                        metrics_log.write(json.dumps(rec) + "\n"); metrics_log.flush()
                        loss_log.flush()
                        if wandb_run:
                            wandb_run.log({"val/" + k: v for k, v in _log_safe(met).items()
                                           if isinstance(v, (int, float))}, step=step)
                        f1 = met["macro_f1"]
                        logger.info("eval step=%d macro_f1=%.4f acc=%.4f", step, f1, met["accuracy"])
                        if f1 > best_f1:
                            best_f1 = f1; best_epoch = epoch; patience = 0
                            save_state(model, optimizer, scheduler, scaler, step, epoch, out_root / "best")
                            logger.info("saved best @ macro_f1=%.4f", f1)
                        else:
                            patience += 1
                            if patience >= cfg.patience:
                                logger.info("early stop at step=%d (best macro_f1=%.4f @ epoch %d)",
                                            step, best_f1, best_epoch)
                                save_state(model, optimizer, scheduler, scaler, step, epoch, out_root / "last")
                                metrics_log.close(); loss_log.close()
                                return _final_report(model, dl_test, device, cfg.task, out_root, cfg)
                        model.train()

            pbar.close()
            save_state(model, optimizer, scheduler, scaler, step, epoch, out_root / "last")
    except Exception:
        logger.error("training loop failed:\n%s", traceback.format_exc())
        metrics_log.close(); loss_log.close()
        raise

    metrics_log.close(); loss_log.close()
    logger.info("training complete (best macro_f1=%.4f @ epoch %d)", best_f1, best_epoch)
    return _final_report(model, dl_test, device, cfg.task, out_root, cfg)


def _log_safe(met: dict) -> dict:
    # Drop non-numeric fields (confusion matrix etc.) for flat jsonl/wandb logs.
    flat = {}
    for k, v in met.items():
        if isinstance(v, (int, float)):
            flat[k] = v
    # Flatten per-class F1s too
    for cls, stats in met.get("per_class", {}).items():
        flat[f"f1[{cls}]"] = stats["f1"]
    return flat


def _final_report(model, dl_test, device, task: str, out_root: Path, cfg: Config) -> int:
    best_dir = out_root / "best"
    if best_dir.exists() and (best_dir / "pytorch_model.bin").exists():
        state = torch.load(best_dir / "pytorch_model.bin", map_location=device)
        model.load_state_dict(state)
    met = evaluate(model, dl_test, device, task)
    met["seed"] = cfg.seed
    met["task"] = task
    (out_root / "test_metrics.json").write_text(json.dumps(met, indent=2), encoding="utf-8")
    logger.info("=== TEST METRICS ===")
    logger.info("accuracy=%.4f macro_f1=%.4f", met["accuracy"], met["macro_f1"])
    if task == "point":
        logger.info("within_1_tier_accuracy=%.4f", met["within_1_tier_accuracy"])
        logger.info("\n%s", pretty_confusion(met["confusion_matrix"], POINT_LABELS))
    else:
        logger.info("\n%s", pretty_confusion(met["confusion_matrix"], PAIR_LABELS))
    # Auto-plot (best-effort; matplotlib may be absent)
    try:
        from plot_metrics import plot_all as _plot_all
        _plot_all(out_root)
    except Exception as e:
        logger.warning("plotting failed: %s", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
