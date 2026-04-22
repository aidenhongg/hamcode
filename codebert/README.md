# codebert — GraphCodeBERT complexity classifier

Fine-tunes [microsoft/graphcodebert-base](https://huggingface.co/microsoft/graphcodebert-base)
for Python time-complexity classification. Two modes share one script:

- `python train.py --point` — pointwise 11-class classification (one snippet → class).
- `python train.py --pair`  — pairwise ternary ranking (two snippets → A / same / B faster).

## Classes (11)

Single-variable (7): `O(1)`, `O(log n)`, `O(n)`, `O(n log n)`, `O(n^2)`, `O(n^3)`, `exponential` (covers `O(2^n)` and `O(n!)`).

Multi-variable (4): `O(m+n)`, `O(m*n)`, `O(m log n)`, `O((m+n) log(m+n))`.

## Pairwise label

Ordinal tier (assumes `m ≈ n`). Same tier → `same`; otherwise `A_faster` / `B_faster`.

## What's in the box

| Script | What it does |
|--------|--------------|
| `train.py --point \| --pair` | AdamW + linear warmup/decay, bf16, early stop on dev macro-F1, best-ckpt save, resume, file logging (`train.log`), train-loss JSONL, auto-plots at end |
| `predict.py` | Load checkpoint, infer on file/stdin/pair, emit JSON |
| `tune.py`   | Optuna HP search (lr, warmup, weight decay, batch, epochs, seed, label smoothing); TPE sampler + median pruner; resumable via SQLite study DB |
| `pick_best.py` | Scan multi-seed runs, pick the one with best test macro-F1, copy to a unified `best/` |
| `plot_metrics.py` | Eval curves, train-loss EMA, row-normalized confusion matrix, per-class F1 bar chart |
| `run_pipeline.sh [--smoke]` | End-to-end data pipeline (steps 01–11) |

## Quick start (RunPod RTX 5090)

Pick base image `nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04` — Blackwell (sm_120) needs CUDA 12.8+.

```bash
pip install -r requirements.txt
# Torch wheels with cu128 come from a non-default index:
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.6.0

# Build the dataset (clones doocs/leetcode, downloads CodeComplex, generates synthetics)
./run_pipeline.sh

# Train pointwise
python train.py --point --data_dir data/processed --output_dir runs/point-v1

# Warm-start pairwise
python train.py --pair  --data_dir data/processed --warm_start_from runs/point-v1/best \
                         --output_dir runs/pair-v1

# Inference
python predict.py --model_dir runs/point-v1/best --input examples/linear.py
python predict.py --model_dir runs/pair-v1/best  --pair examples/linear.py examples/quadratic.py
```

## Smoke test (local / tiny data)

```bash
./run_pipeline.sh --smoke
python train.py --point --dry_run --data_dir data/processed
python train.py --pair  --dry_run --data_dir data/processed
```

## Multi-seed + HP tuning

```bash
# Train 5 seeds, pick winner
for s in 42 43 44 45 46; do
  python train.py --point --seed $s --output_dir runs/point-s$s --data_dir data/processed
done
python pick_best.py --globs 'runs/point-s*' --out_dir runs/point-best

# Optuna HP search (12 trials, best params written to runs/tune/best.json)
python tune.py --point --n_trials 12 --study point-tune \
               --data_dir data/processed --base_output_dir runs/tune/point
```

## Artifacts per run (`runs/<name>/`)

```
best/                  # best-dev-F1 checkpoint (HF save_pretrained format + state for resume)
last/                  # most recent checkpoint
config.json            # resolved config
train.log              # full training log (stderr + file)
metrics.jsonl          # one line per dev eval (step, macro_f1, per-class F1, ...)
train_loss.jsonl       # per-step loss + LR
test_metrics.json      # final metrics on test (incl confusion matrix)
curves.png             # accuracy & macro-F1 over steps
train_loss.png         # train loss with EMA smoothing
confusion_matrix.png   # row-normalized, labeled
per_class_f1.png       # bar chart with support annotations
```

## Data sources

| Source | Role | Python est. |
|--------|------|-------------|
| [doocs/leetcode](https://github.com/doocs/leetcode) | Primary (multi-variable annotations) | ~5,000 |
| [CodeComplex](https://arxiv.org/abs/2401.08719) | Primary (single-variable, expert-annotated) | ~4,900 |
| Synthetic templates (this repo) | Supplemental for rare classes | ~500 |
| IBM CodeNet Python (optional, hook) | Additional supplemental | — |

CodeComplex auto-fetch tries several URLs; if none reach, drop the jsonl at
`data/raw/codecomplex/python.jsonl` and pass `--skip_codecomplex` on re-run.

## Layout

```
train.py           # CLI: --point | --pair
predict.py         # inference CLI
data.py            # Dataset + DFG-aware collator
model.py           # GraphCodeBERT + classification head
metrics.py         # pointwise/pairwise + confusion matrix
common/            # labels, schemas, normalizer, DFG cache
parser/            # vendored microsoft/CodeBERT parser (MIT)
pipeline/          # 01-11 data scripts (DAG-ordered by numeric prefix)
configs/           # point.yaml, pair.yaml (CLI overrides them)
data/raw/          # cloned repos + downloaded jsonl
data/processed/    # final parquet (pointwise + pairwise; per-split files)
data/audit/        # parse failures, reject reasons, stats.json
runs/              # checkpoints (best/ + last/)
```

## License

This project vendors `parser/DFG.py` and `parser/utils.py` from
[microsoft/CodeBERT](https://github.com/microsoft/CodeBERT) under the MIT License.
