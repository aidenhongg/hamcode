# stacking — pairwise complexity head on top of frozen pointwise LongCoder

Second-stage classifier on top of frozen LongCoder. Consumes pre-softmax
pointwise logits (11-d × A and B), AST features (21 per snippet, differenced
+ |diff|), and pooled-vector cosine similarity; predicts the binary `same` vs
`A_faster` label.

Target deployment: **RunPod RTX 5090** (32GB VRAM, bf16, CUDA 12.8+).
End-to-end orchestrator: `stacking/scripts/run_runpod.sh`.

## Task

The head is binary. The pipeline canonicalizes every pair so tier_A <= tier_B
— given any candidate (a, b), if tier_a > tier_b they are swapped. After the
rewrite the dataset contains only two classes:

- `0 = same`     — A and B are in the same complexity tier
- `1 = A_faster` — B is strictly slower than A (i.e. f(A) ∈ o(f(B)))

Callers of head inference must submit pairs in canonical order
(B is same-speed-or-slower than A). `pipeline/10_make_pairwise.py`
enforces this at data-generation time; `stacking/dataset.filter_b_ge_a`
remains as a defensive safety net.

## Pointwise encoder only

Pairwise encoder fine-tuning was retired. The head learns to compare two
snippets using only:

- **A's pointwise LongCoder logits** (11-d), **B's pointwise LongCoder logits**
  (11-d), plus their `diff` and `|diff|`.
- **AST diff features** between A and B (~84-d, derived from per-snippet
  AST counts/booleans extracted by `stacking/features/ast_features.py`).
- **Pooled-vector similarity** features (cosine, L2, mean/max abs diff)
  between A's and B's last-token pooled hidden states. The parquet column
  names retain the `cls_*` prefix for naming-stability with downstream
  readers — values are LongCoder pooled vectors, not [CLS] activations.

This is the only supported variant (formerly `v1`).

## Leakage fix

The original recipe (`bert_logits.py`) reuses a pointwise BERT trained on the
same `train.parquet` codes that end up inside the head's training features.
Those train-split logits are over-confident and the head overfits. The new
default replaces it with K-fold out-of-fold (OOF) extraction:

| Module | Old (leaky) | New (OOF, default) |
|--------|-------------|--------------------|
| Pointwise logits + CLS | `stacking/features/bert_logits.py` | `stacking/features/oof_point.py` |

`oof_point.py` splits train by **problem_id** (so no code's problem
appears across folds), trains pointwise BERT K times on K-1-fold unions,
and concatenates the held-out fold predictions into a single
`point_logits_train.parquet`. A final full-train model covers val/test.

Legacy `bert_logits.py` remains available for quick sanity checks but its
results should be treated as upper-bound train metrics only.

## Install (on a fresh RunPod 5090 pod)

```bash
pip install -r requirements.txt
# Torch wheels with CUDA 12.8 come from a non-default index. No pin — any of
# the 2.7.0+cu128 / 2.8 / 2.9 / 2.10 / 2.11 wheels work.
pip install --index-url https://download.pytorch.org/whl/cu128 torch
# If the base image ships torchvision/torchaudio, uninstall them — they'll
# almost certainly be ABI-incompatible with the cu128 torch wheel and make
# transformers crash on first model load ("operator torchvision::nms does
# not exist"). We don't use vision anywhere.
pip uninstall -y torchvision torchaudio
pip install -r requirements-stacking.txt    # xgboost, lightgbm, radon, joblib
```

`stacking/scripts/run_runpod.sh` runs this `uninstall` step defensively
every time, so you can skip it here if you prefer one-shot execution.

## Why the runpod script sets DISABLE_SAFETENSORS_CONVERSION=1

`microsoft/longcoder-base`'s `main` branch only ships `pytorch_model.bin`.
For every such repo, `transformers.modeling_utils._get_resolved_checkpoint_files`
launches a **background thread** (`Thread-auto_conversion`) that walks every
open PR ref (`refs/pr/N`) looking for a safetensors conversion. Any orphan
LFS-backed conversion PR can hang the redirect indefinitely — the thread runs
alongside training, holds network sockets, and blocks the main process on
I/O. `use_safetensors=False` does NOT gate that thread; it only affects the
main-thread file-preference logic.

The only real kill switches (from the `can_auto_convert` gate in transformers):
`DISABLE_SAFETENSORS_CONVERSION=1`, or `TRANSFORMERS_OFFLINE=1` /
`HF_HUB_OFFLINE=1`. `run_runpod.sh` exports the first flag at script start, then
pre-downloads the model from `main` via `snapshot_download` (which is
huggingface_hub, not transformers, so it never does the PR-ref dance), then
exports the offline flags so every downstream subprocess reads from the cache
and never calls home.

## One-shot pipeline

```bash
bash stacking/scripts/run_runpod.sh
```

Runs end-to-end: data → AST → OOF pointwise → semantic → grid sweep
(12 configs) → HP search on top heads (Optuna, 40 trials each).

Flags (all optional):
- `--skip-data` — dataset already built
- `--skip-oof-point` — OOF pointwise extraction already done
- `--skip-sweep` — skip the 12-config grid (go straight to HP search)
- `--skip-hp` — skip Optuna HP search
- `--hp-trials N` — Optuna trials per head (default 40)
- `--hp-heads` — csv, default `xgb,lgbm,mlp,stacked`
- `--n_folds N` — number of OOF folds (default 5)

Expected wallclock on 5090:
- OOF pointwise (40-epoch cap, 6 runs): ~2 h
- Grid sweep (12 head configs): ~3 min CPU
- HP search (4 heads × 40 trials): ~2-4 h CPU
- Total: ~4-6 h

The grid sweep and HP search run on CPU so they don't contend with the BERT extraction.

## Manual pipeline (same steps)

```bash
# 1. Build dataset (~5 min)
bash run_pipeline.sh

# 2. AST features (~15s CPU)
python -m stacking.features.ast_features \
    --in_splits data/processed --out_dir runs/heads/extraction

# 3. OOF pointwise BERT (leakage-fixed)
#    5 folds + final full-train model = 6 pointwise training runs (~2h on 5090)
python -m stacking.features.oof_point \
    --data_dir data/processed \
    --out_dir runs/heads/extraction \
    --n_folds 5 \
    --epochs 40 --batch_size 16 --grad_accum 2 --lr 2e-5 --bf16 \
    --eval_every_steps 100 --patience 3 \
    --extract_batch 64 --resume

# 4. CLS cosine similarity
python -m stacking.features.semantic \
    --in_splits data/processed \
    --extraction_dir runs/heads/extraction

# 5. Grid sweep — fixed HPs, 4 heads x 1 variant x 3 seeds = 12 configs (~3 min CPU)
python -m stacking.sweep \
    --config stacking/configs/sweep.yaml \
    --in_splits data/processed \
    --extraction_dir runs/heads/extraction \
    --out_dir runs/heads

# 6. HP search — Optuna TPE on top heads, best seed-avg on test
python -m stacking.hp_search \
    --heads xgb,lgbm,mlp,stacked \
    --trials 40 --seeds 42,43,44 \
    --in_splits data/processed \
    --extraction_dir runs/heads/extraction \
    --out_root runs/heads/hp
```

## Grid sweep vs HP search

The pipeline runs both by design:

**Grid sweep** (`stacking.sweep`): fixed HPs across 4 heads x 1 variant x 3 seeds.
Cheap comparison of heads *under the same defaults*.

**HP search** (`stacking.hp_search`): Optuna TPE, per head. Rich search
spaces — XGBoost over depth/lr/n_estimators/regularization, LightGBM similar,
MLP over hidden_layers/hidden_dim/activation/optimizer/dropout/layer_norm,
Stacked over base-head subset and meta choice. Pruned with MedianPruner.
Winning HPs are re-run at 3 seeds on TEST for variance bars.

Read the grid sweep first (`runs/heads/SUMMARY.md`) for head ranking; read
the HP search (`runs/heads/hp/HP_SUMMARY.md`) for the final best numbers.

Artifacts per HP cell (`runs/heads/hp/<head>-v1/`):
- `study.db` — SQLite Optuna storage (resumable across runs)
- `best_params.json` — winning HPs + best val macro-F1
- `trials.jsonl` — every trial's params + val metrics
- `seed_results.jsonl` — final test eval at each seed
- `summary.json` — headline numbers (test acc mean ± std, etc.)

## Resumability

All training + extraction commands respect `--resume`. Kill a run mid-way;
rerun the same command and it picks up at the last completed fold /
chunk. Fold outputs live at `runs/heads/extraction/oof/fold_<k>/`.

## Heads (top-4)

| # | Head | When it wins |
|---|------|--------------|
| `xgb` | XGBoost | Strong default for 50-150 dim tabular |
| `lgbm` | LightGBM | Often faster + complementary to xgb |
| `mlp` | 2x128 + dropout | Smooth boundary for logit/cos space |
| `stacked` | LogReg meta on (xgb, lgbm, mlp) probs | Final ensemble |

`logreg` is registered for use as the default meta classifier inside
`stacked`, but it is not exposed for standalone sweeping. The lower-ranked
heads (gnb, knn, rf) were retired after the HP_SUMMARY established the
top-4 ordering.

## Outputs

Per experiment (`runs/heads/<head>-v1-s<seed>/`):

```
config.json              -- resolved config + HPs
scaler.joblib            -- fit on train only
schema.json              -- column order + scaled mask
metrics.jsonl            -- val + test metrics
test_metrics.json        -- accuracy, macro-F1, ROC-AUC, Brier, ECE
predictions.parquet      -- per-pair predictions for error analysis
feature_importance.json  -- gain/coef (tree + logreg only)
confusion_matrix.png
roc_curve.png
head/                    -- saved head artifact(s)
```

Aggregate (`runs/heads/SUMMARY.md` + `SUMMARY.csv`):
- All experiments ranked by test_acc
- Best-seed pivot per head
- Failures logged to `runs/heads/_failures.jsonl` (sweep is fail-soft)

OOF intermediates:
```
runs/heads/extraction/oof/
  splits/                -- per-fold train/heldout parquets
  fold_00/ .. fold_04/   -- per-fold pointwise run directories (train.log + best/)
  full/                  -- final full-train pointwise model
```

## Head-only inference

Once a winning config is selected (say `xgb v1 seed 42`):

```bash
python -m stacking.predict_head \
    --head_dir runs/heads/xgb-v1-s42 \
    --point_ckpt runs/heads/extraction/oof/full/best \
    --pair examples/linear.py examples/quadratic.py
```

Pass pairs in canonical order (B is same-speed-or-slower). Output is JSON
with `label` (same / A_faster), `prob_same`, `prob_A_faster`.

## Tests

```bash
python -m pytest stacking/tests/ -v
```

Unit tests cover AST extractor correctness, head fit/predict/save/load
round-trips, dataset filter + label construction, imbalance detection,
reproducibility under same seed, class-weight effect on minority recall.
The OOF drivers are currently covered by subprocess-level integration
(the underlying `train.py` is already battle-tested); unit tests for fold
disjointness live in `tests/test_oof_folds.py`.

## Known limitations

- **Recursion detection** is direct-only: `foo() calling foo`. Mutual recursion
  (`a→b→a`) is not caught (~5% miss rate in typical corpora).
- **`no_of_switches`** counts Python 3.10+ `match` statements only. Still
  near-zero in practice; tree heads ignore it.
