# stacking — stacked head for pairwise complexity ranking

Second-stage classifier on top of frozen GraphCodeBERT. Consumes pre-softmax
logits (pointwise 11-d + **pairwise 2-d**), AST features (21 per snippet,
differenced + |diff|), and CLS-cosine similarity; predicts the binary
`same` vs `A_faster` label.

Target deployment: **RunPod RTX 5090** (32GB VRAM, bf16, CUDA 12.8+).
End-to-end orchestrator: `stacking/scripts/run_runpod.sh`.

## Task

**The pairwise BERT head is now binary, not ternary.** The pipeline
canonicalizes every pair so tier_A <= tier_B — given any candidate
(a, b), if tier_a > tier_b they are swapped. After the rewrite the
dataset contains only two classes:

- `0 = same`     — A and B are in the same complexity tier
- `1 = A_faster` — B is strictly slower than A (i.e. f(A) ∈ o(f(B)))

Callers of head inference must submit pairs in canonical order
(B is same-speed-or-slower than A). `pipeline/10_make_pairwise.py`
enforces this at data-generation time; `stacking/dataset.filter_b_ge_a`
remains as a defensive safety net.

## Leakage fix (vs. original plan)

The original recipe (`bert_logits.py --point/--pair`) reuses a pointwise
and pairwise BERT trained on the same `train.parquet`/`pair_train.parquet`
codes that end up inside the head's training features. Those train-split
logits are therefore over-confident and the head overfits. The new
default replaces both with K-fold out-of-fold (OOF) extraction:

| Module | Old (leaky) | New (OOF, default) |
|--------|-------------|--------------------|
| Pointwise logits + CLS | `stacking/features/bert_logits.py --point` | `stacking/features/oof_point.py` |
| Pairwise logits | `stacking/features/bert_logits.py --pair` | `stacking/features/oof_pair.py` |

`oof_point.py` splits train by **problem_id** (so no code's problem
appears across folds), trains pointwise BERT K times on K-1-fold unions,
and concatenates the held-out fold predictions into a single
`point_logits_train.parquet`. A final full-train model covers val/test.

`oof_pair.py` does the same for pair BERT, grouping pairs by a union-find
over their code SHA256s so no code is shared across fold training sets.

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

`microsoft/graphcodebert-base`'s `main` branch only ships `pytorch_model.bin`.
For every such repo, `transformers.modeling_utils._get_resolved_checkpoint_files`
launches a **background thread** (`Thread-auto_conversion`) that walks every
open PR ref (`refs/pr/N`) looking for a safetensors conversion. PR #8 of this
model is exactly such a conversion, but its LFS redirects hang indefinitely —
the thread runs alongside training, holds network sockets, and blocks the
main process on I/O. `use_safetensors=False` does NOT gate that thread; it
only affects the main-thread file-preference logic.

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

Runs end-to-end: data → AST → OOF pointwise → semantic → OOF pairwise →
grid sweep (72 configs) → HP search on top heads (Optuna, 40 trials each).

Flags (all optional):
- `--skip-data` — dataset already built
- `--skip-oof-point` — OOF pointwise extraction already done
- `--skip-oof-pair` — accept pair leakage (cuts ~2.5 GPU-hours)
- `--skip-sweep` — skip the 72-config grid (go straight to HP search)
- `--skip-hp` — skip Optuna HP search
- `--hp-trials N` — Optuna trials per (head, variant) cell (default 40)
- `--hp-heads` — csv, default `xgb,lgbm,mlp,stacked`
- `--hp-variants` — csv, default `v1,v3` (v2 consistently loses on this task)
- `--n_folds N` — number of OOF folds (default 5)

Expected wallclock on 5090 (post epoch-bump + HP search):
- OOF pointwise (40-epoch cap, 6 runs): ~2 h
- OOF pairwise (30-epoch cap, 6 runs): ~4 h
- Grid sweep (72 head configs): ~10 min CPU
- HP search (4 heads x 2 variants x 40 trials): ~2-4 h CPU
- Total: ~8-10 h

The grid sweep and HP search run on CPU so they don't contend with the BERT extraction.

## Manual pipeline (same steps)

```bash
# 1. Build dataset (~5 min)
bash run_pipeline.sh

# 2. AST features (~15s CPU)
python -m stacking.features.ast_features \
    --in_splits data/processed --out_dir runs/heads/extraction

# 3. OOF pointwise BERT (leakage-fixed, 8-epoch cap)
#    5 folds + final full-train model = 6 pointwise training runs (~45 min on 5090)
python -m stacking.features.oof_point \
    --data_dir data/processed \
    --out_dir runs/heads/extraction \
    --n_folds 5 \
    --epochs 8 --batch_size 16 --grad_accum 2 --lr 2e-5 --bf16 \
    --eval_every_steps 100 --patience 2 \
    --extract_batch 64 --resume

# 4. CLS cosine similarity
python -m stacking.features.semantic \
    --in_splits data/processed \
    --extraction_dir runs/heads/extraction

# 5. OOF pairwise BERT (binary task, 8-epoch cap)
#    Warm-starts from the OOF pointwise full encoder for faster convergence.
python -m stacking.features.oof_pair \
    --data_dir data/processed \
    --out_dir runs/heads/extraction \
    --n_folds 5 \
    --epochs 8 --batch_size 12 --grad_accum 2 --lr 1e-5 --bf16 \
    --eval_every_steps 100 --patience 2 \
    --label_smoothing 0.05 --class_weights none \
    --warm_start_from runs/heads/extraction/oof/full/best \
    --extract_batch 32 --resume

# 6. Grid sweep — fixed HPs, 8 heads x 3 variants x 3 seeds = 72 configs (~10 min CPU)
python -m stacking.sweep \
    --config stacking/configs/sweep.yaml \
    --in_splits data/processed \
    --extraction_dir runs/heads/extraction \
    --out_dir runs/heads

# 7. HP search — Optuna TPE on top heads (XGB/LGBM/MLP/Stacked), best seed-avg on test
python -m stacking.hp_search \
    --heads xgb,lgbm,mlp,stacked \
    --variants v1,v3 \
    --trials 40 --seeds 42,43,44 \
    --in_splits data/processed \
    --extraction_dir runs/heads/extraction \
    --out_root runs/heads/hp
```

## Grid sweep vs HP search

The pipeline runs both by design:

**Grid sweep** (`stacking.sweep`): fixed HPs across 8 heads x 3 variants x 3 seeds.
Cheap comparison of heads *under the same defaults*. Tells you which head
architecture responds best to these features out of the box.

**HP search** (`stacking.hp_search`): Optuna TPE, per (head, variant). Rich search
spaces — XGBoost over depth/lr/n_estimators/regularization, LightGBM similar,
MLP over hidden_layers/hidden_dim/activation/optimizer/dropout/layer_norm,
Stacked over base-head subset and meta choice. Pruned with MedianPruner.
Winning HPs are re-run at 3 seeds on TEST for variance bars.

Read the grid sweep first (`runs/heads/SUMMARY.md`) for head ranking; read
the HP search (`runs/heads/hp/HP_SUMMARY.md`) for the final best numbers.

Artifacts per HP cell (`runs/heads/hp/<head>-<variant>/`):
- `study.db` — SQLite Optuna storage (resumable across runs)
- `best_params.json` — winning HPs + best val macro-F1
- `trials.jsonl` — every trial's params + val metrics
- `seed_results.jsonl` — final test eval at each seed
- `summary.json` — headline numbers (test acc mean ± std, etc.)

## Resumability

All training + extraction commands respect `--resume`. Kill a run mid-way;
rerun the same command and it picks up at the last completed fold /
chunk. Fold outputs live at `runs/heads/extraction/oof/fold_<k>/` and
`oof_pair/fold_<k>/`.

## BERT variants (head features)

- `v1` — A + B pointwise logits with [raw A; raw B; diff; |diff|] (44 dims BERT)
- `v2` — pairwise logit only (**2 dims** BERT after the binary rewrite)
- `v3` — v1 + v2 combined (**46 dims** BERT)

All variants also include the 84-dim AST diff block + 4-dim CLS similarity block.

## Heads (8)

| # | Head | When it wins |
|---|------|--------------|
| `xgb` | XGBoost | Strong default for 50-150 dim tabular |
| `lgbm` | LightGBM | Often faster + complementary to xgb |
| `mlp` | 2x128 + dropout | Smooth boundary for logit/cos space |
| `logreg` | Logistic Regression | Linear baseline (calibration sanity) |
| `rf` | Random Forest | Uncorrelated with GBM |
| `knn` | KNN (cosine, k=11) | Instance-based |
| `gnb` | Gaussian NB | Naive baseline |
| `stacked` | LogReg meta on (xgb, lgbm, mlp) probs | Final ensemble |

## Outputs

Per experiment (`runs/heads/<head>-<variant>-s<seed>/`):

```
config.json              -- resolved config + HPs
scaler.joblib            -- fit on train only
schema.json              -- column order + scaled mask
metrics.jsonl            -- val + test metrics
test_metrics.json        -- accuracy, macro-F1, ROC-AUC, Brier, ECE,
                            McNemar p vs BERT baseline
predictions.parquet      -- per-pair predictions for error analysis
feature_importance.json  -- gain/coef (tree + logreg only)
confusion_matrix.png
roc_curve.png
head/                    -- saved head artifact(s)
```

Aggregate (`runs/heads/SUMMARY.md` + `SUMMARY.csv`):
- All experiments ranked by test_acc
- Best-seed pivot per (head, variant)
- Failures logged to `runs/heads/_failures.jsonl` (sweep is fail-soft)

OOF intermediates:
```
runs/heads/extraction/oof/
  splits/                -- per-fold train/heldout parquets
  fold_00/ .. fold_04/   -- per-fold pointwise run directories (train.log + best/)
  full/                  -- final full-train pointwise model
runs/heads/extraction/oof_pair/
  splits/
  fold_00/ .. fold_04/   -- per-fold pair run directories
  full/                  -- final full-train pair model
```

## Head-only inference

Once a winning config is selected (say `xgb v3 seed 42`):

```bash
python -m stacking.predict_head \
    --head_dir runs/heads/xgb-v3-s42 \
    --point_ckpt runs/heads/extraction/oof/full/best \
    --pair_ckpt  runs/heads/extraction/oof_pair/full/best \
    --pair examples/linear.py examples/quadratic.py
```

Pass pairs in canonical order (B is same-speed-or-slower). Output is JSON
with `label` (same / A_faster), `prob_same`, `prob_A_faster`.

## Tests

```bash
python -m pytest stacking/tests/ -v
```

47 unit tests cover AST extractor correctness, head fit/predict/save/load
round-trips, dataset filter + label construction, imbalance detection,
reproducibility under same seed, class-weight effect on minority recall.
The OOF drivers are currently covered by subprocess-level integration
(the underlying `train.py` is already battle-tested); unit tests for fold
disjointness live in `tests/test_oof_folds.py` (see below).

## Known limitations

- **Recursion detection** is direct-only: `foo() calling foo`. Mutual recursion
  (`a→b→a`) is not caught (~5% miss rate in typical corpora).
- **`no_of_switches`** counts Python 3.10+ `match` statements only. Still
  near-zero in practice; tree heads ignore it.
- **Pair BERT warm-start**: by default each OOF pair fold warm-starts from
  the OOF pointwise full encoder. That encoder saw all train codes, so
  there's residual encoder-level leakage into pair fold training. It is
  bounded (the classifier head is fresh per fold) but non-zero. A fully
  clean run would warm-start pair fold `k` from pointwise fold `k`; this
  is deferred as TODO — the expected gain is small relative to the 5x
  cost of fold-aligned warm starts.
