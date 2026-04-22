# stacking — stacked head for pairwise complexity ranking

Second-stage classifier on top of frozen GraphCodeBERT. Consumes pre-softmax
logits (pointwise 11-d + pairwise 3-d), AST features (21 per snippet,
differenced + |diff|), and CLS-cosine similarity; predicts the binary
`same` vs `A_faster` label on the B>=A subset of pairs.

Target deployment: **RunPod RTX 5090** (32GB VRAM, bf16, CUDA 12.8+).
End-to-end orchestrator: `stacking/scripts/run_runpod.sh`.

## Task

Caller submits pairs in canonical order (B is same-speed-or-slower than A).
Head outputs:

- `0 = same`     — A and B are in the same complexity tier
- `1 = A_faster` — B is strictly slower than A

Pairs with B_faster are filtered out of training and test splits.

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
# Torch wheels with CUDA 12.8 come from a non-default index:
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.6.0
pip install -r requirements-stacking.txt    # xgboost, lightgbm, radon, joblib
```

## One-shot pipeline

```bash
bash stacking/scripts/run_runpod.sh
```

Flags (all optional):
- `--skip-data` — dataset already built
- `--skip-oof-point` — OOF pointwise extraction already done
- `--skip-oof-pair` — accept pair leakage (cuts ~2.5 GPU-hours)
- `--n_folds N` — number of OOF folds (default 5)

Expected wallclock on 5090: ~3.5 hours end-to-end.

## Manual pipeline (same steps)

```bash
# 1. Build dataset (~5 min)
bash run_pipeline.sh

# 2. AST features (~15s CPU)
python -m stacking.features.ast_features \
    --in_splits data/processed --out_dir runs/heads/extraction

# 3. OOF pointwise BERT (leakage-fixed)
#    5 folds + final full-train model = 6 pointwise training runs (~60 min on 5090)
python -m stacking.features.oof_point \
    --data_dir data/processed \
    --out_dir runs/heads/extraction \
    --n_folds 5 \
    --epochs 8 --batch_size 16 --grad_accum 2 --lr 2e-5 --bf16 \
    --extract_batch 64 --resume

# 4. CLS cosine similarity
python -m stacking.features.semantic \
    --in_splits data/processed \
    --extraction_dir runs/heads/extraction

# 5. OOF pairwise BERT (leakage-fixed)
#    Warm-starts from the OOF pointwise full encoder for faster convergence.
python -m stacking.features.oof_pair \
    --data_dir data/processed \
    --out_dir runs/heads/extraction \
    --n_folds 5 \
    --epochs 6 --batch_size 12 --grad_accum 2 --lr 1e-5 --bf16 \
    --label_smoothing 0.05 --class_weights none \
    --warm_start_from runs/heads/extraction/oof/full/best \
    --extract_batch 32 --resume

# 6. Head sweep (72 experiments, ~10 min CPU)
python -m stacking.sweep \
    --config stacking/configs/sweep.yaml \
    --in_splits data/processed \
    --extraction_dir runs/heads/extraction \
    --out_dir runs/heads
```

## Resumability

All training + extraction commands respect `--resume`. Kill a run mid-way;
rerun the same command and it picks up at the last completed fold /
chunk. Fold outputs live at `runs/heads/extraction/oof/fold_<k>/` and
`oof_pair/fold_<k>/`.

## BERT variants (head features)

- `v1` — A + B pointwise logits with [raw A; raw B; diff; |diff|] (44 dims BERT)
- `v2` — pairwise logit only (3 dims BERT)
- `v3` — v1 + v2 combined (47 dims BERT)

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
