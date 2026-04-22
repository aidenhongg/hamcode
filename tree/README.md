# tree — random forest for code complexity classification

Point-task classifier: given a Python snippet, predict one of 11 worst-case
time-complexity classes using 18 hand-crafted tree-sitter features.

```
O(1)  O(log n)  O(n)  O(n log n)  O(n^2)  O(n^3)  exponential
O(m+n)  O(m*n)  O(m log n)  O((m+n) log(m+n))
```

Companion to the `codebert/` sibling project (neural approach on the same
label space via GraphCodeBERT). Runs on CPU; the RTX 5090 pod is for the
neural twin, not this RF.

## Layout

```
tree/
├── complexity/                  # importable package
│   ├── labels.py                # 11 canonical classes + tier map
│   ├── normalizer.py            # raw O(...) strings -> canonical labels
│   ├── schemas.py               # PointRecord + pyarrow schemas
│   ├── features.py              # 18 tree-sitter features
│   ├── split.py                 # leak-free problem-id split
│   ├── train.py                 # RandomizedSearchCV + ensemble trainer
│   ├── predict.py               # inference wrapper
│   └── ingest/
│       ├── codecomplex.py       # codeparrot/codecomplex on HF
│       ├── doocs_leetcode.py    # mine "# Time: O(...)" comments
│       ├── codeforces.py        # open-r1/codeforces editorial mining
│       └── utils.py             # shared ingestion helpers
├── scripts/
│   ├── 01_ingest.py             # run all three sources + merge + dedup
│   ├── 02_extract_features.py   # features + split assignment
│   ├── 03_train.py              # train model
│   ├── 04_eval.py               # re-evaluate saved model
│   └── 05_predict.py            # CLI: predict class of a .py file
├── configs/point_rf.yaml        # hyperparameters + search grid
├── tests/                       # pytest suite
├── data/                        # raw/interim/features/processed/audit — gitignored
├── models/                      # gitignored trained artifacts
└── requirements.txt
```

## Quick start (local)

```bash
pip install -r requirements.txt

# 1) Pull data from all three sources (~20-40 min depending on Codeforces cap)
python scripts/01_ingest.py

# 2) Extract features + assign splits by problem_id
python scripts/02_extract_features.py

# 3) Train the random forest ensemble
python scripts/03_train.py

# Optional: predict on a file
python scripts/05_predict.py --input some_solution.py
```

Each ingest source can also be run standalone:

```bash
python -m complexity.ingest.doocs_leetcode --max-files 500
python -m complexity.ingest.codecomplex --limit 1000
python -m complexity.ingest.codeforces --submissions-limit 50000
```

## RunPod deployment

Random forest is CPU-bound. The RTX 5090 blackwell pod provisioned for the
codebert twin is not needed here. Choose a **cheap CPU pod** (~8 vCPU, 16 GB RAM
is plenty for ~50k training rows):

```bash
# From the pod, after git-cloning tree/
pip install -r requirements.txt

# doocs/leetcode clone takes a minute; Codeforces submissions streaming takes
# the most time. Tune via --submissions-limit.
python scripts/01_ingest.py --codeforces-submissions-limit 100000

python scripts/02_extract_features.py
python scripts/03_train.py

# The trained artifact is ~10-40 MB. Copy to object storage or bake into the
# inference image.
```

If you *are* deploying both this and codebert on the same pod, keep both
`tree/` and `codebert/` under the same parent directory. The label definitions
in `complexity/labels.py` mirror `codebert/common/labels.py` — keep them in
sync when extending.

## Training knobs

All in `configs/point_rf.yaml`:

- `rf.n_estimators`, `rf.max_depth`, `rf.min_samples_leaf` — tree hyperparams
- `rf.class_weight: balanced_subsample` — handles imbalance without undersampling
- `search.enabled: true` — `RandomizedSearchCV` over `search.distributions`
- `ensemble.enabled: true` / `ensemble.seeds` — 5-seed ensemble, majority-vote

Data pipeline knobs:

- `max_code_chars: 20000` — skip pathologically long snippets
- `min_ast_nodes: 5` — skip empty/trivial snippets
- `split_strategy: by_problem_id` — leak-free splitting
- `train_ratio / val_ratio / test_ratio` — 80/10/10 default

## Data sources — what each contributes

| Source | Label origin | Primary classes | Expected yield |
|---|---|---|---|
| `codeparrot/codecomplex` (HF) | Expert annotation | 7 single-var classes | 0-4,900 Python rows (HF version is Java-only today; filters Python defensively, drops in a local JSONL if you have the paper's Python split) |
| `doocs/leetcode` (mined comments) | Editorial `# Time: O(...)` | **All 11, incl. multi-var** | Varies; primary source for `O(m+n)`, `O(m*n)`, `O(m log n)` |
| `open-r1/codeforces` editorials | Mined prose | 7 single-var + occasional multi-var | Supplementary |

Labels are tagged with `origin` in the parquet — `"dataset"` for hand-curated
sources, `"comment"` for mined text. You can filter by origin in training if
you want to train only on hand-curated labels and evaluate on mined ones.

## Testing

```bash
python -m pytest tests/
```

Covers all 18 features with per-feature unit tests plus end-to-end tests on
canonical complexity snippets (bubble sort, binary search, Fibonacci, etc.),
and the normalizer + comment-mining regex.

The `tests/_smoke_e2e.py` script runs the full pipeline on ~88 synthetic
records — useful after making changes to confirm nothing is wired wrong:

```bash
PYTHONPATH=. python tests/_smoke_e2e.py
```

## Known limits

- **Multi-var disambiguation is structural-feature-limited.** The extractor
  doesn't track which variables come from distinct inputs, so `for i in range(len(a)): for j in range(len(b)):`
  (O(m*n)) looks identical to `for i in range(n): for j in range(n):` (O(n^2)).
  The model has to guess from context (presence of `matrix`, `grid`,
  multi-param signatures, etc.). For better multi-var accuracy, add
  `num_size_params` and `distinct_iterables` features in a future revision.
- **Python 3.10+ `match_statement` required.** Tree-sitter-languages 1.10.2
  supports it, but if you downgrade you'll lose `no_of_switches` on match
  statements.
- **Comprehensions are treated as one compound loop.** A list comp with `k`
  `for_in_clause`s counts as `k` loops with depth `k`. This matches the
  semantic depth in most cases but undercounts nested body complexity when
  mixed with explicit loops inside the body expression.
- **Mutual recursion isn't detected.** `recursion_present` catches direct
  self-calls only.

## Labels

```python
from complexity.labels import POINT_LABELS, LABEL_TO_IDX, TIER

# POINT_LABELS[idx]  -> human-readable class string
# LABEL_TO_IDX[lab]  -> int (used as sklearn label)
# TIER[lab]          -> 0..6 ordinal (for ranking diagnostics)
```
