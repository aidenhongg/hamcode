"""Stacked head for pairwise complexity ranking.

Consumes frozen BERT pre-softmax logits (pointwise 11-d and/or pairwise 3-d),
AST features (21 per snippet, differenced), and CodeGraphBERT CLS cosine.
Predicts the binary "same vs A_faster" label on the B>=A subset of pairs.

Layout:
  features/    -- extractors for AST, BERT logits, CLS cosine
  heads/       -- 8 classifier heads with a shared protocol
  dataset.py   -- filter B>=A, join, scaler, build feature matrix per variant
  train_head.py -- single-experiment CLI
  predict_head.py -- pair inference CLI
  sweep.py     -- cartesian (head x variant x seed) sweep
"""

from __future__ import annotations

__all__ = []
