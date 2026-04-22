"""Dataset + collator for GraphCodeBERT complexity classification.

This is where the DFG magic happens. For each code string we:
  1) Parse with tree-sitter-python.
  2) Extract DFG edges via vendored parser/DFG.py.
  3) Tokenize each source-level token with the GraphCodeBERT BPE tokenizer,
     tracking a mapping from source-token index -> BPE span.
  4) Build the final sequence:
         [CLS] <code BPE tokens> [SEP] <DFG variable tokens> <PAD ...>
     plus, for --pair, we splice two snippets separated by [SEP] and keep
     each snippet's DFG as a disjoint graph subregion.
  5) Build:
         input_ids        [max_seq]
         position_ids     [max_seq]    (code: 2..; DFG: 0; pad: 1)
         attention_mask   [max_seq, max_seq]   (3D graph-guided)
  6) Cache the whole bundle keyed by code SHA256 so tree-sitter runs once.
"""

from __future__ import annotations

import hashlib
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# AST on LeetCode code occasionally triggers Py3.12 invalid-escape warnings.
# The tokenizer fires a "too long" warning per over-budget sample. Both are
# expected and handled downstream — suppress so training logs stay readable.
warnings.filterwarnings("ignore", category=SyntaxWarning)
try:
    import transformers.utils.logging as _hf_log  # type: ignore
    _hf_log.set_verbosity_error()
except Exception:
    pass

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from common.dfg_cache import DFGCache
from common.labels import (
    LABEL_TO_IDX,
    NUM_PAIR_LABELS,
    NUM_POINT_LABELS,
    PAIR_LABEL_TO_IDX,
)

# -----------------------------------------------------------------------------
# tree-sitter + parser
# -----------------------------------------------------------------------------

def _make_parser():
    """Try several ways to get a tree-sitter Python parser."""
    try:
        from tree_sitter_languages import get_parser  # prebuilt binary bundle
        return get_parser("python")
    except Exception:
        pass
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_python as tsp
        parser = Parser()
        lang = Language(tsp.language())
        if hasattr(parser, "set_language"):
            parser.set_language(lang)
        else:  # tree_sitter >= 0.22 style
            parser.language = lang
        return parser
    except Exception as e:
        raise RuntimeError(
            "Cannot load tree-sitter-python. Install `tree_sitter_languages` "
            "or `tree_sitter_python` plus `tree_sitter`."
        ) from e


_PY_PARSER = None

def get_python_parser():
    global _PY_PARSER
    if _PY_PARSER is None:
        _PY_PARSER = _make_parser()
    return _PY_PARSER


# -----------------------------------------------------------------------------
# DFG extraction (closely adapted from microsoft/CodeBERT run.py)
# -----------------------------------------------------------------------------

_PARSER_SYMBOLS = None

def _parser_symbols():
    global _PARSER_SYMBOLS
    if _PARSER_SYMBOLS is None:
        from parser import (
            DFG_python,
            index_to_code_token,
            remove_comments_and_docstrings,
            tree_to_token_index,
        )
        _PARSER_SYMBOLS = (DFG_python, index_to_code_token,
                           remove_comments_and_docstrings, tree_to_token_index)
    return _PARSER_SYMBOLS


def extract_dataflow(code: str) -> tuple[list[str], list[tuple]]:
    """Return (code_tokens, dfg_edges).

    code_tokens: list of source-level leaf tokens (identifier / literal / punct).
    dfg_edges:   list of (var_name, idx, edge_type, source_vars, source_indices).
                 `idx` indexes into code_tokens.
    """
    parser = get_python_parser()
    DFG_python, index_to_code_token, remove_comments_and_docstrings, tree_to_token_index = _parser_symbols()
    try:
        clean = remove_comments_and_docstrings(code, "python")
    except Exception:
        clean = code
    try:
        tree = parser.parse(bytes(clean, "utf-8"))
        root = tree.root_node
        indices = tree_to_token_index(root)
        lines = clean.split("\n")
        code_tokens = [index_to_code_token(x, lines) for x in indices]
        index_to_code: dict[tuple, tuple[int, str]] = {}
        for idx, (pos, tok) in enumerate(zip(indices, code_tokens)):
            index_to_code[pos] = (idx, tok)
        try:
            dfg, _ = DFG_python(root, index_to_code, {})
        except Exception:
            dfg = []
        dfg = sorted(dfg, key=lambda x: x[1])
        used: set[int] = set()
        for d in dfg:
            if len(d[-1]) != 0:
                used.add(d[1])
            for x in d[-1]:
                used.add(x)
        dfg = [d for d in dfg if d[1] in used]
        return code_tokens, dfg
    except Exception:
        return [], []


# -----------------------------------------------------------------------------
# Tokenization + mapping from source-token index to BPE span
# -----------------------------------------------------------------------------

def _tokenize_with_mapping(
    code_tokens: list[str], tokenizer
) -> tuple[list[str], list[tuple[int, int]]]:
    """Tokenize each source token; return flat BPE tokens + per-source-token spans.

    Uses the "@" trick from GraphCodeBERT: prepend "@" and drop first piece so
    subword tokens get the space-prefix Ġ consistently even without leading
    whitespace.
    """
    flat: list[str] = []
    spans: list[tuple[int, int]] = []
    for i, tok in enumerate(code_tokens):
        if i == 0:
            sub = tokenizer.tokenize(tok) if tok else []
        else:
            sub = tokenizer.tokenize("@" + tok)[1:] if tok else []
        start = len(flat)
        flat.extend(sub)
        spans.append((start, len(flat)))
    return flat, spans


# -----------------------------------------------------------------------------
# Input bundle builder — the core thing
# -----------------------------------------------------------------------------

@dataclass
class InputBundle:
    input_ids: np.ndarray        # [seq]
    position_ids: np.ndarray     # [seq]
    attention_mask: np.ndarray   # [seq, seq]  (bool)


def build_point_inputs(
    code: str,
    tokenizer,
    max_seq_len: int = 512,
    max_dfg_nodes: int = 64,
) -> InputBundle:
    """Build a single-snippet bundle for --point."""
    cls, sep, pad, unk, pad_id = _special_ids(tokenizer)

    raw_tokens, dfg = extract_dataflow(code)
    flat, spans = _tokenize_with_mapping(raw_tokens, tokenizer)

    # Budget: CLS + code + SEP + DFG nodes
    # Reserve 2 special tokens; leave the rest split between code and DFG.
    max_code = max_seq_len - 2 - max_dfg_nodes
    if max_code < 32:
        max_code = 32  # defensive; tiny

    # Truncate code tokens (respecting source-token spans when possible).
    if len(flat) > max_code:
        flat = flat[:max_code]
        # drop spans that fall outside the new code region
        spans = [(a, b) for (a, b) in spans if b <= max_code]

    # Cap DFG by available slots and by max_dfg_nodes.
    dfg_budget = min(max_dfg_nodes, max_seq_len - 2 - len(flat))
    dfg = _keep_valid_dfg(dfg, len(spans))[:dfg_budget]

    # Assemble: [CLS] flat [SEP] dfg_vars [PAD...]
    code_part = [cls] + tokenizer.convert_tokens_to_ids(flat) + [sep]
    # DFG tokens: use UNK as the placeholder id (like upstream).
    dfg_part = [unk] * len(dfg)
    used = len(code_part) + len(dfg_part)
    pad_len = max_seq_len - used
    input_ids = code_part + dfg_part + [pad_id] * pad_len
    input_ids = input_ids[:max_seq_len]

    # position_ids
    position_ids: list[int] = []
    # [CLS] and code tokens use positions starting at pad_id + 1
    # (RoBERTa convention — pad is 1, so code starts at 2). [CLS] is token 0 effectively,
    # but we follow upstream: give CLS position 2 too (start + 0).
    for i in range(len(code_part)):
        position_ids.append(pad_id + 1 + i)  # 2, 3, 4, ...
    position_ids.extend([0] * len(dfg_part))           # DFG: UNK_POS
    position_ids.extend([pad_id] * pad_len)            # padding
    position_ids = position_ids[:max_seq_len]

    # attention mask [seq, seq]
    seq = max_seq_len
    attn = np.zeros((seq, seq), dtype=bool)

    # indices
    n_code = len(code_part)              # [CLS] ... [SEP]
    n_dfg = len(dfg_part)
    code_start, code_end = 0, n_code     # [CLS] is at 0
    dfg_start = n_code

    # 1) Code tokens attend to all code tokens (full within code)
    attn[code_start:code_end, code_start:code_end] = True

    # 2) DFG-to-code: each DFG node attends to the code tokens of its variable
    #    occurrence. Var's source index = dfg[k][1]; find its BPE span.
    var_indexd: dict[int, int] = {}   # src_token_idx -> dfg_order
    for j, d in enumerate(dfg):
        src_idx = d[1]
        # +1 for [CLS] at position 0 pushing code tokens to +1
        if 0 <= src_idx < len(spans):
            span_start, span_end = spans[src_idx]
            attn[dfg_start + j, 1 + span_start : 1 + span_end] = True
            attn[1 + span_start : 1 + span_end, dfg_start + j] = True  # bidir
        var_indexd[src_idx] = j

    # 3) DFG-to-DFG: edge exists iff DFG_j's source_indices includes DFG_k's source_idx
    for j, d in enumerate(dfg):
        for src in d[4]:
            if src in var_indexd:
                attn[dfg_start + j, dfg_start + var_indexd[src]] = True

    # 4) CLS row already True within code block (above). Good.

    # 5) Padding rows/cols stay 0 by construction.

    return InputBundle(
        input_ids=np.asarray(input_ids, dtype=np.int64),
        position_ids=np.asarray(position_ids, dtype=np.int64),
        attention_mask=attn,
    )


def build_pair_inputs(
    code_a: str,
    code_b: str,
    tokenizer,
    max_seq_len: int = 512,
    max_dfg_nodes_total: int = 64,
) -> InputBundle:
    """Build a cross-encoder bundle: [CLS] A [SEP] B [SEP] DFG_A DFG_B [PAD]."""
    cls, sep, pad, unk, pad_id = _special_ids(tokenizer)

    per_dfg = max_dfg_nodes_total // 2
    # Each side's code budget ≈ (seq_len - 3 special - 2*per_dfg) / 2
    max_code_each = (max_seq_len - 3 - max_dfg_nodes_total) // 2

    def _prep(code: str):
        raw_tokens, dfg = extract_dataflow(code)
        flat, spans = _tokenize_with_mapping(raw_tokens, tokenizer)
        if len(flat) > max_code_each:
            flat = flat[:max_code_each]
            spans = [(a, b) for (a, b) in spans if b <= max_code_each]
        dfg = _keep_valid_dfg(dfg, len(spans))[:per_dfg]
        return flat, spans, dfg

    flat_a, spans_a, dfg_a = _prep(code_a)
    flat_b, spans_b, dfg_b = _prep(code_b)

    # Assemble ids: [CLS] flat_a [SEP] flat_b [SEP] dfg_a dfg_b [PAD...]
    ids = (
        [cls]
        + tokenizer.convert_tokens_to_ids(flat_a)
        + [sep]
        + tokenizer.convert_tokens_to_ids(flat_b)
        + [sep]
        + [unk] * len(dfg_a)
        + [unk] * len(dfg_b)
    )
    pad_len = max_seq_len - len(ids)
    if pad_len < 0:
        ids = ids[:max_seq_len]; pad_len = 0
    ids = ids + [pad_id] * pad_len

    # Index offsets
    cls_idx = 0
    flat_a_start = 1                       # right after CLS
    flat_a_end = flat_a_start + len(flat_a)
    sep1_idx = flat_a_end
    flat_b_start = sep1_idx + 1
    flat_b_end = flat_b_start + len(flat_b)
    sep2_idx = flat_b_end
    dfg_a_start = sep2_idx + 1
    dfg_a_end = dfg_a_start + len(dfg_a)
    dfg_b_start = dfg_a_end
    dfg_b_end = dfg_b_start + len(dfg_b)
    code_end = sep2_idx + 1  # CLS..SEP2 inclusive region
    total_used = dfg_b_end

    # position_ids:
    #   CLS & flat_a: pad_id+1, pad_id+2, ...
    #   [SEP] after A: continues
    #   flat_b: restarts at pad_id+1+offset? Upstream uses a single continuous
    #     numbering — we follow that (matches RoBERTa's usual behavior; the
    #     model was pretrained with one sequence). This keeps inference simple.
    pos: list[int] = []
    for i in range(code_end):
        pos.append(pad_id + 1 + i)
    pos.extend([0] * (len(dfg_a) + len(dfg_b)))
    pos.extend([pad_id] * pad_len)
    pos = pos[:max_seq_len]

    # attention mask [seq, seq]
    seq = max_seq_len
    attn = np.zeros((seq, seq), dtype=bool)

    # Code-to-code attention: split into two disjoint blocks so A's code doesn't
    # attend to B's DFG (and vice versa). We DO allow A-code <-> B-code at the
    # transformer level — that's the whole point of a cross-encoder.
    attn[:code_end, :code_end] = True

    # DFG-A connections
    var_a: dict[int, int] = {}
    for j, d in enumerate(dfg_a):
        src_idx = d[1]
        if 0 <= src_idx < len(spans_a):
            span_s, span_e = spans_a[src_idx]
            attn[dfg_a_start + j, flat_a_start + span_s : flat_a_start + span_e] = True
            attn[flat_a_start + span_s : flat_a_start + span_e, dfg_a_start + j] = True
        var_a[src_idx] = j
    for j, d in enumerate(dfg_a):
        for src in d[4]:
            if src in var_a:
                attn[dfg_a_start + j, dfg_a_start + var_a[src]] = True

    # DFG-B connections (disjoint from DFG-A — no cross-snippet edges)
    var_b: dict[int, int] = {}
    for j, d in enumerate(dfg_b):
        src_idx = d[1]
        if 0 <= src_idx < len(spans_b):
            span_s, span_e = spans_b[src_idx]
            attn[dfg_b_start + j, flat_b_start + span_s : flat_b_start + span_e] = True
            attn[flat_b_start + span_s : flat_b_start + span_e, dfg_b_start + j] = True
        var_b[src_idx] = j
    for j, d in enumerate(dfg_b):
        for src in d[4]:
            if src in var_b:
                attn[dfg_b_start + j, dfg_b_start + var_b[src]] = True

    return InputBundle(
        input_ids=np.asarray(ids, dtype=np.int64),
        position_ids=np.asarray(pos, dtype=np.int64),
        attention_mask=attn,
    )


def _special_ids(tokenizer) -> tuple[int, int, int, int, int]:
    cls = tokenizer.cls_token_id
    sep = tokenizer.sep_token_id
    pad = tokenizer.pad_token_id
    unk = tokenizer.unk_token_id
    pad_id = tokenizer.pad_token_id
    return cls, sep, pad, unk, pad_id


def _keep_valid_dfg(dfg: list[tuple], n_src_tokens: int) -> list[tuple]:
    # Drop edges whose source index falls outside the (truncated) code region.
    out = []
    for d in dfg:
        if 0 <= d[1] < n_src_tokens and all(0 <= s < n_src_tokens for s in d[4]):
            out.append(d)
    return out


# -----------------------------------------------------------------------------
# Datasets (Parquet-backed)
# -----------------------------------------------------------------------------

class PointDataset(Dataset):
    """Loads a pointwise parquet. Defaults to simple 1D tokenization (no DFG).

    Pass `use_dfg=True` to enable the tree-sitter + 2D-mask path. Simple mode
    is ~10x faster per sample and needs no prewarming.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        tokenizer,
        max_seq_len: int = 512,
        max_dfg_nodes: int = 64,
        cache_dir: str | None = None,
        use_dfg: bool = False,
    ) -> None:
        super().__init__()
        self.table = pq.read_table(parquet_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_dfg_nodes = max_dfg_nodes
        self.use_dfg = use_dfg
        self.cache = (DFGCache(cache_dir) if cache_dir is not None else DFGCache()) if use_dfg else None
        self._codes = self.table.column("code").to_pylist()
        self._labels = self.table.column("label").to_pylist()
        self._ids = self.table.column("id").to_pylist()

    def __len__(self) -> int:
        return len(self._codes)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        code = self._codes[i]
        label_idx = LABEL_TO_IDX[self._labels[i]]

        if not self.use_dfg:
            enc = self.tokenizer(
                code,
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_len,
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],            # 1D — standard
                "position_ids": torch.arange(self.max_seq_len, dtype=torch.long),
                "labels": torch.tensor(label_idx, dtype=torch.long),
            }

        key = f"point:{self.max_seq_len}:{self.max_dfg_nodes}"
        cached = self.cache.get(code + "|" + key)
        if cached is not None:
            b: InputBundle = cached
        else:
            b = build_point_inputs(code, self.tokenizer, self.max_seq_len, self.max_dfg_nodes)
            try:
                self.cache.put(code + "|" + key, b)
            except OSError:
                pass
        return {
            "input_ids": torch.from_numpy(b.input_ids),
            "position_ids": torch.from_numpy(b.position_ids),
            "attention_mask": torch.from_numpy(b.attention_mask).to(torch.bool),
            "labels": torch.tensor(label_idx, dtype=torch.long),
        }


class PairDataset(Dataset):
    """Loads a pairwise parquet."""

    def __init__(
        self,
        parquet_path: str | Path,
        tokenizer,
        max_seq_len: int = 512,
        max_dfg_nodes: int = 64,
        cache_dir: str | None = None,
        use_dfg: bool = False,
    ) -> None:
        super().__init__()
        self.table = pq.read_table(parquet_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_dfg_nodes = max_dfg_nodes
        self.use_dfg = use_dfg
        self.cache = (DFGCache(cache_dir) if cache_dir is not None else DFGCache()) if use_dfg else None
        self._a = self.table.column("code_a").to_pylist()
        self._b = self.table.column("code_b").to_pylist()
        self._t = self.table.column("ternary").to_pylist()
        self._ids = self.table.column("pair_id").to_pylist()

    def __len__(self) -> int:
        return len(self._a)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        label_idx = PAIR_LABEL_TO_IDX[self._t[i]]

        if not self.use_dfg:
            enc = self.tokenizer(
                self._a[i], self._b[i],                # HF handles [CLS] A [SEP] B [SEP]
                truncation="longest_first",
                padding="max_length",
                max_length=self.max_seq_len,
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "position_ids": torch.arange(self.max_seq_len, dtype=torch.long),
                "labels": torch.tensor(label_idx, dtype=torch.long),
            }

        key = f"pair:{self.max_seq_len}:{self.max_dfg_nodes}"
        cache_key = (self._a[i] + "\n---\n" + self._b[i]) + "|" + key
        cached = self.cache.get(cache_key)
        if cached is not None:
            b: InputBundle = cached
        else:
            b = build_pair_inputs(
                self._a[i], self._b[i],
                self.tokenizer, self.max_seq_len, self.max_dfg_nodes,
            )
            try:
                self.cache.put(cache_key, b)
            except OSError:
                pass
        return {
            "input_ids": torch.from_numpy(b.input_ids),
            "position_ids": torch.from_numpy(b.position_ids),
            "attention_mask": torch.from_numpy(b.attention_mask).to(torch.bool),
            "labels": torch.tensor(label_idx, dtype=torch.long),
        }


def make_collator() -> Callable[[list[dict]], dict[str, torch.Tensor]]:
    """All items are already fixed-shape; just stack."""
    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
            "position_ids": torch.stack([b["position_ids"] for b in batch], dim=0),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch], dim=0),
            "labels": torch.stack([b["labels"] for b in batch], dim=0),
        }
    return collate


# -----------------------------------------------------------------------------
# Inferred class-weight helper
# -----------------------------------------------------------------------------

def compute_class_weights(parquet_path: str | Path) -> list[float]:
    """w_c ∝ sqrt(N_total / N_c), normalized to mean 1."""
    table = pq.read_table(parquet_path)
    labels = table.column("label").to_pylist()
    counts = np.zeros(NUM_POINT_LABELS, dtype=np.float64)
    for lab in labels:
        counts[LABEL_TO_IDX[lab]] += 1
    total = counts.sum()
    # Avoid divide-by-zero for missing classes.
    safe = np.where(counts > 0, counts, 1.0)
    w = np.sqrt(total / safe)
    w = w / w.mean()
    # Zero out classes that have no samples so their loss contribution is 0.
    w[counts == 0] = 0.0
    return w.tolist()
