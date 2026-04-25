"""Dataset + LongCoder input encoder for pointwise complexity classification.

For each Python snippet we build an encoder-friendly bundle:
  1) Tokenize the source with the LongCoder BPE.
  2) Locate `import` / `def` / `class` statements via tree-sitter; the
     newline BPE that ends each such statement becomes a memory token.
  3) Insert bridge tokens at uniform stride throughout the sequence.
     Bridges are global summary slots; together with memory tokens they
     form the global-attention set that LongCoder routes long-range
     information through.
  4) Pack into fixed-shape tensors:
        input_ids               [seq]
        attention_mask          [seq]   1 on real tokens, 0 on pad
        global_attention_mask   [seq]   1 on bridge + memory + last-token
        token_type_ids          [seq]   0 normal, 1 bridge, 2 memory

We cache the full bundle keyed by (code_sha, encoder_config_hash) so
tree-sitter runs once per snippet across runs.
"""

from __future__ import annotations

import hashlib
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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

from common.labels import LABEL_TO_IDX, NUM_POINT_LABELS
from common.parsers import memory_byte_offsets

TOKEN_TYPE_NORMAL = 0
TOKEN_TYPE_BRIDGE = 1
TOKEN_TYPE_MEMORY = 2


# Language-aware memory-line offsets: dispatches to the correct tree-sitter
# parser via common.parsers. The Python-only helper that lived here previously
# is gone; callers must pass `language`.


# -----------------------------------------------------------------------------
# Input bundle builder
# -----------------------------------------------------------------------------

@dataclass
class InputBundle:
    input_ids: np.ndarray              # [seq]
    attention_mask: np.ndarray         # [seq]
    global_attention_mask: np.ndarray  # [seq]
    token_type_ids: np.ndarray         # [seq]


def _bridge_token_id(tokenizer) -> int:
    """LongCoder doesn't ship a dedicated bridge id; use the unk id as the
    placeholder and let `token_type_ids=BRIDGE` route it through LongCoder's
    bridge-specific Q/K/V projections."""
    return tokenizer.unk_token_id


def _memory_token_id(tokenizer) -> int:
    """Same convention as bridges — the per-type Q/K/V projections do the
    real work; the literal id is a placeholder."""
    return tokenizer.unk_token_id


def build_point_inputs(
    code: str,
    tokenizer,
    max_seq_len: int = 4096,
    bridge_stride: int = 128,
    language: str = "python",
) -> InputBundle:
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    bridge_id = _bridge_token_id(tokenizer)
    memory_id = _memory_token_id(tokenizer)

    enc = tokenizer(
        code,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    code_ids: list[int] = list(enc["input_ids"])
    offsets: list[tuple[int, int]] = list(enc["offset_mapping"])

    mem_byte_offsets = memory_byte_offsets(language, code)
    mem_token_indices: set[int] = set()
    if mem_byte_offsets:
        offset_set = set(mem_byte_offsets)
        for i, (a, b) in enumerate(offsets):
            if a in offset_set or (b - 1) in offset_set:
                mem_token_indices.add(i)

    ids: list[int] = [cls_id]
    types: list[int] = [TOKEN_TYPE_NORMAL]
    globals_: list[int] = [0]

    stride = max(1, int(bridge_stride))
    for i, tid in enumerate(code_ids):
        if stride and i > 0 and i % stride == 0:
            ids.append(bridge_id)
            types.append(TOKEN_TYPE_BRIDGE)
            globals_.append(1)
        if i in mem_token_indices:
            ids.append(memory_id)
            types.append(TOKEN_TYPE_MEMORY)
            globals_.append(1)
            ids.append(tid)
            types.append(TOKEN_TYPE_NORMAL)
            globals_.append(0)
        else:
            ids.append(tid)
            types.append(TOKEN_TYPE_NORMAL)
            globals_.append(0)

    if len(ids) > max_seq_len - 1:
        ids = ids[: max_seq_len - 1]
        types = types[: max_seq_len - 1]
        globals_ = globals_[: max_seq_len - 1]

    ids.append(sep_id)
    types.append(TOKEN_TYPE_NORMAL)
    globals_.append(1)

    real_len = len(ids)
    pad_len = max_seq_len - real_len
    if pad_len > 0:
        ids.extend([pad_id] * pad_len)
        types.extend([TOKEN_TYPE_NORMAL] * pad_len)
        globals_.extend([0] * pad_len)

    attention = [1] * real_len + [0] * pad_len

    return InputBundle(
        input_ids=np.asarray(ids, dtype=np.int64),
        attention_mask=np.asarray(attention, dtype=np.int64),
        global_attention_mask=np.asarray(globals_, dtype=np.int64),
        token_type_ids=np.asarray(types, dtype=np.int64),
    )


# -----------------------------------------------------------------------------
# Bundle cache (keyed by code + encoder config)
# -----------------------------------------------------------------------------

class _BundleCache:
    def __init__(self, root: str | os.PathLike | None = None) -> None:
        if root is None:
            root = Path.home() / ".cache" / "codebert" / "longcoder"
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(code: str, cfg: str) -> str:
        return hashlib.sha256((cfg + "|" + code).encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.npz"

    def get(self, code: str, cfg: str) -> InputBundle | None:
        p = self._path(self._key(code, cfg))
        if not p.exists():
            return None
        try:
            with np.load(p) as z:
                return InputBundle(
                    input_ids=z["input_ids"],
                    attention_mask=z["attention_mask"],
                    global_attention_mask=z["global_attention_mask"],
                    token_type_ids=z["token_type_ids"],
                )
        except Exception:
            return None

    def put(self, code: str, cfg: str, b: InputBundle) -> None:
        p = self._path(self._key(code, cfg))
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp.npz")
        try:
            np.savez(
                tmp,
                input_ids=b.input_ids,
                attention_mask=b.attention_mask,
                global_attention_mask=b.global_attention_mask,
                token_type_ids=b.token_type_ids,
            )
            os.replace(tmp, p)
        except Exception:
            try:
                tmp.unlink()
            except OSError:
                pass


# -----------------------------------------------------------------------------
# Datasets (Parquet-backed)
# -----------------------------------------------------------------------------

class PointDataset(Dataset):
    def __init__(
        self,
        parquet_path: str | Path,
        tokenizer,
        max_seq_len: int = 1024,
        bridge_stride: int = 128,
        cache_dir: str | None = None,
        language_filter: str | None = None,
    ) -> None:
        super().__init__()
        table = pq.read_table(parquet_path)
        if language_filter is not None:
            import pyarrow.compute as pc
            mask = pc.equal(table["language"], language_filter)
            table = table.filter(mask)
        self.table = table
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.bridge_stride = bridge_stride
        self.cache = _BundleCache(cache_dir) if cache_dir is not None else _BundleCache()
        # v3 cache namespace: keyed on (code, language, max_seq_len, bridge_stride)
        # because per-language tree-sitter visitors emit different memory-token sets.
        self.cache_cfg = f"v3:{max_seq_len}:{bridge_stride}"
        self._codes = self.table.column("code").to_pylist()
        self._labels = self.table.column("label").to_pylist()
        self._ids = self.table.column("id").to_pylist()
        if "language" in self.table.column_names:
            self._langs = self.table.column("language").to_pylist()
        else:
            # Backwards-compat for pre-overhaul parquets (Python-only).
            self._langs = ["python"] * len(self._codes)

    def __len__(self) -> int:
        return len(self._codes)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        code = self._codes[i]
        lang = self._langs[i]
        label_idx = LABEL_TO_IDX[self._labels[i]]

        cache_key = f"{lang}|{code}"
        bundle = self.cache.get(cache_key, self.cache_cfg)
        if bundle is None:
            bundle = build_point_inputs(
                code,
                self.tokenizer,
                self.max_seq_len,
                self.bridge_stride,
                language=lang,
            )
            try:
                self.cache.put(cache_key, self.cache_cfg, bundle)
            except OSError:
                pass

        return {
            "input_ids": torch.from_numpy(bundle.input_ids),
            "attention_mask": torch.from_numpy(bundle.attention_mask),
            "global_attention_mask": torch.from_numpy(bundle.global_attention_mask),
            "token_type_ids": torch.from_numpy(bundle.token_type_ids),
            "labels": torch.tensor(label_idx, dtype=torch.long),
        }


def make_collator() -> Callable[[list[dict]], dict[str, torch.Tensor]]:
    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch], dim=0),
            "global_attention_mask": torch.stack([b["global_attention_mask"] for b in batch], dim=0),
            "token_type_ids": torch.stack([b["token_type_ids"] for b in batch], dim=0),
            "labels": torch.stack([b["labels"] for b in batch], dim=0),
        }
    return collate


# -----------------------------------------------------------------------------
# Inferred class-weight helper
# -----------------------------------------------------------------------------

def compute_class_weights(parquet_path: str | Path) -> list[float]:
    table = pq.read_table(parquet_path)
    labels = table.column("label").to_pylist()
    counts = np.zeros(NUM_POINT_LABELS, dtype=np.float64)
    for lab in labels:
        counts[LABEL_TO_IDX[lab]] += 1
    total = counts.sum()
    safe = np.where(counts > 0, counts, 1.0)
    w = np.sqrt(total / safe)
    w = w / w.mean()
    w[counts == 0] = 0.0
    return w.tolist()
